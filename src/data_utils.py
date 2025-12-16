from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.transforms import v2 as T
from torchvision.io import read_image, ImageReadMode


@dataclass
class DataConfig:
    image_size: int = 224
    num_workers: int = 4
    batch_size: int = 32
    pin_memory: bool = True

    # IMAGENET нормализация под предобученные torchvision модели
    imagenet_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    imagenet_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    path_col: str = "processed_path"
    label_col: str = "label"
    conf_col: str = "confidence"

def build_train_transforms(cfg: DataConfig) -> T.Compose:
    return T.Compose([
        T.ToImage(),
        T.RandomResizedCrop(cfg.image_size, scale=(0.6, 1.0), ratio=(0.75, 1.33), antialias=True),
        T.RandomHorizontalFlip(p=0.5),
        T.RandAugment(num_ops=2, magnitude=9),

        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=cfg.imagenet_mean, std=cfg.imagenet_std),
    ])


def build_val_transforms(cfg: DataConfig) -> T.Compose:
    return T.Compose([
        T.ToImage(),
        T.Resize(cfg.image_size, antialias=True),
        T.CenterCrop(cfg.image_size),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=cfg.imagenet_mean, std=cfg.imagenet_std),
    ])

def stratified_split(
    df: pd.DataFrame,
    label_col: str,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    df = df.copy()
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce").astype("Int64")
    df = df[df[label_col].notna()].reset_index(drop=True)

    train_idx = []
    val_idx = []

    for cls in sorted(df[label_col].unique()):
        cls_idx = df.index[df[label_col] == cls].to_numpy()
        rng.shuffle(cls_idx)
        n_val = int(round(len(cls_idx) * val_ratio))
        val_part = cls_idx[:n_val]
        train_part = cls_idx[n_val:]
        val_idx.append(val_part)
        train_idx.append(train_part)

    train_idx = np.concatenate(train_idx) if len(train_idx) else np.array([], dtype=int)
    val_idx = np.concatenate(val_idx) if len(val_idx) else np.array([], dtype=int)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)

class ConiferDataset(Dataset):
    """
    Возвращает:
      x: Tensor [3, H, W] float32 нормализованный (если transforms содержит Normalize)
      y: Tensor [] float32 (0/1) — удобно для BCEWithLogitsLoss
      meta: dict (опционально) — id/path/confidence, если нужно для дебага/логов
    """
    def __init__(
        self,
        df: pd.DataFrame,
        cfg: DataConfig,
        transforms: Optional[T.Compose] = None,
        return_meta: bool = False,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.cfg = cfg
        self.transforms = transforms
        self.return_meta = return_meta

        for col in (cfg.path_col, cfg.label_col):
            if col not in self.df.columns:
                raise ValueError(f"Column '{col}' not found. Available: {list(self.df.columns)}")

        # убедимся, что label — числовой 0/1
        self.df[cfg.label_col] = pd.to_numeric(self.df[cfg.label_col], errors="coerce").astype("Int64")

        # выкинем строки без лейбла или пути
        self.df = self.df[self.df[cfg.label_col].notna() & self.df[cfg.path_col].notna()].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def _load_image_rgb(self, path: str) -> torch.Tensor:
        img = read_image(path, mode=ImageReadMode.RGB)  # гарантирует 3 канала
        return img

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = str(row[self.cfg.path_col])
        y_int = int(row[self.cfg.label_col])

        img = self._load_image_rgb(path)

        if self.transforms is not None:
            img = self.transforms(img)

        y = torch.tensor(y_int, dtype=torch.float32)

        if not self.return_meta:
            return img, y

        meta = {
            "path": path,
            "label": y_int,
        }
        if self.cfg.conf_col in self.df.columns:
            meta["confidence"] = (None if pd.isna(row[self.cfg.conf_col]) else float(row[self.cfg.conf_col]))
        return img, y, meta

class UnlabeledDataset(Dataset):
    def __init__(self, df: pd.DataFrame, path_col: str = "processed_path", transforms=None):
        self.df = df.reset_index(drop=True)
        self.path_col = path_col
        self.transforms = transforms

        if path_col not in self.df.columns:
            raise ValueError(f"Column '{path_col}' not found. Available: {list(self.df.columns)}")

        # оставим только строки с путём
        self.df = self.df[self.df[path_col].notna()].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = str(row[self.path_col])
        img = read_image(path, mode=ImageReadMode.RGB)  # uint8 [3,H,W]

        if self.transforms is not None:
            img = self.transforms(img)

        # возвращаем path/id чтобы потом смёржить
        item_id = row["id"] if "id" in row.index else idx
        url = row["url"] if "url" in row.index else None
        return img, item_id, path, url

def _make_weighted_sampler(
    df: pd.DataFrame,
    label_col: str,
    conf_col: Optional[str] = None,
    use_confidence: bool = True,
    conf_floor: float = 0.05,
) -> WeightedRandomSampler:
    """
    Вес = (1 / freq[class]) * confidence_weight
    confidence_weight = max(conf, conf_floor) если есть confidence
    """
    labels = pd.to_numeric(df[label_col], errors="coerce").astype("Int64").to_numpy()
    labels = labels.astype(int)

    # частоты классов
    unique, counts = np.unique(labels, return_counts=True)
    freq = {int(u): int(c) for u, c in zip(unique, counts)}
    inv_freq = {k: 1.0 / v for k, v in freq.items()}

    base_w = np.array([inv_freq[int(y)] for y in labels], dtype=np.float64)

    if use_confidence and conf_col is not None and conf_col in df.columns:
        conf = pd.to_numeric(df[conf_col], errors="coerce").to_numpy(dtype=np.float64)
        conf = np.where(np.isnan(conf), 1.0, conf)  # если нет — пусть будет 1
        conf = np.clip(conf, conf_floor, 1.0)
        w = base_w * conf
    else:
        w = base_w

    w_t = torch.as_tensor(w, dtype=torch.double)
    return WeightedRandomSampler(weights=w_t, num_samples=len(w_t), replacement=True)

def build_train_loader(
    csv_path: str | Path,
    cfg: DataConfig,
    val_ratio: float = 0.2,
    seed: int = 42,
    use_weighted_sampler: bool = True,
    return_meta_in_val: bool = False,
):
    df = pd.read_csv(csv_path)

    train_df, val_df = stratified_split(df, label_col=cfg.label_col, val_ratio=val_ratio, seed=seed)

    train_ds = ConiferDataset(train_df, cfg, transforms=build_train_transforms(cfg), return_meta=False)
    val_ds = ConiferDataset(val_df, cfg, transforms=build_val_transforms(cfg), return_meta=return_meta_in_val)

    if use_weighted_sampler:
        sampler = _make_weighted_sampler(
            train_df,
            label_col=cfg.label_col,
            conf_col=cfg.conf_col,
            use_confidence=True,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            sampler=sampler,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
        )
        shuffle = False
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
        )
        shuffle = True

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    return train_loader, val_loader, train_df, val_df, {"train_shuffle": shuffle, "use_weighted_sampler": use_weighted_sampler}


def build_test_loader(
    csv_path: str,
    cfg: DataConfig,
    batch_size: int | None = None,
    num_workers: int | None = None,
    return_meta: bool = True,
):
    df = pd.read_csv(csv_path)

    ds = ConiferDataset(
        df=df,
        cfg=cfg,
        transforms=build_val_transforms(cfg),
        return_meta=return_meta,
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size or cfg.batch_size,
        shuffle=False,
        num_workers=num_workers if num_workers is not None else cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    return loader, df

def build_unlabeled_loader(
    csv_path: str,
    cfg: DataConfig,
    batch_size: int | None = None,
    num_workers: int | None = None
):
    df = pd.read_csv(csv_path)
    ds = UnlabeledDataset(df, path_col="processed_path", transforms=build_val_transforms(cfg))
    loader = DataLoader(
        ds,
        batch_size=batch_size or cfg.batch_size,
        shuffle=False,
        num_workers=num_workers if num_workers is not None else cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    return loader, df
