from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn, optim
from tqdm import tqdm

from .metrics import binary_metrics_from_logits


@dataclass
class TrainConfig:
    epochs_head: int = 5          # сколько эпох учим только голову
    epochs_finetune: int = 10     # потом размораживаем верх и доучиваем

    lr_head: float = 1e-3
    lr_finetune: float = 1e-4

    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    use_amp: bool = True
    threshold: float = 0.5

    # class imbalance: pos_weight = Nneg/Npos (если нужно)
    pos_weight: float | None = None

    # разморозить последние блоки (layer4 + fc) на finetune
    unfreeze: bool = False
    unfreeze_layer4 : bool = False

def make_optimizer(model: nn.Module, lr: float, weight_decay: float) -> optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    return optim.AdamW(params, lr=lr, weight_decay=weight_decay)

def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    criterion,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None,
    cfg: TrainConfig,
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    agg = {"acc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    n_batches = 0

    for batch in tqdm(loader, desc="train", leave=False):
        x, y = batch[:2]  # на случай если val возвращает meta
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).view(-1)  # [B]

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(x).view(-1)  # [B]
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            if cfg.grad_clip is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x).view(-1)
            loss = criterion(logits, y)
            loss.backward()
            if cfg.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

        running_loss += loss.item()
        m = binary_metrics_from_logits(logits.detach(), y.detach(), thr=cfg.threshold)
        for k in agg:
            agg[k] += m[k]
        n_batches += 1

    out = {"loss": running_loss / max(n_batches, 1)}
    for k in agg:
        out[k] = agg[k] / max(n_batches, 1)
    return out

@torch.no_grad()
def eval_one_epoch(
    model: nn.Module,
    loader,
    criterion,
    device: torch.device,
    cfg: TrainConfig,
) -> Dict[str, float]:
    model.eval()
    running_loss = 0.0
    agg = {"acc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    n_batches = 0

    for batch in tqdm(loader, desc="val", leave=False):
        x, y = batch[:2]
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).view(-1)

        logits = model(x).view(-1)
        loss = criterion(logits, y)

        running_loss += loss.item()
        m = binary_metrics_from_logits(logits, y, thr=cfg.threshold)
        for k in agg:
            agg[k] += m[k]
        n_batches += 1

    out = {"loss": running_loss / max(n_batches, 1)}
    for k in agg:
        out[k] = agg[k] / max(n_batches, 1)
    return out

@torch.no_grad()
def eval_on_gold(model: nn.Module, loader, device, threshold: float = 0.5):
    model.eval()
    all_logits = []
    all_targets = []

    for batch in tqdm(loader, desc="eval", leave=False):
        if len(batch) == 3:
            x, y, _meta = batch
        else:
            x, y = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).view(-1)

        logits = model(x).view(-1)
        all_logits.append(logits.cpu())
        all_targets.append(y.cpu())

    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)

    # метрики
    m = binary_metrics_from_logits(logits, targets, thr=threshold)
    return m
