import os
import re
import hashlib
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

import torch
import torchvision
from torchvision.io import decode_image, write_jpeg
from torchvision.transforms import v2 as T


# --------- Настройки ----------
OUT_DIR = Path("dataset")
RAW_DIR = OUT_DIR / "raw"
PROCESSED_DIR = OUT_DIR / "processed"
META_OUT = OUT_DIR / "metadata_processed.csv"

GOLDEN_DIR = Path("golden")
GOLDEN_RAW_DIR = GOLDEN_DIR / "raw"
GOLDEN_PROCESSED_DIR = GOLDEN_DIR / "processed"
GOLDEN_META_OUT = GOLDEN_DIR / "metadata_processed.csv"
GOLDEN_CLEAN_META_OUT = GOLDEN_DIR / "metadata_processed_clean.csv"

UNLABELED_DIR = Path("unlabeled")
UNLABELED_RAW_DIR = UNLABELED_DIR / "raw"
UNLABELED_PROCESSED_DIR = UNLABELED_DIR / "processed"
UNLABELED_META_OUT = UNLABELED_DIR / "metadata_processed.csv"
UNLABELED_CLEAN_META_OUT = UNLABELED_DIR / "metadata_processed_clean.csv"

IMAGE_SIZE = 224
TIMEOUT = 20
RETRIES = 3
USER_AGENT = "dataset-prep/1.0"
JPEG_QUALITY = 92


# --------- Парсинг полей ----------
def parse_bool(v: str) -> int | None:
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ("true", "1", "yes", "y"):
        return 1
    if s in ("false", "0", "no", "n"):
        return 0
    return None


def parse_percent(v: str) -> float | None:
    if v is None:
        return None
    s = str(v).strip()
    m = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*%?\s*$", s)
    if not m:
        return None
    return float(m.group(1)) / 100.0


def stable_stem(row_id: str, url: str) -> str:
    h = hashlib.md5(url.encode("utf-8")).hexdigest()[:10]
    return f"id_{row_id}_{h}"


def ext_from_url(url: str) -> str:
    base = url.split("?")[0]
    ext = os.path.splitext(base)[1].lower()
    if ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
        return ext
    return ".jpg"


def download_bytes(url: str) -> bytes:
    headers = {"User-Agent": USER_AGENT}
    last_err = None
    for _ in range(RETRIES):
        try:
            r = requests.get(url, headers=headers, timeout=TIMEOUT)
            r.raise_for_status()
            return r.content
        except Exception as e:
            last_err = e
    raise last_err


# --------- Preprocessing ----------

preprocess = T.Compose([
    T.ToImage(),
    T.ToDtype(torch.uint8, scale=False),
    T.Resize(IMAGE_SIZE, antialias=True),
    T.CenterCrop(IMAGE_SIZE),
])

def ensure_3ch_uint8(img: torch.Tensor) -> torch.Tensor:
    """
    img: Tensor [C,H,W], dtype uint8
    """
    if img.ndim != 3:
        raise ValueError(f"Expected [C,H,W], got {tuple(img.shape)}")
    c, h, w = img.shape
    if c == 1:
        img = img.repeat(3, 1, 1)
    elif c == 4:
        img = img[:3]  # отбрасываем alpha
    elif c != 3:
        raise ValueError(f"Unsupported channels: {c}")
    if img.dtype != torch.uint8:
        img = img.to(torch.uint8)
    return img


def parse(dataset_tsv_path : str = "data.tsv"):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset_tsv_path, sep="\t", dtype=str)

    url_col = "INPUT:downloadUrl"
    id_col = "INPUT:id"
    y_col = "OUTPUT:is_conifer"
    conf_col = "CONFIDENCE:is_conifer"

    missing = [c for c in [url_col, id_col, y_col, conf_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Не найдены колонки в tsv: {missing}. Есть: {list(df.columns)}")

    records = []
    ok, fail = 0, 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Download + preprocess"):
        url = row[url_col]
        row_id = row[id_col]
        label = parse_bool(row[y_col])
        conf = parse_percent(row[conf_col])

        if not url or pd.isna(url):
            fail += 1
            records.append({
                "id": row_id,
                "url": url,
                "label": label,
                "confidence": conf,
                "raw_path": None,
                "processed_path": None,
                "status": "bad_url",
            })
            continue

        stem = stable_stem(row_id, url)
        raw_ext = ext_from_url(url)
        raw_path = RAW_DIR / f"{stem}{raw_ext}"
        processed_path = PROCESSED_DIR / f"{stem}.jpg"  # приводим всё к jpeg

        # 1) download (кэшируем)
        try:
            if not raw_path.exists():
                raw = download_bytes(url)
                raw_path.write_bytes(raw)
        except Exception as e:
            fail += 1
            records.append({
                "id": row_id,
                "url": url,
                "label": label,
                "confidence": conf,
                "raw_path": str(raw_path),
                "processed_path": None,
                "status": f"download_fail:{type(e).__name__}",
            })
            continue

        # 2) preprocess -> save jpeg
        try:
            if not processed_path.exists():
                # read_image умеет читать с диска, но не все форматы одинаково хорошо,
                # поэтому для универсальности читаем байты и decode_image
                raw_bytes = raw_path.read_bytes()
                img = decode_image(torch.frombuffer(raw_bytes, dtype=torch.uint8), mode=torchvision.io.ImageReadMode.UNCHANGED)  # noqa
                img = preprocess(img)
                img = ensure_3ch_uint8(img)
                write_jpeg(img, str(processed_path), quality=JPEG_QUALITY)

            ok += 1
            records.append({
                "id": row_id,
                "url": url,
                "label": label,
                "confidence": conf,
                "raw_path": str(raw_path),
                "processed_path": str(processed_path),
                "status": "ok",
            })

        except Exception as e:
            fail += 1
            records.append({
                "id": row_id,
                "url": url,
                "label": label,
                "confidence": conf,
                "raw_path": str(raw_path),
                "processed_path": None,
                "status": f"preprocess_fail:{type(e).__name__}",
            })

    meta = pd.DataFrame.from_records(records)
    meta["label"] = meta["label"].astype("Int64")
    meta["confidence"] = meta["confidence"].astype("Float64")
    meta.to_csv(META_OUT, index=False)

    clean = meta[(meta["status"] == "ok") & (meta["label"].notna())]
    clean_out = OUT_DIR / "metadata_processed_clean.csv"
    clean.to_csv(clean_out, index=False)

    print(f"\nDone. OK={ok}, FAIL={fail}")
    print(f"Metadata: {META_OUT.resolve()}")
    print(f"Clean metadata: {clean_out.resolve()} (rows={len(clean)})")
    print(f"Processed images: {PROCESSED_DIR.resolve()}")

def parse_golden(dataset_tsv_path : str = "golden_data.tsv"):
    GOLDEN_RAW_DIR.mkdir(parents=True, exist_ok=True)
    GOLDEN_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Читаем "пробелы/табы", с заголовком
    # Если у тебя иногда есть пустые строки — engine="python" устойчивее
    df = pd.read_csv(dataset_tsv_path, sep=r"\t", engine="python", dtype=str)

    # Нормализуем названия колонок (на случай странных пробелов)
    df.columns = [c.strip() for c in df.columns]

    required = ["OUTPUT:is_conifer", "INPUT:downloadUrl"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Не найдены колонки {missing}. Есть: {list(df.columns)}")

    records = []
    ok, fail = 0, 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Download + preprocess (ref)"):
        url = row["INPUT:downloadUrl"]
        y = parse_bool(row["OUTPUT:is_conifer"])

        if not url or pd.isna(url):
            fail += 1
            records.append({
                "id": idx,
                "url": url,
                "label": y,
                "raw_path": None,
                "processed_path": None,
                "status": "bad_url",
            })
            continue

        stem = stable_stem(str(idx), url)
        raw_path = GOLDEN_RAW_DIR / f"{stem}{ext_from_url(url)}"
        processed_path = GOLDEN_PROCESSED_DIR / f"{stem}.jpg"

        # 1) download (кэшируем)
        try:
            if not raw_path.exists():
                raw = download_bytes(url)
                raw_path.write_bytes(raw)
        except Exception as e:
            fail += 1
            records.append({
                "id": idx,
                "url": url,
                "label": y,
                "raw_path": str(raw_path),
                "processed_path": None,
                "status": f"download_fail:{type(e).__name__}",
            })
            continue

        # 2) preprocess -> save jpeg
        try:
            if not processed_path.exists():
                raw_bytes = raw_path.read_bytes()
                img = decode_image(
                    torch.frombuffer(raw_bytes, dtype=torch.uint8),
                    mode=torchvision.io.ImageReadMode.UNCHANGED,
                )
                img = preprocess(img)
                img = ensure_3ch_uint8(img)
                write_jpeg(img, str(processed_path), quality=JPEG_QUALITY)

            ok += 1
            records.append({
                "id": idx,
                "url": url,
                "label": y,
                "raw_path": str(raw_path),
                "processed_path": str(processed_path),
                "status": "ok",
            })
        except Exception as e:
            fail += 1
            records.append({
                "id": idx,
                "url": url,
                "label": y,
                "raw_path": str(raw_path),
                "processed_path": None,
                "status": f"preprocess_fail:{type(e).__name__}",
            })

    meta = pd.DataFrame.from_records(records)
    meta["label"] = pd.to_numeric(meta["label"], errors="coerce").astype("Int64")
    meta.to_csv(GOLDEN_META_OUT, index=False)

    clean = meta[(meta["status"] == "ok") & (meta["label"].notna())].reset_index(drop=True)
    clean.to_csv(GOLDEN_CLEAN_META_OUT, index=False)

    print(f"\nDone. OK={ok}, FAIL={fail}")
    print(f"Metadata: {GOLDEN_META_OUT.resolve()}")
    print(f"Clean metadata: {GOLDEN_CLEAN_META_OUT.resolve()} (rows={len(clean)})")
    print(f"Processed images: {GOLDEN_PROCESSED_DIR.resolve()}")

def parse_unlabeled(dataset_tsv_path: str = "unlabeled_data.tsv"):
    UNLABELED_DIR.mkdir(parents=True, exist_ok=True)
    UNLABELED_RAW_DIR.mkdir(parents=True, exist_ok=True)
    UNLABELED_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset_tsv_path, sep="\t", dtype=str)
    df.columns = [c.strip() for c in df.columns]

    url_col = "INPUT:downloadUrl"
    if url_col not in df.columns:
        raise ValueError(f"Не найдена колонка '{url_col}'. Есть: {list(df.columns)}")

    records = []
    ok, fail = 0, 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Download + preprocess (unlabeled)"):
        url = row[url_col]

        if not url or pd.isna(url):
            fail += 1
            records.append({
                "id": idx,
                "url": url,
                "raw_path": None,
                "processed_path": None,
                "status": "bad_url",
            })
            continue

        # стабильное имя: по индексу + хэш url
        stem = stable_stem(str(idx), url)  # переиспользуем твою функцию
        raw_path = UNLABELED_RAW_DIR / f"{stem}{ext_from_url(url)}"
        processed_path = UNLABELED_PROCESSED_DIR / f"{stem}.jpg"

        # download
        try:
            if not raw_path.exists():
                raw = download_bytes(url)
                raw_path.write_bytes(raw)
        except Exception as e:
            fail += 1
            records.append({
                "id": idx,
                "url": url,
                "raw_path": str(raw_path),
                "processed_path": None,
                "status": f"download_fail:{type(e).__name__}",
            })
            continue

        # preprocess
        try:
            if not processed_path.exists():
                raw_bytes = raw_path.read_bytes()
                img = decode_image(
                    torch.frombuffer(raw_bytes, dtype=torch.uint8),
                    mode=torchvision.io.ImageReadMode.UNCHANGED,
                )
                img = preprocess(img)
                img = ensure_3ch_uint8(img)
                write_jpeg(img, str(processed_path), quality=JPEG_QUALITY)

            ok += 1
            records.append({
                "id": idx,
                "url": url,
                "raw_path": str(raw_path),
                "processed_path": str(processed_path),
                "status": "ok",
            })
        except Exception as e:
            fail += 1
            records.append({
                "id": idx,
                "url": url,
                "raw_path": str(raw_path),
                "processed_path": None,
                "status": f"preprocess_fail:{type(e).__name__}",
            })

    meta = pd.DataFrame.from_records(records)
    meta.to_csv(UNLABELED_META_OUT, index=False)

    clean = meta[meta["status"] == "ok"].reset_index(drop=True)
    clean.to_csv(UNLABELED_CLEAN_META_OUT, index=False)

    print(f"\nDone. OK={ok}, FAIL={fail}")
    print(f"Metadata: {UNLABELED_META_OUT.resolve()}")
    print(f"Clean metadata: {UNLABELED_CLEAN_META_OUT.resolve()} (rows={len(clean)})")
    print(f"Processed images: {UNLABELED_PROCESSED_DIR.resolve()}")

if __name__ == "__main__":
    parse()
    parse_golden()
    parse_unlabeled()
