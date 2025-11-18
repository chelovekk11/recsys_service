# app/utils.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd


# --- числовая безопасность ---
def ensure_float32_no_inf(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Приводит выбранные столбцы к float32, заменяет inf/-inf на NaN и NaN -> 0.
    Не модифицирует исходный df.
    """
    out = df.copy()
    for c in cols:
        if c in out.columns:
            v = pd.to_numeric(out[c], errors="coerce").astype("float32")
            v = v.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            out[c] = v.values
    return out


# --- маленькие помощники ранжирования ---
def softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x)  # числовая стабильность
    ex = np.exp(x, dtype=np.float64)
    s = ex.sum()
    return (ex / s).astype(np.float32) if s > 0 else np.zeros_like(x, dtype=np.float32)


def topk_indices(scores: Sequence[float] | np.ndarray, k: int) -> np.ndarray:
    """
    Возвращает индексы топ-k по убыванию, без полного сортинга.
    """
    a = np.asarray(scores)
    k = max(0, min(k, a.size))
    if k == 0:
        return np.empty((0,), dtype=int)
    part = np.argpartition(-a, k - 1)[:k]
    return part[np.argsort(-a[part])]


def dedup_keep_order(seq: Iterable[int]) -> List[int]:
    """
    Удаляет дубли, сохраняя порядок.
    """
    seen = set()
    out: List[int] = []
    for x in seq:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


# --- работа с артефактами ---
def safe_read_parquet(path: str | Path) -> pd.DataFrame | None:
    """
    Пробует прочитать parquet. Возвращает None при ошибке вместо исключения.
    """
    p = Path(path)
    try:
        if p.exists():
            return pd.read_parquet(p)
    except Exception:
        return None
    return None


def load_metrics_json(path: str | Path, default: dict) -> dict:
    """
    Безопасно читает JSON с метриками. Возвращает default при ошибке.
    """
    p = Path(path)
    try:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default


def pop_rank_from_item_agg(item_agg: pd.DataFrame, topk: int = 50) -> list[int]:
    """
    Быстрый топ популярных товаров по trx_cnt (если колонка есть).
    """
    if isinstance(item_agg, pd.DataFrame) and "trx_cnt" in item_agg.columns:
        return (
            item_agg.sort_values("trx_cnt", ascending=False)["itemid"]
            .head(topk)
            .astype(int)
            .tolist()
        )
    return []


# --- воспроизводимость ---
def seed_everything(seed: int = 42) -> None:
    try:
        import random
        random.seed(seed)
    except Exception:
        pass
    try:
        np.random.seed(seed)
    except Exception:
        pass

