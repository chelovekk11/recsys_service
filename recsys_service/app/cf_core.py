# cf_core.py
# Простой item-based CF: считаем похожесть по ко-встречаемости с весами и затуханием.
from __future__ import annotations
from typing import Dict, Iterable, List, Tuple, Optional
import numpy as np
import pandas as pd

from .utils import time_decay, topk_from_scores, dedup_preserve_order, filter_available

# Вес события по умолчанию
DEFAULT_EVENT_WEIGHTS = {
    "view": 1.0,
    "addtocart": 10.0,
    "transaction": 18.0,
}

def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def build_item_sim_by_cooccur(
    events: pd.DataFrame,
    max_user_history: int = 30,
    lambda_decay: float = 0.12,
    event_weights: Optional[Dict[str, float]] = None,
    min_item_trx: int = 3,
    sim_topk_per_item: int = 500,
) -> Dict[int, List[Tuple[int, float]]]:
    """
    Строим «top-K похожих товаров для каждого товара».
    Упрощённо: для каждого пользователя берём его последние max_user_history событий,
    формируем пары (seed -> candidate) и накапливаем взвешенные очки с декей-коэффициентом.
    Далее для каждого seed оставляем topK кандидатов по косинусной мере.
    Возвращает словарь: item_id -> [(other_item_id, sim), ...]
    """
    if event_weights is None:
        event_weights = DEFAULT_EVENT_WEIGHTS

    # Берём только нужные колонки и сортируем по времени
    df = events[["visitorid", "itemid", "event", "datetime"]].copy()
    df = df.sort_values(["visitorid", "datetime"])

    # (1) посчитаем простую популярность по транзакциям (для фильтра хвоста)
    item_trx = (events.loc[events["event"] == "transaction", "itemid"]
                .value_counts())
    head_items = set(item_trx[item_trx >= int(min_item_trx)].index.astype(np.int64).tolist())

    # (2) идём по пользователям, берём последние max_user_history событий
    co_counts: Dict[int, Dict[int, float]] = {}
    self_pow: Dict[int, float] = {}

    def push_pair(a: int, b: int, w: float):
        if a == b:
            self_pow[a] = self_pow.get(a, 0.0) + w*w
            return
        co_counts.setdefault(a, {})
        co_counts[a][b] = co_counts[a].get(b, 0.0) + w

    for uid, g in df.groupby("visitorid", sort=False):
        g = g.tail(max_user_history)
        items = g["itemid"].astype(np.int64).to_numpy()
        evs = g["event"].astype(str).to_numpy()
        times = g["datetime"].to_numpy()

        # нормируем на «последнюю дату» юзера
        t_last = times[-1]
        # вектор весов каждого события
        w = np.empty(len(g), dtype=np.float64)
        for i in range(len(g)):
            days = (t_last - times[i]) / np.timedelta64(1, "D")
            w[i] = time_decay(days, lambda_decay) * event_weights.get(evs[i], 0.0)

        # накопим по всем парам (a,b) из корзины пользователя
        # сложность ~ O(n^2) на юзера, но n <= max_user_history (маленькое)
        for i in range(len(items)):
            ai = int(items[i])
            wi = float(w[i])
            # сам себе — для нормы
            self_pow[ai] = self_pow.get(ai, 0.0) + wi*wi
            for j in range(i+1, len(items)):
                bj = int(items[j])
                wj = float(w[j])
                # симметричный вклад
                val = wi * wj
                push_pair(ai, bj, val)
                push_pair(bj, ai, val)

    # (3) из ко-встречаемости -> косинусная похожесть и topK для каждого item
    sim_index: Dict[int, List[Tuple[int, float]]] = {}
    for a, neigh in co_counts.items():
        if a not in head_items:
            continue  # фильтр хвоста
        na = np.sqrt(self_pow.get(a, 1e-12))
        scores: Dict[int, float] = {}
        for b, ab in neigh.items():
            if b not in head_items:
                continue
            nb = np.sqrt(self_pow.get(b, 1e-12))
            sim = ab / (na * nb + 1e-12)
            if sim > 0:
                scores[b] = sim
        sim_index[a] = topk_from_scores(scores, sim_topk_per_item)
    return sim_index

def recommend_cf_for_user(
    recent_items: Iterable[int],
    sim_index: Dict[int, List[Tuple[int, float]]],
    k: int = 10,
    cand_topk_per_user: int = 300,
    exclude_items: Optional[Iterable[int]] = None,
    item_agg: Optional[pd.DataFrame] = None,
) -> List[Tuple[int, float]]:
    """
    Простой ранж без ML: суммируем похожести по сид-айтемам, убираем то, что уже было у юзера,
    фильтруем недоступное, берём top-k.
    """
    seeds = [int(x) for x in dedup_preserve_order(recent_items)]
    if not seeds:
        # холодный старт: можно вернуть «популярное» (если понадобилось) — тут оставим пусто
        return []

    exclude = set(int(x) for x in (exclude_items or []))
    exclude |= set(seeds)

    # агрегируем кандидатов
    score: Dict[int, float] = {}
    for s in seeds:
        for b, sim in sim_index.get(s, []):
            if b in exclude:
                continue
            score[b] = score.get(b, 0.0) + float(sim)

    # укоротим кандидатный пул, чтобы не тянуть всё дерево
    top_cand = topk_from_scores(score, cand_topk_per_user)
    cand_ids = [it for it, _ in top_cand]

    # фильтр на доступность (если есть артефакт)
    cand_ids = filter_available(cand_ids, item_agg)
    # восстановим скоры после фильтра
    score = {it: score.get(it, 0.0) for it in cand_ids}

    # финальный top-k
    return topk_from_scores(score, k)
