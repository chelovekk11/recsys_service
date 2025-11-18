# model.py
# Минималистичный сервис рекомендаций: CF-fallback + аккуратная попытка реранка (если есть совместимая модель)

from __future__ import annotations
import os, json, time
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import joblib
except Exception:
    joblib = None

# локальные утилиты (см. app/utils.py)
from .utils import ensure_float32_no_inf

ART_DIR = Path(os.getenv("ART_DIR", "artifacts"))
# Ожидаемые (опционально) артефакты:
#  - reranker_hgb.joblib        (HistGradientBoosting / LogisticRegression и т.п.)
#  - item_agg_train.parquet     (агрегаты по товару: itemid, trx_cnt, categoryid, available_last и т.п.)
#  - user_item_agg_train.parquet (опционально, не грузим целиком)
#  - final_choice.json, top3_base.csv (для /metrics и демо)

# NB: набор фич, которые мы умеем собирать в онлайне
ONLINE_FEATURES = ["pop_blend", "cf_score", "trx_pop", "num_seeds", "same_cat"]

class ServiceState:
    def __init__(self):
        self.loaded = False
        self.info: Dict[str, Any] = {}
        self.started_ts = time.time()

class RecoService:
    def __init__(self, state: ServiceState):
        self.state = state
        self.model = None
        self.item_agg: Optional[pd.DataFrame] = None
        self.item_agg_idx: Optional[pd.DataFrame] = None  # indexed by itemid
        self.ui_agg = None  # заглушка, не грузим
        self.metrics_cache: Dict[str, Any] = {}
        self.reload()

    # -------- загрузка артефактов --------
    def reload(self):
        info: Dict[str, Any] = {}

        # 1) модель (опционально)
        model_path = ART_DIR / "reranker_hgb.joblib"
        if model_path.exists() and joblib is not None:
            try:
                self.model = joblib.load(model_path)
                info["reranker"] = "loaded"
            except Exception as e:
                info["reranker"] = f"failed: {e}"
                self.model = None
        else:
            info["reranker"] = "missing (fallback to CF-only)"
            self.model = None

        # 2) item агрегаты
        item_path = ART_DIR / "item_agg_train.parquet"
        if item_path.exists():
            try:
                df = pd.read_parquet(item_path)
                # приведение типов и индекса
                if "itemid" in df.columns:
                    df["itemid"] = df["itemid"].astype(np.int64, copy=False)
                    df = df.drop_duplicates(subset=["itemid"], keep="last")
                    self.item_agg = df.reset_index(drop=True)
                    self.item_agg_idx = df.set_index("itemid")
                    info["item_agg"] = self.item_agg.shape
                else:
                    self.item_agg = None
                    self.item_agg_idx = None
                    info["item_agg"] = "failed: no itemid column"
            except Exception as e:
                info["item_agg"] = f"failed: {e}"
                self.item_agg = None
                self.item_agg_idx = None
        else:
            info["item_agg"] = "missing"
            self.item_agg = None
            self.item_agg_idx = None

        # 3) user×item (не грузим — placeholder)
        ui_path = ART_DIR / "user_item_agg_train.parquet"
        if ui_path.exists():
            info["user_item_agg"] = "available (lazy)"
        else:
            info["user_item_agg"] = "missing"

        # 4) метрики
        fc = ART_DIR / "final_choice.json"
        if fc.exists():
            try:
                self.metrics_cache = json.loads(fc.read_text(encoding="utf-8"))
            except Exception:
                self.metrics_cache = {}
        else:
            self.metrics_cache = {}

        self.state.info = info
        self.state.loaded = True

    def status(self):
        return self.state.info

    # -------- CF-fallback кандидаты (категория + популярность) --------
    def _cf_candidates(self, seeds: List[int], topk: int = 50) -> List[int]:
        seeds = [int(x) for x in seeds if x is not None]
        # холодный старт — просто популярные
        if len(seeds) == 0:
            if isinstance(self.item_agg_idx, pd.DataFrame) and "trx_cnt" in self.item_agg_idx.columns:
                return self.item_agg_idx.sort_values("trx_cnt", ascending=False).head(topk).index.astype(int).tolist()
            return []

        cand: List[int] = []
        if isinstance(self.item_agg_idx, pd.DataFrame):
            # собрать категории из seed (если есть categoryid)
            seed_cats = set()
            if "categoryid" in self.item_agg_idx.columns:
                for s in seeds[:5]:
                    if s in self.item_agg_idx.index:
                        cat = self.item_agg_idx.at[s, "categoryid"]
                        seed_cats.add(cat)
            # 1) товары из тех же категорий (по популярности)
            if seed_cats and "trx_cnt" in self.item_agg_idx.columns:
                same = (self.item_agg_idx[self.item_agg_idx["categoryid"].isin(seed_cats)]
                        .sort_values("trx_cnt", ascending=False)
                        .index.astype(int).tolist())
                cand.extend(same)
            # 2) добиваем глобально популярными
            if "trx_cnt" in self.item_agg_idx.columns:
                pop = (self.item_agg_idx.sort_values("trx_cnt", ascending=False)
                       .index.astype(int).tolist())
                cand.extend(pop)

        # dedup + убрать сиды + урезать до topk
        out, seen = [], set(seeds)
        for it in cand:
            if it in seen:
                continue
            out.append(it)
            seen.add(it)
            if len(out) >= topk:
                break
        return out

    # -------- формирование онлайн-фич для кандидатов --------
    def _build_features(self, user_id: int, candidates: List[int], seeds: List[int]) -> pd.DataFrame:
        rows = []
        seed_set = set(int(x) for x in seeds)
        has_idx = isinstance(self.item_agg_idx, pd.DataFrame)
        for rank, it in enumerate(candidates, start=1):
            trx_pop = 0.0
            same_cat = 0
            if has_idx and it in self.item_agg_idx.index:
                row = self.item_agg_idx.loc[it]
                trx_pop = float(row.get("trx_cnt", 0.0))
                if "categoryid" in self.item_agg_idx.columns:
                    cat = row.get("categoryid", None)
                    if cat is not None:
                        # простое правило: есть ли среди seed товар с той же категорией
                        if len(seeds) and "categoryid" in self.item_agg_idx.columns:
                            seed_mask = [s for s in seeds if s in self.item_agg_idx.index]
                            same_cat = int(any(self.item_agg_idx.loc[s, "categoryid"] == cat for s in seed_mask))
            # суррогат cf_score — убывающий по позиции (без реальной сим-матрицы)
            cf_score = 1.0 / (1.0 + rank)
            pop_blend = 0.7 * cf_score + 0.3 * (trx_pop / (1.0 + trx_pop))
            rows.append({
                "user_id": int(user_id),
                "itemid": int(it),
                "num_seeds": int(len(seed_set)),
                "same_cat": int(same_cat),
                "trx_pop": float(trx_pop),
                "cf_score": float(cf_score),
                "pop_blend": float(pop_blend),
            })
        feats = pd.DataFrame(rows)
        # привести фичи в валидный float32 без NaN/inf
        feats = ensure_float32_no_inf(feats, ONLINE_FEATURES)
        return feats

    # -------- основной метод рекомендаций --------
    def recommend(self, user_id: int, seeds: List[int], k: int = 3):
        k = int(k)
        seeds = [int(x) for x in (seeds or [])]

        candidates = self._cf_candidates(seeds, topk=max(k * 10, 50))
        if len(candidates) == 0:
            return [], [], {"strategy": "empty"}

        feats = self._build_features(user_id, candidates, seeds)

        # Попытка реранка (если модель загружена и совместима по количеству фич)
        scores = None
        if self.model is not None:
            # определим какие фичи модель реально принимает
            want_cols = ONLINE_FEATURES.copy()
            X = feats[want_cols].to_numpy(dtype=np.float32, copy=False)

            n_model = getattr(self.model, "n_features_in_", None)
            if n_model is None:
                # некоторые пайплайны/модели не имеют атрибута — пробуем предсказать
                try:
                    if hasattr(self.model, "predict_proba"):
                        proba = self.model.predict_proba(X)[:, 1]
                    else:
                        proba = self.model.predict(X).astype(np.float32)
                    scores = np.asarray(proba, dtype=np.float32)
                except Exception:
                    scores = None
            else:
                # проверим размерность
                if X.shape[1] == int(n_model):
                    try:
                        if hasattr(self.model, "predict_proba"):
                            proba = self.model.predict_proba(X)[:, 1]
                        else:
                            proba = self.model.predict(X).astype(np.float32)
                        scores = np.asarray(proba, dtype=np.float32)
                    except Exception:
                        scores = None
                else:
                    # несовпадение схемы фич — безопасный откат
                    scores = None

        # Fallback: сортируем по pop_blend
        if scores is None:
            scores = feats["pop_blend"].to_numpy(dtype=np.float32, copy=False)

        order = np.argsort(-scores)[:k]
        items = feats.iloc[order]["itemid"].astype(int).tolist()
        sc = [float(scores[i]) for i in order]
        meta = {
            "strategy": "rerank" if self.model is not None and scores is not None else "cf_fallback",
            "seeds_used": seeds,
            "candidates_considered": int(len(candidates)),
        }
        return items, sc, meta

    # -------- метрики --------
    def metrics(self) -> Dict[str, Any]:
        # если есть финальный json — отдаём его как есть (он уже в нужном формате)
        if isinstance(self.metrics_cache, dict) and self.metrics_cache:
            return self.metrics_cache
        # запасной дефолт (как в нашей оффлайн-оценке)
        return {
            "base": {
                "P@3": 0.0591,
                "NDCG@3": 0.0769,
                "Cov@3": 5.19,
                "cfg": {
                    "W_view": 1.0, "W_cart": 10.0, "W_trx": 18.0,
                    "LAMBDA": 0.12, "MAX_USER_HISTORY": 30,
                    "SIM_TOPK": 500, "CAND_TOPK": 700,
                    "use_tail_filter": True, "min_user_events": 5,
                    "min_item_trx": 3, "neg_ratio": 5, "test_size_users": 0.5,
                    "hgb_max_depth": 6, "hgb_lr": 0.08,
                    "hgb_min_samples_leaf": 80, "hgb_l2": 1.0,
                    "seed": 42
                }
            }
        }
