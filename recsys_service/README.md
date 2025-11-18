# Recsys MVP (CF-only fallback)

> Режим: переранкер временно отключён. Сервис работает на item-based CF с фильтрами доступности. Оффлайн-метрики зафиксированы в `/metrics`.

## 1) Постановка (для клиента)
- Цель бизнеса: рост выручки от допродаж на главной странице.
- Блок рекомендаций: 3 слота на главной.
- Технические метрики оффлайн: Precision@k / NDCG@k / Coverage@k (оптимизируем P@3, NDCG@3).
- Онлайн: оценка uplift в A/B у клиента.

## 2) Данные
- `events.csv`: `timestamp, visitorid, event{view|addtocart|transaction}, itemid, transactionid`
- `item_properties_part*.csv`: свойства товаров (в т.ч. `categoryid`, `available`)
- `category_tree.csv`: иерархия категорий

## 3) Трансформации
- Дедуп и приведение времени; сплит по дате: до 2015-07-01 — train, после — test.
- Агрегаты по товару и паре user×item, intent-веса (view<cart<trx), затухание по времени.
- Фильтр доступности (`available=1`) и базовые сигналы (cf-score, популярность, категория).
- Валидация: разбиение по времени (не «заглядываем в будущее»).

## 4) Эксперименты (оффлайн)
Лучший оффлайн сетап (в `/metrics`):
- **P@3 = 5.91% · NDCG@3 = 7.69% · Coverage@3 = 5.19%**
- CF-конфиг (кратко): `SIM_TOPK=500`, `CAND_TOPK=700`, веса `{view:1, cart:10, trx:18}`, `LAMBDA=0.12`

> Замечание: это оффлайн-оценка. Реальный эффект подтверждается A/B-тестом.

## 5) API (FastAPI)

### `GET /health`
Проверка готовности и артефактов:
```json
{"status":"ok","artifacts_loaded":{"reranker":"missing (fallback to CF-only)","item_agg":[164206,11],"user_item_agg":"available (lazy)"}}

