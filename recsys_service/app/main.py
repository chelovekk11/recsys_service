from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from .model import ServiceState, RecoService

app = FastAPI(title="Recsys MVP", version="1.0")

# Глобальное состояние и сервис
state = ServiceState()
svc = RecoService(state)


class PredictRequest(BaseModel):
    user_id: int = Field(..., ge=1, description="ID пользователя (целое, >= 1)")
    recent_items: list[int] = Field(default_factory=list, description="Недавно просмотренные/купленные айтемы")
    k: int = Field(default=3, ge=1, le=50, description="Сколько рекомендаций вернуть (1..50)")


@app.get("/health")
def health():
    """
    Простой healthcheck: показывает, какие артефакты подхватились.
    """
    return {"status": "ok", "artifacts_loaded": svc.status()}


@app.get("/metrics")
def metrics():
    """
    Оффлайн-метрики и конфиг (подхватываются из artifacts/final_choice.json, если есть).
    """
    return svc.metrics()


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Вернуть топ-k рекомендаций. Если нет переранкера — работает CF-fallback.
    """
    try:
        items, scores, meta = svc.recommend(req.user_id, req.recent_items, req.k)
        return {"user_id": req.user_id, "items": items, "scores": scores, "meta": meta}
    except Exception as e:
        # Чтобы клиент видел понятное сообщение об ошибке
        raise HTTPException(status_code=500, detail=str(e))


