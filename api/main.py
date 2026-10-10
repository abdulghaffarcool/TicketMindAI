
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.inference import TicketIntentPredictor
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="TicketMind AI",
    description="Customer support ticket intent classification API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

predictor = TicketIntentPredictor()


class TicketRequest(BaseModel):
    text: str = Field(
        min_length=1,
        max_length=5000,
        description="Customer support ticket text"
    )


@app.get("/health")
def health():
    return {"status": "healthy"}


# @app.post("/predict")
# def predict_ticket(request: TicketRequest):
#     return predictor.predict(request.text)

@app.post("/predict")
def predict_ticket(request: TicketRequest):
    result = predictor.predict(request.text)
    confidence = result["confidence"]

    if confidence >= 0.85:
        result["review_status"] = "auto_accepted"
    elif confidence >= 0.60:
        result["review_status"] = "needs_review"
    else:
        result["review_status"] = "manual_classification"

    return result