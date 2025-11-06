from fastapi import APIRouter
from models import PredictRequest, PredictResponse

router = APIRouter(prefix="", tags=["predict"])

@router.post("/predict", response_model=PredictResponse, summary="Run RAG prediction")
async def predict_endpoint(request: PredictRequest):
    """
    Executes the retrieval-augmented generation (RAG) pipeline for a given prompt and conversation history.
    """
    # Lógica real por implementar
    return {"response": "", "contexts": []}
