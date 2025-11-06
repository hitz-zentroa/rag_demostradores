from fastapi import APIRouter
from models import GetConfigResponse

router = APIRouter(prefix="", tags=["config"])

@router.get("/get_config", response_model=GetConfigResponse, summary="Get available language-domain modes")
async def get_config():
    """
    Retrieve the list of `(language, domain)` combinations supported by this backend.
    """
    # Lógica real por implementar
    return {"modes": []}
