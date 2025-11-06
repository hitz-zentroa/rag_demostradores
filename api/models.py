from typing import List, Optional
from pydantic import BaseModel

# ----- /get_config -----
class Mode(BaseModel):
    language: str
    domain: str

class GetConfigResponse(BaseModel):
    modes: List[Mode]


# ----- /configure -----
class ConfigureRequest(BaseModel):
    prompt: str
    language: Optional[str] = None
    domain: Optional[str] = None

class ConfigureResponse(BaseModel):
    language: str
    domain: str


# ----- /predict -----
class HistoryItem(BaseModel):
    role: str
    content: str

class PredictRequest(BaseModel):
    history: List[HistoryItem]
    prompt: str
    domain: str
    language: str

class ContextItem(BaseModel):
    id: str
    title: str
    passage: str
    timestamp: Optional[str] = None
    url: Optional[str] = None
    metadata: Optional[dict] = None

class PredictResponse(BaseModel):
    response: str
    contexts: Optional[List[ContextItem]] = None
