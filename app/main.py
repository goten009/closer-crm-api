from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from datetime import datetime

app = FastAPI(title="Closer CRM API")

# --------- CORS (ESTO ES LO NUEVO Y CLAVE) ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # frontend local
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ---------------------------------------------------

# ----- "Base de datos" temporal en memoria -----
MODELS = []
NEXT_ID = 1

# ----- Esquemas -----
class ModelCreate(BaseModel):
    name: str

class ModelOut(BaseModel):
    id: int
    name: str
    created_at: str

# ----- Endpoints -----
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/models")
def list_models() -> List[ModelOut]:
    return MODELS

@app.post("/models")
def create_model(payload: ModelCreate) -> ModelOut:
    global NEXT_ID

    model = {
        "id": NEXT_ID,
        "name": payload.name,
        "created_at": datetime.utcnow().isoformat()
    }

    NEXT_ID += 1
    MODELS.append(model)

    return model
