import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from supabase import create_client


# =========================
# App
# =========================
app = FastAPI(title="Closer CRM API")


# =========================
# Supabase client (usa variables de entorno en Render)
# Deben existir:
# - SUPABASE_URL
# - SUPABASE_SERVICE_ROLE_KEY
# =========================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    # Esto te ayuda a detectar rápido si Render no tiene bien las variables
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY env vars")

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


# =========================
# Schemas
# =========================
class ModelCreate(BaseModel):
    stage_name: str


# =========================
# Endpoints
# =========================
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models")
def list_models():
    try:
        res = supabase.table("models").select("*").order("created_at", desc=True).execute()
        return res.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models")
def create_model(payload: ModelCreate):
    try:
        res = supabase.table("models").insert(
            {"stage_name": payload.stage_name}
        ).execute()

        if not res.data:
            raise HTTPException(status_code=500, detail="Insert failed: empty response")

        return res.data[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
