import os
from uuid import UUID
from typing import Optional, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, EmailStr, Field
from supabase import create_client


# =========================
# App
# =========================
app = FastAPI(title="Closer CRM API")


# =========================
# Supabase client (Render env vars)
# - SUPABASE_URL
# - SUPABASE_SERVICE_ROLE_KEY
# =========================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY env vars")

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


# =========================
# Schemas
# =========================
TurnType = Literal["day", "night", "mixed"]

class ModelCreate(BaseModel):
    stage_name: str = Field(..., min_length=1)
    turn_type: TurnType

    base_pct_day: int = Field(..., ge=0, le=100)
    base_pct_night: int = Field(..., ge=0, le=100)

    bonus_threshold_usd: int = Field(..., ge=0)
    bonus_pct: int = Field(..., ge=0, le=100)

    active: bool = True

    # opcionales
    legal_name: Optional[str] = None
    document: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[EmailStr] = None


class ModelUpdate(BaseModel):
    # TODO opcional para poder editar lo que quieras sin obligar todo
    stage_name: Optional[str] = Field(None, min_length=1)
    turn_type: Optional[TurnType] = None

    base_pct_day: Optional[int] = Field(None, ge=0, le=100)
    base_pct_night: Optional[int] = Field(None, ge=0, le=100)

    bonus_threshold_usd: Optional[int] = Field(None, ge=0)
    bonus_pct: Optional[int] = Field(None, ge=0, le=100)

    active: Optional[bool] = None

    legal_name: Optional[str] = None
    document: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[EmailStr] = None


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


@app.get("/models/{model_id}")
def get_model(model_id: UUID):
    try:
        res = supabase.table("models").select("*").eq("id", str(model_id)).limit(1).execute()
        if not res.data:
            raise HTTPException(status_code=404, detail="Model not found")
        return res.data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models")
def create_model(payload: ModelCreate):
    try:
        insert_data = payload.model_dump(exclude_none=True)

        res = supabase.table("models").insert(insert_data).execute()
        if not res.data:
            raise HTTPException(status_code=500, detail="Insert failed: empty response")

        return res.data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.patch("/models/{model_id}")
def update_model(model_id: UUID, payload: ModelUpdate):
    try:
        update_data = payload.model_dump(exclude_none=True)
        if not update_data:
            raise HTTPException(status_code=422, detail="No fields provided to update")

        res = supabase.table("models").update(update_data).eq("id", str(model_id)).execute()

        if not res.data:
            raise HTTPException(status_code=404, detail="Model not found (nothing updated)")

        return res.data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/models/{model_id}")
def deactivate_model(model_id: UUID):
    """
    Recomendado (soft delete): NO borramos el registro.
    Solo lo desactivamos: active=false
    """
    try:
        res = supabase.table("models").update({"active": False}).eq("id", str(model_id)).execute()

        if not res.data:
            raise HTTPException(status_code=404, detail="Model not found (nothing updated)")

        return {"status": "ok", "id": str(model_id), "active": False}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
