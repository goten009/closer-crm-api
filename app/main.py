import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from supabase import create_client


# =========================
# App
# =========================
app = FastAPI(title="Closer CRM API")


# =========================
# CORS (frontend en localhost:3000)
# =========================
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    # "https://tu-frontend.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# Supabase client
# =========================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY env vars")

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


# =========================
# Health
# =========================
@app.get("/health")
def health():
    return {"status": "ok"}


# ==========================================================
# MODELS
# ==========================================================
class ModelCreate(BaseModel):
    stage_name: str
    turn_type: str  # "day" or "night"
    legal_name: Optional[str] = None
    document: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[EmailStr] = None
    base_pct_day: int
    base_pct_night: int
    bonus_threshold_usd: int
    bonus_pct: int
    active: bool = True


class ModelUpdate(BaseModel):
    stage_name: Optional[str] = None
    turn_type: Optional[str] = None
    legal_name: Optional[str] = None
    document: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[EmailStr] = None
    base_pct_day: Optional[int] = None
    base_pct_night: Optional[int] = None
    bonus_threshold_usd: Optional[int] = None
    bonus_pct: Optional[int] = None
    active: Optional[bool] = None


@app.get("/models")
def list_models():
    try:
        res = (
            supabase.table("models")
            .select("*")
            .order("created_at", desc=True)
            .execute()
        )
        return res.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models")
def create_model(payload: ModelCreate):
    try:
        res = supabase.table("models").insert(payload.model_dump()).execute()
        if not res.data:
            raise HTTPException(status_code=500, detail="Insert failed: empty response")
        return res.data[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/{model_id}")
def get_model(model_id: str):
    try:
        res = supabase.table("models").select("*").eq("id", model_id).single().execute()
        if not res.data:
            raise HTTPException(status_code=404, detail="Model not found")
        return res.data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.patch("/models/{model_id}")
def update_model(model_id: str, payload: ModelUpdate):
    try:
        data = {k: v for k, v in payload.model_dump().items() if v is not None}
        if not data:
            raise HTTPException(status_code=400, detail="No fields to update")

        res = supabase.table("models").update(data).eq("id", model_id).execute()

        if not res.data:
            raise HTTPException(status_code=404, detail="Model not found")

        return res.data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/models/{model_id}")
def deactivate_model(model_id: str):
    """
    Soft delete: no borramos el registro, solo lo dejamos active=false.
    """
    try:
        res = (
            supabase.table("models")
            .update({"active": False})
            .eq("id", model_id)
            .execute()
        )

        if not res.data:
            raise HTTPException(status_code=404, detail="Model not found")

        return {"ok": True, "id": model_id, "active": False}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================================
# PLATFORMS
# ==========================================================
class PlatformCreate(BaseModel):
    name: str
    calc_type: str  # "tokens" | "usd_value" | "usd_delta"
    token_usd_rate: Optional[float] = None
    active: bool = True


class PlatformUpdate(BaseModel):
    name: Optional[str] = None
    calc_type: Optional[str] = None
    token_usd_rate: Optional[float] = None
    active: Optional[bool] = None


@app.get("/platforms")
def list_platforms():
    try:
        res = (
            supabase.table("platforms")
            .select("*")
            .order("name", desc=False)
            .execute()
        )
        return res.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/platforms")
def create_platform(payload: PlatformCreate):
    try:
        res = supabase.table("platforms").insert(payload.model_dump()).execute()
        if not res.data:
            raise HTTPException(status_code=500, detail="Insert failed: empty response")
        return res.data[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/platforms/{platform_id}")
def get_platform(platform_id: str):
    try:
        res = (
            supabase.table("platforms")
            .select("*")
            .eq("id", platform_id)
            .single()
            .execute()
        )
        if not res.data:
            raise HTTPException(status_code=404, detail="Platform not found")
        return res.data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.patch("/platforms/{platform_id}")
def update_platform(platform_id: str, payload: PlatformUpdate):
    try:
        data = {k: v for k, v in payload.model_dump().items() if v is not None}
        if not data:
            raise HTTPException(status_code=400, detail="No fields to update")

        res = (
            supabase.table("platforms")
            .update(data)
            .eq("id", platform_id)
            .execute()
        )

        if not res.data:
            raise HTTPException(status_code=404, detail="Platform not found")

        return res.data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/platforms/{platform_id}")
def deactivate_platform(platform_id: str):
    """
    Soft delete: no borramos el registro, solo lo dejamos active=false.
    """
    try:
        res = (
            supabase.table("platforms")
            .update({"active": False})
            .eq("id", platform_id)
            .execute()
        )

        if not res.data:
            raise HTTPException(status_code=404, detail="Platform not found")

        return {"ok": True, "id": platform_id, "active": False}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
