import os
from typing import Optional, Literal, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from supabase import create_client


# =========================
# App
# =========================
app = FastAPI(title="Closer CRM API", version="0.1.0")


# =========================
# Supabase client
# =========================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY env vars")

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


# =========================
# Schemas - Models (simple)
# (si ya tienes el schema grande en tu backend, déjalo como lo tengas;
# esto no estorba a Platforms)
# =========================
class ModelCreate(BaseModel):
    stage_name: str


# =========================
# Schemas - Platforms
# =========================
CalcType = Literal["tokens", "usd_value", "usd_delta"]


class PlatformCreate(BaseModel):
    name: str = Field(..., min_length=1)
    calc_type: CalcType
    token_usd_rate: Optional[float] = None
    active: bool = True


class PlatformUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1)
    calc_type: Optional[CalcType] = None
    token_usd_rate: Optional[float] = None
    active: Optional[bool] = None


# =========================
# Helpers
# =========================
def _normalize_platform_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ajusta reglas:
    - Si calc_type == 'tokens' y token_usd_rate viene vacío -> error
    - Si calc_type != 'tokens' -> token_usd_rate se pone None
    """
    calc_type = data.get("calc_type")
    if calc_type == "tokens":
        if data.get("token_usd_rate") in (None, "", 0):
            raise HTTPException(status_code=422, detail="token_usd_rate is required when calc_type='tokens'")
    elif calc_type in ("usd_value", "usd_delta"):
        data["token_usd_rate"] = None
    return data


# =========================
# Endpoints
# =========================
@app.get("/health")
def health():
    return {"status": "ok"}


# -------- MODELS (lo que ya tenías) --------
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
        res = supabase.table("models").insert({"stage_name": payload.stage_name}).execute()
        if not res.data:
            raise HTTPException(status_code=500, detail="Insert failed: empty response")
        return res.data[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------- PLATFORMS (nuevo) --------
@app.get("/platforms")
def list_platforms():
    try:
        res = supabase.table("platforms").select("*").order("created_at", desc=True).execute()
        return res.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/platforms")
def create_platform(payload: PlatformCreate):
    try:
        data = payload.model_dump()
        data = _normalize_platform_payload(data)

        res = supabase.table("platforms").insert(data).execute()
        if not res.data:
            raise HTTPException(status_code=500, detail="Insert failed: empty response")
        return res.data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/platforms/{platform_id}")
def get_platform(platform_id: str):
    try:
        res = supabase.table("platforms").select("*").eq("id", platform_id).limit(1).execute()
        if not res.data:
            raise HTTPException(status_code=404, detail="Platform not found")
        return res.data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.patch("/platforms/{platform_id}")
def update_platform(platform_id: str, payload: PlatformUpdate):
    try:
        data = payload.model_dump(exclude_unset=True)

        # Si están cambiando calc_type, aplico reglas
        if "calc_type" in data:
            data = _normalize_platform_payload(data)

        res = supabase.table("platforms").update(data).eq("id", platform_id).execute()
        if not res.data:
            raise HTTPException(status_code=404, detail="Platform not found (update returned empty)")
        return res.data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/platforms/{platform_id}")
def deactivate_platform(platform_id: str):
    """
    Soft-delete: active = false
    """
    try:
        res = supabase.table("platforms").update({"active": False}).eq("id", platform_id).execute()
        if not res.data:
            raise HTTPException(status_code=404, detail="Platform not found")
        return res.data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
