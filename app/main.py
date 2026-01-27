import os
from typing import Optional, Dict, Any, List
from datetime import date

from fastapi import FastAPI, HTTPException, Query
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
# Helpers
# ==========================================================
def _is_unique_violation(e: Exception) -> bool:
    msg = str(e).lower()
    return ("duplicate" in msg) or ("unique" in msg) or ("violates unique constraint" in msg)


def _get_model_or_404(model_id: str) -> Dict[str, Any]:
    res = supabase.table("models").select("id, active").eq("id", model_id).single().execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="Model not found")
    return res.data


def _get_platform_or_404(platform_id: str) -> Dict[str, Any]:
    res = supabase.table("platforms").select("id, active").eq("id", platform_id).single().execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="Platform not found")
    return res.data


def _get_session_or_404(session_id: str) -> Dict[str, Any]:
    res = supabase.table("sessions").select("*").eq("id", session_id).single().execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="Session not found")
    return res.data


def _get_room_or_404(room_id: str) -> Dict[str, Any]:
    res = supabase.table("rooms").select("id, active").eq("id", room_id).single().execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="Room not found")
    return res.data


def _validate_turn_type(turn_type: str) -> None:
    if turn_type not in ("day", "night"):
        raise HTTPException(status_code=400, detail="turn_type must be 'day' or 'night'")


def _validate_shift_slot(shift_slot: str) -> None:
    if shift_slot not in ("morning", "afternoon", "night"):
        raise HTTPException(status_code=400, detail="shift_slot must be 'morning', 'afternoon', or 'night'")


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
        _validate_turn_type(payload.turn_type)
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
        if "turn_type" in data:
            _validate_turn_type(str(data["turn_type"]))
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


# ==========================================================
# MODEL ↔ PLATFORM (model_platforms)
# ==========================================================
class ModelPlatformCreate(BaseModel):
    platform_id: str
    pct_day: Optional[int] = None
    pct_night: Optional[int] = None
    bonus_threshold_usd: Optional[int] = None
    bonus_pct: Optional[int] = None


class ModelPlatformUpdate(BaseModel):
    pct_day: Optional[int] = None
    pct_night: Optional[int] = None
    bonus_threshold_usd: Optional[int] = None
    bonus_pct: Optional[int] = None
    active: Optional[bool] = None


@app.get("/models/{model_id}/platforms")
def list_model_platforms(
    model_id: str,
    include_inactive: bool = Query(False, description="If true, includes inactive relations"),
):
    """
    Lista plataformas asignadas a una modelo (incluye info de la plataforma para UI).
    """
    try:
        _get_model_or_404(model_id)

        q = supabase.table("model_platforms").select("*").eq("model_id", model_id)
        if not include_inactive:
            q = q.eq("active", True)

        rel: List[Dict[str, Any]] = q.order("created_at", desc=True).execute().data or []
        if not rel:
            return []

        platform_ids = list({r["platform_id"] for r in rel if r.get("platform_id")})
        plats: List[Dict[str, Any]] = (
            supabase.table("platforms")
            .select("id, name, calc_type, token_usd_rate, active")
            .in_("id", platform_ids)
            .execute()
            .data
            or []
        )
        plat_map = {p["id"]: p for p in plats}

        out = []
        for r in rel:
            p = plat_map.get(r["platform_id"], {})
            out.append(
                {
                    "model_id": r.get("model_id"),
                    "platform_id": r.get("platform_id"),
                    "platform_name": p.get("name"),
                    "calc_type": p.get("calc_type"),
                    "token_usd_rate": p.get("token_usd_rate"),
                    "platform_active": p.get("active"),
                    "pct_day": r.get("pct_day"),
                    "pct_night": r.get("pct_night"),
                    "bonus_threshold_usd": r.get("bonus_threshold_usd"),
                    "bonus_pct": r.get("bonus_pct"),
                    "active": r.get("active"),
                    "created_at": r.get("created_at"),
                }
            )

        out.sort(key=lambda x: (x["platform_name"] or "").lower())
        return out

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/{model_id}/platforms")
def add_platform_to_model(model_id: str, payload: ModelPlatformCreate):
    """
    Asigna una plataforma a una modelo (con overrides opcionales).
    """
    try:
        m = _get_model_or_404(model_id)
        if m.get("active") is False:
            raise HTTPException(status_code=400, detail="Model is inactive")

        p = _get_platform_or_404(payload.platform_id)
        if p.get("active") is False:
            raise HTTPException(status_code=400, detail="Platform is inactive")

        insert_data = {
            "model_id": model_id,
            "platform_id": payload.platform_id,
            "pct_day": payload.pct_day,
            "pct_night": payload.pct_night,
            "bonus_threshold_usd": payload.bonus_threshold_usd,
            "bonus_pct": payload.bonus_pct,
            "active": True,
        }

        res = supabase.table("model_platforms").insert(insert_data).execute()
        if not res.data:
            raise HTTPException(status_code=500, detail="Insert failed: empty response")
        return res.data[0]

    except HTTPException:
        raise
    except Exception as e:
        if _is_unique_violation(e):
            raise HTTPException(status_code=409, detail="Platform already assigned to this model")
        raise HTTPException(status_code=500, detail=str(e))


@app.patch("/models/{model_id}/platforms/{platform_id}")
def update_model_platform(model_id: str, platform_id: str, payload: ModelPlatformUpdate):
    """
    Edita overrides / activa-desactiva la relación modelo ↔ plataforma.
    """
    try:
        _get_model_or_404(model_id)
        _get_platform_or_404(platform_id)

        data = {k: v for k, v in payload.model_dump().items() if v is not None}
        if not data:
            raise HTTPException(status_code=400, detail="No fields to update")

        res = (
            supabase.table("model_platforms")
            .update(data)
            .eq("model_id", model_id)
            .eq("platform_id", platform_id)
            .execute()
        )

        if not res.data:
            raise HTTPException(status_code=404, detail="Relation not found")

        return res.data[0]

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/models/{model_id}/platforms/{platform_id}")
def deactivate_model_platform(model_id: str, platform_id: str):
    """
    Soft delete de la relación: active=false
    """
    try:
        _get_model_or_404(model_id)
        _get_platform_or_404(platform_id)

        res = (
            supabase.table("model_platforms")
            .update({"active": False})
            .eq("model_id", model_id)
            .eq("platform_id", platform_id)
            .execute()
        )

        if not res.data:
            raise HTTPException(status_code=404, detail="Relation not found")

        return {"ok": True, "model_id": model_id, "platform_id": platform_id, "active": False}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================================
# SESSIONS (sessions)
# ==========================================================
class SessionCreate(BaseModel):
    model_id: str
    session_date: date
    turn_type: str  # "day" | "night"
    notes: Optional[str] = None
    active: bool = True


class SessionUpdate(BaseModel):
    session_date: Optional[date] = None
    turn_type: Optional[str] = None
    notes: Optional[str] = None
    active: Optional[bool] = None


@app.get("/sessions")
def list_sessions(
    session_date: Optional[date] = Query(None, description="YYYY-MM-DD"),
    model_id: Optional[str] = None,
    include_inactive: bool = False,
):
    """
    Lista sesiones. Filtra por fecha y/o model_id.
    Por defecto devuelve active=true, a menos que include_inactive=true.
    """
    try:
        q = supabase.table("sessions").select("*").order("session_date", desc=True)

        if session_date:
            q = q.eq("session_date", str(session_date))
        if model_id:
            q = q.eq("model_id", model_id)
        if not include_inactive:
            q = q.eq("active", True)

        res = q.execute()
        return res.data or []
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sessions")
def create_session(payload: SessionCreate):
    """
    Crea una sesión (model_id + session_date + turn_type).
    En BD suele existir unique(model_id, session_date, turn_type).
    """
    try:
        _get_model_or_404(payload.model_id)
        _validate_turn_type(payload.turn_type)

        res = supabase.table("sessions").insert(payload.model_dump()).execute()
        if not res.data:
            raise HTTPException(status_code=500, detail="Insert failed: empty response")
        return res.data[0]
    except HTTPException:
        raise
    except Exception as e:
        if _is_unique_violation(e):
            raise HTTPException(status_code=409, detail="Session already exists for that model/date/turn")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    try:
        return _get_session_or_404(session_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.patch("/sessions/{session_id}")
def update_session(session_id: str, payload: SessionUpdate):
    try:
        _get_session_or_404(session_id)

        data = {k: v for k, v in payload.model_dump().items() if v is not None}
        if "turn_type" in data:
            _validate_turn_type(str(data["turn_type"]))
        if not data:
            raise HTTPException(status_code=400, detail="No fields to update")

        res = supabase.table("sessions").update(data).eq("id", session_id).execute()
        if not res.data:
            raise HTTPException(status_code=404, detail="Session not found")

        return res.data[0]
    except HTTPException:
        raise
    except Exception as e:
        if _is_unique_violation(e):
            raise HTTPException(status_code=409, detail="Session already exists for that model/date/turn")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/sessions/{session_id}")
def deactivate_session(session_id: str):
    """
    Soft delete: sessions.active=false
    """
    try:
        _get_session_or_404(session_id)

        res = supabase.table("sessions").update({"active": False}).eq("id", session_id).execute()
        if not res.data:
            raise HTTPException(status_code=404, detail="Session not found")

        return {"ok": True, "id": session_id, "active": False}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================================
# SESSION ENTRIES (session_platform_entries)  <-- columnas reales
# ==========================================================
class SessionEntryUpsert(BaseModel):
    platform_id: str

    value_in: Optional[float] = None
    value_out: Optional[float] = None

    tokens: Optional[float] = None
    usd_value: Optional[float] = None
    usd_earned: Optional[float] = None

    locked_in: Optional[bool] = None
    active: Optional[bool] = True


@app.get("/sessions/{session_id}/entries")
def list_session_entries(
    session_id: str,
    include_inactive: bool = False,
):
    """
    Lista entries de una sesión.
    """
    try:
        _get_session_or_404(session_id)

        q = supabase.table("session_platform_entries").select("*").eq("session_id", session_id)
        if not include_inactive:
            q = q.eq("active", True)

        res = q.order("created_at", desc=False).execute()
        return res.data or []
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sessions/{session_id}/entries")
def upsert_session_entry(
    session_id: str,
    payload: SessionEntryUpsert,
    allow_update_locked: bool = Query(False, description="If true, allows updating locked_in=true rows"),
):
    """
    Upsert por (session_id, platform_id).
    - Si ya existe, actualiza.
    - Si no existe, crea.

    Si la fila está locked_in=true y allow_update_locked=false, bloquea el update.
    """
    try:
        _get_session_or_404(session_id)
        _get_platform_or_404(payload.platform_id)

        # ¿Existe ya?
        existing = (
            supabase.table("session_platform_entries")
            .select("*")
            .eq("session_id", session_id)
            .eq("platform_id", payload.platform_id)
            .execute()
        ).data

        data = payload.model_dump()
        data["session_id"] = session_id

        if existing:
            row = existing[0]
            if row.get("locked_in") is True and not allow_update_locked:
                raise HTTPException(status_code=400, detail="Entry is locked_in and cannot be updated")

            # update solo campos no-None
            upd = {k: v for k, v in data.items() if v is not None}
            # nunca actualizar session_id/platform_id por accidente
            upd.pop("session_id", None)
            upd.pop("platform_id", None)

            if not upd:
                raise HTTPException(status_code=400, detail="No fields to update")

            res = (
                supabase.table("session_platform_entries")
                .update(upd)
                .eq("session_id", session_id)
                .eq("platform_id", payload.platform_id)
                .execute()
            )

            if not res.data:
                raise HTTPException(status_code=500, detail="Update failed: empty response")
            return res.data[0]

        # insert
        ins = {
            "session_id": session_id,
            "platform_id": payload.platform_id,
            "value_in": payload.value_in,
            "value_out": payload.value_out,
            "tokens": payload.tokens,
            "usd_value": payload.usd_value,
            "usd_earned": payload.usd_earned,
            "locked_in": payload.locked_in if payload.locked_in is not None else False,
            "active": True if payload.active is None else bool(payload.active),
        }

        res = supabase.table("session_platform_entries").insert(ins).execute()
        if not res.data:
            raise HTTPException(status_code=500, detail="Insert failed: empty response")
        return res.data[0]

    except HTTPException:
        raise
    except Exception as e:
        if _is_unique_violation(e):
            # si por alguna razón hay unique (session_id, platform_id), intentaron crear repetido
            raise HTTPException(status_code=409, detail="Entry already exists for this session/platform")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/sessions/{session_id}/entries/{platform_id}")
def deactivate_session_entry(session_id: str, platform_id: str):
    """
    Soft delete: entry.active=false (por session_id+platform_id)
    """
    try:
        _get_session_or_404(session_id)
        _get_platform_or_404(platform_id)

        res = (
            supabase.table("session_platform_entries")
            .update({"active": False})
            .eq("session_id", session_id)
            .eq("platform_id", platform_id)
            .execute()
        )

        if not res.data:
            raise HTTPException(status_code=404, detail="Entry not found")

        return {"ok": True, "session_id": session_id, "platform_id": platform_id, "active": False}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================================
# ROOMS (rooms)
# ==========================================================
class RoomCreate(BaseModel):
    name: str
    active: bool = True


class RoomUpdate(BaseModel):
    name: Optional[str] = None
    active: Optional[bool] = None


@app.get("/rooms")
def list_rooms(include_inactive: bool = Query(False)):
    try:
        q = supabase.table("rooms").select("*").order("name")
        if not include_inactive:
            q = q.eq("active", True)
        res = q.execute()
        return res.data or []
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rooms")
def create_room(payload: RoomCreate):
    try:
        res = supabase.table("rooms").insert(payload.model_dump()).execute()
        if not res.data:
            raise HTTPException(status_code=500, detail="Insert failed: empty response")
        return res.data[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.patch("/rooms/{room_id}")
def update_room(room_id: str, payload: RoomUpdate):
    try:
        _get_room_or_404(room_id)

        data = {k: v for k, v in payload.model_dump().items() if v is not None}
        if not data:
            raise HTTPException(status_code=400, detail="No fields to update")

        res = supabase.table("rooms").update(data).eq("id", room_id).execute()
        if not res.data:
            raise HTTPException(status_code=404, detail="Room not found")
        return res.data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/rooms/{room_id}")
def deactivate_room(room_id: str):
    try:
        _get_room_or_404(room_id)
        res = supabase.table("rooms").update({"active": False}).eq("id", room_id).execute()
        if not res.data:
            raise HTTPException(status_code=404, detail="Room not found")
        return {"ok": True, "room_id": room_id, "active": False}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================================
# ROOM ASSIGNMENTS (room_assignments)
# ==========================================================
class RoomAssignmentCreate(BaseModel):
    assignment_date: Optional[date] = None  # YYYY-MM-DD. Si viene None -> hoy
    shift_slot: str  # morning | afternoon | night
    model_id: str
    room_id: str
    active: bool = True


class RoomAssignmentUpdate(BaseModel):
    room_id: Optional[str] = None
    shift_slot: Optional[str] = None
    active: Optional[bool] = None


@app.get("/assignments")
def list_assignments(
    assignment_date: Optional[date] = Query(None, description="YYYY-MM-DD"),
    shift_slot: Optional[str] = Query(None, description="morning|afternoon|night"),
    room_id: Optional[str] = None,
    model_id: Optional[str] = None,
    include_inactive: bool = False,
):
    """
    Lista asignaciones de room por fecha y turno operativo.
    """
    try:
        q = supabase.table("room_assignments").select("*").order("created_at", desc=False)

        if assignment_date:
            q = q.eq("assignment_date", str(assignment_date))
        if shift_slot:
            _validate_shift_slot(shift_slot)
            q = q.eq("shift_slot", shift_slot)
        if room_id:
            q = q.eq("room_id", room_id)
        if model_id:
            q = q.eq("model_id", model_id)
        if not include_inactive:
            q = q.eq("active", True)

        res = q.execute()
        return res.data or []
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/assignments")
def create_assignment(payload: RoomAssignmentCreate):
    """
    Crea una asignación diaria para una modelo en un room, por turno operativo.
    Recomendado en DB: unique(assignment_date, shift_slot, model_id)
    """
    try:
        _get_model_or_404(payload.model_id)
        _get_room_or_404(payload.room_id)
        _validate_shift_slot(payload.shift_slot)

        data = payload.model_dump()
        if data.get("assignment_date") is None:
            data["assignment_date"] = date.today()

        res = supabase.table("room_assignments").insert(data).execute()
        if not res.data:
            raise HTTPException(status_code=500, detail="Insert failed: empty response")
        return res.data[0]
    except HTTPException:
        raise
    except Exception as e:
        if _is_unique_violation(e):
            raise HTTPException(status_code=409, detail="Assignment already exists for that date/shift/model")
        raise HTTPException(status_code=500, detail=str(e))


@app.patch("/assignments/{assignment_id}")
def update_assignment(assignment_id: str, payload: RoomAssignmentUpdate):
    try:
        # Validar existencia
        existing = (
            supabase.table("room_assignments")
            .select("*")
            .eq("id", assignment_id)
            .single()
            .execute()
        )
        if not existing.data:
            raise HTTPException(status_code=404, detail="Assignment not found")

        data = {k: v for k, v in payload.model_dump().items() if v is not None}
        if "shift_slot" in data:
            _validate_shift_slot(str(data["shift_slot"]))
        if "room_id" in data:
            _get_room_or_404(str(data["room_id"]))
        if not data:
            raise HTTPException(status_code=400, detail="No fields to update")

        res = supabase.table("room_assignments").update(data).eq("id", assignment_id).execute()
        if not res.data:
            raise HTTPException(status_code=404, detail="Assignment not found")
        return res.data[0]
    except HTTPException:
        raise
    except Exception as e:
        if _is_unique_violation(e):
            raise HTTPException(status_code=409, detail="Assignment already exists for that date/shift/model")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/assignments/{assignment_id}")
def deactivate_assignment(assignment_id: str):
    try:
        res = supabase.table("room_assignments").update({"active": False}).eq("id", assignment_id).execute()
        if not res.data:
            raise HTTPException(status_code=404, detail="Assignment not found")
        return {"ok": True, "assignment_id": assignment_id, "active": False}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
