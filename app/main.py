import os
from typing import Optional, Dict, Any, List
from datetime import date, datetime

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field, field_validator
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


def _parse_date_any(v) -> Optional[date]:
    """
    Acepta:
      - date
      - 'YYYY-MM-DD'
      - 'DD/MM/YYYY'
    Devuelve date o lanza HTTPException 400 si no puede parsear.
    """
    if v is None:
        return None
    if isinstance(v, date):
        return v

    if isinstance(v, str):
        s = v.strip()
        # YYYY-MM-DD
        try:
            return datetime.strptime(s, "%Y-%m-%d").date()
        except Exception:
            pass
        # DD/MM/YYYY
        try:
            return datetime.strptime(s, "%d/%m/%Y").date()
        except Exception:
            pass

    raise HTTPException(status_code=400, detail=f"Invalid date format: {v}. Use YYYY-MM-DD or DD/MM/YYYY")


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


def _get_room_or_404(room_id: str) -> Dict[str, Any]:
    res = supabase.table("rooms").select("id, active").eq("id", room_id).single().execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="Room not found")
    return res.data


def _get_shift_session_or_404(session_id: str) -> Dict[str, Any]:
    res = supabase.table("shift_sessions").select("*").eq("id", session_id).single().execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="Session not found")
    return res.data


def _validate_turn_type(turn_type: str) -> None:
    if turn_type not in ("day", "night"):
        raise HTTPException(status_code=400, detail="turn_type must be 'day' or 'night'")


def _validate_shift(shift: str) -> None:
    if shift not in ("morning", "afternoon", "night"):
        raise HTTPException(status_code=400, detail="shift must be 'morning', 'afternoon', or 'night'")


def _turn_type_from_shift(shift: str) -> str:
    return "night" if shift == "night" else "day"


def _default_shift_from_turn_type(turn_type: str) -> str:
    # si el frontend manda day/no shift, asumimos morning
    return "night" if turn_type == "night" else "morning"


def _find_assignment_room_for_model(model_id: str, assignment_date: date, shift: str) -> Optional[Dict[str, Any]]:
    """
    Busca la asignación activa de room para esa modelo + fecha + shift_slot.
    Si no existe, intenta fallback:
      - si shift=morning: busca afternoon
      - si shift=afternoon: busca morning
    """
    q = (
        supabase.table("room_assignments")
        .select("*")
        .eq("model_id", model_id)
        .eq("assignment_date", str(assignment_date))
        .eq("active", True)
    )

    # 1) intento exacto
    exact = q.eq("shift_slot", shift).execute().data
    if exact:
        return exact[0]

    # 2) fallback dentro del día
    if shift in ("morning", "afternoon"):
        alt = "afternoon" if shift == "morning" else "morning"
        alt_res = q.eq("shift_slot", alt).execute().data
        if alt_res:
            return alt_res[0]

    return None


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
        res = supabase.table("models").select("*").order("created_at", desc=True).execute()
        return res.data or []
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
    try:
        res = supabase.table("models").update({"active": False}).eq("id", model_id).execute()
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
        res = supabase.table("platforms").select("*").order("name", desc=False).execute()
        return res.data or []
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
        res = supabase.table("platforms").select("*").eq("id", platform_id).single().execute()
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

        res = supabase.table("platforms").update(data).eq("id", platform_id).execute()
        if not res.data:
            raise HTTPException(status_code=404, detail="Platform not found")
        return res.data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/platforms/{platform_id}")
def deactivate_platform(platform_id: str):
    try:
        res = supabase.table("platforms").update({"active": False}).eq("id", platform_id).execute()
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
def list_model_platforms(model_id: str, include_inactive: bool = Query(False)):
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
# SESSIONS (API contract)  -> backed by shift_sessions table
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
    try:
        q = supabase.table("shift_sessions").select("*").order("date", desc=True)

        if session_date:
            q = q.eq("date", str(session_date))
        if model_id:
            q = q.eq("model_id", model_id)

        if not include_inactive:
            q = q.neq("status", "deleted")

        rows = q.execute().data or []

        out = []
        for r in rows:
            shift = r.get("shift")
            out.append(
                {
                    "id": r.get("id"),
                    "model_id": r.get("model_id"),
                    "room_id": r.get("room_id"),
                    "shift": shift,
                    "session_date": r.get("date"),
                    "turn_type": _turn_type_from_shift(shift) if shift else None,
                    "checkin_at": r.get("checkin_at"),
                    "checkout_at": r.get("checkout_at"),
                    "notes": r.get("notes"),
                    "status": r.get("status"),
                    "active": (r.get("status") != "deleted"),
                }
            )
        return out

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sessions")
def create_session(payload: SessionCreate):
    try:
        _get_model_or_404(payload.model_id)
        _validate_turn_type(payload.turn_type)

        desired_shift = _default_shift_from_turn_type(payload.turn_type)

        asg = _find_assignment_room_for_model(payload.model_id, payload.session_date, desired_shift)
        if not asg or not asg.get("room_id"):
            raise HTTPException(
                status_code=400,
                detail=f"No hay asignación activa (room_assignments) para model_id={payload.model_id} date={payload.session_date} shift={desired_shift}",
            )

        room_id = asg["room_id"]
        _get_room_or_404(room_id)

        ins = {
            "model_id": payload.model_id,
            "room_id": room_id,
            "shift": desired_shift,
            "date": str(payload.session_date),
            "notes": payload.notes,
            "status": "open",
        }

        res = supabase.table("shift_sessions").insert(ins).execute()
        if not res.data:
            raise HTTPException(status_code=500, detail="Insert failed: empty response")

        r = res.data[0]
        return {
            "id": r.get("id"),
            "model_id": r.get("model_id"),
            "room_id": r.get("room_id"),
            "shift": r.get("shift"),
            "session_date": r.get("date"),
            "turn_type": _turn_type_from_shift(r.get("shift")),
            "checkin_at": r.get("checkin_at"),
            "checkout_at": r.get("checkout_at"),
            "notes": r.get("notes"),
            "status": r.get("status"),
            "active": (r.get("status") != "deleted"),
        }

    except HTTPException:
        raise
    except Exception as e:
        if _is_unique_violation(e):
            raise HTTPException(status_code=409, detail="Session already exists (unique constraint)")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    try:
        r = _get_shift_session_or_404(session_id)
        shift = r.get("shift")
        return {
            "id": r.get("id"),
            "model_id": r.get("model_id"),
            "room_id": r.get("room_id"),
            "shift": shift,
            "session_date": r.get("date"),
            "turn_type": _turn_type_from_shift(shift) if shift else None,
            "checkin_at": r.get("checkin_at"),
            "checkout_at": r.get("checkout_at"),
            "notes": r.get("notes"),
            "status": r.get("status"),
            "active": (r.get("status") != "deleted"),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.patch("/sessions/{session_id}")
def update_session(session_id: str, payload: SessionUpdate):
    try:
        existing = _get_shift_session_or_404(session_id)

        data_in = payload.model_dump()
        upd: Dict[str, Any] = {}

        if data_in.get("session_date") is not None:
            upd["date"] = str(data_in["session_date"])

        if data_in.get("turn_type") is not None:
            _validate_turn_type(str(data_in["turn_type"]))
            upd["shift"] = _default_shift_from_turn_type(str(data_in["turn_type"]))

        if data_in.get("notes") is not None:
            upd["notes"] = data_in["notes"]

        if data_in.get("active") is not None:
            upd["status"] = "deleted" if data_in["active"] is False else (existing.get("status") or "open")

        if not upd:
            raise HTTPException(status_code=400, detail="No fields to update")

        res = supabase.table("shift_sessions").update(upd).eq("id", session_id).execute()
        if not res.data:
            raise HTTPException(status_code=404, detail="Session not found")

        r = res.data[0]
        shift = r.get("shift")
        return {
            "id": r.get("id"),
            "model_id": r.get("model_id"),
            "room_id": r.get("room_id"),
            "shift": shift,
            "session_date": r.get("date"),
            "turn_type": _turn_type_from_shift(shift) if shift else None,
            "checkin_at": r.get("checkin_at"),
            "checkout_at": r.get("checkout_at"),
            "notes": r.get("notes"),
            "status": r.get("status"),
            "active": (r.get("status") != "deleted"),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/sessions/{session_id}")
def deactivate_session(session_id: str):
    try:
        _get_shift_session_or_404(session_id)
        res = supabase.table("shift_sessions").update({"status": "deleted"}).eq("id", session_id).execute()
        if not res.data:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"ok": True, "id": session_id, "active": False}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================================
# SESSION ENTRIES (session_platform_entries)
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
def list_session_entries(session_id: str, include_inactive: bool = False):
    try:
        _get_shift_session_or_404(session_id)

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
    allow_update_locked: bool = Query(False),
):
    try:
        _get_shift_session_or_404(session_id)
        _get_platform_or_404(payload.platform_id)

        existing = (
            supabase.table("session_platform_entries")
            .select("*")
            .eq("session_id", session_id)
            .eq("platform_id", payload.platform_id)
            .execute()
        ).data

        if existing:
            row = existing[0]
            if row.get("locked_in") is True and not allow_update_locked:
                raise HTTPException(status_code=400, detail="Entry is locked_in and cannot be updated")

            upd = {k: v for k, v in payload.model_dump().items() if v is not None}
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
            raise HTTPException(status_code=409, detail="Entry already exists for this session/platform")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/sessions/{session_id}/entries/{platform_id}")
def deactivate_session_entry(session_id: str, platform_id: str):
    try:
        _get_shift_session_or_404(session_id)
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
    # el frontend manda "date" con DD/MM/YYYY o YYYY-MM-DD
    assignment_date: Optional[date] = Field(default=None, validation_alias="date")

    # acepta {shift:"morning"} o {shift_slot:"morning"}
    shift_slot: str = Field(validation_alias="shift")

    model_id: str
    room_id: str
    active: bool = True

    @field_validator("assignment_date", mode="before")
    @classmethod
    def _v_assignment_date(cls, v):
        return _parse_date_any(v)


class RoomAssignmentUpdate(BaseModel):
    room_id: Optional[str] = None
    shift_slot: Optional[str] = Field(default=None, validation_alias="shift")
    active: Optional[bool] = None


@app.get("/assignments")
def list_assignments(
    # IMPORTANT: el frontend te manda date=01/02/2026 => lo recibimos aquí
    assignment_date: Optional[str] = Query(None, description="YYYY-MM-DD o DD/MM/YYYY", alias="date"),
    shift_slot: Optional[str] = Query(None, description="morning|afternoon|night", alias="shift"),
    room_id: Optional[str] = None,
    model_id: Optional[str] = None,
    include_inactive: bool = False,
):
    try:
        q = supabase.table("room_assignments").select("*").order("created_at", desc=False)

        if assignment_date:
            d = _parse_date_any(assignment_date)
            q = q.eq("assignment_date", str(d))

        if shift_slot:
            _validate_shift(shift_slot)
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
    try:
        _get_model_or_404(payload.model_id)
        _get_room_or_404(payload.room_id)
        _validate_shift(payload.shift_slot)

        data = payload.model_dump(by_alias=False)

        if data.get("assignment_date") is None:
            data["assignment_date"] = date.today()

        # guardar siempre en formato YYYY-MM-DD (como columna date en Postgres)
        data["assignment_date"] = str(data["assignment_date"])

        data["shift_slot"] = payload.shift_slot

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
            _validate_shift(str(data["shift_slot"]))
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
