import os
from typing import Optional, Dict, Any, List
from datetime import date, datetime

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, EmailStr, Field, field_validator, AliasChoices
from supabase import create_client


# =========================
# App
# =========================
app = FastAPI(title="Closer CRM API", version="0.1.0")


# =========================
# CORS
# =========================
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    # cuando subas el frontend, agrega aquí tu dominio Vercel/Render:
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
# Validation error handler
# =========================
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    parts = []
    for err in exc.errors():
        loc = ".".join([str(x) for x in err.get("loc", [])])
        msg = err.get("msg", "Invalid value")
        parts.append(f"{loc}: {msg}")
    human = " | ".join(parts) if parts else "Validation error"
    return JSONResponse(status_code=422, content={"detail": human})


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
    """
    if v is None:
        return None
    if isinstance(v, date):
        return v

    if isinstance(v, str):
        s = v.strip()
        try:
            return datetime.strptime(s, "%Y-%m-%d").date()
        except Exception:
            pass
        try:
            return datetime.strptime(s, "%d/%m/%Y").date()
        except Exception:
            pass

    raise HTTPException(status_code=400, detail=f"Invalid date format: {v}. Use YYYY-MM-DD or DD/MM/YYYY")


def _validate_turn_type(turn_type: str) -> None:
    if turn_type not in ("day", "night"):
        raise HTTPException(status_code=400, detail="turn_type must be 'day' or 'night'")


def _validate_shift(shift: str) -> None:
    if shift not in ("morning", "afternoon", "night"):
        raise HTTPException(status_code=400, detail="shift must be 'morning', 'afternoon', or 'night'")


def _turn_type_from_shift(shift: Optional[str]) -> Optional[str]:
    if not shift:
        return None
    return "night" if shift == "night" else "day"


def _default_shift_from_turn_type(turn_type: str) -> str:
    return "night" if turn_type == "night" else "morning"


def _get_model_or_404(model_id: str) -> Dict[str, Any]:
    res = supabase.table("models").select("id, active").eq("id", model_id).single().execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="Model not found")
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


def _find_assignment_room_for_model(model_id: str, assignment_date: date, shift: str) -> Optional[Dict[str, Any]]:
    q = (
        supabase.table("room_assignments")
        .select("*")
        .eq("model_id", model_id)
        .eq("assignment_date", str(assignment_date))
        .eq("active", True)
    )

    exact = q.eq("shift_slot", shift).execute().data
    if exact:
        return exact[0]

    if shift in ("morning", "afternoon"):
        alt = "afternoon" if shift == "morning" else "morning"
        alt_res = q.eq("shift_slot", alt).execute().data
        if alt_res:
            return alt_res[0]

    return None


def _map_shift_session_row(r: Dict[str, Any]) -> Dict[str, Any]:
    """
    Devuelve el shape que consume el frontend:
    Session = {id, model_id, session_date, turn_type, notes, active}
    """
    shift = r.get("shift")
    status = r.get("status")
    return {
        "id": r.get("id"),
        "model_id": r.get("model_id"),
        "session_date": r.get("date"),
        "turn_type": _turn_type_from_shift(shift),
        "notes": r.get("notes"),
        "active": (status != "deleted"),
    }


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
# MODELS -> PLATFORMS (LO QUE TE FALTA PARA EL FRONTEND)
# ==========================================================
@app.get("/models/{model_id}/platforms")
def get_model_platforms(model_id: str):
    """
    Devuelve un array con fields que consume SessionPage:
    ModelPlatform = {
      model_id, platform_id, platform_name, calc_type, token_usd_rate, platform_active,
      pct_day, pct_night, bonus_threshold_usd, bonus_pct, active
    }
    """
    try:
        _get_model_or_404(model_id)

        # model_platforms + platforms
        # OJO: este select asume que model_platforms.platform_id FK -> platforms.id
        res = (
            supabase.table("model_platforms")
            .select(
                "model_id, platform_id, pct_day, pct_night, bonus_threshold_usd, bonus_pct, active, "
                "platforms:platform_id (id, name, calc_type, token_usd_rate, active)"
            )
            .eq("model_id", model_id)
            .execute()
        )

        rows = res.data or []
        out = []
        for r in rows:
            p = r.get("platforms") or {}
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
                }
            )

        return out
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
# SESSIONS (backed by shift_sessions)
# ==========================================================
class SessionCreate(BaseModel):
    model_id: str
    session_date: date
    turn_type: str  # "day" | "night"
    notes: Optional[str] = None
    active: bool = True


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
        return [_map_shift_session_row(r) for r in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ✅ ESTA ES LA RUTA QUE TE FALTABA (FRONTEND LA LLAMA)
@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    try:
        r = _get_shift_session_or_404(session_id)
        return _map_shift_session_row(r)
    except HTTPException:
        raise
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
                detail=(
                    f"No hay asignación activa (room_assignments) para "
                    f"model_id={payload.model_id} date={payload.session_date} shift={desired_shift}"
                ),
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

        return _map_shift_session_row(res.data[0])

    except HTTPException:
        raise
    except Exception as e:
        if _is_unique_violation(e):
            raise HTTPException(status_code=409, detail="Session already exists (unique constraint)")
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================================
# SESSION ENTRIES (session_platform_entries)
# ==========================================================
# IMPORTANT:
# - Este nombre debe existir en tu DB. En tu Supabase se ve como "session_platform_en..."
#   Si tu tabla se llama distinto, cambia SESSION_ENTRIES_TABLE.
SESSION_ENTRIES_TABLE = os.getenv("SESSION_ENTRIES_TABLE", "session_platform_entries")


class EntryUpsert(BaseModel):
    platform_id: str

    value_in: Optional[float] = None
    value_out: Optional[float] = None
    tokens: Optional[float] = None
    usd_value: Optional[float] = None
    usd_earned: Optional[float] = None

    locked_in: Optional[bool] = False
    active: Optional[bool] = True


@app.get("/sessions/{session_id}/entries")
def list_session_entries(session_id: str):
    try:
        _get_shift_session_or_404(session_id)

        res = (
            supabase.table(SESSION_ENTRIES_TABLE)
            .select("*")
            .eq("session_id", session_id)
            .execute()
        )
        rows = res.data or []
        return rows
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sessions/{session_id}/entries")
def upsert_session_entry(
    session_id: str,
    payload: EntryUpsert,
    allow_update_locked: bool = Query(False),
):
    """
    Upsert por (session_id, platform_id).
    Si existe y locked_in=true, bloquea salvo allow_update_locked=true.
    """
    try:
        _get_shift_session_or_404(session_id)

        # buscar existente
        existing = (
            supabase.table(SESSION_ENTRIES_TABLE)
            .select("*")
            .eq("session_id", session_id)
            .eq("platform_id", payload.platform_id)
            .maybe_single()
            .execute()
        )

        ex = existing.data

        if ex and ex.get("locked_in") and not allow_update_locked:
            raise HTTPException(status_code=409, detail="Entry is locked (locked_in=true)")

        data = payload.model_dump()
        data["session_id"] = session_id

        # si existe -> update, si no -> insert
        if ex:
            res = (
                supabase.table(SESSION_ENTRIES_TABLE)
                .update(data)
                .eq("id", ex.get("id"))
                .execute()
            )
            if not res.data:
                raise HTTPException(status_code=500, detail="Update failed: empty response")
            return res.data[0]

        res = supabase.table(SESSION_ENTRIES_TABLE).insert(data).execute()
        if not res.data:
            raise HTTPException(status_code=500, detail="Insert failed: empty response")
        return res.data[0]

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================================
# ROOMS
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
# ROOM ASSIGNMENTS
# ==========================================================
class RoomAssignmentCreate(BaseModel):
    assignment_date: Optional[date] = Field(
        default=None,
        validation_alias=AliasChoices("date", "assignment_date"),
    )
    shift_slot: str = Field(
        validation_alias=AliasChoices("shift", "shift_slot"),
    )
    model_id: str
    room_id: str
    active: bool = True

    @field_validator("assignment_date", mode="before")
    @classmethod
    def _v_assignment_date(cls, v):
        return _parse_date_any(v)


class RoomAssignmentUpdate(BaseModel):
    room_id: Optional[str] = None
    shift_slot: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("shift", "shift_slot"),
    )
    active: Optional[bool] = None


@app.get("/assignments")
def list_assignments(
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
        res = (
            supabase.table("room_assignments")
            .update({"active": False})
            .eq("id", assignment_id)
            .execute()
        )
        if not res.data:
            raise HTTPException(status_code=404, detail="Assignment not found")
        return {"ok": True, "assignment_id": assignment_id, "active": False}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
