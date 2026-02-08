import os
from typing import Optional, Dict, Any, List
from datetime import date, datetime
from uuid import UUID

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
def _load_origins() -> List[str]:
    raw = os.getenv("CORS_ORIGINS", "").strip()
    if not raw:
        return ["http://localhost:3000", "http://127.0.0.1:3000"]
    return [x.strip() for x in raw.split(",") if x.strip()]


ALLOWED_ORIGINS = _load_origins()

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


def _require_uuid(value: str, label: str) -> str:
    """
    Evita que Supabase reviente con:
      invalid input syntax for type uuid: "undefined"
    """
    if value is None:
        raise HTTPException(status_code=400, detail=f"{label} is required")
    if isinstance(value, str) and value.strip().lower() in ("undefined", "null", ""):
        raise HTTPException(status_code=400, detail=f"{label} is invalid")
    try:
        UUID(str(value))
        return str(value)
    except Exception:
        raise HTTPException(status_code=400, detail=f"{label} must be a valid uuid")


def _sb_execute(builder, label: str):
    """
    Ejecuta una query de supabase-py de forma segura.
    - Si Supabase devuelve None, lo convertimos en 502 con contexto
    - Si hay error, lo devolvemos con detalle
    """
    try:
        res = builder.execute()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Supabase call failed ({label}): {str(e)}")

    if res is None:
        raise HTTPException(status_code=502, detail=f"Supabase returned None ({label})")

    err = getattr(res, "error", None)
    if err:
        msg = getattr(err, "message", None) or str(err)
        raise HTTPException(status_code=400, detail=f"Supabase error ({label}): {msg}")

    return res


def _get_model_or_404(model_id: str) -> Dict[str, Any]:
    model_id = _require_uuid(model_id, "model_id")
    res = _sb_execute(
        supabase.table("models").select("id, active").eq("id", model_id).single(),
        "get model",
    )
    if not res.data:
        raise HTTPException(status_code=404, detail="Model not found")
    return res.data


def _get_platform_or_404(platform_id: str) -> Dict[str, Any]:
    platform_id = _require_uuid(platform_id, "platform_id")
    res = _sb_execute(
        supabase.table("platforms").select("id, active").eq("id", platform_id).single(),
        "get platform",
    )
    if not res.data:
        raise HTTPException(status_code=404, detail="Platform not found")
    return res.data


def _get_room_or_404(room_id: str) -> Dict[str, Any]:
    room_id = _require_uuid(room_id, "room_id")
    res = _sb_execute(
        supabase.table("rooms").select("id, active").eq("id", room_id).single(),
        "get room",
    )
    if not res.data:
        raise HTTPException(status_code=404, detail="Room not found")
    return res.data


def _get_shift_session_or_404(session_id: str) -> Dict[str, Any]:
    session_id = _require_uuid(session_id, "session_id")
    res = _sb_execute(
        supabase.table("shift_sessions").select("*").eq("id", session_id).single(),
        "get shift_session",
    )
    if not res.data:
        raise HTTPException(status_code=404, detail="Session not found")
    return res.data


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


def _find_assignment_room_for_model(model_id: str, assignment_date: date, shift: str) -> Optional[Dict[str, Any]]:
    model_id = _require_uuid(model_id, "model_id")
    q = (
        supabase.table("room_assignments")
        .select("*")
        .eq("model_id", model_id)
        .eq("assignment_date", str(assignment_date))
        .eq("active", True)
    )

    exact = _sb_execute(q.eq("shift_slot", shift), "find assignment exact").data
    if exact:
        return exact[0]

    if shift in ("morning", "afternoon"):
        alt = "afternoon" if shift == "morning" else "morning"
        alt_res = _sb_execute(q.eq("shift_slot", alt), "find assignment alt").data
        if alt_res:
            return alt_res[0]

    return None


def _map_shift_session_row(r: Dict[str, Any]) -> Dict[str, Any]:
    shift = r.get("shift")
    return {
        "id": r.get("id"),
        "model_id": r.get("model_id"),
        "session_date": r.get("date"),
        "turn_type": _turn_type_from_shift(shift),
        "notes": r.get("notes"),
        "active": (r.get("status") != "deleted"),
        "room_id": r.get("room_id"),
        "shift": shift,
        "status": r.get("status"),
        "checkin_at": r.get("checkin_at"),
        "checkout_at": r.get("checkout_at"),
    }


# ==========================================================
# MODELS
# ==========================================================
class ModelCreate(BaseModel):
    stage_name: str
    turn_type: str
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
    res = _sb_execute(supabase.table("models").select("*").order("created_at", desc=True), "list models")
    return res.data or []


@app.post("/models")
def create_model(payload: ModelCreate):
    _validate_turn_type(payload.turn_type)
    res = _sb_execute(supabase.table("models").insert(payload.model_dump()), "create model")
    if not res.data:
        raise HTTPException(status_code=500, detail="Insert failed: empty response")
    return res.data[0]


@app.get("/models/{model_id}")
def get_model(model_id: str):
    model_id = _require_uuid(model_id, "model_id")
    res = _sb_execute(supabase.table("models").select("*").eq("id", model_id).single(), "get model by id")
    if not res.data:
        raise HTTPException(status_code=404, detail="Model not found")
    return res.data


@app.patch("/models/{model_id}")
def update_model(model_id: str, payload: ModelUpdate):
    model_id = _require_uuid(model_id, "model_id")
    data = {k: v for k, v in payload.model_dump().items() if v is not None}
    if "turn_type" in data:
        _validate_turn_type(str(data["turn_type"]))
    if not data:
        raise HTTPException(status_code=400, detail="No fields to update")

    res = _sb_execute(supabase.table("models").update(data).eq("id", model_id), "update model")
    if not res.data:
        raise HTTPException(status_code=404, detail="Model not found")
    return res.data[0]


@app.delete("/models/{model_id}")
def deactivate_model(model_id: str):
    model_id = _require_uuid(model_id, "model_id")
    res = _sb_execute(supabase.table("models").update({"active": False}).eq("id", model_id), "deactivate model")
    if not res.data:
        raise HTTPException(status_code=404, detail="Model not found")
    return {"ok": True, "id": model_id, "active": False}


# ==========================================================
# PLATFORMS
# ==========================================================
class PlatformCreate(BaseModel):
    name: str
    calc_type: str
    token_usd_rate: Optional[float] = None
    active: bool = True


class PlatformUpdate(BaseModel):
    name: Optional[str] = None
    calc_type: Optional[str] = None
    token_usd_rate: Optional[float] = None
    active: Optional[bool] = None


@app.get("/platforms")
def list_platforms():
    res = _sb_execute(supabase.table("platforms").select("*").order("name", desc=False), "list platforms")
    return res.data or []


@app.post("/platforms")
def create_platform(payload: PlatformCreate):
    res = _sb_execute(supabase.table("platforms").insert(payload.model_dump()), "create platform")
    if not res.data:
        raise HTTPException(status_code=500, detail="Insert failed: empty response")
    return res.data[0]


@app.get("/platforms/{platform_id}")
def get_platform(platform_id: str):
    platform_id = _require_uuid(platform_id, "platform_id")
    res = _sb_execute(supabase.table("platforms").select("*").eq("id", platform_id).single(), "get platform by id")
    if not res.data:
        raise HTTPException(status_code=404, detail="Platform not found")
    return res.data


@app.patch("/platforms/{platform_id}")
def update_platform(platform_id: str, payload: PlatformUpdate):
    platform_id = _require_uuid(platform_id, "platform_id")
    data = {k: v for k, v in payload.model_dump().items() if v is not None}
    if not data:
        raise HTTPException(status_code=400, detail="No fields to update")

    res = _sb_execute(supabase.table("platforms").update(data).eq("id", platform_id), "update platform")
    if not res.data:
        raise HTTPException(status_code=404, detail="Platform not found")
    return res.data[0]


@app.delete("/platforms/{platform_id}")
def deactivate_platform(platform_id: str):
    platform_id = _require_uuid(platform_id, "platform_id")
    res = _sb_execute(supabase.table("platforms").update({"active": False}).eq("id", platform_id), "deactivate platform")
    if not res.data:
        raise HTTPException(status_code=404, detail="Platform not found")
    return {"ok": True, "id": platform_id, "active": False}


# ==========================================================
# MODEL PLATFORMS
# ==========================================================
@app.get("/models/{model_id}/platforms")
def get_model_platforms(model_id: str):
    model_id = _require_uuid(model_id, "model_id")

    mp_res = _sb_execute(
        supabase.table("model_platforms")
        .select("model_id, platform_id, pct_day, pct_night, bonus_threshold_usd, bonus_pct, active")
        .eq("model_id", model_id),
        "model_platforms for model",
    )
    mp = mp_res.data or []

    if not mp:
        return []

    platform_ids = list({row["platform_id"] for row in mp if row.get("platform_id")})
    if not platform_ids:
        return []

    pl_res = _sb_execute(
        supabase.table("platforms")
        .select("id, name, calc_type, token_usd_rate, active")
        .in_("id", platform_ids),
        "platforms for model_platforms",
    )
    pl = pl_res.data or []
    pl_map = {p["id"]: p for p in pl}

    out = []
    for row in mp:
        pid = row.get("platform_id")
        p = pl_map.get(pid, {})
        out.append(
            {
                "model_id": row.get("model_id"),
                "platform_id": pid,
                "platform_name": p.get("name"),
                "calc_type": p.get("calc_type"),
                "token_usd_rate": p.get("token_usd_rate"),
                "platform_active": p.get("active", True),
                "pct_day": row.get("pct_day"),
                "pct_night": row.get("pct_night"),
                "bonus_threshold_usd": row.get("bonus_threshold_usd"),
                "bonus_pct": row.get("bonus_pct"),
                "active": row.get("active", True),
            }
        )
    return out


# ==========================================================
# SESSIONS
# ==========================================================
class SessionCreate(BaseModel):
    model_id: str
    session_date: date
    turn_type: str
    notes: Optional[str] = None
    active: bool = True


@app.get("/sessions")
def list_sessions(
    session_date: Optional[date] = Query(None, description="YYYY-MM-DD"),
    model_id: Optional[str] = None,
    include_inactive: bool = False,
):
    q = supabase.table("shift_sessions").select("*").order("date", desc=True)

    if session_date:
        q = q.eq("date", str(session_date))
    if model_id:
        q = q.eq("model_id", _require_uuid(model_id, "model_id"))

    if not include_inactive:
        q = q.neq("status", "deleted")

    rows = _sb_execute(q, "list sessions").data or []
    return [_map_shift_session_row(r) for r in rows]


@app.post("/sessions")
def create_session(payload: SessionCreate):
    _get_model_or_404(payload.model_id)
    _validate_turn_type(payload.turn_type)

    desired_shift = _default_shift_from_turn_type(payload.turn_type)

    asg = _find_assignment_room_for_model(payload.model_id, payload.session_date, desired_shift)
    if not asg or not asg.get("room_id"):
        raise HTTPException(
            status_code=400,
            detail=(
                f"No hay asignación activa (room_assignments) para model_id={payload.model_id} "
                f"date={payload.session_date} shift={desired_shift}"
            ),
        )

    room_id = _require_uuid(asg["room_id"], "room_id")
    _get_room_or_404(room_id)

    ins = {
        "model_id": _require_uuid(payload.model_id, "model_id"),
        "room_id": room_id,
        "shift": desired_shift,
        "date": str(payload.session_date),
        "notes": payload.notes,
        "status": "open",
    }

    try:
        res = _sb_execute(supabase.table("shift_sessions").insert(ins), "create session")
        if not res.data:
            raise HTTPException(status_code=500, detail="Insert failed: empty response")
        return _map_shift_session_row(res.data[0])
    except Exception as e:
        if _is_unique_violation(e):
            raise HTTPException(status_code=409, detail="Session already exists (unique constraint)")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    r = _get_shift_session_or_404(session_id)
    return _map_shift_session_row(r)


# ==========================================================
# SESSION PLATFORM ENTRIES  ✅ FIXED
# tabla: session_platform_entries
# ==========================================================
class SessionEntryUpsert(BaseModel):
    platform_id: str

    value_in: Optional[float] = None
    value_out: Optional[float] = None
    tokens: Optional[float] = None
    usd_value: Optional[float] = None
    usd_earned: Optional[float] = None

    locked_in: bool = False
    active: bool = True


@app.get("/sessions/{session_id}/entries")
def list_session_entries(session_id: str):
    session_id = _require_uuid(session_id, "session_id")
    _get_shift_session_or_404(session_id)

    rows = _sb_execute(
        supabase.table("session_platform_entries").select("*").eq("session_id", session_id),
        "list session entries",
    ).data or []
    return rows


@app.post("/sessions/{session_id}/entries")
def upsert_session_entry(
    session_id: str,
    payload: SessionEntryUpsert,
    allow_update_locked: bool = Query(False),
):
    session_id = _require_uuid(session_id, "session_id")
    _get_shift_session_or_404(session_id)

    platform_id = _require_uuid(payload.platform_id, "platform_id")
    _get_platform_or_404(platform_id)

    # 1) Lookup existing (SIN maybe_single, SIN cosas raras)
    existing_res = _sb_execute(
        supabase.table("session_platform_entries")
        .select("id, locked_in")
        .eq("session_id", session_id)
        .eq("platform_id", platform_id)
        .limit(1),
        "entries lookup",
    )
    existing = (existing_res.data[0] if existing_res.data else None)

    if existing and existing.get("locked_in") and not allow_update_locked:
        raise HTTPException(status_code=409, detail="Entry is locked. Use allow_update_locked=true to update.")

    # 2) Upsert (SIN .select() encadenado)
    data = payload.model_dump()
    data["session_id"] = session_id
    data["platform_id"] = platform_id

    upsert_res = _sb_execute(
        supabase.table("session_platform_entries").upsert(data, on_conflict="session_id,platform_id"),
        "entries upsert",
    )

    # Supabase puede devolver data o no dependiendo config; si no hay data, OK
    if not upsert_res.data:
        return {"ok": True}

    return upsert_res.data[0]


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
    q = supabase.table("rooms").select("*").order("name")
    if not include_inactive:
        q = q.eq("active", True)
    return _sb_execute(q, "list rooms").data or []


@app.post("/rooms")
def create_room(payload: RoomCreate):
    res = _sb_execute(supabase.table("rooms").insert(payload.model_dump()), "create room")
    if not res.data:
        raise HTTPException(status_code=500, detail="Insert failed: empty response")
    return res.data[0]


@app.patch("/rooms/{room_id}")
def update_room(room_id: str, payload: RoomUpdate):
    room_id = _require_uuid(room_id, "room_id")
    _get_room_or_404(room_id)

    data = {k: v for k, v in payload.model_dump().items() if v is not None}
    if not data:
        raise HTTPException(status_code=400, detail="No fields to update")

    res = _sb_execute(supabase.table("rooms").update(data).eq("id", room_id), "update room")
    if not res.data:
        raise HTTPException(status_code=404, detail="Room not found")
    return res.data[0]


@app.delete("/rooms/{room_id}")
def deactivate_room(room_id: str):
    room_id = _require_uuid(room_id, "room_id")
    _get_room_or_404(room_id)

    res = _sb_execute(supabase.table("rooms").update({"active": False}).eq("id", room_id), "deactivate room")
    if not res.data:
        raise HTTPException(status_code=404, detail="Room not found")
    return {"ok": True, "room_id": room_id, "active": False}


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
    q = supabase.table("room_assignments").select("*").order("created_at", desc=False)

    if assignment_date:
        d = _parse_date_any(assignment_date)
        q = q.eq("assignment_date", str(d))

    if shift_slot:
        _validate_shift(shift_slot)
        q = q.eq("shift_slot", shift_slot)

    if room_id:
        q = q.eq("room_id", _require_uuid(room_id, "room_id"))

    if model_id:
        q = q.eq("model_id", _require_uuid(model_id, "model_id"))

    if not include_inactive:
        q = q.eq("active", True)

    return _sb_execute(q, "list assignments").data or []


@app.post("/assignments")
def create_assignment(payload: RoomAssignmentCreate):
    _get_model_or_404(payload.model_id)
    _get_room_or_404(payload.room_id)
    _validate_shift(payload.shift_slot)

    data = payload.model_dump(by_alias=False)

    if data.get("assignment_date") is None:
        data["assignment_date"] = date.today()

    data["assignment_date"] = str(data["assignment_date"])
    data["shift_slot"] = payload.shift_slot
    data["model_id"] = _require_uuid(payload.model_id, "model_id")
    data["room_id"] = _require_uuid(payload.room_id, "room_id")

    try:
        res = _sb_execute(supabase.table("room_assignments").insert(data), "create assignment")
        if not res.data:
            raise HTTPException(status_code=500, detail="Insert failed: empty response")
        return res.data[0]
    except Exception as e:
        if _is_unique_violation(e):
            raise HTTPException(status_code=409, detail="Assignment already exists for that date/shift/model")
        raise HTTPException(status_code=500, detail=str(e))


@app.patch("/assignments/{assignment_id}")
def update_assignment(assignment_id: str, payload: RoomAssignmentUpdate):
    assignment_id = _require_uuid(assignment_id, "assignment_id")

    existing = _sb_execute(
        supabase.table("room_assignments").select("*").eq("id", assignment_id).single(),
        "get assignment for update",
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

    res = _sb_execute(supabase.table("room_assignments").update(data).eq("id", assignment_id), "update assignment")
    if not res.data:
        raise HTTPException(status_code=404, detail="Assignment not found")
    return res.data[0]


@app.delete("/assignments/{assignment_id}")
def deactivate_assignment(assignment_id: str):
    assignment_id = _require_uuid(assignment_id, "assignment_id")
    res = _sb_execute(
        supabase.table("room_assignments").update({"active": False}).eq("id", assignment_id),
        "deactivate assignment",
    )
    if not res.data:
        raise HTTPException(status_code=404, detail="Assignment not found")
    return {"ok": True, "assignment_id": assignment_id, "active": False}
