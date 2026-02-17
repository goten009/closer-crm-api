import os
from pydantic import ConfigDict
from typing import Optional, Dict, Any, List, Tuple
from datetime import date, datetime
from uuid import UUID
from calendar import monthrange

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


def _get_model_platform_or_404(mp_id: str) -> Dict[str, Any]:
    mp_id = _require_uuid(mp_id, "model_platform_id")
    res = _sb_execute(
        supabase.table("model_platforms").select("*").eq("id", mp_id).single(),
        "get model_platform by id",
    )
    if not res.data:
        raise HTTPException(status_code=404, detail="Model platform relation not found")
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


# ✅ Turnos 100% independientes. NO hay fallback morning<->afternoon
def _find_assignment_room_for_model(model_id: str, assignment_date: date, shift: str) -> Optional[Dict[str, Any]]:
    model_id = _require_uuid(model_id, "model_id")
    _validate_shift(shift)

    q = (
        supabase.table("room_assignments")
        .select("*")
        .eq("model_id", model_id)
        .eq("assignment_date", str(assignment_date))
        .eq("shift_slot", shift)
        .eq("active", True)
        .limit(1)
    )

    exact = _sb_execute(q, "find assignment exact").data
    if exact:
        return exact[0]
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


def _normalize_shift_param(shift_raw: Optional[str]) -> Optional[str]:
    """
    Acepta valores del frontend:
      - "mañana" / "manana" -> "morning"
      - "tarde" -> "afternoon"
      - "noche" -> "night"
      - "todos" / "all" / None -> None (sin filtro)
    También acepta directamente: morning/afternoon/night.
    """
    if shift_raw is None:
        return None
    s = str(shift_raw).strip().lower()
    if s in ("", "todos", "all", "todas", "todo"):
        return None
    if s in ("mañana", "manana", "morning"):
        return "morning"
    if s in ("tarde", "afternoon"):
        return "afternoon"
    if s in ("noche", "night"):
        return "night"
    raise HTTPException(
        status_code=400,
        detail="shift inválido. Use: mañana|tarde|noche|todos (o morning|afternoon|night|all)",
    )


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
class ModelPlatformAssign(BaseModel):
    platform_id: str
    pct_day: Optional[int] = None
    pct_night: Optional[int] = None
    bonus_threshold_usd: Optional[int] = None
    bonus_pct: Optional[int] = None
    active: bool = True


class ModelPlatformPatch(BaseModel):
    pct_day: Optional[int] = None
    pct_night: Optional[int] = None
    bonus_threshold_usd: Optional[int] = None
    bonus_pct: Optional[int] = None
    active: Optional[bool] = None


@app.get("/models/{model_id}/platforms")
def get_model_platforms(
    model_id: str,
    include_inactive: bool = Query(False, description="true para incluir relaciones inactivas"),
):
    model_id = _require_uuid(model_id, "model_id")

    q = (
        supabase.table("model_platforms")
        .select("id, model_id, platform_id, pct_day, pct_night, bonus_threshold_usd, bonus_pct, active")
        .eq("model_id", model_id)
        .order("created_at", desc=False)
    )
    if not include_inactive:
        q = q.eq("active", True)

    mp_res = _sb_execute(q, "model_platforms for model")
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
                "id": row.get("id"),
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


@app.post("/models/{model_id}/platforms")
def assign_model_platform(model_id: str, payload: ModelPlatformAssign):
    model_id = _require_uuid(model_id, "model_id")
    _get_model_or_404(model_id)

    platform_id = _require_uuid(payload.platform_id, "platform_id")
    _get_platform_or_404(platform_id)

    data = {
        "model_id": model_id,
        "platform_id": platform_id,
        "pct_day": payload.pct_day,
        "pct_night": payload.pct_night,
        "bonus_threshold_usd": payload.bonus_threshold_usd,
        "bonus_pct": payload.bonus_pct,
        "active": payload.active,
    }

    res = _sb_execute(
        supabase.table("model_platforms").upsert(data, on_conflict="model_id,platform_id"),
        "assign model_platform",
    )

    if not res.data:
        return {"ok": True}

    return res.data[0]


@app.patch("/models/{model_id}/platforms/{platform_id}")
def patch_model_platform_by_pair(model_id: str, platform_id: str, payload: ModelPlatformPatch):
    model_id = _require_uuid(model_id, "model_id")
    platform_id = _require_uuid(platform_id, "platform_id")
    _get_model_or_404(model_id)
    _get_platform_or_404(platform_id)

    data = {k: v for k, v in payload.model_dump().items() if v is not None}
    if not data:
        raise HTTPException(status_code=400, detail="No fields to update")

    res = _sb_execute(
        supabase.table("model_platforms")
        .update(data)
        .eq("model_id", model_id)
        .eq("platform_id", platform_id),
        "update model_platform by pair",
    )
    if not res.data:
        raise HTTPException(status_code=404, detail="Model platform relation not found")
    return res.data[0]


@app.delete("/models/{model_id}/platforms/{platform_id}")
def deactivate_model_platform_by_pair(model_id: str, platform_id: str):
    model_id = _require_uuid(model_id, "model_id")
    platform_id = _require_uuid(platform_id, "platform_id")
    _get_model_or_404(model_id)
    _get_platform_or_404(platform_id)

    res = _sb_execute(
        supabase.table("model_platforms")
        .update({"active": False})
        .eq("model_id", model_id)
        .eq("platform_id", platform_id),
        "deactivate model_platform by pair",
    )
    if not res.data:
        raise HTTPException(status_code=404, detail="Model platform relation not found")
    return {"ok": True, "model_id": model_id, "platform_id": platform_id, "active": False}


@app.patch("/model_platforms/{mp_id}")
def patch_model_platform_by_id(mp_id: str, payload: ModelPlatformPatch):
    mp = _get_model_platform_or_404(mp_id)

    data = {k: v for k, v in payload.model_dump().items() if v is not None}
    if not data:
        raise HTTPException(status_code=400, detail="No fields to update")

    res = _sb_execute(
        supabase.table("model_platforms").update(data).eq("id", mp["id"]),
        "update model_platform by id",
    )
    if not res.data:
        raise HTTPException(status_code=404, detail="Model platform relation not found")
    return res.data[0]


@app.delete("/model_platforms/{mp_id}")
def deactivate_model_platform_by_id(mp_id: str):
    mp = _get_model_platform_or_404(mp_id)

    res = _sb_execute(
        supabase.table("model_platforms").update({"active": False}).eq("id", mp["id"]),
        "deactivate model_platform by id",
    )
    if not res.data:
        raise HTTPException(status_code=404, detail="Model platform relation not found")

    return {"ok": True, "id": mp["id"], "active": False}


# ==========================================================
# SESSIONS
# ==========================================================
class SessionCreate(BaseModel):
    model_id: str
    session_date: date
    turn_type: str  # se mantiene para no romper el frontend actual
    shift: Optional[str] = None  # permite afternoon real
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

    if payload.shift:
        _validate_shift(payload.shift)
        desired_shift = payload.shift
    else:
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
# SESSION PLATFORM ENTRIES
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

    data = payload.model_dump()
    data["session_id"] = session_id
    data["platform_id"] = platform_id

    upsert_res = _sb_execute(
        supabase.table("session_platform_entries").upsert(data, on_conflict="session_id,platform_id"),
        "entries upsert",
    )

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


# ==========================================================
# DASHBOARD (Gross vs Payout)
# ==========================================================
def _safe_float(v) -> float:
    try:
        if v is None:
            return 0.0
        return float(v)
    except Exception:
        return 0.0


def _entry_gross_amount(entry: Dict[str, Any], platform: Dict[str, Any]) -> float:
    """
    GROSS = facturación real de la plataforma (lo que entra antes de % modelo).
    Prioridad:
      1) usd_value si viene (ideal, porque tu UI lo está guardando como gross)
      2) según calc_type:
         - tokens: tokens * token_usd_rate
         - usd_value: usd_value (si no venía arriba)
         - usd_delta: (value_out - value_in)
      3) fallback: 0
    """
    if entry.get("usd_value") is not None:
        return _safe_float(entry.get("usd_value"))

    calc_type = platform.get("calc_type")
    rate = _safe_float(platform.get("token_usd_rate"))

    if calc_type == "tokens":
        tokens = _safe_float(entry.get("tokens"))
        return tokens * rate

    if calc_type == "usd_value":
        tokens = _safe_float(entry.get("tokens"))
        if tokens and rate:
            return tokens * rate
        return 0.0

    vin = _safe_float(entry.get("value_in"))
    vout = _safe_float(entry.get("value_out"))
    return vout - vin


def _entry_payout_amount(entry: Dict[str, Any]) -> float:
    """
    PAYOUT = lo que gana la modelo (ya con % + bono) si tu UI lo guardó en usd_earned.
    """
    if entry.get("usd_earned") is None:
        return 0.0
    return _safe_float(entry.get("usd_earned"))


@app.get("/dashboard/daily")
def dashboard_daily(
    session_date: Optional[str] = Query(None, description="YYYY-MM-DD o DD/MM/YYYY", alias="date"),
    shift: Optional[str] = Query(None, description="mañana|tarde|noche|todos", alias="shift"),
):
    d = _parse_date_any(session_date) if session_date else date.today()
    d_str = str(d)

    shift_norm = _normalize_shift_param(shift)

    q = (
        supabase.table("shift_sessions")
        .select("*")
        .eq("date", d_str)
        .neq("status", "deleted")
        .order("id", desc=False)
    )
    if shift_norm:
        q = q.eq("shift", shift_norm)

    sessions_res = _sb_execute(q, "dashboard daily sessions")
    sessions = sessions_res.data or []
    mapped_sessions = [_map_shift_session_row(s) for s in sessions]

    if not sessions:
        return {
            "date": d_str,
            "shift_selected": shift_norm,  # morning/afternoon/night/None
            "sessions": [],
            "totals": {
                "usd_total": 0.0,        # GROSS
                "payout_total": 0.0,     # PAYOUT
                "by_platform": [],
                "by_model": [],
                "by_shift": [],
                "by_shift_payout": [],
                "by_platform_payout": [],
                "by_model_payout": [],
                "by_room": [],
                "by_room_payout": [],
            },
        }

    session_ids = [s["id"] for s in sessions if s.get("id")]
    model_ids = list({s.get("model_id") for s in sessions if s.get("model_id")})
    room_ids = list({s.get("room_id") for s in sessions if s.get("room_id")})

    entries_res = _sb_execute(
        supabase.table("session_platform_entries")
        .select("*")
        .in_("session_id", session_ids),
        "dashboard daily entries",
    )
    entries = entries_res.data or []
    platform_ids = list({e.get("platform_id") for e in entries if e.get("platform_id")})

    models_map: Dict[str, Dict[str, Any]] = {}
    if model_ids:
        models_res = _sb_execute(
            supabase.table("models")
            .select("id, stage_name, turn_type, active")
            .in_("id", model_ids),
            "dashboard daily models",
        )
        for m in (models_res.data or []):
            models_map[m["id"]] = m

    rooms_map: Dict[str, Dict[str, Any]] = {}
    if room_ids:
        rooms_res = _sb_execute(
            supabase.table("rooms").select("id, name, active").in_("id", room_ids),
            "dashboard daily rooms",
        )
        for r in (rooms_res.data or []):
            rooms_map[r["id"]] = r

    platforms_map: Dict[str, Dict[str, Any]] = {}
    if platform_ids:
        plats_res = _sb_execute(
            supabase.table("platforms")
            .select("id, name, calc_type, token_usd_rate, active")
            .in_("id", platform_ids),
            "dashboard daily platforms",
        )
        for p in (plats_res.data or []):
            platforms_map[p["id"]] = p

    entries_by_session: Dict[str, List[Dict[str, Any]]] = {}
    for e in entries:
        sid = e.get("session_id")
        if sid:
            entries_by_session.setdefault(sid, []).append(e)

    gross_total = 0.0
    payout_total = 0.0

    by_platform_gross: Dict[str, float] = {}
    by_model_gross: Dict[str, float] = {}
    by_shift_gross: Dict[str, float] = {}
    by_room_gross: Dict[str, float] = {}

    by_platform_payout: Dict[str, float] = {}
    by_model_payout: Dict[str, float] = {}
    by_shift_payout: Dict[str, float] = {}
    by_room_payout: Dict[str, float] = {}

    enriched_sessions = []
    for s_raw, s_map in zip(sessions, mapped_sessions):
        mid = s_raw.get("model_id")
        rid = s_raw.get("room_id")
        s_shift = s_raw.get("shift")  # morning/afternoon/night

        s_entries = entries_by_session.get(s_raw.get("id"), [])
        session_gross = 0.0
        session_payout = 0.0
        entries_out = []

        for e in s_entries:
            pid = e.get("platform_id")
            p = platforms_map.get(pid, {})

            gross_amt = _entry_gross_amount(e, p)
            payout_amt = _entry_payout_amount(e)

            session_gross += gross_amt
            session_payout += payout_amt

            if pid:
                by_platform_gross[pid] = by_platform_gross.get(pid, 0.0) + gross_amt
                by_platform_payout[pid] = by_platform_payout.get(pid, 0.0) + payout_amt

            if mid:
                by_model_gross[mid] = by_model_gross.get(mid, 0.0) + gross_amt
                by_model_payout[mid] = by_model_payout.get(mid, 0.0) + payout_amt

            entries_out.append(
                {
                    "id": e.get("id"),
                    "platform_id": pid,
                    "platform_name": p.get("name"),
                    "calc_type": p.get("calc_type"),
                    "token_usd_rate": p.get("token_usd_rate"),
                    "gross_amount": gross_amt,
                    "payout_amount": payout_amt,
                    "value_in": e.get("value_in"),
                    "value_out": e.get("value_out"),
                    "tokens": e.get("tokens"),
                    "usd_value": e.get("usd_value"),
                    "usd_earned": e.get("usd_earned"),
                    "locked_in": e.get("locked_in"),
                    "active": e.get("active"),
                }
            )

        gross_total += session_gross
        payout_total += session_payout

        if s_shift:
            by_shift_gross[s_shift] = by_shift_gross.get(s_shift, 0.0) + session_gross
            by_shift_payout[s_shift] = by_shift_payout.get(s_shift, 0.0) + session_payout

        if rid:
            by_room_gross[rid] = by_room_gross.get(rid, 0.0) + session_gross
            by_room_payout[rid] = by_room_payout.get(rid, 0.0) + session_payout

        m = models_map.get(mid, {})
        r = rooms_map.get(rid, {})

        enriched_sessions.append(
            {
                **s_map,
                "model_stage_name": m.get("stage_name"),
                "room_name": r.get("name"),
                "session_gross": session_gross,
                "session_payout": session_payout,
                "entries": entries_out,
            }
        )

    by_platform_out = []
    for pid, amt in sorted(by_platform_gross.items(), key=lambda x: x[1], reverse=True):
        p = platforms_map.get(pid, {})
        by_platform_out.append({"platform_id": pid, "platform_name": p.get("name"), "usd_total": amt})

    by_model_out = []
    for mid, amt in sorted(by_model_gross.items(), key=lambda x: x[1], reverse=True):
        m = models_map.get(mid, {})
        by_model_out.append({"model_id": mid, "model_stage_name": m.get("stage_name"), "usd_total": amt})

    by_shift_out = [{"shift": k, "usd_total": v} for k, v in sorted(by_shift_gross.items(), key=lambda x: x[1], reverse=True)]

    by_room_out = []
    for rid, amt in sorted(by_room_gross.items(), key=lambda x: x[1], reverse=True):
        r = rooms_map.get(rid, {})
        by_room_out.append({"room_id": rid, "room_name": r.get("name"), "usd_total": amt})

    by_platform_payout_out = []
    for pid, amt in sorted(by_platform_payout.items(), key=lambda x: x[1], reverse=True):
        p = platforms_map.get(pid, {})
        by_platform_payout_out.append({"platform_id": pid, "platform_name": p.get("name"), "payout_total": amt})

    by_model_payout_out = []
    for mid, amt in sorted(by_model_payout.items(), key=lambda x: x[1], reverse=True):
        m = models_map.get(mid, {})
        by_model_payout_out.append({"model_id": mid, "model_stage_name": m.get("stage_name"), "payout_total": amt})

    by_shift_payout_out = [{"shift": k, "payout_total": v} for k, v in sorted(by_shift_payout.items(), key=lambda x: x[1], reverse=True)]

    by_room_payout_out = []
    for rid, amt in sorted(by_room_payout.items(), key=lambda x: x[1], reverse=True):
        r = rooms_map.get(rid, {})
        by_room_payout_out.append({"room_id": rid, "room_name": r.get("name"), "payout_total": amt})

    return {
        "date": d_str,
        "shift_selected": shift_norm,
        "sessions": enriched_sessions,
        "totals": {
            "usd_total": gross_total,
            "payout_total": payout_total,
            "by_platform": by_platform_out,
            "by_model": by_model_out,
            "by_shift": by_shift_out,
            "by_shift_payout": by_shift_payout_out,
            "by_platform_payout": by_platform_payout_out,
            "by_model_payout": by_model_payout_out,
            "by_room": by_room_out,
            "by_room_payout": by_room_payout_out,
        },
    }


# ==========================================================
# DASHBOARD (Fortnight: 1-15 / 16-fin)
# ==========================================================
def _fortnight_range(d: date) -> Tuple[date, date, int]:
    last_day = monthrange(d.year, d.month)[1]
    if d.day <= 15:
        return date(d.year, d.month, 1), date(d.year, d.month, 15), 1
    return date(d.year, d.month, 16), date(d.year, d.month, last_day), 2


@app.get("/dashboard/fortnight")
def dashboard_fortnight(
    session_date: Optional[str] = Query(None, description="YYYY-MM-DD o DD/MM/YYYY", alias="date"),
    shift: Optional[str] = Query(None, description="mañana|tarde|noche|todos", alias="shift"),
):
    d = _parse_date_any(session_date) if session_date else date.today()
    start_d, end_d, half = _fortnight_range(d)

    start_str = str(start_d)
    end_str = str(end_d)

    shift_norm = _normalize_shift_param(shift)

    q = (
        supabase.table("shift_sessions")
        .select("*")
        .gte("date", start_str)
        .lte("date", end_str)
        .neq("status", "deleted")
        .order("date", desc=False)
    )
    if shift_norm:
        q = q.eq("shift", shift_norm)

    sessions_res = _sb_execute(q, "dashboard fortnight sessions")
    sessions = sessions_res.data or []
    mapped_sessions = [_map_shift_session_row(s) for s in sessions]

    if not sessions:
        return {
            "range": {"start": start_str, "end": end_str, "half": half},
            "shift_selected": shift_norm,
            "sessions": [],
            "totals": {
                "usd_total": 0.0,
                "payout_total": 0.0,
                "by_platform": [],
                "by_model": [],
                "by_shift": [],
                "by_shift_payout": [],
                "by_platform_payout": [],
                "by_model_payout": [],
                "by_room": [],
                "by_room_payout": [],
            },
        }

    session_ids = [s["id"] for s in sessions if s.get("id")]
    model_ids = list({s.get("model_id") for s in sessions if s.get("model_id")})
    room_ids = list({s.get("room_id") for s in sessions if s.get("room_id")})

    entries_res = _sb_execute(
        supabase.table("session_platform_entries")
        .select("*")
        .in_("session_id", session_ids),
        "dashboard fortnight entries",
    )
    entries = entries_res.data or []
    platform_ids = list({e.get("platform_id") for e in entries if e.get("platform_id")})

    models_map: Dict[str, Dict[str, Any]] = {}
    if model_ids:
        models_res = _sb_execute(
            supabase.table("models")
            .select("id, stage_name, active")
            .in_("id", model_ids),
            "dashboard fortnight models",
        )
        for m in (models_res.data or []):
            models_map[m["id"]] = m

    rooms_map: Dict[str, Dict[str, Any]] = {}
    if room_ids:
        rooms_res = _sb_execute(
            supabase.table("rooms").select("id, name, active").in_("id", room_ids),
            "dashboard fortnight rooms",
        )
        for r in (rooms_res.data or []):
            rooms_map[r["id"]] = r

    platforms_map: Dict[str, Dict[str, Any]] = {}
    if platform_ids:
        plats_res = _sb_execute(
            supabase.table("platforms")
            .select("id, name, calc_type, token_usd_rate, active")
            .in_("id", platform_ids),
            "dashboard fortnight platforms",
        )
        for p in (plats_res.data or []):
            platforms_map[p["id"]] = p

    entries_by_session: Dict[str, List[Dict[str, Any]]] = {}
    for e in entries:
        sid = e.get("session_id")
        if sid:
            entries_by_session.setdefault(sid, []).append(e)

    gross_total = 0.0
    payout_total = 0.0

    by_model_gross: Dict[str, float] = {}
    by_platform_gross: Dict[str, float] = {}
    by_shift_gross: Dict[str, float] = {}
    by_room_gross: Dict[str, float] = {}

    by_model_payout: Dict[str, float] = {}
    by_platform_payout: Dict[str, float] = {}
    by_shift_payout: Dict[str, float] = {}
    by_room_payout: Dict[str, float] = {}

    enriched_sessions = []
    for s_raw, s_map in zip(sessions, mapped_sessions):
        sid = s_raw.get("id")
        mid = s_raw.get("model_id")
        rid = s_raw.get("room_id")
        s_shift = s_raw.get("shift")  # morning/afternoon/night

        s_entries = entries_by_session.get(sid, [])
        session_gross = 0.0
        session_payout = 0.0

        for e in s_entries:
            pid = e.get("platform_id")
            p = platforms_map.get(pid, {})

            gross_amt = _entry_gross_amount(e, p)
            payout_amt = _entry_payout_amount(e)

            session_gross += gross_amt
            session_payout += payout_amt

            if mid:
                by_model_gross[mid] = by_model_gross.get(mid, 0.0) + gross_amt
                by_model_payout[mid] = by_model_payout.get(mid, 0.0) + payout_amt

            if pid:
                by_platform_gross[pid] = by_platform_gross.get(pid, 0.0) + gross_amt
                by_platform_payout[pid] = by_platform_payout.get(pid, 0.0) + payout_amt

        gross_total += session_gross
        payout_total += session_payout

        if s_shift:
            by_shift_gross[s_shift] = by_shift_gross.get(s_shift, 0.0) + session_gross
            by_shift_payout[s_shift] = by_shift_payout.get(s_shift, 0.0) + session_payout

        if rid:
            by_room_gross[rid] = by_room_gross.get(rid, 0.0) + session_gross
            by_room_payout[rid] = by_room_payout.get(rid, 0.0) + session_payout

        m = models_map.get(mid, {})
        r = rooms_map.get(rid, {})
        enriched_sessions.append(
            {
                **s_map,
                "model_stage_name": m.get("stage_name"),
                "room_name": r.get("name"),
                "session_gross": session_gross,
                "session_payout": session_payout,
            }
        )

    by_platform_out = [
        {"platform_id": pid, "platform_name": platforms_map.get(pid, {}).get("name"), "usd_total": amt}
        for pid, amt in sorted(by_platform_gross.items(), key=lambda x: x[1], reverse=True)
    ]
    by_model_out = [
        {"model_id": mid, "model_stage_name": models_map.get(mid, {}).get("stage_name"), "usd_total": amt}
        for mid, amt in sorted(by_model_gross.items(), key=lambda x: x[1], reverse=True)
    ]
    by_shift_out = [
        {"shift": sh, "usd_total": amt}
        for sh, amt in sorted(by_shift_gross.items(), key=lambda x: x[1], reverse=True)
    ]
    by_room_out = [
        {"room_id": rid, "room_name": rooms_map.get(rid, {}).get("name"), "usd_total": amt}
        for rid, amt in sorted(by_room_gross.items(), key=lambda x: x[1], reverse=True)
    ]

    by_platform_payout_out = [
        {"platform_id": pid, "platform_name": platforms_map.get(pid, {}).get("name"), "payout_total": amt}
        for pid, amt in sorted(by_platform_payout.items(), key=lambda x: x[1], reverse=True)
    ]
    by_model_payout_out = [
        {"model_id": mid, "model_stage_name": models_map.get(mid, {}).get("stage_name"), "payout_total": amt}
        for mid, amt in sorted(by_model_payout.items(), key=lambda x: x[1], reverse=True)
    ]
    by_shift_payout_out = [
        {"shift": sh, "payout_total": amt}
        for sh, amt in sorted(by_shift_payout.items(), key=lambda x: x[1], reverse=True)
    ]
    by_room_payout_out = [
        {"room_id": rid, "room_name": rooms_map.get(rid, {}).get("name"), "payout_total": amt}
        for rid, amt in sorted(by_room_payout.items(), key=lambda x: x[1], reverse=True)
    ]

    return {
        "range": {"start": start_str, "end": end_str, "half": half},
        "shift_selected": shift_norm,
        "sessions": enriched_sessions,
        "totals": {
            "usd_total": gross_total,
            "payout_total": payout_total,
            "by_platform": by_platform_out,
            "by_model": by_model_out,
            "by_shift": by_shift_out,
            "by_shift_payout": by_shift_payout_out,
            "by_platform_payout": by_platform_payout_out,
            "by_model_payout": by_model_payout_out,
            "by_room": by_room_out,
            "by_room_payout": by_room_payout_out,
        },
    }

# ==========================================================
# STORE / TIENDA (DESCUENTOS POR NÓMINA)
# ==========================================================

# Reglas: tienda por tipo (monitores/juguetes/adelantos/multas)
# No hay pagos. Solo cargos a modelos + cuotas en 1-3 quincenas.

_STORE_SHOP_TYPES = {"monitores", "juguetes", "adelantos", "multas"}
_STORE_PRODUCT_TYPES = {"physical", "digital", "service"}
_STORE_CHARGE_STATUS = {"draft", "scheduled", "void"}
_STORE_INSTALLMENT_STATUS = {"pending", "applied", "void"}


def _validate_shop_type(shop_type: str) -> None:
    s = (shop_type or "").strip().lower()
    if s not in _STORE_SHOP_TYPES:
        raise HTTPException(status_code=400, detail="shop_type inválido. Use: monitores|juguetes|adelantos|multas")


def _validate_product_type(product_type: str) -> None:
    s = (product_type or "").strip().lower()
    if s not in _STORE_PRODUCT_TYPES:
        raise HTTPException(status_code=400, detail="product_type inválido. Use: physical|digital|service")


def _validate_installments_count(n: int) -> None:
    if n is None:
        raise HTTPException(status_code=400, detail="installments_count is required")
    if int(n) < 1 or int(n) > 3:
        raise HTTPException(status_code=400, detail="installments_count must be 1..3")


def _fortnight_key_from_date(d: date) -> Tuple[int, int, int]:
    half = 1 if d.day <= 15 else 2
    return d.year, d.month, half


def _next_fortnight_key(year: int, month: int, half: int) -> Tuple[int, int, int]:
    if half == 1:
        return year, month, 2
    # half==2 => next month half 1
    if month == 12:
        return year + 1, 1, 1
    return year, month + 1, 1


def _get_charge_or_404(charge_id: str) -> Dict[str, Any]:
    charge_id = _require_uuid(charge_id, "charge_id")
    res = _sb_execute(
        supabase.table("store_charges").select("*").eq("id", charge_id).single(),
        "get store_charge",
    )
    if not res.data:
        raise HTTPException(status_code=404, detail="Charge not found")
    return res.data


def _get_product_or_404(product_id: str) -> Dict[str, Any]:
    product_id = _require_uuid(product_id, "product_id")
    res = _sb_execute(
        supabase.table("store_products").select("*").eq("id", product_id).single(),
        "get store_product",
    )
    if not res.data:
        raise HTTPException(status_code=404, detail="Product not found")
    return res.data


def _recalc_charge_totals(charge_id: str) -> Dict[str, float]:
    charge_id = _require_uuid(charge_id, "charge_id")
    items = _sb_execute(
        supabase.table("store_charge_items").select("line_total_usd").eq("charge_id", charge_id),
        "recalc charge items",
    ).data or []

    subtotal = 0.0
    for it in items:
        try:
            subtotal += float(it.get("line_total_usd") or 0)
        except Exception:
            subtotal += 0.0

    total = subtotal

    _sb_execute(
        supabase.table("store_charges").update(
            {"subtotal_usd": subtotal, "total_usd": total, "updated_at": datetime.utcnow().isoformat()}
        ).eq("id", charge_id),
        "update charge totals",
    )

    return {"subtotal_usd": subtotal, "total_usd": total}


def _validate_stock_qty(v: Optional[int]) -> Optional[int]:
    if v is None:
        return None
    try:
        iv = int(v)
    except Exception:
        raise HTTPException(status_code=400, detail="stock_qty must be integer or null")
    if iv < 0:
        raise HTTPException(status_code=400, detail="stock_qty cannot be negative")
    return iv


def _qty_to_int_for_stock(qty: float) -> int:
    """
    Para inventario físico, exigimos cantidad entera.
    """
    try:
        q = float(qty)
    except Exception:
        raise HTTPException(status_code=400, detail="qty must be a number")

    if q <= 0:
        raise HTTPException(status_code=400, detail="qty must be > 0")

    if abs(q - round(q)) > 1e-9:
        raise HTTPException(status_code=400, detail="qty must be an integer for physical stock items")

    return int(round(q))


def _store_decrement_stock(product_id: str, qty_int: int) -> Dict[str, Any]:
    """
    Llama al RPC: public.store_decrement_stock(p_product_id uuid, p_qty integer)
    Retorna: { ok: bool, new_stock: int|null }
    """
    product_id = _require_uuid(product_id, "product_id")
    if qty_int <= 0:
        raise HTTPException(status_code=400, detail="qty must be >= 1")

    res = _sb_execute(
        supabase.rpc("store_decrement_stock", {"p_product_id": product_id, "p_qty": int(qty_int)}),
        "rpc store_decrement_stock",
    )

    data = res.data
    if isinstance(data, list):
        row = data[0] if data else None
    else:
        row = data

    if not row:
        return {"ok": True, "new_stock": None}

    return {"ok": bool(row.get("ok")), "new_stock": row.get("new_stock")}


def _store_increment_stock_direct(product_id: str, qty_int: int) -> None:
    """
    Re-stock simple (sin RPC) para deshacer un item.
    - Solo se usa si el producto tiene stock_qty != NULL y es physical.
    """
    product_id = _require_uuid(product_id, "product_id")
    if qty_int <= 0:
        return

    prod = _get_product_or_404(product_id)
    current = prod.get("stock_qty")
    if current is None:
        # sin control de stock, no hacemos nada
        return

    new_stock = int(current) + int(qty_int)
    _sb_execute(
        supabase.table("store_products")
        .update({"stock_qty": new_stock, "updated_at": datetime.utcnow().isoformat()})
        .eq("id", product_id),
        "restock product",
    )


def _has_any_applied_installment(charge_id: str) -> bool:
    rows = _sb_execute(
        supabase.table("store_charge_installments")
        .select("id,status")
        .eq("charge_id", _require_uuid(charge_id, "charge_id")),
        "check applied installments",
    ).data or []
    for r in rows:
        if (r.get("status") or "").strip().lower() == "applied":
            return True
    return False


def _replan_installments_keep_schedule(charge: Dict[str, Any]) -> None:
    """
    Recalcula cuotas (amount_usd) manteniendo:
      - mismo start_year/start_month/start_half
      - mismo installments_count
    Solo si NO hay cuotas applied.
    """
    if charge.get("status") != "scheduled":
        return

    if _has_any_applied_installment(charge["id"]):
        raise HTTPException(status_code=409, detail="No se puede ajustar: ya hay cuotas aplicadas (applied).")

    # recalcular total desde items
    totals = _recalc_charge_totals(charge["id"])
    total = float(totals.get("total_usd") or 0.0)

    inst_count = int(charge.get("installments_count") or 1)
    _validate_installments_count(inst_count)

    # borrar cuotas pendientes/void anteriores y recrear en pending
    _sb_execute(
        supabase.table("store_charge_installments").delete().eq("charge_id", charge["id"]),
        "delete installments for replan",
    )

    base = round(total / inst_count, 2)
    amounts = [base] * inst_count
    diff = round(total - sum(amounts), 2)
    amounts[-1] = round(amounts[-1] + diff, 2)

    y = int(charge.get("start_year"))
    m = int(charge.get("start_month"))
    half = int(charge.get("start_half"))

    rows = []
    for i in range(inst_count):
        rows.append(
            {
                "charge_id": charge["id"],
                "due_year": y,
                "due_month": m,
                "due_half": half,
                "amount_usd": float(amounts[i]),
                "status": "pending",
            }
        )
        y, m, half = _next_fortnight_key(y, m, half)

    _sb_execute(
        supabase.table("store_charge_installments").insert(rows),
        "insert replanned installments",
    )


# =========================
# Schemas - Products
# =========================
class StoreProductCreate(BaseModel):
    shop_type: str = Field(default="monitores")
    name: str
    sku: Optional[str] = None
    product_type: str = Field(default="physical")
    unit_price_usd: float = Field(default=0, ge=0)
    unit_cost_usd: Optional[float] = None
    stock_qty: Optional[int] = None
    is_active: bool = True


class StoreProductUpdate(BaseModel):
    shop_type: Optional[str] = None
    name: Optional[str] = None
    sku: Optional[str] = None
    product_type: Optional[str] = None
    unit_price_usd: Optional[float] = None
    unit_cost_usd: Optional[float] = None
    stock_qty: Optional[int] = None
    is_active: Optional[bool] = None


# =========================
# Schemas - Stock adjustments
# =========================
class StoreAdjustStock(BaseModel):
    qty: int = Field(..., ge=0)
    mode: str = Field(..., description="add | set")

    @field_validator("mode")
    @classmethod
    def _v_mode(cls, v):
        s = (v or "").strip().lower()
        if s not in ("add", "set"):
            raise ValueError("mode must be 'add' or 'set'")
        return s


# =========================
# Schemas - Charges
# =========================
class StoreChargeCreate(BaseModel):
    model_id: str
    shop_type: str = Field(default="monitores")
    installments_count: int = Field(default=1, ge=1, le=3)
    start_date: Optional[date] = None
    notes: Optional[str] = None
    created_by: Optional[str] = None


class StoreChargeItemAdd(BaseModel):
    product_id: Optional[str] = None
    description: Optional[str] = None
    qty: float = Field(default=1, gt=0)
    unit_price_usd: float = Field(default=0, ge=0)


class StoreChargeFinalize(BaseModel):
    # ✅ opcional para no exigir body nunca
    installments_count: Optional[int] = Field(default=None, ge=1, le=3)


# =========================
# PRODUCTS endpoints
# =========================
@app.get("/store/products")
def store_list_products(
    shop_type: Optional[str] = Query(default=None),
    active_only: bool = True,
):
    q = supabase.table("store_products").select("*").order("name", desc=False)
    if active_only:
        q = q.eq("is_active", True)
    if shop_type:
        _validate_shop_type(shop_type)
        q = q.eq("shop_type", shop_type)

    return {"items": _sb_execute(q, "store list products").data or []}


@app.post("/store/products")
def store_create_product(payload: StoreProductCreate):
    _validate_shop_type(payload.shop_type)
    _validate_product_type(payload.product_type)

    data = payload.model_dump()
    data["stock_qty"] = _validate_stock_qty(data.get("stock_qty"))
    data["updated_at"] = datetime.utcnow().isoformat()

    res = _sb_execute(supabase.table("store_products").insert(data), "store create product")
    if not res.data:
        raise HTTPException(status_code=500, detail="Insert failed: empty response")
    return {"item": res.data[0]}


@app.patch("/store/products/{product_id}")
def store_update_product(product_id: str, payload: StoreProductUpdate):
    product_id = _require_uuid(product_id, "product_id")
    _get_product_or_404(product_id)

    data = {k: v for k, v in payload.model_dump().items() if v is not None}
    if not data:
        raise HTTPException(status_code=400, detail="No fields to update")

    if "shop_type" in data:
        _validate_shop_type(str(data["shop_type"]))
    if "product_type" in data:
        _validate_product_type(str(data["product_type"]))
    if "stock_qty" in data:
        data["stock_qty"] = _validate_stock_qty(data.get("stock_qty"))

    data["updated_at"] = datetime.utcnow().isoformat()

    res = _sb_execute(supabase.table("store_products").update(data).eq("id", product_id), "store update product")
    if not res.data:
        raise HTTPException(status_code=404, detail="Product not found")
    return {"item": res.data[0]}


@app.post("/store/products/{product_id}/adjust-stock")
def store_adjust_stock(product_id: str, payload: StoreAdjustStock):
    """
    Ajuste de inventario:
      - mode=add : suma qty al stock actual
      - mode=set : deja el stock EXACTAMENTE en qty
    Nota:
      - Si stock_qty es NULL y mode=add -> error (no hay control de stock).
      - Si stock_qty es NULL y mode=set -> se habilita control dejando stock en qty.
    """
    product_id = _require_uuid(product_id, "product_id")
    prod = _get_product_or_404(product_id)

    current = prod.get("stock_qty")

    if payload.mode == "add":
        if current is None:
            raise HTTPException(status_code=400, detail="Product has no stock control (stock_qty is null). Use mode='set' first.")
        new_stock = int(current) + int(payload.qty)
    else:  # set
        new_stock = int(payload.qty)

    if new_stock < 0:
        raise HTTPException(status_code=400, detail="Stock cannot be negative")

    res = _sb_execute(
        supabase.table("store_products")
        .update({"stock_qty": new_stock, "updated_at": datetime.utcnow().isoformat()})
        .eq("id", product_id),
        "store adjust stock",
    )

    if not res.data:
        raise HTTPException(status_code=404, detail="Product not found")

    return {"ok": True, "item": res.data[0]}


@app.delete("/store/products/{product_id}")
def store_deactivate_product(product_id: str):
    """
    "Eliminar" producto = desactivarlo (soft delete):
      - is_active = false
    (No borramos histórico.)
    """
    product_id = _require_uuid(product_id, "product_id")
    _get_product_or_404(product_id)

    res = _sb_execute(
        supabase.table("store_products")
        .update({"is_active": False, "updated_at": datetime.utcnow().isoformat()})
        .eq("id", product_id),
        "store deactivate product",
    )

    if not res.data:
        raise HTTPException(status_code=404, detail="Product not found")

    return {"ok": True, "id": product_id, "is_active": False}


# =========================
# CHARGES endpoints
# =========================
@app.post("/store/charges")
def store_create_charge(payload: StoreChargeCreate):
    _validate_shop_type(payload.shop_type)
    _validate_installments_count(payload.installments_count)

    model_id = _require_uuid(payload.model_id, "model_id")
    _get_model_or_404(model_id)

    d = payload.start_date or date.today()
    y, m, half = _fortnight_key_from_date(d)

    data = {
        "model_id": model_id,
        "shop_type": payload.shop_type,
        "status": "draft",
        "installments_count": int(payload.installments_count),
        "start_year": int(y),
        "start_month": int(m),
        "start_half": int(half),
        "notes": payload.notes,
        "subtotal_usd": 0,
        "total_usd": 0,
        "created_by": payload.created_by,
        "updated_at": datetime.utcnow().isoformat(),
    }

    res = _sb_execute(supabase.table("store_charges").insert(data), "store create charge")
    if not res.data:
        raise HTTPException(status_code=500, detail="Insert failed: empty response")
    return {"charge": res.data[0]}


@app.get("/store/charges")
def store_list_charges(
    status: Optional[str] = Query(default=None),
    model_id: Optional[str] = Query(default=None),
    shop_type: Optional[str] = Query(default=None),
    limit: int = 50,
):
    q = supabase.table("store_charges").select("*").order("created_at", desc=True).limit(limit)

    if status:
        s = status.strip().lower()
        if s not in _STORE_CHARGE_STATUS:
            raise HTTPException(status_code=400, detail="status inválido. Use: draft|scheduled|void")
        q = q.eq("status", s)

    if model_id:
        q = q.eq("model_id", _require_uuid(model_id, "model_id"))

    if shop_type:
        _validate_shop_type(shop_type)
        q = q.eq("shop_type", shop_type)

    return {"items": _sb_execute(q, "store list charges").data or []}


@app.get("/store/charges/{charge_id}")
def store_get_charge(charge_id: str):
    charge = _get_charge_or_404(charge_id)

    items = _sb_execute(
        supabase.table("store_charge_items").select("*").eq("charge_id", charge["id"]).order("created_at", desc=False),
        "store get charge items",
    ).data or []

    installments = _sb_execute(
        supabase.table("store_charge_installments")
        .select("*")
        .eq("charge_id", charge["id"])
        .order("due_year", desc=False)
        .order("due_month", desc=False)
        .order("due_half", desc=False),
        "store get charge installments",
    ).data or []

    return {"charge": charge, "items": items, "installments": installments}


# ✅ NUEVO: ver TODO lo asignado a una modelo (cargos + items + cuotas)
@app.get("/store/models/{model_id}/assignments")
def store_model_assignments(model_id: str, limit: int = 200):
    model_id = _require_uuid(model_id, "model_id")
    _get_model_or_404(model_id)

    charges = _sb_execute(
        supabase.table("store_charges")
        .select("*")
        .eq("model_id", model_id)
        .order("created_at", desc=True)
        .limit(limit),
        "store model charges",
    ).data or []

    if not charges:
        return {"model_id": model_id, "items": []}

    charge_ids = [c["id"] for c in charges if c.get("id")]

    items = _sb_execute(
        supabase.table("store_charge_items")
        .select("*")
        .in_("charge_id", charge_ids)
        .order("created_at", desc=True),
        "store model charge items",
    ).data or []

    installments = _sb_execute(
        supabase.table("store_charge_installments")
        .select("*")
        .in_("charge_id", charge_ids)
        .order("due_year", desc=False)
        .order("due_month", desc=False)
        .order("due_half", desc=False),
        "store model charge installments",
    ).data or []

    items_by_charge: Dict[str, List[Dict[str, Any]]] = {}
    for it in items:
        items_by_charge.setdefault(it.get("charge_id"), []).append(it)

    inst_by_charge: Dict[str, List[Dict[str, Any]]] = {}
    for ins in installments:
        inst_by_charge.setdefault(ins.get("charge_id"), []).append(ins)

    out = []
    for ch in charges:
        cid = ch.get("id")
        out.append(
            {
                "charge": ch,
                "items": items_by_charge.get(cid, []),
                "installments": inst_by_charge.get(cid, []),
            }
        )

    return {"model_id": model_id, "items": out}


@app.post("/store/charges/{charge_id}/items")
def store_add_charge_item(charge_id: str, payload: StoreChargeItemAdd):
    charge = _get_charge_or_404(charge_id)
    if charge.get("status") != "draft":
        raise HTTPException(status_code=400, detail="Cannot modify items when charge is not draft")

    product_id = None
    description = (payload.description or "").strip() if payload.description else ""
    qty = float(payload.qty or 1)
    unit_price = float(payload.unit_price_usd or 0)

    product_row: Optional[Dict[str, Any]] = None
    product_type = None

    if payload.product_id:
        product_id = _require_uuid(payload.product_id, "product_id")
        product_row = _get_product_or_404(product_id)
        product_type = (product_row.get("product_type") or "").strip().lower()

        if not description:
            description = product_row.get("name") or "Item"

        if unit_price == 0:
            try:
                unit_price = float(product_row.get("unit_price_usd") or 0)
            except Exception:
                unit_price = 0.0

    if not description:
        description = "Item"

    line_total = round(qty * unit_price, 2)

    ins = {
        "charge_id": charge["id"],
        "product_id": product_id,
        "description": description,
        "qty": qty,
        "unit_price_usd": unit_price,
        "line_total_usd": line_total,
    }

    inserted = _sb_execute(supabase.table("store_charge_items").insert(ins), "store add charge item")
    if not inserted.data:
        raise HTTPException(status_code=500, detail="Insert failed: empty response")

    item = inserted.data[0]

    # Descontar inventario si aplica
    if product_id and product_row:
        try:
            if (product_type or "") == "physical" and product_row.get("stock_qty") is not None:
                qty_int = _qty_to_int_for_stock(qty)
                r = _store_decrement_stock(product_id, qty_int)

                if not r.get("ok"):
                    _sb_execute(
                        supabase.table("store_charge_items").delete().eq("id", item["id"]),
                        "rollback item (no stock)",
                    )
                    _recalc_charge_totals(charge["id"])
                    raise HTTPException(
                        status_code=409,
                        detail=f"Stock insuficiente para '{product_row.get('name')}'. Disponible: {r.get('new_stock')}",
                    )
        except HTTPException:
            raise
        except Exception as e:
            _sb_execute(
                supabase.table("store_charge_items").delete().eq("id", item["id"]),
                "rollback item (rpc error)",
            )
            _recalc_charge_totals(charge["id"])
            raise HTTPException(status_code=500, detail=f"Stock update failed: {str(e)}")

    _recalc_charge_totals(charge["id"])

    return {"item": item, "charge": _get_charge_or_404(charge["id"])}


# ✅ MEJORADO: eliminar item (draft) y ADEMÁS permitir en scheduled si NO hay applied
@app.delete("/store/charges/{charge_id}/items/{item_id}")
def store_remove_charge_item(charge_id: str, item_id: str):
    charge = _get_charge_or_404(charge_id)
    item_id = _require_uuid(item_id, "item_id")

    # traer item completo para poder restock
    item_res = _sb_execute(
        supabase.table("store_charge_items").select("*").eq("id", item_id).eq("charge_id", charge["id"]).single(),
        "get item to delete",
    )
    item = item_res.data
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")

    status = (charge.get("status") or "").strip().lower()

    if status == "void":
        raise HTTPException(status_code=400, detail="Charge is void. No se puede modificar.")

    if status == "scheduled":
        # si ya se aplicó alguna cuota, no dejamos borrar
        if _has_any_applied_installment(charge["id"]):
            raise HTTPException(status_code=409, detail="No se puede borrar: ya hay cuotas aplicadas (applied).")

    # borrar item
    _sb_execute(
        supabase.table("store_charge_items").delete().eq("id", item_id).eq("charge_id", charge["id"]),
        "store delete charge item",
    )

    # restock si aplica
    try:
        pid = item.get("product_id")
        if pid:
            prod = _get_product_or_404(pid)
            ptype = (prod.get("product_type") or "").strip().lower()
            if ptype == "physical" and prod.get("stock_qty") is not None:
                qty_int = _qty_to_int_for_stock(float(item.get("qty") or 0))
                _store_increment_stock_direct(pid, qty_int)
    except Exception:
        # no tumbamos la operación por restock (pero queda loggable si lo deseas después)
        pass

    totals = _recalc_charge_totals(charge["id"])

    # si estaba scheduled, replanea cuotas para que cuadre con el nuevo total
    if status == "scheduled":
        _replan_installments_keep_schedule(_get_charge_or_404(charge["id"]))

    return {
        "ok": True,
        "totals": totals,
        "charge": _get_charge_or_404(charge["id"]),
        "full": store_get_charge(charge["id"]),
    }


# ✅ FIX: finalize SIN body obligatorio (payload opcional)
@app.post("/store/charges/{charge_id}/finalize")
def store_finalize_charge(charge_id: str, payload: Optional[StoreChargeFinalize] = None):
    payload = payload or StoreChargeFinalize()

    charge = _get_charge_or_404(charge_id)

    if charge.get("status") != "draft":
        raise HTTPException(status_code=400, detail="Only draft charges can be finalized")

    # Debe tener items
    items = _sb_execute(
        supabase.table("store_charge_items").select("id").eq("charge_id", charge["id"]).limit(1),
        "store charge items existence",
    ).data or []
    if not items:
        raise HTTPException(status_code=400, detail="Add at least one item before finalizing")

    # Recalcular total
    totals = _recalc_charge_totals(charge["id"])
    total = float(totals.get("total_usd") or 0.0)

    # installments_count (puede venir para ajustar antes de programar)
    inst_count = int(charge.get("installments_count") or 1)
    if payload.installments_count is not None:
        _validate_installments_count(payload.installments_count)
        inst_count = int(payload.installments_count)

        _sb_execute(
            supabase.table("store_charges").update(
                {"installments_count": inst_count, "updated_at": datetime.utcnow().isoformat()}
            ).eq("id", charge["id"]),
            "store update charge installments_count",
        )

    # Borrar cuotas previas (si re-finalizas)
    _sb_execute(
        supabase.table("store_charge_installments").delete().eq("charge_id", charge["id"]),
        "store delete prior installments",
    )

    # Dividir total en inst_count cuotas (redondeo exacto)
    base = round(total / inst_count, 2)
    amounts = [base] * inst_count
    diff = round(total - sum(amounts), 2)
    amounts[-1] = round(amounts[-1] + diff, 2)

    y = int(charge.get("start_year"))
    m = int(charge.get("start_month"))
    half = int(charge.get("start_half"))

    rows = []
    for i in range(inst_count):
        rows.append(
            {
                "charge_id": charge["id"],
                "due_year": y,
                "due_month": m,
                "due_half": half,
                "amount_usd": float(amounts[i]),
                "status": "pending",
            }
        )
        y, m, half = _next_fortnight_key(y, m, half)

    _sb_execute(
        supabase.table("store_charge_installments").insert(rows),
        "store insert installments",
    )

    _sb_execute(
        supabase.table("store_charges")
        .update({"status": "scheduled", "updated_at": datetime.utcnow().isoformat()})
        .eq("id", charge["id"]),
        "store set charge scheduled",
    )

    return store_get_charge(charge["id"])


@app.post("/store/charges/{charge_id}/void")
def store_void_charge(charge_id: str):
    charge = _get_charge_or_404(charge_id)

    if charge.get("status") == "void":
        return {"ok": True, "charge": charge}

    _sb_execute(
        supabase.table("store_charges")
        .update({"status": "void", "updated_at": datetime.utcnow().isoformat()})
        .eq("id", charge["id"]),
        "store void charge",
    )
    _sb_execute(
        supabase.table("store_charge_installments").update({"status": "void"}).eq("charge_id", charge["id"]),
        "store void installments",
    )

    return {"ok": True, "charge": _get_charge_or_404(charge["id"])}
# ==========================================================
# PAYROLL SOURCES + MODEL PLATFORM ACCOUNTS + REPORT GENERATE
# (NEUM-style)
# ==========================================================

def _norm_user(u: Optional[str]) -> str:
    return (u or "").strip().lower()

# -------------------------
# Schemas: Income Sources
# -------------------------
class IncomeSourceCreate(BaseModel):
    platform_id: str
    alias: str
    api_key_enc: str
    source_type: str = "studio_master"  # studio_master | individual
    username: Optional[str] = None      # master username si aplica
    monetizer_name: Optional[str] = None
    is_active: bool = True
    is_hidden: bool = False

class IncomeSourceUpdate(BaseModel):
    alias: Optional[str] = None
    api_key_enc: Optional[str] = None
    source_type: Optional[str] = None
    username: Optional[str] = None
    monetizer_name: Optional[str] = None
    is_active: Optional[bool] = None
    is_hidden: Optional[bool] = None

@app.get("/income-sources")
def list_income_sources(
    platform_id: Optional[str] = None,
    active_only: bool = True,
    include_hidden: bool = False,
):
    q = supabase.table("platform_income_sources").select("*").order("created_at", desc=True)
    if platform_id:
        q = q.eq("platform_id", _require_uuid(platform_id, "platform_id"))
    if active_only:
        q = q.eq("is_active", True)
    if not include_hidden:
        q = q.eq("is_hidden", False)

    return _sb_execute(q, "list income_sources").data or []

@app.post("/income-sources")
def create_income_source(payload: IncomeSourceCreate):
    platform_id = _require_uuid(payload.platform_id, "platform_id")
    _get_platform_or_404(platform_id)

    data = payload.model_dump()
    data["platform_id"] = platform_id

    res = _sb_execute(supabase.table("platform_income_sources").insert(data), "create income_source")
    if not res.data:
        raise HTTPException(status_code=500, detail="Insert failed: empty response")
    return res.data[0]

@app.get("/income-sources/{income_source_id}")
def get_income_source(income_source_id: str):
    income_source_id = _require_uuid(income_source_id, "income_source_id")
    res = _sb_execute(
        supabase.table("platform_income_sources").select("*").eq("id", income_source_id).single(),
        "get income_source",
    )
    if not res.data:
        raise HTTPException(status_code=404, detail="Income source not found")
    return res.data

@app.patch("/income-sources/{income_source_id}")
def update_income_source(income_source_id: str, payload: IncomeSourceUpdate):
    income_source_id = _require_uuid(income_source_id, "income_source_id")
    data = {k: v for k, v in payload.model_dump().items() if v is not None}
    if not data:
        raise HTTPException(status_code=400, detail="No fields to update")

    res = _sb_execute(
        supabase.table("platform_income_sources").update(data).eq("id", income_source_id),
        "update income_source",
    )
    if not res.data:
        raise HTTPException(status_code=404, detail="Income source not found")
    return res.data[0]

@app.delete("/income-sources/{income_source_id}")
def deactivate_income_source(income_source_id: str):
    income_source_id = _require_uuid(income_source_id, "income_source_id")
    res = _sb_execute(
        supabase.table("platform_income_sources").update({"is_active": False}).eq("id", income_source_id),
        "deactivate income_source",
    )
    if not res.data:
        raise HTTPException(status_code=404, detail="Income source not found")
    return {"ok": True, "id": income_source_id, "is_active": False}


# -------------------------
# Schemas: Model Platform Accounts
# (esta es la grilla de "Plataformas" por modelo)
# -------------------------
class ModelPlatformAccountUpsert(BaseModel):
    platform_id: str
    username: str
    password_enc: Optional[str] = None
    autologin_enabled: bool = False
    is_active: bool = True

class ModelPlatformAccountPatch(BaseModel):
    username: Optional[str] = None
    password_enc: Optional[str] = None
    autologin_enabled: Optional[bool] = None
    is_active: Optional[bool] = None

@app.get("/models/{model_id}/accounts")
def list_model_accounts(model_id: str, include_inactive: bool = False):
    model_id = _require_uuid(model_id, "model_id")
    _get_model_or_404(model_id)

    q = (
        supabase.table("model_platform_accounts")
        .select("*")
        .eq("model_id", model_id)
        .order("created_at", desc=False)
    )
    if not include_inactive:
        q = q.eq("is_active", True)

    rows = _sb_execute(q, "list model_platform_accounts").data or []
    if not rows:
        return []

    platform_ids = list({r.get("platform_id") for r in rows if r.get("platform_id")})
    pl_map = {}
    if platform_ids:
        pl = _sb_execute(
            supabase.table("platforms").select("id,name,calc_type,token_usd_rate,active").in_("id", platform_ids),
            "platforms for accounts",
        ).data or []
        pl_map = {p["id"]: p for p in pl}

    out = []
    for r in rows:
        p = pl_map.get(r.get("platform_id"), {})
        out.append(
            {
                **r,
                "platform_name": p.get("name"),
                "calc_type": p.get("calc_type"),
                "token_usd_rate": p.get("token_usd_rate"),
                "platform_active": p.get("active", True),
            }
        )
    return out

@app.post("/models/{model_id}/accounts")
def upsert_model_account(model_id: str, payload: ModelPlatformAccountUpsert):
    model_id = _require_uuid(model_id, "model_id")
    _get_model_or_404(model_id)

    platform_id = _require_uuid(payload.platform_id, "platform_id")
    _get_platform_or_404(platform_id)

    if not payload.username or not payload.username.strip():
        raise HTTPException(status_code=400, detail="username is required")

    data = payload.model_dump()
    data["model_id"] = model_id
    data["platform_id"] = platform_id
    data["username"] = payload.username.strip()

    # Upsert por (model_id, platform_id) según tu unique index
    res = _sb_execute(
        supabase.table("model_platform_accounts").upsert(data, on_conflict="model_id,platform_id"),
        "upsert model_platform_accounts",
    )
    if not res.data:
        return {"ok": True}
    return res.data[0]

@app.patch("/models/{model_id}/accounts/{platform_id}")
def patch_model_account(model_id: str, platform_id: str, payload: ModelPlatformAccountPatch):
    model_id = _require_uuid(model_id, "model_id")
    platform_id = _require_uuid(platform_id, "platform_id")
    _get_model_or_404(model_id)
    _get_platform_or_404(platform_id)

    data = {k: v for k, v in payload.model_dump().items() if v is not None}
    if not data:
        raise HTTPException(status_code=400, detail="No fields to update")

    if "username" in data and data["username"] is not None:
        if not str(data["username"]).strip():
            raise HTTPException(status_code=400, detail="username cannot be empty")
        data["username"] = str(data["username"]).strip()

    res = _sb_execute(
        supabase.table("model_platform_accounts")
        .update(data)
        .eq("model_id", model_id)
        .eq("platform_id", platform_id),
        "patch model_platform_accounts",
    )
    if not res.data:
        raise HTTPException(status_code=404, detail="Account not found for model/platform")
    return res.data[0]

@app.delete("/models/{model_id}/accounts/{platform_id}")
def deactivate_model_account(model_id: str, platform_id: str):
    model_id = _require_uuid(model_id, "model_id")
    platform_id = _require_uuid(platform_id, "platform_id")
    _get_model_or_404(model_id)
    _get_platform_or_404(platform_id)

    res = _sb_execute(
        supabase.table("model_platform_accounts")
        .update({"is_active": False})
        .eq("model_id", model_id)
        .eq("platform_id", platform_id),
        "deactivate model_platform_accounts",
    )
    if not res.data:
        raise HTTPException(status_code=404, detail="Account not found for model/platform")
    return {"ok": True, "model_id": model_id, "platform_id": platform_id, "is_active": False}


# -------------------------
# Payroll Generate
# -------------------------
class PayrollRowOverride(BaseModel):
    username: str
    tokens: float = 0
    raw: Optional[Dict[str, Any]] = None

class PayrollGenerateRequest(BaseModel):
    income_source_id: str
    date_from: date
    date_to: date

    # ✅ Para probar YA sin integración: manda rows_override
    rows_override: Optional[List[PayrollRowOverride]] = None

    # ✅ Para ver qué trae "por debajo" (como tu duda del Network)
    debug_return_raw: bool = True

@app.post("/payroll/generate")
def payroll_generate(payload: PayrollGenerateRequest):
    income_source_id = _require_uuid(payload.income_source_id, "income_source_id")

    # 1) Traer source
    src_res = _sb_execute(
        supabase.table("platform_income_sources").select("*").eq("id", income_source_id).single(),
        "get income_source for payroll",
    )
    src = src_res.data
    if not src:
        raise HTTPException(status_code=404, detail="Income source not found")

    platform_id = _require_uuid(src.get("platform_id"), "platform_id")
    _get_platform_or_404(platform_id)

    # 2) Crear report (running)
    report_ins = {
        "income_source_id": income_source_id,
        "date_from": str(payload.date_from),
        "date_to": str(payload.date_to),
        "status": "running",
        "warnings_count": 0,
        "errors_count": 0,
        "created_by": None,
        "created_at": datetime.utcnow().isoformat(),
    }
    rep_res = _sb_execute(
        supabase.table("platform_reports").insert(report_ins),
        "insert platform_reports",
    )
    if not rep_res.data:
        raise HTTPException(status_code=500, detail="Report insert failed")
    report = rep_res.data[0]
    report_id = report["id"]

    # 3) Cargar mapa username->model_id para esa plataforma
    accounts = _sb_execute(
        supabase.table("model_platform_accounts")
        .select("model_id, platform_id, username, is_active")
        .eq("platform_id", platform_id)
        .eq("is_active", True),
        "load model_platform_accounts for matching",
    ).data or []

    user_map = {}  # lower(username)->model_id
    for a in accounts:
        u = _norm_user(a.get("username"))
        if u:
            user_map[u] = a.get("model_id")

    # 4) Obtener filas externas:
    #    - si rows_override viene: usamos eso (para probar YA)
    #    - si no viene: dejamos placeholder (sin romper)
    external_rows: List[Dict[str, Any]] = []
    if payload.rows_override:
        for r in payload.rows_override:
            external_rows.append({"username": r.username, "tokens": r.tokens, "raw": r.raw})
    else:
        # Placeholder: todavía no integramos la API real de la plataforma
        external_rows = []

    # 5) Insertar report_rows
    rows_to_insert = []
    warnings = 0

    for r in external_rows:
        ext_user = (r.get("username") or "").strip()
        tokens = float(r.get("tokens") or 0)
        raw = r.get("raw")

        matched_model_id = user_map.get(_norm_user(ext_user))
        if tokens == 0:
            status = "no_data"
        elif matched_model_id:
            status = "matched"
        else:
            status = "unmatched"
            warnings += 1

        row_ins = {
            "report_id": report_id,
            "platform_id": platform_id,
            "external_username": ext_user,
            "tokens": tokens,
            "raw": raw if payload.debug_return_raw else None,
            "matched_model_id": matched_model_id,
            "status": status,
            "message": None if matched_model_id else ("No match in model_platform_accounts" if tokens != 0 else "0 tokens"),
            "created_at": datetime.utcnow().isoformat(),
        }
        rows_to_insert.append(row_ins)

    if rows_to_insert:
        _sb_execute(
            supabase.table("platform_report_rows").insert(rows_to_insert),
            "insert platform_report_rows",
        )

    # 6) Marcar report done
    rep_upd = _sb_execute(
        supabase.table("platform_reports")
        .update(
            {
                "status": "done",
                "warnings_count": warnings,
                "errors_count": 0,
                "finished_at": datetime.utcnow().isoformat(),
            }
        )
        .eq("id", report_id),
        "update platform_reports done",
    )

    # 7) Respuesta
    rows = _sb_execute(
        supabase.table("platform_report_rows")
        .select("*")
        .eq("report_id", report_id)
        .order("created_at", desc=False),
        "load report rows",
    ).data or []

    return {
        "report": rep_upd.data[0] if rep_upd.data else report,
        "source": {
            "id": src.get("id"),
            "platform_id": platform_id,
            "alias": src.get("alias"),
            "source_type": src.get("source_type"),
            "username": src.get("username"),
            "is_active": src.get("is_active"),
        },
        "summary": {
            "rows": len(rows),
            "warnings_unmatched": warnings,
            "note": "Si rows_override es vacío y aún no hay integración, rows puede salir 0 (normal).",
        },
        "rows": rows,
    }

