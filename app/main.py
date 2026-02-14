import os
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
