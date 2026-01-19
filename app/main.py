# =========================
# PLATFORMS Schemas
# =========================
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


# =========================
# PLATFORMS Endpoints
# =========================
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
