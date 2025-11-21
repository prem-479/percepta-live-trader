from fastapi import APIRouter

router = APIRouter()

@router.post("/check")
def check_risk():
    return {"ok": True, "reason": "risk engine placeholder"}
