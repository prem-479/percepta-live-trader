from fastapi import APIRouter

router = APIRouter()

@router.get("/latest")
def get_latest_news():
    return {"news": []}
