from fastapi import APIRouter

router = APIRouter()

@router.post("/ask")
def ask():
    return {"answer": "This is a placeholder financial coach response."}
