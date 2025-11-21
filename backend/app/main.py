from fastapi.middleware.cors import CORSMiddleware
import sys
import os

# Add project ROOT directory to Python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT_DIR)

from fastapi import FastAPI

# Initialize FastAPI FIRST
app = FastAPI(title="Percepta API")

# CORS settings
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import routers AFTER app is created
from app.api import (
    routes_signals,
    routes_news,
    routes_risk,
    routes_auth,
    routes_coach
)

# Register routes
app.include_router(routes_signals.router, prefix="/signals", tags=["signals"])
app.include_router(routes_news.router, prefix="/news", tags=["news"])
app.include_router(routes_risk.router, prefix="/risk", tags=["risk"])
app.include_router(routes_auth.router, prefix="/auth", tags=["auth"])
app.include_router(routes_coach.router, prefix="/coach", tags=["coach"])


@app.get("/")
def root():
    return {"message": "Percepta backend running 🚀"}
