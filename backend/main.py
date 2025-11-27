from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="Percepta Live Trader API")

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health endpoint
@app.get("/")
def home():
    return {"status": "backend running", "message": "Percepta API online"}

# Test ML endpoint (mock example)
@app.get("/predict")
def predict(symbol: str = "RELIANCE"):
    return {
        "symbol": symbol,
        "signal": "BUY",
        "confidence": 0.87,
        "model": "RandomForest_v1"
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
