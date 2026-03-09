from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware # Add this import
from schemas import DroneTelemetry
from processor import brain
import uvicorn

app = FastAPI(title="AI-Drone Command Center")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, you'd specify http://localhost:3000
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def health_check():
    return {"status": "online", "system": "Drone-V2"}

@app.post("/predict")
async def predict_action(data: DroneTelemetry):
    result = brain.process_telemetry(data.model_dump())
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)