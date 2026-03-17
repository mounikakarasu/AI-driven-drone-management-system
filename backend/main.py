import cv2
import os
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from schemas import TelemetryData
from processor import brain
from cv_engine import detect_obstacle_distance

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

current_obstacle_distance = 100.0
IS_PROD = os.getenv("ENV") == "production"

camera = cv2.VideoCapture(0)

def gen_frames():
    global current_obstacle_distance
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        dist, bbox = detect_obstacle_distance(frame)
        current_obstacle_distance = dist
        
        if bbox:
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, f"OBSTACLE: {dist}m", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "PATH CLEAR", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/predict")
async def predict(data: TelemetryData):
    telemetry_data = data.dict()
    telemetry_data['obstacle_distance'] = current_obstacle_distance
    
    result = brain.process_telemetry(telemetry_data)
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)