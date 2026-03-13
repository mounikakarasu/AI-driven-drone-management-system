from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import cv2

from backend.cv_engine import detect_obstacle_distance
from backend.processor import brain
from backend.schemas import TelemetryData

app = FastAPI()

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

for _ in range(10):
    camera.read()

current_obstacle_distance = 100.0


def gen_frames():
    global current_obstacle_distance

    while True:
        success, frame = camera.read()

        if not success:
            print("Failed to capture frame")
            continue

        dist, bbox = detect_obstacle_distance(frame)

        current_obstacle_distance = dist

        if bbox:
            x, y, w, h = bbox

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            cv2.putText(
                frame,
                f"OBSTACLE: {dist}m",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )
        else:
            cv2.putText(
                frame,
                "PATH CLEAR",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        ret, buffer = cv2.imencode(".jpg", frame)

        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
        )


@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        gen_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.post("/predict")
async def predict(data: TelemetryData):

    telemetry_data = data.dict()

    telemetry_data["obstacle_distance"] = current_obstacle_distance

    result = brain.process_telemetry(telemetry_data)

    return result