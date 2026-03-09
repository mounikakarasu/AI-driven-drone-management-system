This is a small full-stack project that simulates autonomous drone decision making from telemetry data. A machine-learning model receives drone telemetry (battery, altitude, velocity, GPS signal, obstacle distance) and outputs a flight decision - continue, holdm return home
The React dashboard visualizes the decision and explains the model logic using SHAP feature attribution.
Stack
Backend: FastAPI + Scikit-Learn
Frontend: React + Tailwind HUD interface
Explainability: SHAP
Infrastructure: Docker Compose

To run it:

1. Clone
2. Start the system (docker build)
3. Open the interface
Frontend dashboard: http://localhost:3000
API documentation: http://localhost:8000/docs

to see it working

1. Open the dashboard in the browser.
2. Move the telemetry sliders (battery, altitude, GPS, etc.).
3. The backend ML model returns a real-time flight decision.

The SHAP chart explains which telemetry factors influenced the decision.

Each prediction is also written to a flight log csv file acting as a black-box recorder.
Additionally, Monitor is run to check logged telemetry for anomalies or drift. the command is docker-compose exec api python monitor.py
