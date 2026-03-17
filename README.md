It is a full-stack system that acts as the "brain" for autonomous drone operations. It processes telemetry data through a three-layer decision pipeline — computer vision, hard-coded safety rules, and machine learning — then explains every decision to the human operator in real time. The system prioritizes transparent autonomy: operators don't just see what the drone decides, they see why.

Decision flow: Safety layer runs first and if a hard constraint is violated, the C++ engine overrides the ML model entirely. If the environment is "SAFE," the Random Forest evaluates all 5 telemetry features and SHAP deconstructs the prediction into per-feature importance percentages.

Stack:
-Frontend: React 19, Tailwind CSS, Recharts, Framer Motion  
-API: FastAPI, Uvicorn 
-Safety Engine: C++ (compiled via Pybind11)
-scikit-learn (RandomForest)
-SHAP (TreeExplainer)
-Computer Vision: OpenCV
-Infra: Docker, Redis 


to run:

Clone the repository
cd AI-driven-drone-management-system
docker-compose up --build -d

Frontend (HUD):  http://localhost:3000
Backend (API):   http://localhost:8000
API Docs:        http://localhost:8000/docs

1. Open the frontend at `http://localhost:3000`
2. Adjust the telemetry sliders on the left panel (battery, altitude, velocity, GPS signal, obstacle distance)
3. Watch the AI decision update in real time on the right panel
4. The SHAP bar chart below shows exactly which features drove the decision

to test API directly:

curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "battery_level": 85,
    "altitude": 100,
    "velocity": 12,
    "gps_signal": 9,
    "obstacle_distance": 25
  }'
