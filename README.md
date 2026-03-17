# Drone OS // Command Center

An autonomous drone decision engine with a hybrid C++/Python pipeline, explainable AI, and a real-time tactical HUD.

## Description

Drone OS is a full-stack system that acts as the "brain" for autonomous drone operations. It processes telemetry data through a three-layer decision pipeline — computer vision, hard-coded safety rules, and machine learning — then explains every decision to the human operator in real time. The system prioritizes transparent autonomy: operators don't just see what the drone decides, they see why.

## Architecture & Design

```
┌──────────────────────────────────────────────────────────────┐
│  FRONTEND (React + Tailwind)  :3000                          │
│  Tactical HUD with telemetry sliders + SHAP visualization    │
└──────────────────┬───────────────────────────────────────────┘
                   │ POST /predict
┌──────────────────▼───────────────────────────────────────────┐
│  BACKEND (FastAPI + Uvicorn)  :8000                          │
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Layer 1: PERCEPTION (OpenCV)                            │ │
│  │ HSV color detection → contour analysis → distance calc  │ │
│  ├─────────────────────────────────────────────────────────┤ │
│  │ Layer 2: SAFETY (C++ via Pybind11)                      │ │
│  │ Hard constraints: battery < 20% → RETURN_HOME           │ │
│  │                   GPS < 3 → RETURN_HOME                 │ │
│  │                   obstacle < 5m → HOLD                  │ │
│  ├─────────────────────────────────────────────────────────┤ │
│  │ Layer 3: INTELLIGENCE (scikit-learn + SHAP)             │ │
│  │ RandomForest classifier → SHAP TreeExplainer            │ │
│  │ Returns: decision, confidence, feature importance       │ │
│  └─────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

**Decision flow:** Safety layer runs first — if a hard constraint is violated, the C++ engine overrides the ML model entirely. If the environment is "SAFE," the Random Forest evaluates all 5 telemetry features and SHAP deconstructs the prediction into per-feature importance percentages.

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Frontend | React 19, Tailwind CSS, Recharts, Framer Motion | Tactical HUD dashboard |
| API | FastAPI, Uvicorn | Low-latency REST API |
| Safety Engine | C++ (compiled via Pybind11) | Deterministic safety overrides |
| ML Model | scikit-learn (RandomForest) | Flight decision classification |
| Explainability | SHAP (TreeExplainer) | Per-feature decision breakdown |
| Computer Vision | OpenCV | Obstacle detection via HSV + contours |
| Infrastructure | Docker, Docker Compose, Redis | Containerized deployment |

## Tools & Libraries

**Backend:** Python 3.11, FastAPI, Uvicorn, scikit-learn, SHAP, OpenCV, Pandas, NumPy, Pybind11, Joblib

**Frontend:** React, Tailwind CSS, Recharts, Framer Motion, Lucide React

**DevOps:** Docker, Docker Compose, Redis

## How to Run

### Prerequisites
- Docker and Docker Compose installed
- Git

### Quick Start

```bash
# Clone the repository
git clone <repo-url>
cd drone_system

# Build and start all services
docker-compose up --build -d

# Services will be available at:
# Frontend (HUD):  http://localhost:3000
# Backend (API):   http://localhost:8000
# API Docs:        http://localhost:8000/docs
```

### Usage

1. Open the frontend at `http://localhost:3000`
2. Adjust the telemetry sliders on the left panel (battery, altitude, velocity, GPS signal, obstacle distance)
3. Watch the AI decision update in real time on the right panel
4. The SHAP bar chart below shows exactly which features drove the decision

### Test the API Directly

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "battery_level": 85,
    "altitude": 100,
    "velocity": 12,
    "gps_signal": 9,
    "obstacle_distance": 25
  }'
```

### Stop

```bash
docker-compose down
```

## Project Structure

```
drone_system/
├── backend/
│   ├── core/
│   │   └── safety_engine.cpp    # C++ safety rules (Pybind11)
│   ├── models/
│   │   └── drone_model.pkl      # Trained RandomForest model
│   ├── main.py                  # FastAPI server + endpoints
│   ├── processor.py             # DroneBrain: ML + SHAP pipeline
│   ├── cv_engine.py             # OpenCV obstacle detection
│   ├── schemas.py               # Pydantic request models
│   ├── train_model.py           # Model training script
│   ├── setup.py                 # C++ module compilation config
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   └── App.js               # React tactical HUD
│   ├── package.json
│   └── Dockerfile
└── docker-compose.yml
```
