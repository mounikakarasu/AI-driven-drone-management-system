import os
import joblib
import pandas as pd
import shap
import numpy as np
from backend import drone_safety

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "drone_model.pkl")


class DroneBrain:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
        self.explainer = shap.TreeExplainer(self.model)
        self.classes = ["HOLD", "CONTINUE", "RETURN_HOME"]

    def process_telemetry(self, data_dict):

        safety_status = drone_safety.evaluate_safety(
            float(data_dict["battery_level"]),
            float(data_dict["obstacle_distance"]),
            int(data_dict["gps_signal"])
        )

        if safety_status != "SAFE":
            return {
                "decision": safety_status,
                "confidence": 1.0,
                "primary_factor": "C++_SAFETY_ENGINE",
                "logic_breakdown": {k: 0.0 for k in data_dict.keys()}
            }

        model_input = {
            "battery_level": data_dict["battery_level"],
            "altitude": data_dict["altitude"],
            "velocity": data_dict["velocity"],
            "gps_signal": data_dict["gps_signal"],
            "obstacle_distance": data_dict["obstacle_distance"]
        }

        df = pd.DataFrame([model_input])

        pred_idx = int(self.model.predict(df)[0])
        probs = self.model.predict_proba(df)[0]

        return {
            "decision": self.classes[pred_idx],
            "confidence": round(float(np.max(probs)), 2),
            "primary_factor": "ML_MODEL",
            "logic_breakdown": model_input
        }


brain = DroneBrain()