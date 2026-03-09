import joblib
import pandas as pd
import shap
import numpy as np
import os
import csv
from datetime import datetime


class DroneBrain:
    def __init__(self):
        model_path = os.path.join("models", "drone_model.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. Please run train_model.py first."
            )

        self.model = joblib.load(model_path)
        self.explainer = shap.TreeExplainer(self.model)

        self.classes = ["HOLD", "CONTINUE", "RETURN_HOME"]

    def process_telemetry(self, data_dict):
        df = pd.DataFrame([data_dict])

        pred_idx = int(self.model.predict(df)[0])
        probs = self.model.predict_proba(df)[0]

        shap_results = self.explainer.shap_values(df)

        if isinstance(shap_results, list):
            class_shap = shap_results[pred_idx]
        else:
            class_shap = shap_results[..., pred_idx]

        class_shap = np.array(class_shap).flatten()

        feature_importance = dict(zip(df.columns, class_shap))

        top_contributor = max(
            feature_importance,
            key=lambda k: abs(feature_importance[k])
        )

        log_entry = {
            **data_dict,
            "decision": self.classes[pred_idx],
            "timestamp": datetime.now()
        }

        self.log_to_csv(log_entry)

        return {
            "decision": self.classes[pred_idx],
            "confidence": round(float(np.max(probs)), 2),
            "primary_factor": top_contributor,
            "logic_breakdown": {
                k: round(float(v), 3)
                for k, v in feature_importance.items()
            }
        }

    def log_to_csv(self, entry):
        file_exists = os.path.isfile("flight_log.csv")

        with open("flight_log.csv", "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=entry.keys())

            if not file_exists:
                writer.writeheader()

            writer.writerow(entry)


brain = DroneBrain()