import joblib
import pandas as pd
import shap
import numpy as np
import drone_safety
import os

class DroneBrain:
    def __init__(self):
        model_path = os.path.join(os.path.dirname(__file__), 'models/drone_model.pkl')
        self.model = joblib.load(model_path)
        self.explainer = shap.TreeExplainer(self.model)
        self.classes = ["HOLD", "CONTINUE", "RETURN_HOME"]
        self.feature_names = ['battery_level', 'altitude', 'velocity', 'gps_signal', 'obstacle_distance']

    def process_telemetry(self, data_dict):
        # 1. Safety check (rule-based override)
        safety_status = drone_safety.evaluate_safety(
            data_dict['battery_level'],
            data_dict['obstacle_distance'],
            data_dict['gps_signal']
        )

        # 2. Prepare Data for ML & SHAP (strict feature order)
        df = pd.DataFrame([data_dict], columns=self.feature_names)

        # 3. Model prediction
        probs = self.model.predict_proba(df)[0]
        pred_idx = int(np.argmax(probs))

        # 4. SHAP values
        shap_values = self.explainer.shap_values(df)

        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            current_shap = shap_values[pred_idx][0]
        else:
            # Covers newer SHAP versions
            try:
                current_shap = shap_values[0, pred_idx, :]
            except:
                current_shap = shap_values[0]

        # 5. Normalize feature impacts → percentages
        raw_impacts = [abs(float(v)) for v in current_shap]
        total = sum(raw_impacts) if sum(raw_impacts) > 0 else 1.0

        logic_breakdown = {
            col: round((raw_impacts[i] / total) * 100, 2)
            for i, col in enumerate(self.feature_names)
        }

        # 6. Final decision (safety overrides ML)
        decision = safety_status if safety_status != "SAFE" else self.classes[pred_idx]

        return {
            "decision": decision,
            "confidence": round(float(np.max(probs)) * 100, 2),
            "primary_factor": "C++_SAFETY_ENGINE" if safety_status != "SAFE" else "NEURAL_INFERENCE",
            "logic_breakdown": logic_breakdown
        }

brain = DroneBrain()
