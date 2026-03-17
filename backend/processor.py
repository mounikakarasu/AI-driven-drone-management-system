import joblib
import pandas as pd
import shap
import numpy as np
import os

try:
    import drone_safety
    SAFETY_ENGINE = "C++"
except ImportError:
    SAFETY_ENGINE = "PYTHON"
    print("[WARN] C++ drone_safety module not found — using Python fallback")

    class _DroneSafetyFallback:
        @staticmethod
        def evaluate_safety(battery: float, obstacle_dist: float, gps_signal: int) -> str:
            if battery < 20.0 or gps_signal < 3:
                return "RETURN_HOME"
            if obstacle_dist < 5.0:
                return "HOLD"
            return "SAFE"

    drone_safety = _DroneSafetyFallback()

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

        # Extract SHAP values for the predicted class
        # shap_values can be: list of arrays (one per class), or 3D array
        try:
            if isinstance(shap_values, list):
                # List of arrays: each is (n_samples, n_features)
                sv = np.array(shap_values[pred_idx])
                current_shap = sv.flatten()[:len(self.feature_names)]
            elif shap_values.ndim == 3:
                # 3D array: (n_samples, n_features, n_classes) or (n_samples, n_classes, n_features)
                if shap_values.shape[1] == len(self.feature_names):
                    current_shap = shap_values[0, :, pred_idx]
                else:
                    current_shap = shap_values[0, pred_idx, :]
            else:
                # 2D array: (n_samples, n_features)
                current_shap = shap_values[0]
        except Exception:
            # Fallback: equal weights
            current_shap = np.ones(len(self.feature_names))

        # 5. Normalize feature impacts → percentages
        raw_impacts = [abs(float(v)) for v in current_shap[:len(self.feature_names)]]
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
