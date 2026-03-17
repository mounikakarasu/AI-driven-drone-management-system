import joblib
import pandas as pd
import shap
import numpy as np
import drone_safety

class DroneBrain:
    def __init__(self):
        self.model = joblib.load('models/drone_model.pkl')
        self.explainer = shap.TreeExplainer(self.model)
        self.classes = ["HOLD", "CONTINUE", "RETURN_HOME"]

    def process_telemetry(self, data_dict):
        safety_status = drone_safety.evaluate_safety(
            data_dict['battery_level'],
            data_dict['obstacle_distance'],
            data_dict['gps_signal']
        )
        
        df = pd.DataFrame([data_dict])
        
        probs = self.model.predict_proba(df)[0]
        pred_idx = int(np.argmax(probs))
        
        shap_output = self.explainer.shap_values(df)
        
        if isinstance(shap_output, list):
            current_shap = shap_output[pred_idx][0]
        else:
            current_shap = shap_output[0, pred_idx, :]

        logic_breakdown = {}
        raw_impacts = [abs(float(val)) for val in current_shap]
        total_impact = sum(raw_impacts) if sum(raw_impacts) > 0 else 1.0
        
        columns = ['battery_level', 'altitude', 'velocity', 'gps_signal', 'obstacle_distance']
        for i, col in enumerate(columns):
            impact_percent = (raw_impacts[i] / total_impact) * 100
            logic_breakdown[col] = round(impact_percent, 2)

        decision = safety_status if safety_status != "SAFE" else self.classes[pred_idx]
        
        return {
            "decision": decision,
            "confidence": round(float(np.max(probs)) * 100, 2),
            "primary_factor": "C++_SAFETY_ENGINE" if safety_status != "SAFE" else "NEURAL_INFERENCE",
            "logic_breakdown": logic_breakdown
        }

brain = DroneBrain()
