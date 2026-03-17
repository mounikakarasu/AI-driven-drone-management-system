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
        shap_values = self.explainer.shap_values(df)
        pred_idx = int(self.model.predict(df)[0])
        
        if isinstance(shap_values, list):
            current_shap = shap_values[pred_idx][0]
        else:
            current_shap = shap_values[0]

        logic_breakdown = {}
        for i, col in enumerate(df.columns):
            impact = abs(float(current_shap[i]))
            logic_breakdown[col] = round(impact, 4)

        decision = safety_status if safety_status != "SAFE" else self.classes[pred_idx]
        
        return {
            "decision": decision,
            "confidence": round(float(np.max(self.model.predict_proba(df)[0])), 2),
            "primary_factor": "C++_SAFETY_ENGINE" if safety_status != "SAFE" else "NEURAL_INFERENCE",
            "logic_breakdown": logic_breakdown
        }

brain = DroneBrain()
