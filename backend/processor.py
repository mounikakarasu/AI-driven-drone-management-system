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
        # 1. Check C++ Safety Engine
        safety_status = drone_safety.evaluate_safety(
            data_dict['battery_level'],
            data_dict['obstacle_distance'],
            data_dict['gps_signal']
        )
        
        # 2. Prepare Data for SHAP
        df = pd.DataFrame([data_dict])
        
        # 3. Get Prediction and SHAP Values
        probs = self.model.predict_proba(df)[0]
        pred_idx = int(np.argmax(probs))
        shap_values = self.explainer.shap_values(df)

        # Handle different SHAP output formats (RandomForest vs others)
        if isinstance(shap_values, list):
            # Use values for the predicted class
            current_shap = shap_values[pred_idx][0]
        else:
            # Multi-class output in one array
            current_shap = shap_values[0][pred_idx] if len(shap_values.shape) > 2 else shap_values[0]

        # 4. Map SHAP to 0-100 scale for the Frontend
        logic_breakdown = {}
        raw_impacts = [abs(val) for val in current_shap]
        total = sum(raw_impacts) if sum(raw_impacts) > 0 else 1
        
        for i, col in enumerate(df.columns):
            # Calculate percentage of total impact
            percentage = (abs(current_shap[i]) / total) * 100
            logic_breakdown[col] = round(percentage, 2)

        decision = safety_status if safety_status != "SAFE" else self.classes[pred_idx]
        
        return {
            "decision": decision,
            "confidence": round(float(np.max(probs)) * 100, 2),
            "primary_factor": "C++_SAFETY_ENGINE" if safety_status != "SAFE" else "NEURAL_INFERENCE",
            "logic_breakdown": logic_breakdown
        }

brain = DroneBrain()
