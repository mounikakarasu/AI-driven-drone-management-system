import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

os.makedirs('backend/models', exist_ok=True)

data = {
    'battery_level':     [100, 50, 10, 80, 15, 90, 5, 20, 80, 40],
    'altitude':          [100, 200, 50, 400, 300, 100, 50, 100, 150, 250],
    'velocity':          [10, 20, 5, 30, 25, 10, 2, 10, 15, 12],
    'gps_signal':        [10, 8, 2, 9, 1, 10, 0, 5, 8, 7],
    'obstacle_distance': [50, 30, 20, 2, 40, 10, 50, 1, 35, 12],
    'target':            [1, 1, 2, 0, 2, 1, 2, 0, 1, 1] # 0:HOLD, 1:CONTINUE, 2:RETURN_HOME
}

df = pd.DataFrame(data)
X = df.drop('target', axis=1)
y = df['target']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

joblib.dump(model, 'backend/models/drone_model.pkl')
print("Model trained and saved with 5 features.")