import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

data = {
    'battery_level': np.random.uniform(10, 100, 1000),
    'altitude': np.random.uniform(0, 500, 1000),
    'velocity': np.random.uniform(0, 50, 1000),
    'gps_signal': np.random.randint(0, 10, 1000),
    'obstacle_distance': np.random.uniform(0, 50, 1000)
}
df = pd.DataFrame(data)

def labeling(row):
    if row['battery_level'] < 20 or row['gps_signal'] < 3: return 2
    if row['obstacle_distance'] < 5: return 0
    return 1

df['action'] = df.apply(labeling, axis=1)

X = df.drop('action', axis=1)
y = df['action']
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/drone_model.pkl')
print("Model trained and saved to models/drone_model.pkl")