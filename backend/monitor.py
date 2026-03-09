import pandas as pd
import numpy as np

def run_drift_analysis():
    try:
        df = pd.read_csv('flight_log.csv')
        if len(df) < 5:
            print("Status: COLLECTING_DATA (Need more logs for analysis)")
            return

        bounds = {
            'battery_level': (10, 100),
            'altitude': (0, 500),
            'gps_signal': (0, 10)
        }

        print("--- SYSTEM DRIFT REPORT ---")
        drift_found = False

        for feature, (min_val, max_val) in bounds.items():
            current_avg = df[feature].tail(10).mean() # Look at last 10 readings
            
            if current_avg < min_val or current_avg > max_val:
                print(f" {feature} drifting Avg: {current_avg:.2f} (Expected {min_val}-{max_val})")
                drift_found = True
        
        if not drift_found:
            print(" stable (telemetry within training distribution)")
        
        print(f"Total Flight Logs Analyzed: {len(df)}")
        print("---------------------------")

    except FileNotFoundError:
        print("no flight logs found.")

if __name__ == "__main__":
    run_drift_analysis()