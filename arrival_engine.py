import numpy as np
import scipy.stats as st
from datetime import timedelta
import pandas as pd

class ArrivalEngine:
    def __init__(self, log = None, seed = 1):
        self.seed = seed
        np.random.seed(seed)
        self.global_scale = 1002.20
        self.weekday_multipliers = {
            0: 1.4, #Monday
            1: 1.2, #Tuesday
            2: 1.1, #Wednesday
            3: 1.1, #Thursday
            4: 1.3, #Friday
            5: 0.5, #Saturday
            6: 0.3 #Sunday
        }

        self.hourly_multipliers ={
            0: 0.3, 1: 0.2, 2: 0.1, 3: 0.1, 4: 0.2, 5: 0.4,
            6: 0.8, 7: 1.2, 8: 1.5, 9: 1.8, 10: 2.0, 11: 1.9,
            12: 1.7, 13: 1.8, 14: 1.9, 15: 1.8, 16: 1.6, 17: 1.4,
            18: 1.1, 19: 0.9, 20: 0.7, 21: 0.6, 22: 0.5, 23: 0.4
        }
        if log is not None and not log.empty:
            self.train(log)

    def train(self, log_df):
        try:
            log_df['time:timestamp'] = pd.to_datetime(log_df['time:timestamp'])

            case_starts = log_df.sort_values('time:timestamp').groupby('case:concept:name').first().reset_index()
            case_starts = case_starts.sort_values('time:timestamp')

            case_starts['inter_arrival'] = case_starts['time:timestamp'].diff().dt.total_seconds()
            inter_arrivals = case_starts['inter_arrival'].dropna().values

            if len(inter_arrivals) > 0:
                self.global_scale = np.mean(inter_arrivals)
                print(f"[ArrivalEngine] Trained successfully. Base Global Scale: {self.global_scale:.2f}s")
        except Exception as e:
            print(f"[ArrivalEngine] Train failed, using default global scale. Error: {e}")

    def nextArrivalTime(self, current_time) -> timedelta:
        if hasattr(current_time, 'weekday') and hasattr(current_time, 'hour'):
            weekday = current_time.weekday()
            hour = current_time.hour
        else:
            total_hours = int(float(current_time) / 3600)
            hour = total_hours % 24
            weekday = (total_hours // 24) % 7
        
        day_mult = self.weekday_multipliers.get(weekday, 1.0)
        hour_mult = self.hourly_multipliers.get(hour, 1.0)
        dynamic_scale = self.global_scale / (day_mult * hour_mult)
        seconds = np.random.exponential(scale=dynamic_scale)

        return timedelta(seconds=float(seconds))
        