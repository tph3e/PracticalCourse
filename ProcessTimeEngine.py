from Helper import *
import pm4py
import pandas as pd
from datetime import datetime, timedelta
from scipy import stats
import numpy as np
import json
import os

PATH_MODEL = "data/processing_time_models_basic.json"

class ProcessTimeEngine:

    models_basic = dict()
    rndm_state = 0

    def __init__(self, log: pd.DataFrame=pd.DataFrame()):

        #if the models are already trained they do not have to be retrained
        if os.path.exists(PATH_MODEL):
            with open(PATH_MODEL, "r") as f:
                self.models_basic = json.load(f)
            return
        #if the log is empty (no propper log was given), the log is manually loaded
        if(log.empty):
            log = pm4py.convert_to_dataframe(pm4py.read_xes('data/logData.xes', variant="r4pm"))
        
        event_times = self.format_data(log)
        self.train(event_times)

    def train(self, event_times):

        for activity, group in event_times.groupby("concept:name"):
            for kind in ("processing", "waiting"):
                potential_distribs = []
                times_data = group[kind+"_time"]
                null_count = sum(times_data==0)
                times_data=times_data[times_data>0]
                
                #if there is not enough data to make a good statement, return
                if(len(times_data)<4):
                    continue

                is_constant = np.unique(times_data).size == 1
                if is_constant:
                    continue
                #pois
                lam = times_data.mean()
                loglikelihood = np.sum(stats.poisson.logpmf(times_data, lam))
                aic = 2*1-2*loglikelihood
                potential_distribs.append({
                    "distribution": "poisson",
                    "parameters": {"lambda": lam},
                    "AIC": aic
                })
                #gamma
                shape, loc, scale = stats.gamma.fit(times_data, floc=0)
                loglikelihood = np.sum(stats.gamma.logpdf(times_data, shape, loc, scale))
                aic = 2*2-2*loglikelihood
                potential_distribs.append({
                    "distribution": "gamma",
                    "parameters": {"shape": shape, "scale": scale},
                    "AIC": aic
                })


                #lognorm
                shape, loc, scale = stats.lognorm.fit(times_data, floc=0)
                loglikelihood = np.sum(stats.lognorm.logpdf(times_data, shape, loc, scale))
                aic = 2*2-2*loglikelihood
                potential_distribs.append({
                    "distribution": "lognorm",
                    "parameters": {"shape": shape, "scale": scale},
                    "AIC": aic
                })

                potential_distribs = pd.DataFrame(potential_distribs)

                # Best distribution nach AIC
                distrib_best = potential_distribs.loc[potential_distribs["AIC"].idxmin(), :]

                self.models_basic[str(activity)+ " "+str(kind)]={
                    "distribution": distrib_best["distribution"],
                    "parameters": distrib_best["parameters"],
                    "0-proportion": null_count/len(times_data)
                }
        print("Trained the ProcessTime model")
        with open(PATH_MODEL, "w") as f:
            json.dump(self.models_basic, f, indent=4)

    def format_data(self, log: pd.DataFrame):
        # Sort events chronologisch
        log = log.sort_values(["case:concept:name", "concept:name", "time:timestamp"])
        event_times = []
        # Gruppieren nach Case und Aktivität
        for (case, activity), event_data in log.groupby(["case:concept:name", "concept:name"]):
            event_data = event_data.sort_values("time:timestamp")
            active_start = None
            total_active_time = timedelta(0)
            activity_waiting = None
            total_waiting_time = timedelta(0)
            for _, log_entry in event_data.iterrows():
                event = log_entry["lifecycle:transition"]
                if event == "start":
                    active_start = log_entry["time:timestamp"]
                    total_active_time = timedelta(0)
                elif event == "resume":
                    active_start = log_entry["time:timestamp"]

                    total_waiting_time = activity_waiting - log_entry["time:timestamp"]
                elif event == "suspend":
                    if active_start is not None:
                        total_active_time += (log_entry["time:timestamp"] - active_start)
                        active_start = None

                        activity_waiting = log_entry["time:timestamp"]
                elif event == "complete":
                    if active_start is not None:
                        total_active_time += (log_entry["time:timestamp"] - active_start)
                        active_start = None
                        #resets activity so that if it reoccurs another entry is written
                    event_times.append({
                        "case:concept:name": case,
                        "concept:name": activity,
                        "processing_time": total_active_time.seconds,
                        "waiting_time": total_waiting_time.seconds
                    })
        return pd.DataFrame.from_records(event_times)
    
    def sample_distrib(self, distrib, param):
        if distrib == "poisson":
            return stats.poisson.rvs(mu = param["lambda"], random_state= self.rndm_state)
        if distrib == "gamma":
            return stats.gamma.rvs(param["shape"], loc=0, scale=param["scale"], random_state= self.rndm_state)
        if distrib == "lognorm":
            return stats.lognorm.rvs(param["shape"], loc=0, scale=param["scale"], random_state= self.rndm_state)
        return timedelta(0)
    
    def getProcessingTime_(self, event: Event):
        return self.getProcessingTime_basic(event.activity)
    
    def getWaitingTime(self, event: Event):
        return self.getWaitingTime_basic(event.activity)
        
    
    def getProcessingTime_basic(self, activity):
        if(activity in self.models_basic):
            if(np.random.rand() < self.models_basic[activity+" processing"]["0-proportion"]):
                return timedelta(0)
            return self.sample_distrib(self.models_basic[activity+" processing"]["distribution"], self.models_basic[activity]["parameters"])
        else:
            return timedelta(0)

    def getWaitingTime_basic(self, activity):
        if(activity in self.models_basic):
            if(np.random.rand() < self.models_basic[activity+" processing"]["0-proportion"]):
                return timedelta(0)
            return self.sample_distrib(self.models_basic[activity+" waiting"]["distribution"], self.models_basic[activity]["parameters"])
        else:
            return timedelta(0)   


if __name__ == "__main__":
    processTimeEngine = ProcessTimeEngine()
    print(processTimeEngine.models_basic)