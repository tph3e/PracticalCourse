import sys
import os
import pm4py
import pandas as pd
from datetime import datetime, timedelta
from scipy import stats
import sklearn.ensemble, sklearn.pipeline, sklearn.compose, sklearn.preprocessing, sklearn.impute
import numpy as np
import json
import joblib

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from Helper import *

PATH_MODEL_BASIC = "processTimes/processing_time_models_basic.pkl"
PATH_MODEL_ADVANCED = "processTimes/processing_time_models_advanced.pkl"

class ProcessTimeEngine:

    rndm_state = 1
    models_basic = dict()
    models_advanced = dict()

    def __init__(self, log: pd.DataFrame=pd.DataFrame(), seed=1):

        self.rndm_state = seed
        #if the models are already trained they do not have to be retrained
        if os.path.exists(PATH_MODEL_BASIC) and os.path.exists(PATH_MODEL_ADVANCED):
            self.models_basic = joblib.load(PATH_MODEL_BASIC)
        
            self.models_advanced = joblib.load(PATH_MODEL_ADVANCED)
            return
        #if the log is empty (no propper log was given), the log is manually loaded
        if(log.empty):
            log = pm4py.convert_to_dataframe(pm4py.read_xes('data/logData.xes', variant="r4pm"))
        
        event_times = self.format_data(log)
        self.train_basic(event_times)
        self.train_advanced(event_times)

    def train_basic(self, event_times):
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
        joblib.dump(self.models_basic, PATH_MODEL_BASIC)


    

    def train_advanced(self, event_times: pd.DataFrame):
        self.models_advanced = {}
        numeric_features = ["case:RequestedAmount", "HourOfDay", "Weekday"]
        categorical_features = ["case:ApplicationType"]

        preprocessor = sklearn.compose.ColumnTransformer(
            transformers=[
                ('num', sklearn.impute.SimpleImputer(strategy='median'), numeric_features),
                ('cat', sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])

        for activity, group in event_times.groupby("concept:name"):
            #model can only be trained with sufficient data
            if len(group) < 10:
                continue
            X = group[numeric_features + categorical_features]
            for kind in ("processing", "waiting"):
                y = group[f"{kind}_time"]
                if y.sum() == 0:
                    continue
                #create and train model pipeline
                model = sklearn.pipeline.Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('regressor', sklearn.ensemble.RandomForestRegressor(n_estimators=10, random_state=self.rndm_state, max_depth=5))
                ])
                
                model.fit(X, y)
                self.models_advanced[f"{activity}_{kind}"] = model
        joblib.dump(self.models_advanced, PATH_MODEL_ADVANCED)

    def format_data(self, log: pd.DataFrame):
        # Sort events chronologisch
        log = log.sort_values(["case:concept:name", "concept:name", "time:timestamp"])
        event_times = []
        
        # Gruppieren nach Case und Aktivität
        for (case, activity), event_data in log.groupby(["case:concept:name", "concept:name"]):

            first_date = datetime(2000,1,1)
            applicationType = None
            requestedAmount = None

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
                    first_date = log_entry["time:timestamp"]
                    applicationType = log_entry["case:ApplicationType"]
                    requestedAmount = log_entry["case:RequestedAmount"]
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
                        "waiting_time": total_waiting_time.seconds,
                        "case:RequestedAmount": requestedAmount,
                        "case:ApplicationType": applicationType,
                        "HourOfDay": first_date.hour,
                        "Weekday": first_date.weekday()
                    })
        return pd.DataFrame.from_records(event_times)
    
    def sample_distrib(self, distrib, param) -> timedelta:
        if distrib == "poisson":
            return timedelta(stats.poisson.rvs(mu = param["lambda"], random_state= self.rndm_state))
        if distrib == "gamma":
            return timedelta(stats.gamma.rvs(param["shape"], loc=0, scale=param["scale"], random_state= self.rndm_state))
        if distrib == "lognorm":
            return timedelta(stats.lognorm.rvs(param["shape"], loc=0, scale=param["scale"], random_state= self.rndm_state))
        return timedelta(0)
    
    def getProcessingTime(self, event: Event, activity=None) -> timedelta:
        return self.sampleTime_advanced(event, "processing")
    
    def getWaitingTime(self, event: Event, activity=None) -> timedelta:
        return self.sampleTime_advanced(event, "waiting")
    
    def getProcessingTimes_advanced(self, event: Event) -> timedelta:
        return self.sampleTime_advanced(event, "waiting")       

    def sampleTime_basic(self, activity, kind) -> timedelta:
        if(activity in self.models_basic):
            if(np.random.rand() < self.models_basic[f"{activity} {kind}"]["0-proportion"]):
                return timedelta(0)
            return self.sample_distrib(self.models_basic[f"{activity} {kind}"]["distribution"], self.models_basic[activity]["parameters"])
        else:
            return timedelta(0)
    
    def sampleTime_advanced(self, event: Event, kind) -> timedelta:        
        #If the activity is not modelled, return 0
        if f"{event.activity}_{kind}" not in self.models_advanced:
            return timedelta(0)
            
        model = self.models_advanced[f"{event.activity}_{kind}"]
        
        context_df = pd.DataFrame([{
            "case:RequestedAmount": float(event.eventCase.requestedAmount),
            "case:ApplicationType": str(event.eventCase.applicationType),
            "HourOfDay": event.time.hour,
            "Weekday": event.time.weekday()
        }])
        predicted_seconds = model.predict(context_df)[0]
        predicted_seconds = max(0.0, float(predicted_seconds))
        return timedelta(int(predicted_seconds))


if __name__ == "__main__":
    processTimeEngine = ProcessTimeEngine()
    print(processTimeEngine.models_basic)