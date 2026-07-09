import sys
import os
import pm4py
import pandas as pd
from datetime import datetime, timedelta
from scipy import stats
import sklearn.ensemble, sklearn.pipeline, sklearn.compose, sklearn.preprocessing, sklearn.impute
from sklearn.linear_model import QuantileRegressor
import numpy as np
import json
import joblib

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from Helper import *

PATH_MODELS = "processTimes/processing_time_models.pkl"

class ProcessTimeEngine:

    rndm_state = 1
    models_basic = dict()
    models_advanced = dict()
    models_quantiles = dict()

    fallback_models_basic = dict()

    def __init__(self, log: pd.DataFrame=pd.DataFrame(), seed=1):

        self.rndm_state = seed
        self.rng = np.random.default_rng(seed)
        #if the models are already trained they do not have to be retrained
        if os.path.exists(PATH_MODELS):
            models = joblib.load(PATH_MODELS)
            self.models_basic = models["basic"]
            self.models_quantiles = models["quantiles"]
            self.models_advanced = models["advanced"]
            self.fallback_models_basic = models.get("fallback_basic", {})
            print("[ProcessTimeEngine] Loaded models successfully")
            return
        
        else:
            #if the log is empty (no propper log was given), the log is manually loaded
            if(log.empty):
                log = pm4py.convert_to_dataframe(pm4py.read_xes('data/logData.xes', variant="r4pm"))
            
            event_times = self.format_data(log)
            self.train_basic(event_times)
            self.train_advanced(event_times)
            print("[ProcessTimeEngine] Trained models successfully")

            models={}
            models["basic"]=self.models_basic
            models["quantiles"]=self.models_quantiles
            models["advanced"]=self.models_advanced
            models["fallback_basic"]=self.fallback_models_basic
            joblib.dump(models, PATH_MODELS)

    def _fit_distribution(self, times_data, null_count):
        """Helper method to fit distributions and return the best one based on AIC."""
        potential_distribs = []
        
        # poisson
        lam = times_data.mean()
        loglikelihood = np.sum(stats.poisson.logpmf(times_data, lam))
        potential_distribs.append({
            "distribution": "poisson", "parameters": {"lambda": lam}, "AIC": 2*1 - 2*loglikelihood
        })
        
        # gamma
        shape, loc, scale = stats.gamma.fit(times_data, floc=0)
        loglikelihood = np.sum(stats.gamma.logpdf(times_data, shape, loc, scale))
        potential_distribs.append({
            "distribution": "gamma", "parameters": {"shape": shape, "scale": scale}, "AIC": 2*2 - 2*loglikelihood
        })

        # lognorm
        shape, loc, scale = stats.lognorm.fit(times_data, floc=0)
        loglikelihood = np.sum(stats.lognorm.logpdf(times_data, shape, loc, scale))
        potential_distribs.append({
            "distribution": "lognorm", "parameters": {"shape": shape, "scale": scale}, "AIC": 2*2 - 2*loglikelihood
        })

        potential_distribs = pd.DataFrame(potential_distribs)
        distrib_best = potential_distribs.loc[potential_distribs["AIC"].idxmin(), :]
        
        return {
            "distribution": distrib_best["distribution"],
            "parameters": distrib_best["parameters"],
            "0-proportion": null_count / (len(times_data) + null_count)
        }

    def train_basic(self, event_times):
            #fallback training
            for activity, group in event_times.groupby("concept:name"):
                for kind in ("processing", "waiting"):
                    times_data = group[kind+"_time"]
                    null_count = sum(times_data == 0)
                    times_data = times_data[times_data > 0]
                    
                    if len(times_data) >= 4 and np.unique(times_data).size > 1:
                        self.fallback_models_basic[f"{activity}_{kind}"] = self._fit_distribution(times_data, null_count)

            for (activity, resource), group in event_times.groupby(["concept:name", "org:resource"]):
                for kind in ("processing", "waiting"):
                    times_data = group[kind+"_time"]
                    null_count = sum(times_data == 0)
                    times_data = times_data[times_data > 0]
                    
                    if len(times_data) < 4 or np.unique(times_data).size <= 1:
                        continue

                    self.models_basic[f"{activity}_{resource}_{kind}"] = self._fit_distribution(times_data, null_count)

    def train_advanced(self, event_times: pd.DataFrame):
        self.models_advanced = {}
        quantiles = [i/10 for i in range(1,10)]
        numeric_features = ["case:RequestedAmount", "HourOfDay", "Weekday"]
        categorical_features = ["case:ApplicationType", "org:resource"]

        preprocessor = sklearn.compose.ColumnTransformer(
            transformers=[
                ('num', sklearn.impute.SimpleImputer(strategy='median'), numeric_features),
                ('cat', sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])

        for (activity, resource), group in event_times.groupby(["concept:name", "org:resource"]):
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
                self.models_advanced[f"{activity}_{resource}_{kind}"] = model

                results_quantiles ={}
                for q in quantiles:

                    regressor = QuantileRegressor(quantile=q, alpha=0.0, solver='highs')
                    pipeline = sklearn.pipeline.Pipeline(steps=[('preprocessor', preprocessor),('regressor', regressor)])
                    pipeline.fit(X, y)
                    results_quantiles[q]=pipeline
                
                    self.models_quantiles[f"{activity}_{resource}_{kind}"]= results_quantiles

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

                    total_waiting_time = log_entry["time:timestamp"] - activity_waiting
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
                        "processing_time": total_active_time.total_seconds(),
                        "waiting_time": total_waiting_time.total_seconds(),
                        "org:resource": log_entry["org:resource"],
                        "case:RequestedAmount": requestedAmount,
                        "case:ApplicationType": applicationType,
                        "HourOfDay": first_date.hour,
                        "Weekday": first_date.weekday()
                    })
        return pd.DataFrame.from_records(event_times)
    
    def sample_distrib(self, distrib, param) -> timedelta:
        if distrib == "poisson":
            return timedelta(seconds=stats.poisson.rvs(mu = param["lambda"], random_state=self.rng))
        if distrib == "gamma":
            return timedelta(seconds=stats.gamma.rvs(param["shape"], loc=0, scale=param["scale"], random_state=self.rng))
        if distrib == "lognorm":
            return timedelta(seconds=stats.lognorm.rvs(param["shape"], loc=0, scale=param["scale"], random_state=self.rng))
        return timedelta(0)
    
    def getProcessingTime(self, event: Event, activity=None) -> timedelta:
        return self.sampleTime_basic(event.activity, event.resource, "processing")
        #return self.sampleTime_advanced(event, "processing")
    
    def getWaitingTime(self, event: Event, activity=None) -> timedelta:
        return self.sampleTime_basic(event.activity, event.resource, "waiting")
        #return self.sampleTime_advanced(event, "waiting")
    
    def getProcessingTimes_advanced(self, event: Event) -> timedelta:
        return self.sampleTime_advanced(event, "waiting")       

    def sampleTime_basic(self, activity, resource="", kind="processing") -> timedelta:
        model = self.models_basic.get(f"{activity}_{resource}_{kind}") \
            or self.fallback_models_basic.get(f"{activity}_{kind}")
        if model is None:
            return timedelta(0)
        if np.random.rand() < model["0-proportion"]:
            return timedelta(0)
        return self.sample_distrib(model["distribution"], model["parameters"])
    
    def sampleTime_advanced(self, event: Event, kind="processing") -> timedelta:        
        key = f"{event.activity}_{event.resource}_{kind}"
        if key not in self.models_advanced:
            return self.sampleTime_basic(event.activity, event.resource, kind)
            
        model = self.models_advanced[key]
        
        context_df = pd.DataFrame([{
            "case:RequestedAmount": float(event.eventCase.requestedAmount),
            "case:ApplicationType": str(event.eventCase.applicationType),
            "HourOfDay": event.time.hour,
            "Weekday": event.time.weekday(),
            "org:resource": str(event.resource)
        }])
        predicted_seconds = model.predict(context_df)[0]
        predicted_seconds = max(0.0, float(predicted_seconds))
        return timedelta(seconds=int(predicted_seconds))
    
    def getQuantileValue(self, activity, resource, kind, q_value):
        key = f"{activity}_{resource}_{kind}"
        if key in self.models_quantiles:
            return self.models_quantiles[key][q_value]
        return None

if __name__ == "__main__":
    processTimeEngine = ProcessTimeEngine()
    print(processTimeEngine.models_basic)