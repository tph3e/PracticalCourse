import os
import sys
import pm4py
import pandas as pd
from datetime import datetime, timedelta
from scipy import stats
import sklearn.ensemble, sklearn.pipeline, sklearn.compose, sklearn.preprocessing, sklearn.impute
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import make_friedman1
from sklearn.tree import plot_tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_pinball_loss, make_scorer
import numpy as np
import joblib
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy.stats")
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from Helper import *

PATH_MODELS = "processTimes/processing_time_models.pkl"
PATH_LOG_TRAINING = "data/BPI Challenge 2017.xes"

class ProcessTimeEngine:

    def __init__(self, log=None, seed=1, waiting_advanced=False, processing_advanced=False, metricProcessing=False):
        self.metricProcessing=metricProcessing
        if metricProcessing:
            return

        self.rndm_state = seed
        self.rng = np.random.default_rng(seed)

        self.waiting_advanced=waiting_advanced
        self.processing_advanced=processing_advanced

        self.models_basic = {}
        self.models_quantiles = {}
        self.models_advanced = {}
        self.fallback_models_basic = {}
        self.models_median={}
        #if the models are already trained they do not have to be retrained
        if os.path.exists(PATH_MODELS):
            models = joblib.load(PATH_MODELS)
            self.models_basic = models["basic"]
            self.models_quantiles = models["quantiles"]
            self.models_advanced = models["advanced"]
            self.fallback_models_basic = models.get("fallback_basic", {})
            self.models_median = models["median"]
            print("[ProcessTimeEngine] Loaded models successfully")
            return
        
        else:
            #if the log is empty (no propper log was given), the log is manually loaded
            if log is None or log.empty:
                log = pm4py.convert_to_dataframe(pm4py.read_xes(PATH_LOG_TRAINING, variant="r4pm"))
            event_times = self.format_data(log)
            self.train_basic(event_times)
            self.train_advanced(event_times)

            models = {
                "basic": self.models_basic,
                "quantiles": self.models_quantiles,
                "advanced": self.models_advanced,
                "fallback_basic": self.fallback_models_basic,
                "median": self.models_median
            }
            #ensure directory exists before dumping
            os.makedirs(os.path.dirname(PATH_MODELS), exist_ok=True)
            joblib.dump(models, PATH_MODELS)

    def _fit_distribution(self, times_data, null_count):
        """Helper method to fit distributions and return the best one based on AIC."""
        potential_distribs = []
        try:
            #gamma
            shape, loc, scale = stats.gamma.fit(times_data, floc=0)
            shape, loc, scale = [round(float(x), 2) for x in (shape, loc, scale)]
            loglikelihood = np.sum(stats.gamma.logpdf(times_data, shape, loc, scale))
            potential_distribs.append({
                "distribution": "gamma", "parameters": {"shape": shape, "scale": scale}, "AIC": 2*2 - 2*loglikelihood
            })

            #lognorm
            shape, loc, scale = stats.lognorm.fit(times_data, floc=0)
            shape, loc, scale = [round(float(x), 2) for x in (shape, loc, scale)]
            loglikelihood = np.sum(stats.lognorm.logpdf(times_data, shape, loc, scale))
            potential_distribs.append({
                "distribution": "lognorm", "parameters": {"shape": shape, "scale": scale}, "AIC": 2*2 - 2*loglikelihood
            })

            #poisson
            lam = times_data.mean().round().astype(int)
            loglikelihood = np.sum(stats.poisson.logpmf(times_data, lam))
            potential_distribs.append({
                "distribution": "poisson", "parameters": {"lambda": lam}, "AIC": 2*1 - 2*loglikelihood
            })
        except Exception:
            pass

        potential_distribs = pd.DataFrame(potential_distribs)
        distrib_best = potential_distribs.loc[potential_distribs["AIC"].idxmin(), :]
        
        return {
            "distribution": distrib_best["distribution"],
            "parameters": distrib_best["parameters"],
            "0-proportion": round(null_count / (len(times_data) + null_count),2)
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

                self.models_median[f"{activity}_{resource}"] = group["processing_time"].median()
                for kind in ("processing", "waiting"):
                    times_data = group[kind+"_time"]
                    null_count = sum(times_data == 0)
                    times_data = times_data[times_data > 0]
                    
                    if len(times_data) < 4 or np.unique(times_data).size <= 1:
                        continue

                    self.models_basic[f"{activity}_{resource}_{kind}"] = self._fit_distribution(times_data, null_count)

    def train_advanced(self, event_times: pd.DataFrame):
        self.models_advanced = {}
        quantiles = [i/4 for i in range(1,4)]
        numeric_features = ["case:RequestedAmount", "hour_of_day", "rework_count", "weekday"]
        categorical_features = ["case:ApplicationType", "org:resource", "concept:name"]

        preprocessor = sklearn.compose.ColumnTransformer(
            transformers=[
                ('num', sklearn.impute.SimpleImputer(strategy='median'), numeric_features),
                ('cat', sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])

        X = event_times[numeric_features + categorical_features]
        for kind in ("processing", "waiting"):
            y = event_times[f"{kind}_time"]
            if y.sum() == 0:
                continue

            print(f"[ProcessTimeEngine] Starting Grid Search for '{kind}' time model...")
            
            pipeline = sklearn.pipeline.Pipeline(steps=[
                ('preprocessor', sklearn.base.clone(preprocessor)),
                ('regressor', sklearn.ensemble.RandomForestRegressor(random_state=self.rndm_state))
            ])
            #simplified to get faster calculations
            param_grid = {
                'regressor__n_estimators': [25, 50],
                'regressor__max_depth': [5, 10, 15]
            }

            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                cv=3,
                scoring='neg_mean_absolute_error',
                n_jobs=-1,
                verbose=0
            )
            grid_search.fit(X, y)
            print(f"[ProcessTimeEngine] {datetime.now().strftime('%H:%M:%S')} Best parameters for '{kind}': {grid_search.best_params_}")

            self.models_advanced[f"{kind}"] = grid_search.best_estimator_

            results_quantiles = {}
            quantiles = [0.10, 0.50, 0.90]

            results_quantiles = {}
            for q in quantiles:
                print(f"[ProcessTimeEngine] Starting Grid Search for '{kind}' time quantile {q}...")
                
                regressor = GradientBoostingRegressor(loss='quantile', alpha=q, random_state=self.rndm_state)
                pipeline = sklearn.pipeline.Pipeline(steps=[
                    ('preprocessor', sklearn.base.clone(preprocessor)), 
                    ('regressor', regressor)
                ])
                #also simplified to get faster calculations
                param_grid_gb = {
                    'regressor__n_estimators': [50, 100],
                    'regressor__max_depth': [3, 5],
                }
                
                pinball_scorer = make_scorer(mean_pinball_loss, alpha=q, greater_is_better=False)
                
                grid_search_gb = GridSearchCV(
                    estimator=pipeline,
                    param_grid=param_grid_gb,
                    cv=3,
                    scoring=pinball_scorer,
                    n_jobs=-1,
                    verbose=0
                )
                
                grid_search_gb.fit(X, y)
                print(f"[ProcessTimeEngine] Best parameters for quantile {q}: {grid_search_gb.best_params_}")
                
                results_quantiles[round(q, 2)] = grid_search_gb.best_estimator_

            self.models_quantiles[f"{kind}"] = results_quantiles

    def format_data(self, log: pd.DataFrame):
            log = log.sort_values(["case:concept:name", "time:timestamp"]).reset_index(drop=True)
            
            cases = log["case:concept:name"].values
            activities = log["concept:name"].values
            transitions = log["lifecycle:transition"].values
            resources = log["org:resource"].values
            app_types = log["case:ApplicationType"].values
            req_amounts = log["case:RequestedAmount"].values
            timestamps = log["time:timestamp"].values
            
            ts_seconds = log["time:timestamp"].astype('int64').values // 10**9
            hours = log["time:timestamp"].dt.hour.values
            weekdays = log["time:timestamp"].dt.weekday.values

            event_times = []
            
            last_complete_time = {}
            active_starts = {}
            total_active_times = {}
            first_hours = {}
            first_weekdays = {}
            rework_counts = {}
            
            case_app_types = {}
            case_req_amounts = {}
            waiting_time={}

            for i in range(len(log)):
                case = cases[i]
                activity = activities[i]
                t_sec = ts_seconds[i]
                transition = transitions[i]
                group = (case, activity)

                if pd.notna(app_types[i]):
                    case_app_types[case] = app_types[i]
                if pd.notna(req_amounts[i]):
                    case_req_amounts[case] = req_amounts[i]

                if group not in total_active_times:
                    total_active_times[group] = 0.0
                if group not in rework_counts:
                    rework_counts[group] = 0
                
                if transition == "start":
                    active_starts[group] = t_sec
                    total_active_times[group] = 0.0
                    first_hours[group] = hours[i]
                    first_weekdays[group] = weekdays[i]
                    rework_counts[group] += 1

                    waiting_time[case] = float(t_sec - last_complete_time[case]) if case in last_complete_time else 0.0
                    
                elif transition == "resume":
                    active_starts[group] = t_sec
                    
                elif transition == "suspend":
                    if active_starts.get(group) is not None:
                        total_active_times[group] += (t_sec - active_starts[group])
                        active_starts[group] = None
                    
                elif transition == "complete":
                    if active_starts.get(group) is not None:
                        total_active_times[group] += (t_sec - active_starts[group])
                        active_starts[group] = None

                    infos = {
                        "case:concept:name": case,
                        "concept:name": activity,
                        "processing_time": float(total_active_times[group]),
                        "waiting_time": waiting_time.get(case,0),
                        "org:resource": resources[i],
                        "case:RequestedAmount": case_req_amounts.get(case, np.nan),
                        "case:ApplicationType": case_app_types.get(case, "Unknown"),
                        "hour_of_day": first_hours.get(group, hours[i]),
                        "weekday": first_weekdays.get(group, weekdays[i]),
                        "rework_count": rework_counts[group],
                        "final_timepoint": timestamps[i]
                    }
                    
                    if self.metricProcessing and "strain_time_difference" in log.columns:
                        infos["strain_time"] = log["strain_time_difference"].values[i]
                        
                    event_times.append(infos)
                    
                    total_active_times[group] = 0.0
                    last_complete_time[case] = t_sec

            return pd.DataFrame.from_records(event_times)
    
    def sample_distrib(self, distrib, param) -> timedelta:
        if distrib == "poisson":
            return timedelta(seconds=stats.poisson.rvs(mu = param["lambda"], random_state=self.rng))
        if distrib == "gamma":
            return timedelta(seconds=stats.gamma.rvs(param["shape"], loc=0, scale=param["scale"], random_state=self.rng))
        if distrib == "lognorm":
            return timedelta(seconds=stats.lognorm.rvs(param["shape"], loc=0, scale=param["scale"], random_state=self.rng)) 
        return timedelta(0)
    
    def getProcessingTime(self, event: Event) -> timedelta:
        if self.processing_advanced:
            return self.sampleTime_advanced(event, "processing")
        else:
            return self.sampleTime_basic(event.activity, event.resource, "processing")
        
    def getWaitingTime(self, event: Event) -> timedelta:
        if self.waiting_advanced:
            return self.sampleTime_advanced(event, "waiting")
        else:
            return self.sampleTime_basic(event.activity, event.resource, "waiting")
           

    def sampleTime_basic(self, activity, resource="", kind="processing") -> timedelta:
        key = f"{activity}_{resource}_{kind}"
        if key in self.models_basic:
            if(self.rng.random() < self.models_basic[key]["0-proportion"]):
                return timedelta(0)
            return self.sample_distrib(self.models_basic[key]["distribution"], self.models_basic[key]["parameters"])
        elif f"{activity}_{kind}" in self.fallback_models_basic:
                if(self.rng.random() < self.fallback_models_basic[f"{activity}_{kind}"]["0-proportion"]):
                    return timedelta(0)
                return self.sample_distrib(self.fallback_models_basic[f"{activity}_{kind}"]["distribution"], self.fallback_models_basic[f"{activity}_{kind}"]["parameters"])
        else:
            return timedelta(0)
    
    def sampleTime_advanced(self, event: Event, kind="processing") -> timedelta:        
        if kind not in self.models_advanced:
            return self.sampleTime_basic(event.activity, event.resource, kind)
        
        context_df = pd.DataFrame([{
            "concept:name": event.activity,
            "case:RequestedAmount": float(event.eventCase.requestedAmount),
            "case:ApplicationType": str(event.eventCase.applicationType),
            "hour_of_day": event.time.hour,
            "weekday": event.time.weekday(),
            "org:resource": str(event.resource),
            "rework_count": event.eventCase.getActivityCount(event.activity)
        }])
        predicted_seconds = self.models_advanced[kind].predict(context_df)[0]
        predicted_seconds = max(0.0, float(predicted_seconds))
        return timedelta(seconds=int(predicted_seconds))
    
    def getQuantileTime(self, event: Event, kind: str, q_value: float) -> timedelta:
        if kind in self.models_quantiles and round(q_value,2) in self.models_quantiles[kind]:
            pipeline = self.models_quantiles[kind][round(q_value,2)]
            context_df = pd.DataFrame([{
                "concept:name": event.activity,
                "case:RequestedAmount": float(event.eventCase.requestedAmount),
                "case:ApplicationType": str(event.eventCase.applicationType),
                "hour_of_day": event.time.hour,
                "weekday": event.time.weekday(),
                "org:resource": str(event.resource),
                "rework_count": event.eventCase.getActivityCount(event.activity)
            }])
            pred = max(0.0, float(pipeline.predict(context_df)[0]))
            return timedelta(seconds=int(pred))
        return timedelta(0)
    
    def getMedian(self, activity, resource):
        if f"{activity}_{resource}" in self.models_median:
            return self.models_median[f"{activity}_{resource}"]
        return 0

if __name__ == "__main__":
    processTimeEngine = ProcessTimeEngine()
    print(processTimeEngine.fallback_models_basic)
