import os
import sys
from typing import Dict, Tuple, Any
from datetime import datetime, timedelta


import pandas as pd
import numpy as np
from scipy.stats import entropy
from scipy.linalg import eigh
import pm4py

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from SimulationEngineCore import Engine
from processTimes.process_time_engine import ProcessTimeEngine

from typing import Optional
import joblib

LOG_PATH = "data/BPI Challenge 2017.xes"
#LOG_PATH = "data/generated_log.xes"

MANUAL_CALCULATIONS = False

class ProcessLogAnalyzer:
    def __init__(self, 
        log_df: pd.DataFrame, 
        case_col: str = "case:concept:name",
        activity_col: str = "concept:name",
        timestamp_col: str = "time:timestamp",
        transition_col: str = "lifecycle:transition",
        resource_col: str = "org:resource",
        start_transition: str = "start",
        end_transition: str = "end"):

        self.case_col = case_col
        self.activity_col = activity_col
        self.timestamp_col = timestamp_col
        self.transition_col = transition_col
        self.resource_col = resource_col
        self.start_transition = start_transition
        self.end_transition = end_transition

        self.log_df = log_df.dropna(
            subset=[self.case_col, self.timestamp_col, self.activity_col, self.transition_col, self.resource_col]
        ).copy()

        self.log_df = self.log_df.sort_values(by=[self.case_col, self.timestamp_col]).reset_index(drop=True)

    def avg_handovers_per_case(self) -> float:
        self.log_df['prev_resource'] = self.log_df.groupby(self.case_col)[self.resource_col].shift(1)
        is_handover = (self.log_df[self.resource_col] != self.log_df['prev_resource']) & \
                      self.log_df['prev_resource'].notna() & self.log_df[self.resource_col].notna()
        return is_handover.groupby(self.log_df[self.case_col]).sum().mean()

    def avg_rework_rate(self) -> float:
        starts = self.log_df[self.log_df[self.transition_col] == self.start_transition].copy()
        starts['prev_activity'] = starts.groupby(self.case_col)[self.activity_col].shift(1)
        is_rework = (starts[self.activity_col] == starts['prev_activity']) & starts['prev_activity'].notna()
        return is_rework.groupby(starts[self.case_col]).sum().mean()

    def calculate_activity_resource_entropy(self) -> pd.DataFrame:

        if "lifecycle:transition" not in self.log_df.columns:
            return pd.DataFrame()
        df_filtered = self.log_df[
            self.log_df["lifecycle:transition"] == "start"
        ]
        counts = (
            df_filtered
            .groupby([self.activity_col, self.resource_col])
            .size()
            .reset_index(name="count")
        )
        total_per_activity = counts.groupby(self.activity_col)["count"].transform("sum")
        counts["p_resource"] = counts["count"] / total_per_activity
        activity_entropy = (
            counts.groupby(self.activity_col)["p_resource"]
            .apply(lambda p: entropy(p, base=2))
            .reset_index(name="Resource_Entropy_Bits")
        )
        activity_entropy["Num_Resources"] = (
            counts.groupby(self.activity_col)[self.resource_col]
            .nunique()
            .values
        )
        
        activity_entropy = activity_entropy.sort_values(
            by="Resource_Entropy_Bits",
            ascending=True
        ).reset_index(drop=True)

        return activity_entropy

    def calculate_resource_criticality_score(self) -> pd.DataFrame:

        if "lifecycle:transition" not in self.log_df.columns:
            return pd.DataFrame()

        df_filtered = self.log_df[self.log_df["lifecycle:transition"] == "start"]

        activity_entropy = self.calculate_activity_resource_entropy()

        if activity_entropy.empty:
            return pd.DataFrame()

        max_entropy = np.log2(activity_entropy["Num_Resources"])
        activity_entropy["Activity_Criticality"] = np.where(
            max_entropy > 0,
            1 - activity_entropy["Resource_Entropy_Bits"] / max_entropy,
            1.0)

        # Count executions of each activity by each resource
        counts = (
            df_filtered
            .groupby([self.resource_col, self.activity_col])
            .size()
            .reset_index(name="count")
        )

        counts = counts.merge(
            activity_entropy[[self.activity_col, "Activity_Criticality"]],
            on=self.activity_col,
            how="left"
        )

        #sum of weighted criticality
        counts["Criticality_Contribution"] = (
            counts["count"] * counts["Activity_Criticality"]
        )

        resource_scores = (
            counts.groupby(self.resource_col)["Criticality_Contribution"]
            .sum()
            .reset_index(name="Resource_Criticality_Score")
            .sort_values(
                by="Resource_Criticality_Score",
                ascending=False)
            .reset_index(drop=True))
        return resource_scores

    def calculate_algebraic_connectivity(self) -> float:    
        self.log_df['next_resource'] = self.log_df.groupby(self.case_col)[self.resource_col].shift(-1)
        handoffs = self.log_df.dropna(subset=['next_resource'])
        edge_weights = handoffs.groupby([self.resource_col, 'next_resource']).size().reset_index(name='weight')

        resources = pd.concat([edge_weights[self.resource_col], edge_weights['next_resource']]).unique()
        n = len(resources)
        
        if n <= 1:
            return 0.0
        
        res_to_idx = {res: i for i, res in enumerate(resources)}
        
        #bild directed adjacency matrix
        A_dir = np.zeros((n, n))
        for _, row in edge_weights.iterrows():
            i = res_to_idx[row[self.resource_col]]
            j = res_to_idx[row['next_resource']]
            A_dir[i, j] += row['weight']
            
        #symmetrize the matrix
        A_sym = A_dir + A_dir.T
        
        #build degree matrix and laplacian
        # D_ii is the sum of the i-th row (total interactions for that resource)
        D = np.diag(A_sym.sum(axis=1))
        L = D - A_sym
        
        #compute eigenvalues
        eigenvalues = eigh(L, eigvals_only=True)
        eigenvalues = np.sort(eigenvalues)
        
        lambda_2 = round(eigenvalues[1], 4)
        return lambda_2

    def calculate_supsension_contagion(self)-> float:
        suspensions = self.log_df[self.log_df[self.transition_col]=="suspend"].copy()
        suspensions = suspensions.sort_values(self.timestamp_col).reset_index(drop=True)

        t_current = suspensions[self.timestamp_col]
        t_window_end = t_current + pd.Timedelta(hours=24)

        window_end_indices = t_current.searchsorted(t_window_end, side='right')
        current_indices = np.arange(len(t_current))
        
        subsequent_suspension_counts = window_end_indices - current_indices - 1
        r_susp = subsequent_suspension_counts.mean()

        if np.isnan(r_susp):
            return 0.0
        return round(r_susp,2)

    def calculate_process_strain(self, theta=1):
        self.process_time_engine = ProcessTimeEngine(metricProcessing=True)
        
        self.log_times = self.process_time_engine.format_data(self.log_df)
        self.log_times["waiting_time"] = (
            pd.to_numeric(self.log_times["waiting_time"], errors="coerce")
            .fillna(0.0)
        )

        self.baselines = (
            self.log_times.groupby(self.activity_col)["waiting_time"]
            .mean()
            .fillna(1.0)
            .to_dict()
        )
        self.baselines = {k: (v if v > 0.0 else 1.0) for k, v in self.baselines.items()}

        activity_baselines = self.log_times[self.activity_col].map(
            lambda act: self.baselines.get(act, 1.0)
        )
        
        if MANUAL_CALCULATIONS and "strain_time" in self.log_times.columns:
            self.log_times["strain_time"] = (
                pd.to_numeric(self.log_times["strain_time"], errors="coerce")
                .fillna(0.0)
            )
            raw_strain = self.log_times["strain_time"] / activity_baselines
        else:
            raw_strain = (
                self.log_times["waiting_time"] - activity_baselines
            ) / activity_baselines

        self.log_times["strain"] = (
            raw_strain.fillna(0.0)
            .replace([np.inf, -np.inf], 0.0)
            .clip(lower=0.0)
        )
        self.log_times["jammed"] = (self.log_times["strain"] > theta).astype(int)        
        return self.log_times[self.log_times["jammed"] == 1].groupby(self.activity_col).count()


    def calculate_24h_strain_impact(self, theta=1, beta=1):
        self.calculate_process_strain(theta)
        
        if "final_timepoint" not in self.log_times.columns:
            return self.log_times, pd.DataFrame()
            
        df = self.log_times.sort_values("final_timepoint").copy()

        if df.empty:
            return df, pd.DataFrame()

        # Safely convert to epoch seconds
        times_sec = pd.to_datetime(df["final_timepoint"]).astype("int64").values // 10**9
        strains = df["strain"].values

        # 24 hours in seconds
        window_sec = 24 * 3600 
        right_bounds = np.searchsorted(times_sec, times_sec + window_sec, side="right")
        
        radiated_fields = np.zeros(len(df))
        for i, end_idx in enumerate(right_bounds):
            if end_idx > i + 1:
                t_diff_hours = (times_sec[i + 1 : end_idx] - times_sec[i]) / 3600.0
                decay = np.exp(-beta * t_diff_hours)
                radiated_fields[i] = np.sum(strains[i + 1 : end_idx] * decay)

        df["radiated_field"] = radiated_fields
        df["radiated_impact"] = df["strain"] * df["radiated_field"]

        def calc_gamma(group):
            # No NaNs or Infs will reach here because we sanitized 'strain' at the source
            eps = group["strain"]
            R = group["radiated_field"]
            
            # Guard against single-event groups (covariance requires at least 2 samples)
            if len(eps) < 2:
                gamma = 0.0
            else:
                var_eps = np.var(eps)
                if var_eps > 1e-9:
                    cov_matrix = np.cov(eps, R)
                    # Protect against any unexpected nan output from np.cov
                    gamma = cov_matrix[0, 1] / var_eps if not np.isnan(cov_matrix[0, 1]) else 0.0
                else:
                    gamma = 0.0
            
            return pd.Series({
                "event_count": len(group),
                "max_waiting_time": group["waiting_time"].max(),
                "mean_local_strain": eps.mean(),
                "mean_24h_radiated_field": R.mean(),
                "mean_radiated_impact": group["radiated_impact"].mean(),
                "propagation_elasticity_gamma": gamma
            })
        results_df = (
            df.groupby(self.activity_col)
            .apply(calc_gamma, include_groups=False)
            .sort_values("propagation_elasticity_gamma", ascending=False)
            .reset_index()
        )
        active_results = results_df[(results_df["max_waiting_time"] > 0) & (results_df["mean_local_strain"] > 0)]

        total_events = active_results["event_count"].sum()
        weighted_process_gamma = (
        (active_results["propagation_elasticity_gamma"] * active_results["event_count"]).sum() 
        / total_events)
        
        return 0 if np.isnan(weighted_process_gamma) else round(weighted_process_gamma,2)

    def generate_report(self):
        
        return {
                "Avg Handovers per Case": round(self.avg_handovers_per_case(), 2),
                "Avg Rework Rate (Self-Loops)": round(self.avg_rework_rate(), 2),
                "Total processoral value": round(self.calculate_resource_criticality_score()["Resource_Criticality_Score"].sum(),2),
                "Network Algebraic Connectivity (\u03BB\u2082)": self.calculate_algebraic_connectivity(),
                "Effect of Suspensions": self.calculate_supsension_contagion(),
                "Calculate Strain": self.calculate_24h_strain_impact()
            }

if __name__ == "__main__":
    if not os.path.exists(LOG_PATH):        
        print("Starting Simulation")
        simulation_engine = Engine()
        simulation_engine.run(datetime(2000, 1, 1), datetime(2000, 1, 10), "xes")
    
    simulated_log_df= pm4py.read_xes(LOG_PATH, variant="r4pm")
    print("Loaded simulated log")

    analyzer = ProcessLogAnalyzer(simulated_log_df)

    report = analyzer.generate_report()
    for metric, value in report.items():
        print(f"{metric}: {value}")

    print("Least entropy within activity:")
    print(analyzer.calculate_activity_resource_entropy().head())
    print("Most procedural valuable resources:")
    print(analyzer.calculate_resource_criticality_score().head())