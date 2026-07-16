import os
from typing import Dict, Tuple, Any
from datetime import datetime, timedelta


import pandas as pd
import numpy as np
from scipy.stats import entropy
from scipy.linalg import eigh
import pm4py

from SimulationEngineCore import Engine

from typing import Optional
import sys
import joblib

LOG_PATH = "data/BPI Challenge 2017.xes"
#LOG_PATH = "data/generated_log.xes"

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

        return round(r_susp,2)

    def generate_report(self):
        
        return {
                "Avg Handovers per Case": round(self.avg_handovers_per_case(), 2),
                "Avg Rework Rate (Self-Loops)": round(self.avg_rework_rate(), 2),
                "Total processoral value": self.calculate_resource_criticality_score()["Resource_Criticality_Score"].sum(),
                "Network Algebraic Connectivity (\u03BB\u2082)": self.calculate_algebraic_connectivity(),
                "Efect of Suspensions": self.calculate_supsension_contagion()
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