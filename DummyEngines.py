#This file provides dummy implementations for testing.
from Helper import *
import pm4py
from datetime import datetime, timedelta
import numpy as np
import scipy.stats as st
import pandas as pd
from pm4py.objects.petri_net.utils import petri_utils
from pm4py.objects.petri_net.semantics import is_enabled, execute, enabled_transitions
from pm4py.objects.petri_net.obj import PetriNet, Marking


#for task 1.2
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
    
#for task 1.3
class ProcessTimeEngine:

    def __init__(self, log, seed=1):
        self.seed=seed

    def train(self):
        print("Trained the model")

    def getProcessingTime(self, event: Event)-> timedelta:
        return timedelta(minutes=20)
    
    def getWaitingTime(self, event: Event, newActivity)-> timedelta:
        return timedelta(hours=1)

#for task 1.4
class BPMNEngine:
    def __init__(self):
        self.case_markings = {}
        self.model_filename = "model_heuristic.bpmn"

        try:
            bpmn_graph = pm4py.read_bpmn(self.model_filename)
            self.net, self.init_marking, self.final_marking = pm4py.convert_to_petri_net(bpmn_graph)
            print("[BPMNEngine] Model successfully loaded and converted to Petri net.")
        except Exception as e:
            print(f"[BPMNEngine] Fallback: Failed to load model due to error: {e}")
            from pm4py.objects.petri_net.obj import PetriNet, Marking
            self.net = PetriNet("Safe Fallback")
            self.init_marking = Marking()
            self.final_marking = Marking()

    def initialize_case(self, case_id):
        from copy import copy
        self.case_markings[case_id] = copy(self.init_marking)

    def getStartActivity(self, data=None):
        for transition in self.net.transitions:
            if transition.label is not None:
                if is_enabled(transition, self.net, self.init_marking):
                    return transition.label
        return None
    
    def getPossibleNextActivities(self, current_activity, case_id = None) -> list:
        if case_id is None:
            return [self.getStartActivity()]
        if case_id not in self.case_markings:
            self.initialize_case(case_id)

        current_marking = self.case_markings[case_id]
        possible_next = []

        for transition in self.net.transitions:
            if transition.label is not None:
                if is_enabled(transition,self.net, current_marking):
                    possible_next.append(transition.label)
        return possible_next
    
    def fire_activity(self, activity_name, case_id) -> bool:
        if case_id not in self.case_markings:
            self.initialize_case(case_id)

        current_marking = self.case_markings[case_id]
        for transition in self.net.transitions:
            if transition.label == activity_name:
                if is_enabled(transition,self.net, current_marking):
                    new_marking = execute(transition,self.net, current_marking)
                    self.case_markings[case_id] = new_marking
                    return True
                
        return False

#for task 1.5
class BranchingEngine:

    def __init__(self, seed=1):
        return None

    def getNextActivities(self, event, possibleActivities)-> list:
        if len(possibleActivities)==0:
            return []
        else:
            return possibleActivities

#for task 1.6 - 1.8
class ResourceEngine:

    def __init__(self, log=None, seed=1):
        return None

    def allocateResource(self, data):
        return True

    def releaseResource(self, event):
        return True