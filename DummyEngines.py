#This file provides dummy implementations for testing.
from Helper import *
import pm4py
from datetime import datetime, timedelta
import numpy as np
import scipy.stats as st
import pandas as pd

#for task 1.2
class ArrivalEngine:
    def __init__(self, log, seed):
        self.seed=seed

    def train(self):
        print("Trained the model")
    
    def calc(self, attrib):
        return 1
    
    def nextArrivalTime(self, time) -> timedelta:
        return timedelta(days=1, minutes=1)
    
#for task 1.3
class ProcessTimeEngine:

    def __init__(self, log, seed=1):
        self.seed=seed

    def getProcessingTime(self, event: Event)-> timedelta:
        return timedelta(minutes=20)
    
    def getWaitingTime(self, event: Event, newActivity)-> timedelta:
        return timedelta(hours=1)

#for task 1.4
class BPMNEngine:
    def getStartActivity(self, data):
        return "activity1"
    
    def getPossibleNextActivities(self, activity, caseId=None)-> list:
        if activity=="activity1":
            return ["activity2"]
        elif activity== "activity2":
            return ["activity3", "activity4"]
        else:
            return []
        
    def fire_activity(self, activity_name, case_id) -> bool:
        return True

#for task 1.5
class BranchingEngine:

    def __init__(self, seed=1):
        return None

    def getNextActivities(self, event, possibleActivities)-> list:
        if np.random.random() <0.5:
            return []
        if len(possibleActivities)==0:
            return []
        else:
            return possibleActivities[0]

#for task 1.6 - 1.8
class ResourceEngine:

    def __init__(self, log=None, seed=1):
        return None

    def allocateResource(self, data):
        return True

    def releaseResource(self, event):
        return True