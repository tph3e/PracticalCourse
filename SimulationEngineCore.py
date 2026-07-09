from __future__ import annotations
from enum import Enum, auto
import heapq
import pandas as pd
import csv
import pm4py
import random
import os
import sys
from joao.src.branching.CompositeBranchingEngine import CompositeBranchingEngine as BranchingEngine
from arrival_engine import ArrivalEngine
from BPMN_engine import BPMNEngine
from resources import ResourceEngine
from processTimes import ProcessTimeEngine
from typing import Dict, Any, Optional, List

from datetime import datetime, timedelta
from Helper import *
import xml.etree.ElementTree as ET
import scipy.stats as stats
import numpy as np

PATH_LOG = "data/generated_log"


class EventLogger:
    
    def __init__(self):
        self.records: List[Event] =[]

    def _to_dataframe(self) -> pd.DataFrame:
        if not self.records:
            return pd.DataFrame()
        return pd.DataFrame([record.getAttribs() for record in self.records])

    def log_event(self, event) -> None:
        self.records.append(event)

    def to_csv(self, filepath=PATH_LOG+".csv") -> None:
        df = self._to_dataframe()
        if not df.empty:
            df.to_csv(filepath, index=False, encoding="utf-8")

    def to_xes(self, filepath=PATH_LOG+".xes") -> None:
        df = self._to_dataframe()
        if not df.empty:
            pm4py.write_xes(df, filepath)

    def get_log(self) -> pd.DataFrame:
        return self._to_dataframe()

class Engine:

    def __init__(self, dataPath: str="data/BPI Challenge 2017.xes", seed: int=1)-> None:
        self.event_counter: int = 0
        self.case_counter: int = 0
        self.event_queue: List[Event] = []
        self.cases: List[Case] = []
        self.waiting_processes: List[Event] = []
        self.simulation_time: Optional[datetime] = None

        log = pm4py.read_xes(dataPath, variant="r4pm")

        self.logger = EventLogger()
        self.arrivalEngine = ArrivalEngine(log, seed)
        self.bpmnEngine = BPMNEngine()
        self.resourceEngine = ResourceEngine(log, seed)
        self.branchingEngine = BranchingEngine()
        self.processTimeEngine = ProcessTimeEngine(log, seed)

        self.freq: pd.DataFrame = pd.DataFrame()
        self.amount_dists: Dict[Tuple[str, str], Tuple[float, float, float]] = {}
        self.global_params: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        
        self.train(log)

        random.seed(seed)

    def train(self, log: pd.DataFrame)-> None:
        #The distribution of requestedAmount is fitted for every pair of case:ApplicationType and case:LoanGoal seperatly
        freq = (
            log.groupby(["case:ApplicationType", "case:LoanGoal"])
            .size()
            .rename("count")
            .reset_index()
        )
        freq["prob"] = freq["count"] / freq["count"].sum()
        self.freq = freq

        self.amount_dists: Dict[tuple, tuple] = {}
        
        global_amounts = pd.to_numeric(log["case:RequestedAmount"], errors='coerce').dropna().to_numpy()
        #A global fitting is calculated so that whenever the data for a pair of case:ApplicationType and case:LoanGoal is insufficient, it can be used.
        shape, loc, scale = stats.lognorm.fit(global_amounts)
        #The results are rounded because the additional numbers are not relevant.
        self.global_params = (round(shape,2), round(loc,2), round(scale,2))

        grouped = log.groupby(["case:ApplicationType", "case:LoanGoal"])
        for keys, group in grouped:
            amounts = pd.to_numeric(group["case:RequestedAmount"], errors='coerce').dropna().to_numpy()
            
            if len(amounts) >= 5 and np.var(amounts) > 0:
                try:
                    shape, loc, scale = stats.lognorm.fit(amounts)
                    self.amount_dists[keys] = (round(shape,2), round(loc,2), round(scale,2))
                except Exception:
                    self.amount_dists[keys] = self.global_params
            else:
                self.amount_dists[keys] = self.global_params

    
    def push_event(self, time_point: datetime, event_type: EventType, activity: str, data: dict = dict(), case=None)-> None:
        if case is None:
            case = Case(self.case_counter)
            self.case_counter+=1
        
        if data is not None:
            data.update({"time:timestamp": time_point, "lifecycle:transition": event_type, "concept:name": activity, "EventID": self.event_counter})
        
        event = Event(event_type, activity, time_point, self.event_counter,case, data)
        self.event_counter += 1
        case.addEvent(event)
        heapq.heappush(self.event_queue, event)

    def pop_event(self):
        return heapq.heappop(self.event_queue)
    
    def sample_case_data(self) -> Dict[str, str]:
        #sample according to the frequencies
        sampled_id = random.choices(self.freq.index, weights=self.freq["prob"])[0]
        row = self.freq.loc[sampled_id]
    
        app_type = row["case:ApplicationType"]
        loan_goal = row["case:LoanGoal"]
        lookup_key = (app_type, loan_goal)

        #sample the requestedAmount after looking up its distribution for this pair of applicationType and loanGoal
        shape, loc, scale = self.amount_dists.get(lookup_key, self.global_params)
        requested_amount = stats.lognorm.rvs(shape, loc, scale)

        return {
        "case:ApplicationType": row["case:ApplicationType"],
        "case:LoanGoal": row["case:LoanGoal"],
        "case:RequestedAmount": round(requested_amount,1),
        "EventOrigin": "Application"
        }


    def check_waiting_processes(self) -> None:
            unallocated_events =[]
            while self.waiting_processes:
                event = heapq.heappop(self.waiting_processes)
                event_time = event.time
                event.time = self.simulation_time
                if(self.resourceEngine.allocateResource(event)):

                    event.update({"EventID": self.event_counter,"lifecycle:transition": EventType.ACTIVITY_RESUME, "time:timestamp": self.simulation_time})
                    self.event_counter += 1
                    endTimeActivity = self.processTimeEngine.getProcessingTime(event)+event.time
                    self.push_event(endTimeActivity, EventType.ACTIVITY_END, event.activity, event.getAttribs(), event.eventCase)
                    self.logger.log_event(event)
                else:
                    event.time=event_time
                    unallocated_events.append(event)

            for event in unallocated_events:
                heapq.heappush(self.waiting_processes, event)

    def run(self, start_time: datetime, end_time: datetime, format_type=["csv", "xes"]) -> None:
        event_counter = 0

        self.push_event(start_time,EventType.CASE_ARRIVAL,"", self.sample_case_data(), Case(self.case_counter))
        self.case_counter+=1

        while self.event_queue:
            event = self.pop_event()
            self.simulation_time= event.time
            if self.simulation_time > end_time:
                break
            if event.eventType == EventType.CASE_ARRIVAL:
                data= self.sample_case_data()
                firstActivity = self.bpmnEngine.getStartActivity(data)

                #put event on top of event queue
                self.push_event(event.time, EventType.ACTIVITY_START, firstActivity, data, event.eventCase)
                self.bpmnEngine.initialize_case(event.eventCase)

                newCase = Case(self.case_counter)
                self.case_counter += 1
                self.cases.append(newCase)

                #plan next case arrival
                nextAttrivalTime = self.arrivalEngine.nextArrivalTime(event.time)+event.time

                self.push_event(nextAttrivalTime, EventType.CASE_ARRIVAL,"", dict(), newCase)
                continue
            
            if event.eventType == EventType.ACTIVITY_START:
                validResource = self.resourceEngine.allocateResource(event)
                if(validResource):
                    endTimeActivity = self.processTimeEngine.getProcessingTime(event)+event.time
                    self.push_event(endTimeActivity, EventType.ACTIVITY_END, event.activity, event.getAttribs(), event.eventCase)
                else:
                    event.eventType = EventType.ACTIVITY_SUSPEND
                    heapq.heappush(self.waiting_processes, event)
            elif event.eventType == EventType.ACTIVITY_END:
                self.bpmnEngine.fire_activity(event.activity, event.eventCase.caseId)
                self.resourceEngine.releaseResource(event)
                posNextAcitivites = self.bpmnEngine.getPossibleNextActivities(event.activity, case_id=event.eventCase.caseId)
                newActivities = self.branchingEngine.getNextActivities(event,posNextAcitivites)
                if not(newActivities is None or newActivities==[]):
                    if isinstance(newActivities, str):
                        newActivities = [newActivities]
                    for newActivity in newActivities:
                        if newActivity is None:
                            continue
                        time = self.processTimeEngine.getWaitingTime(event, newActivity)
                        self.push_event(
                            time + event.time,
                            EventType.ACTIVITY_START,
                            newActivity,
                            event.getAttribs(),
                            event.eventCase
                        )
                self.check_waiting_processes()
            
            self.logger.log_event(event)
        if "csv" in format_type:
            self.logger.to_csv()
        if "xes" in format_type:
            self.logger.to_xes()

if __name__ == "__main__":
    simulation_engine = Engine()
    simulation_engine.run(datetime(2000,1,3,9,0), datetime(2000,1,30), "xes")