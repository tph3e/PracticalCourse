from __future__ import annotations
from enum import Enum, auto
import heapq
import pandas as pd
import csv
import pm4py
import random
import os
import sys
from DummyEngines import ArrivalEngine, BPMNEngine, BranchingEngine
from resources import ResourceEngine
from processTimes import ProcessTimeEngine

from datetime import datetime, timedelta
from Helper import *
import xml.etree.ElementTree as ET
import scipy.stats as stats
import numpy as np

class EventLogger:
    
    records = []

    def logEvent(self, event):
        self.records.append(event)

    def toCSV(self, filepath="log.csv"):
        fieldnames = list(self.records[0].getAttribs().keys())

        with open(filepath, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for record in self.records:
                writer.writerow(record.getAttribs())

    def toXES(self, filepath="log.xes"):

        log = ET.Element("log",{"xes.version": "1.0",
                "xes.features": "nested-attributes",
                "xmlns": "http://www.xes-standard.org/"})

        traces = {}
        for event in self.records:
            case_id = event.eventCase.caseId
            if case_id not in traces:
                traces[case_id] = []
            traces[case_id].append(event)


        for case_id, events in traces.items():
            trace = ET.SubElement(log, "trace")

            ET.SubElement(
                trace,
                "string",{"key": "concept:name",
                    "value": str(case_id)})

            for event in events:
                event_elem = ET.SubElement(trace, "event")
                for key, value in event.getAttribs().items():
                    if value is None:
                        continue
                    else:
                        ET.SubElement(event_elem,"string",{"key": key, "value": str(value)})

        tree = ET.ElementTree(log)
        ET.indent(tree, space="  ")

        tree.write(
            filepath,
            encoding="utf-8",
            xml_declaration=True
        )

class Engine:
    
    eventCounter: int
    caseCounter: int
    cases= []
    waitingProcesses= []
    simulationTime: object

    logger: EventLogger
    arrivalEngine: ArrivalEngine
    bpmnEngine: BPMNEngine
    resourseEngine: ResourceEngine
    processTimeEngine: ProcessTimeEngine

    def __init__(self, dataPath="data/logData.xes", seed=1):
        self.eventCounter=0
        self.caseCounter=0
        self.eventQueue = []
        log = pm4py.read_xes(dataPath, variant="r4pm")

        self.arrivalEngine = ArrivalEngine(log, seed)
        self.bpmnEngine = BPMNEngine()
        self.resourseEngine = ResourceEngine(log, seed)
        self.branchingEngine = BranchingEngine()
        self.processTimeEngine = ProcessTimeEngine(log)
        self.logger = EventLogger()
        self.train(log)

    def train(self, log):
        #The distribution of requestedAmount is fitted for every pair of case:ApplicationType and case:LoanGoal seperatly
        freq = (
            log.groupby(["case:ApplicationType", "case:LoanGoal"])
            .size()
            .rename("count")
            .reset_index()
        )
        total = freq["count"].sum()
        freq["prob"] = freq["count"] / total
        self.freq = freq

        self.amount_dists = {}
        
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

        return True

    
    def push_event(self, timePoint: datetime, eventType, activity, data: dict = dict(), case=None):
        if case == None:
            case = Case(self.caseCounter)
            self.caseCounter+=1
        if data!=None:
            data.update({"time:timestamp": timePoint, "lifecycle:transition": eventType, "concept:name": activity, "EventID": self.eventCounter})
        event = Event(eventType, activity, timePoint, self.eventCounter,case, data)
        self.eventCounter += 1
        case.addEvent(event)

        
        heapq.heappush(self.eventQueue, event)

    def pop_event(self):
        return heapq.heappop(self.eventQueue)
    
    def sample(self):
        #sample according to the frequencies
        sampled_idx = random.choices(self.freq.index, weights=self.freq["prob"])[0]
        row = self.freq.loc[sampled_idx]
    
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

    
    def checkWaitingProcesses(self):
            for event in self.waitingProcesses:
                if(self.resourseEngine.allocateResource(event)):
                    self.waitingProcesses.pop(event)
                    event.update({"EventID": self.eventCounter,"lifecycle:transition": EventType.ACTIVITY_RESUME, "time:timestamp": self.simulationTime})
                    self.eventCounter += 1
                    endTimeActivity = self.processTimeEngine.getProcessingTime(event)+event.time
                    self.push_event(endTimeActivity, EventType.ACTIVITY_END, event.activity, event.getAttribs(), event.eventCase)
                    self.logger.logEvent(event)

    def run(self, startTime, endTime):
        eventCounter = 0

        self.push_event(startTime,EventType.CASE_ARRIVAL,Case(self.caseCounter), self.sample())
        self.caseCounter+=1
        while self.eventQueue:
            event = self.pop_event()
            self.simulationTime= event.time
            if self.simulationTime > endTime:
                break
            if event.eventType == EventType.CASE_ARRIVAL:
                data= self.sample()
                firstActivity = self.bpmnEngine.getStartActivity(data)

                #put event on top of event queue
                self.push_event(event.time, EventType.ACTIVITY_START, firstActivity, data, event.eventCase)

                newCase = Case(self.caseCounter)
                self.caseCounter += 1
                self.cases.append(newCase)

                #plan next case arrival
                nextAttrivalTime = self.arrivalEngine.nextArrivalTime(event.time)+event.time

                self.push_event(nextAttrivalTime, EventType.CASE_ARRIVAL,None, dict(), newCase)
                continue
            

            if event.eventType == EventType.ACTIVITY_START:
                validResource = self.resourseEngine.allocateResource(event)
                if(validResource):
                    endTimeActivity = self.processTimeEngine.getProcessingTime(event)+event.time
                    self.push_event(endTimeActivity, EventType.ACTIVITY_END, event.activity, event.getAttribs(), event.eventCase)
                else:
                    event.eventType = EventType.ACTIVITY_SUSPEND
                    heapq.heappush(self.waitingProcesses, event)
            if event.eventType == EventType.ACTIVITY_END:
                self.resourseEngine.releaseResource(event)
                for newActivity in self.branchingEngine.getNextActivities(event,self.bpmnEngine.getPossibleNextActivities(event.activity)):
                    time = self.processTimeEngine.getWaitingTime(event, newActivity)
                    self.push_event(time+event.time, EventType.ACTIVITY_START, newActivity, event.getAttribs(), event.eventCase)
                self.checkWaitingProcesses()
            
            self.logger.logEvent(event)
        self.logger.toXES()

if __name__ == "__main__":
    simulationEngine = Engine()
    simulationEngine.run(datetime(2000,1,1), datetime(2000,1,2))
    print(simulationEngine.amount_dists)