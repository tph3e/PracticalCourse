from __future__ import annotations

from enum import Enum, auto
import heapq
import pandas as pd
import csv
import heapq

from DummyEngines import ArrivalEngine, BPMNEngine, ResourceEngine, ProcessTimeEngine, BranchingEngine
from datetime import datetime, timedelta
from Helper import *
import xml.etree.ElementTree as ET

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

    #def toXes(self, filepath)


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
        log = pd.read_csv(dataPath)

        self.arrivalEngine = ArrivalEngine(log, seed)
        self.bpmnEngine = BPMNEngine()
        self.resourseEngine = ResourceEngine()
        self.branchingEngine = BranchingEngine()
        self.processTimeEngine = ProcessTimeEngine()
        self.logger = EventLogger()
    
    def push_event(self, timePoint: datetime, eventType, activity, data: dict = dict(), case=None):
        if data!=None:
            data.update({"time:timestamp": timePoint, "lifecycle:transition": eventType, "concept:name": activity, "EventID": self.eventCounter})
        event = Event(eventType, activity, timePoint, self.eventCounter,case, data)
        self.eventCounter += 1
        if case!=None:
            case.addEvent(event)

        
        heapq.heappush(self.eventQueue, event)

    def pop_event(self):
        return heapq.heappop(self.eventQueue)
    
    def sample(self):
        res = {}
        return res
    
    def checkWaitingProcesses(self):
            for event in self.waitingProcesses:
                if(self.resourseEngine.allocateResource(event)):
                    event.update({"case:concept:name": self.eventCounter,"lifecycle:transition": EventType.ACTIVITY_RESUME, "time:timestamp": self.simulationTime})
                    self.eventCounter += 1
                    self.logger.logEvent(event)

    def run(self, startTime, endTime):
        eventCounter = 0

        self.push_event(startTime,EventType.CASE_ARRIVAL,None, self.sample())
        while self.eventQueue:
            event = self.pop_event()
            self.simulationTime= event.time
            if self.simulationTime > endTime:
                break
            if event.eventType == EventType.CASE_ARRIVAL:
                data= self.sample()
                firstActivity = self.bpmnEngine.getStartActivity(data)
                newCase = Case(self.caseCounter)
                self.caseCounter += 1
                self.cases.append(newCase)

                #put event on top of event queue
                self.push_event(event.time, EventType.ACTIVITY_START, firstActivity, data, newCase)

                #plan next case arrival
                nextAttrivalTime = self.arrivalEngine.nextArrivalTime(event.time)+event.time

                self.push_event(nextAttrivalTime, EventType.CASE_ARRIVAL,None, dict(), None)
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
                for newActivity in self.branchingEngine.getNextActivities(event,self.bpmnEngine.getPossibleNextActivities(event.activity)):
                    time = self.processTimeEngine.getWaitingTime(event, newActivity)
                    self.push_event(time+event.time, EventType.ACTIVITY_START, newActivity, event.getAttribs(), event.eventCase)
                self.checkWaitingProcesses()
            
            self.logger.logEvent(event)
        self.logger.toXES()

if __name__ == "__main__":
    simulationEngine = Engine()
    simulationEngine.run(datetime(2000,1,1), datetime(2000,1,2))