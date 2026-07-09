from __future__ import annotations

from enum import Enum, auto
import heapq
import pandas as pd
import csv
import pm4py
import random
import sys
import pickle
from pathlib import Path

from DummyEngines import ArrivalEngine, ProcessTimeEngine
from src.branching.ProbabilityBranchingEngine import ProbabilityBranchingEngine
from src.branching.PredictiveBranchingEngine import PredictiveBranchingEngine
from resources import ResourceEngine
from datetime import datetime, timedelta
from Helper import *
import xml.etree.ElementTree as ET
from BPMN_engine import BPMNEngine

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


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

        log = ET.Element(
            "log",
            {
                "xes.version": "1.0",
                "xes.features": "nested-attributes",
                "xmlns": "http://www.xes-standard.org/",
            },
        )

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
                "string",
                {
                    "key": "concept:name",
                    "value": str(case_id),
                },
            )

            for event in events:
                event_elem = ET.SubElement(trace, "event")
                for key, value in event.getAttribs().items():
                    if value is None:
                        continue
                    else:
                        ET.SubElement(
                            event_elem,
                            "string",
                            {
                                "key": key,
                                "value": str(value),
                            },
                        )

        tree = ET.ElementTree(log)
        ET.indent(tree, space="  ")

        tree.write(
            filepath,
            encoding="utf-8",
            xml_declaration=True,
        )


class Engine:
    
    eventCounter: int
    caseCounter: int
    cases = []
    waitingProcesses = []
    simulationTime: object

    logger: EventLogger
    arrivalEngine: ArrivalEngine
    bpmnEngine: BPMNEngine
    resourseEngine: ResourceEngine
    processTimeEngine: ProcessTimeEngine

    def __init__(self, dataPath="../PracticalCourse/data/BPI Challenge 2017.xes", seed=1):
        self.eventCounter = 0
        self.caseCounter = 0
        self.eventQueue = []

        # Read event log.
        # Important: no variant="r4pm", because r4pm/rustxes is not installed locally.
        log = pm4py.read_xes(dataPath)

        self.arrivalEngine = ArrivalEngine(log, seed)
        self.bpmnEngine = BPMNEngine()
        self.resourseEngine = ResourceEngine(log, seed)

        # Task 1.5 final integration:
        # Load the final predictive branching model selected by split comparison.
        model_path = PROJECT_ROOT / "results" / "final_predictive_model_full.pkl"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Final predictive model not found: {model_path}. "
                "Run scripts/train_full_predictive_model.py first."
            )

        with open(model_path, "rb") as file:
            model_bundle = pickle.load(file)

        self.branchingEngine = model_bundle["predictive_engine"]
        self.probabilityFallbackEngine = model_bundle["probability_engine"]
        self.branchingMetrics = model_bundle.get("metrics", {})
        self.branchingTrainingMode = model_bundle.get("training_mode", "validation_split")

        print("[Branching] Engine:", self.branchingEngine.__class__.__name__)
        print("[Branching] Loaded final model:", model_path)
        print("[Branching] Training mode:", self.branchingTrainingMode)
        print("[Branching] Train ratio:", self.branchingMetrics.get("train_ratio", "full_log"))
        print("[Branching] Accuracy:", self.branchingMetrics.get("accuracy"))
        print("[Branching] Macro F1:", self.branchingMetrics.get("macro_f1"))
        print("[Branching] Weighted F1:", self.branchingMetrics.get("weighted_f1"))

        self.processTimeEngine = ProcessTimeEngine(log)
        self.logger = EventLogger()
        self.train(log)

    def train(self, log):
        freq = (
            log.groupby(["case:ApplicationType", "case:LoanGoal"])
            .size()
            .rename("count")
            .reset_index()
        )
        total = freq["count"].sum()
        freq["prob"] = freq["count"] / total
        self.freq = freq
        return True

    def push_event(
        self,
        timePoint: datetime,
        eventType,
        activity,
        data: dict = dict(),
        case=None,
    ):
        if case is None:
            case = Case(self.caseCounter)
            self.caseCounter += 1

        if data is not None:
            data.update(
                {
                    "time:timestamp": timePoint,
                    "lifecycle:transition": eventType,
                    "concept:name": activity,
                    "EventID": self.eventCounter,
                }
            )

        event = Event(
            eventType,
            activity,
            timePoint,
            self.eventCounter,
            case,
            data,
        )

        self.eventCounter += 1
        case.addEvent(event)

        heapq.heappush(self.eventQueue, event)

    def pop_event(self):
        return heapq.heappop(self.eventQueue)
    
    def sample(self):
        idx = random.choices(self.freq.index, self.freq["prob"])
        row = self.freq.loc[idx]

        return {
            "case:ApplicationType": row["case:ApplicationType"],
            "case:LoanGoal": row["case:LoanGoal"],
        }

    def checkWaitingProcesses(self):
        for event in self.waitingProcesses:
            if self.resourseEngine.allocateResource(event):
                self.waitingProcesses.pop(event)

                event.update(
                    {
                        "EventID": self.eventCounter,
                        "lifecycle:transition": EventType.ACTIVITY_RESUME,
                        "time:timestamp": self.simulationTime,
                    }
                )

                self.eventCounter += 1

                endTimeActivity = (
                    self.processTimeEngine.getProcessingTime(event)
                    + event.time
                )

                self.push_event(
                    endTimeActivity,
                    EventType.ACTIVITY_END,
                    event.activity,
                    event.getAttribs(),
                    event.eventCase,
                )

                self.logger.logEvent(event)

    def run(self, startTime, endTime):
        eventCounter = 0

        self.push_event(
            startTime,
            EventType.CASE_ARRIVAL,
            Case(self.caseCounter),
            self.sample(),
        )

        self.caseCounter += 1

        while self.eventQueue:
            event = self.pop_event()
            self.simulationTime = event.time

            if self.simulationTime > endTime:
                break

            if event.eventType == EventType.CASE_ARRIVAL:
                data = self.sample()
                firstActivity = self.bpmnEngine.getStartActivity(data)

                # Put first activity on top of event queue.
                self.push_event(
                    event.time,
                    EventType.ACTIVITY_START,
                    firstActivity,
                    data,
                    event.eventCase,
                )

                newCase = Case(self.caseCounter)
                self.caseCounter += 1
                self.cases.append(newCase)

                # Plan next case arrival.
                nextAttrivalTime = (
                    self.arrivalEngine.nextArrivalTime(event.time)
                    + event.time
                )

                self.push_event(
                    nextAttrivalTime,
                    EventType.CASE_ARRIVAL,
                    None,
                    dict(),
                    newCase,
                )

                continue
            
            if event.eventType == EventType.ACTIVITY_START:
                validResource = self.resourseEngine.allocateResource(event)

                if validResource:
                    endTimeActivity = (
                        self.processTimeEngine.getProcessingTime(event)
                        + event.time
                    )

                    self.push_event(
                        endTimeActivity,
                        EventType.ACTIVITY_END,
                        event.activity,
                        event.getAttribs(),
                        event.eventCase,
                    )

                else:
                    event.eventType = EventType.ACTIVITY_SUSPEND
                    heapq.heappush(self.waitingProcesses, event)

            if event.eventType == EventType.ACTIVITY_END:
                self.resourseEngine.releaseResource(event)

                self.bpmnEngine.fire_activity(
                    event.activity,
                    event.eventCase.caseId,
                )

                possible_next_activities = self.bpmnEngine.getPossibleNextActivities(
                    event.activity,
                    event.eventCase.caseId,
                )

                selected_next_activities = self.branchingEngine.getNextActivities(
                    event,
                    possible_next_activities,
                )

                for newActivity in selected_next_activities:
                    time = self.processTimeEngine.getWaitingTime(
                        event,
                        newActivity,
                    )

                    self.push_event(
                        time + event.time,
                        EventType.ACTIVITY_START,
                        newActivity,
                        event.getAttribs(),
                        event.eventCase,
                    )

                self.checkWaitingProcesses()
            
            self.logger.logEvent(event)

        self.logger.toXES()


if __name__ == "__main__":
    simulationEngine = Engine()
    simulationEngine.run(
        datetime(2000, 1, 1),
        datetime(2000, 1, 2),
    )