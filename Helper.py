from __future__ import annotations

from enum import Enum, auto
from datetime import datetime, timedelta

import pandas as pd

offerDefault = {'FirstWithdrawalAmount': None,
 'NumberOfTerms': None,
 'Accepted': None,
 'MonthlyCost': None,
 'Selected': None,
 'CreditScore': None,
 'OfferedAmount': None,
 'OfferID': None}

class EventType(Enum):
    CASE_ARRIVAL = "schedule"
    ACTIVITY_START = "start"
    ACTIVITY_END =  "complete"
    ACTIVITY_SUSPEND = "suspend"
    ACTIVITY_RESUME = "resume"

    #currently not used types:
    ACTIVITY_WITHDRAW = "withdraw"
    ACTIVITY_ABORT = "ate_abort"


ORDER = {
    EventType.ACTIVITY_WITHDRAW: 0,
    EventType.ACTIVITY_ABORT: 1, 
    EventType.ACTIVITY_END: 2,
    EventType.ACTIVITY_SUSPEND: 3,
    EventType.ACTIVITY_RESUME: 4,
    EventType.ACTIVITY_START: 5,
    EventType.CASE_ARRIVAL: 6
}

class Case:
    caseId: str
    events: list
    activities: list
    applicationType: str
    requestedAmount=0
    loanGoal=0

    def addEvent(self, event: Event):
        self.events.append(event)
        self.activities.append(event.activity)

        event.eventCase=self
    
    def __str__(self):
        return f"Case {self.caseId} with activities {self.activities}"
    
    def getData(self, amount=-1):
        res = {}
        if(amount==-1):
            amount=len(self.events)-1
        for event in reversed(self.events[-amount:]):
            res.update(event.getAttribs())
        return res
    
    def __init__(self, caseId):
        self.caseId=caseId
        self.events=[]
        self.activities=[]
        self.applicationType=""

    def getActivityCount(self, activity):
        count=0
        for event in self.events:
            if event.activity == activity:
                count+=1
        return count

class Event:
    action=""
    activity: str
    resource=""
    time: datetime
    eventId: int
    eventType: EventType
    eventCase: Case
    eventOrigin=""
    eventId: int
    offerData= offerDefault.copy()

    def __init__(self, eventType: EventType, activity, time: datetime, eventId: int, case, data=offerDefault):
        
        self.time = time
        self.eventId = eventId
        self.eventType = eventType
        self.eventCase=case
        self.activity=activity

        self.update(data)
    
    def __lt__(self, other):
        if self.time== other.time:
            if ORDER[self.eventType]<ORDER[other.eventType]:
                return self.eventCase.caseId<other.eventCase.caseId
            else:
                return ORDER[self.eventType]<ORDER[other.eventType]
        return self.time < other.time

    def __str__(self):
        return f"Event {self.eventId} at {self.time} of type {self.eventType} doing {self.activity}"
    
    def getAttribs(self) -> dict:
            base =   {'Action': self.action,
            'org:resource': self.resource,
            'concept:name': self.activity,
            'EventOrigin': self.eventOrigin,
            'EventID': self.eventId,
            'lifecycle:transition': self.eventType.value,
            'time:timestamp': self.time,
            'case:LoanGoal': self.eventCase.loanGoal,
            'case:ApplicationType': self.eventCase.applicationType,
            'case:concept:name': self.eventCase.caseId,
            'case:RequestedAmount': self.eventCase.requestedAmount}
            base.update(self.offerData)
            return base
    
    def update(self, data: dict):
        if data==None:
            return
        for k,v in data.items():
            if k in self.offerData.keys():
                self.offerData[k]=v
                continue
            match k:
                case "Action":
                    self.action=v
                case "org:resource":
                    self.resource=v
                case "concept:name":
                    self.activity=v
                case "EventOrigin":
                    self.eventOrigin=v
                case "EventID":
                    self.eventId=v
                case "lifecycle:transition":
                    self.eventType = v
                case "time:timestamp":
                    self.time=v
                case "case:LoanGoal":
                    self.eventCase.loanGoal=v
                case "case:ApplicationType":
                    self.eventCase.applicationType=v
                case "case:concept:name":
                    self.eventCase.caseId=v
                case "case:RequestedAmount":
                    self.eventCase.requestedAmount=v
                case _:
                    print("Unknown attribute: "+k)

    def getAttribOfLastEvents(self, amount=-1):
        return self.eventCase.getData(amount)