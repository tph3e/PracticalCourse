from __future__ import annotations
import pandas as pd
from enum import Enum
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from collections import Counter

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
    def __init__(self, caseId):
        self.caseId=caseId
        self.events: List[Event] = []
        self.activities: List = []
        self._activity_counts: Counter = Counter()

        self.applicationType=""
        self.requestedAmount=0.0
        self.loanGoal=0.0

    def addEvent(self, event: Event)-> None:
        self.events.append(event)
        self.activities.append(event.activity)
        self._activity_counts[event.activity] += 1

        event.eventCase=self

    def getActivityCount(self, activity: str)-> int:
        return self._activity_counts.get(activity, 0)
    
    def getData(self, amount=-1) -> List[Dict[str, Any]]:
        if amount == -1 or amount >= len(self.events):
            target_events = self.events
        else:
            target_events = self.events[-amount:]
        return [event.getAttribs() for event in target_events]
    
    def __str__(self) -> str:
        return f"Case {self.caseId} with {len(self.events)} events"

class Event:
    _EVENT_ATTR_MAP = {
        "Action": "action",
        "org:resource": "resource",
        "concept:name": "activity",
        "EventOrigin": "eventOrigin",
        "EventID": "eventId",
        "lifecycle:transition": "eventType",
        "time:timestamp": "time",
    }
    
    _CASE_ATTR_MAP = {
        "case:LoanGoal": "loanGoal",
        "case:ApplicationType": "applicationType",
        "case:concept:name": "caseId",
        "case:RequestedAmount": "requestedAmount",
    }

    def __init__(self, eventType: EventType, activity: str, time: datetime, eventId: int, case: Case, data: Optional[Dict] = None):
        
        self.time = time
        self.eventId = eventId
        self.eventType = eventType
        self.eventCase=case
        self.activity=activity

        self.offerData = offerDefault.copy()

        self.action: str = ""
        self.resource: str = ""
        self.eventOrigin: str = ""
        self.time_difference = timedelta(0)

        if data:
            self.update(data)


    def __lt__(self, other) -> bool:
        if not isinstance(other, Event):
            return NotImplemented

        self_rank = ORDER.get(self.eventType, float('inf'))
        other_rank = ORDER.get(other.eventType, float('inf'))
        
        return (self.time, self_rank, str(self.eventCase.caseId)) < \
               (other.time, other_rank, str(other.eventCase.caseId))
    
    def update(self, data: dict) -> None:
        if not data:
            return
            
        for k, v in data.items():
            if k in self.offerData:
                self.offerData[k] = v
                continue
                
            if k in self._EVENT_ATTR_MAP:
                setattr(self, self._EVENT_ATTR_MAP[k], v)
            elif k in self._CASE_ATTR_MAP:
                setattr(self.eventCase, self._CASE_ATTR_MAP[k], v)
            else:
                print(f"Unknown attribute: {k}")
    
    def getAttribs(self, details=False) -> dict:
        base = {
            'Action': self.action,
            'org:resource': self.resource,
            'concept:name': self.activity,
            'EventOrigin': self.eventOrigin,
            'EventID': self.eventId,
            'lifecycle:transition': self.eventType.value if isinstance(self.eventType, EventType) else self.eventType,
            'time:timestamp': self.time,
            'case:LoanGoal': self.eventCase.loanGoal if self.eventCase else None,
            'case:ApplicationType': self.eventCase.applicationType if self.eventCase else None,
            'case:concept:name': str(self.eventCase.caseId) if self.eventCase else None,
            'case:RequestedAmount': self.eventCase.requestedAmount if self.eventCase else None
        }
        if details:
            base.update({
                'strain_time_difference': self.time_difference if self.time_difference else timedelta(0)
            })
        return {**base, **self.offerData}
    
    def getAttribOfLastEvents(self, amount: int = -1) -> List[Dict[str, Any]]:
        if not self.eventCase:
            return []
        return self.eventCase.getData(amount)
    
    def __str__(self):
        return f"Event {self.eventId} at {self.time} of type {self.eventType} doing {self.activity}"