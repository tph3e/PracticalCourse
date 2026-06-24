# EngineCore Anpassungen für die ResourceEngine 


## 1. checkWaitingProcesses crasht
self.waitingProcesses.pop(event) schlägt fehl: list.pop() erwartet einen Index,
kein Event -> TypeError beim ersten Resume. Außerdem wird über die Liste iteriert,
während sie verändert wird. 

**Aktuell (Zeile 176):**
```python
    def checkWaitingProcesses(self):
            for event in self.waitingProcesses:
                if(self.resourseEngine.allocateResource(event)):
                    self.waitingProcesses.pop(event)
                    event.update({"EventID": self.eventCounter,"lifecycle:transition": EventType.ACTIVITY_RESUME, "time:timestamp": self.simulationTime})
                    self.eventCounter += 1
                    endTimeActivity = self.processTimeEngine.getProcessingTime(event)+event.time
                    self.push_event(endTimeActivity, EventType.ACTIVITY_END, event.activity, event.getAttribs(), event.eventCase)
                    self.logger.logEvent(event)
```

**Vorschlag:**
```python
    def checkWaitingProcesses(self):
            for event in list(self.waitingProcesses):          # über eine Kopie iterieren
                if(self.resourseEngine.allocateResource(event)):
                    self.waitingProcesses.remove(event)         # pop braucht Index
                    event.update({"EventID": self.eventCounter,"lifecycle:transition": EventType.ACTIVITY_RESUME, "time:timestamp": self.simulationTime})
                    self.eventCounter += 1
                    endTimeActivity = self.processTimeEngine.getProcessingTime(event)+event.time
                    self.push_event(endTimeActivity, EventType.ACTIVITY_END, event.activity, event.getAttribs(), event.eventCase)
                    self.logger.logEvent(event)
```

## 2, Resume auch bei Zeitfortschritt
checkWaitingProcesses() wird bisher nur im ACTIVITY_END-Block aufgerufen. Also nur,
wenn eine andere Aktivität endet. Ressourcen werden aber auch durch Zeit verfügbar
(Verfügbarkeitskalender, 1.6). Sind nachts alle Fälle suspendiert, feuert kein
ACTIVITY_END -> sie resumen nie -> Deadlock.

**Aktuell (Zeile 222):**
```python
            if event.eventType == EventType.ACTIVITY_END:
                self.resourseEngine.releaseResource(event)
                for newActivity in self.branchingEngine.getNextActivities(event,self.bpmnEngine.getPossibleNextActivities(event.activity)):
                    time = self.processTimeEngine.getWaitingTime(event, newActivity)
                    self.push_event(time+event.time, EventType.ACTIVITY_START, newActivity, event.getAttribs(), event.eventCase)
                self.checkWaitingProcesses()

            self.logger.logEvent(event)
        self.logger.toXES()
```

**Vorschlag:**
```python
            if event.eventType == EventType.ACTIVITY_END:
                self.resourseEngine.releaseResource(event)
                for newActivity in self.branchingEngine.getNextActivities(event,self.bpmnEngine.getPossibleNextActivities(event.activity)):
                    time = self.processTimeEngine.getWaitingTime(event, newActivity)
                    self.push_event(time+event.time, EventType.ACTIVITY_START, newActivity, event.getAttribs(), event.eventCase)
                # checkWaitingProcesses() hier raus (siehe unten)

            self.logger.logEvent(event)
            self.checkWaitingProcesses()   # nach jedem Event prüfen (Ressource frei via END oder via Zeit/Verfügbarkeit)
        self.logger.toXES()
```

