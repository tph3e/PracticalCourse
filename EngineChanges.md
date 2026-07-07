# EngineCore Anpassungen für die ResourceEngine 


## 1. Sim-Fenster ohne Arbeitszeit 
Es ist kein Bug im ResourceEngine, sondern das gewählte Zeitfenster.


**Aktuell (Zeile 223):**
```python
    simulation_engine.run(datetime(2000,1,1), datetime(2000,1,3))
```


2000-01-01 ist ein Samstag, 2000-01-03 00:00 ein Montag Mitternacht. Das
Fenster enthält also keine einzige Arbeitsstunde. 


**Vorschlag:** ein Fenster mit Werktag-Arbeitszeit wählen, z.B.
```python
    simulation_engine.run(datetime(2000,1,3,9,0), datetime(2000,1,31))
```



## 2. Branching: Folgeaktivitäten-Schleife invertiert 
Taucht erst auf, sobald Punkt 1 behoben ist 

getNextActivities() liefert eine Liste. Der aktuelle Code weist die ganze
Liste newActivity zu und prüft == [] 


**Aktuell (Zeile 208):**
```python
            newActivity = self.branchingEngine.getNextActivities(event,self.bpmnEngine.getPossibleNextActivities(event.activity))
            if newActivity ==[]:
                time = self.processTimeEngine.getWaitingTime(event, newActivity)
                self.push_event(time+event.time, EventType.ACTIVITY_START, newActivity, event.getAttribs(), event.eventCase)
```


Folge: bei vorhandenen Folgeaktivitäten passiert nichts (Fall stoppt nach der ersten
Aktivität). Bei Fallende wird ein Müll-Event mit Aktivität [] gepusht.


**Vorschlag:** über die Liste iterieren und nur bei nicht-leerer Liste pushen.
```python
            for na in self.branchingEngine.getNextActivities(event, self.bpmnEngine.getPossibleNextActivities(event.activity)):
                time = self.processTimeEngine.getWaitingTime(event, na)
                self.push_event(time+event.time, EventType.ACTIVITY_START, na, event.getAttribs(), event.eventCase)
```
