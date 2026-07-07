# BPMN Engine Problem

BPMN engine prüft nicht die unsichtbaren Transitionen.
Daher kann nach der 1. Transition nach A_Submitted keine weitere Transition ausgelöst werden.
Hier eine Überlegung für getPossibleNextActivities.
Dies hat auch Auswirkungen auf fire_activity.

```python
def getPossibleNextActivities(self, current_activity, case_id=None) -> list:
        if case_id is None:
            # Falls kein case_id übergeben wird, initialisieren wir Dummy-Markierungen, 
            # um die Startaktivität zu finden.
            dummy_marking = self.init_marking
        else:
            if case_id not in self.case_markings:
                self.initialize_case(case_id)
            dummy_marking = self.case_markings[case_id]

        possible_next = set()
        
        # Queue für die Breitensuche (BFS) durch unsichtbare Transitionen
        queue = [dummy_marking]
        visited = []

        while queue:
            marking = queue.pop(0)
            if marking in visited:
                continue
            visited.append(marking)

            for transition in self.net.transitions:
                if is_enabled(transition, self.net, marking):
                    if transition.label is not None:
                        # Echte Aktivität gefunden!
                        possible_next.add(transition.label)
                    else:
                        # Unsichtbare Transition (z.B. Gateway) gefunden -> abfeuern und weitersuchen
                        new_marking = execute(transition, self.net, marking)
                        if new_marking not in visited:
                            queue.append(new_marking)

        return list(possible_next)
```



# Weitere Bugs

Mit dem Bug oben plus den vier Fixes unten läuft der Sim lokal durch: 2-Tage-Fenster
in unter 1s (vorher Endlos-Loop).


## 1. fire_activity wird im Core nie aufgerufen (SimulationEngineCore.py)

Ohne fire_activity rückt die Marking nie vor, getPossibleNextActivities liefert
immer die Startaktivität, Fälle loopen endlos.

Aktuell (Zeile 202-204):
```python
            elif event.eventType == EventType.ACTIVITY_END:
                self.resourceEngine.releaseResource(event)
                posNextAcitivites = self.bpmnEngine.getPossibleNextActivities(event.activity, case_id=event.eventCase.caseId)
```

Vorschlag:
```python
            elif event.eventType == EventType.ACTIVITY_END:
                self.resourceEngine.releaseResource(event)
                self.bpmnEngine.fire_activity(event.activity, event.eventCase.caseId)
                posNextAcitivites = self.bpmnEngine.getPossibleNextActivities(event.activity, case_id=event.eventCase.caseId)
```


## 2. initialize_case mit caseId statt Case-Objekt (SimulationEngineCore.py)

Aktuell (Zeile 182):
```python
                self.bpmnEngine.initialize_case(event.eventCase)
```

Vorschlag:
```python
                self.bpmnEngine.initialize_case(event.eventCase.caseId)
```


## 3. fire_activity traversiert keine unsichtbaren Transitionen (BPMN_engine.py)

Aktivitäten hinter einem Gateway sind nicht direkt aktiviert, fire_activity gibt
False, die Marking hängt. Gleicher Bug wie bei getPossibleNextActivities.

Aktuell:
```python
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
```

Vorschlag:
```python
    def fire_activity(self, activity_name, case_id) -> bool:
        if case_id not in self.case_markings:
            self.initialize_case(case_id)

        queue = [self.case_markings[case_id]]
        visited = []
        while queue:
            marking = queue.pop(0)
            if marking in visited:
                continue
            visited.append(marking)
            for transition in self.net.transitions:
                if is_enabled(transition, self.net, marking):
                    if transition.label == activity_name:
                        self.case_markings[case_id] = execute(transition, self.net, marking)
                        return True
                    elif transition.label is None:
                        new_marking = execute(transition, self.net, marking)
                        if new_marking not in visited:
                            queue.append(new_marking)
        return False
```


## 4. Wartende Events werden mit veralteter Zeit geprüft (SimulationEngineCore.py)

check_waiting_processes:

Ein suspendiertes Event wird beim Re-Check mit seinem alten Suspend-Zeitstempel
bewertet und resumt nie. Vor der Prüfung die Zeit auf jetzt setzen.

Aktuell (Zeile 152-153):
```python
                event = heapq.heappop(self.waiting_processes)
                if(self.resourceEngine.allocateResource(event)):
```

Vorschlag:
```python
                event = heapq.heappop(self.waiting_processes)
                event.time = self.simulation_time
                if(self.resourceEngine.allocateResource(event)):
```
