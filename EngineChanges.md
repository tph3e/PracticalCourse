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