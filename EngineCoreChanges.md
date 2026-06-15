# Engine Core Anpassungen für die ResourceEngine 

Damit die echte ResourceEngine greifen kann, sind drei
Änderungen im Engine-Core nötig. 

1. Import + Konstruktor mit Log versorgen
Die ResourceEngine braucht das historische Log, um Permissions (1.7) und
Verfügbarkeiten (1.6) zu lernen.

Also konkret:

statt: from DummyEngines import ..., ResourceEngine, ...
-> from resources import ResourceEngine
...
statt: self.resourseEngine = ResourceEngine()
-> self.resourseEngine = ResourceEngine(log, seed)


2. Ressource bei ACTIVITY_END freigeben
Ohne Freigabe bleibt jede Ressource dauerhaft belegt -> die Simulation
deadlockt, sobald die Ressourcen knapp werden. Im ACTIVITY_END-Block
(aktuell Zeile 160) ergänzen:

-> if event.eventType == EventType.ACTIVITY_END:
        self.resourseEngine.releaseResource(event)   # neu
        for newActivity in self.branchingEngine.getNextActivities(...):
        ...
        self.checkWaitingProcesses()


3. checkWaitingProcesses reparieren
Aktuell wird ein wieder aufgenommener Prozess nur geloggt, aber (a) es wird
kein ACTIVITY_END eingeplant, (b) das Event wird nie aus
waitingProcesses entfernt (Doppelbelegung + Leak), und (c) die Case-ID
wird mit dem Event-Counter überschrieben ("case:concept:name": self.eventCounter),
was die Fallidentität im Log zerstört.

