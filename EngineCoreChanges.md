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
