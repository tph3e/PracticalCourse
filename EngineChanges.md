# ProcessTimeEngine: unbrauchbare Bearbeitungszeiten (process_time_engine.py)

Auf dem integrierten Simulator kollabieren alle Zeitstempel eines Falls auf einen Punkt oder der
Fall bleibt für immer stehen, occupation und fairness sind dadurch bedeutungslos. 

Grund: die Bearbeitungszeit ist entweder 0 (unbekanntes Paar) oder
etwa 100 Jahre (Einheitenfehler).

Verifiziert: mit den Fixes liefert der Sim cycle_time 0,58 Tage, occupation 0,255, fairness 0,791 (vorher alles 0).

### 1. Einheiten: sample_distrib, Zeile 208-215

Trainingsziel ist in Sekunden (Zeile 198), Advanced-Pfad liest korrekt timedelta(seconds=...), nur
hier wird der Wert als timedelta(x)=Tage gelesen. 

Aktuell:
```python
    def sample_distrib(self, distrib, param) -> timedelta:
        if distrib == "poisson":
            return timedelta(stats.poisson.rvs(mu = param["lambda"], random_state= self.rndm_state))
        if distrib == "gamma":
            return timedelta(stats.gamma.rvs(param["shape"], loc=0, scale=param["scale"], random_state= self.rndm_state))
        if distrib == "lognorm":
            return timedelta(stats.lognorm.rvs(param["shape"], loc=0, scale=param["scale"], random_state= self.rndm_state))
        return timedelta(0)
```

Vorschlag (self.rng einmal im __init__ nach self.rndm_state = seed anlegen: self.rng = np.random.default_rng(seed)):
```python
    def sample_distrib(self, distrib, param) -> timedelta:
        if distrib == "poisson":
            return timedelta(seconds=stats.poisson.rvs(mu = param["lambda"], random_state=self.rng))
        if distrib == "gamma":
            return timedelta(seconds=stats.gamma.rvs(param["shape"], loc=0, scale=param["scale"], random_state=self.rng))
        if distrib == "lognorm":
            return timedelta(seconds=stats.lognorm.rvs(param["shape"], loc=0, scale=param["scale"], random_state=self.rng))
        return timedelta(0)
```

### 2. Unbekanntes Paar liefert 0: sampleTime_basic, Zeile 228-235

Das aktivitätsweite Fallback-Modell fallback_models_basic wird bereits trainiert (Zeile 102),
gespeichert und geladen, aber nie benutzt. Bei fehlendem Paar darauf zurückfallen.

Aktuell:
```python
    def sampleTime_basic(self, activity, resource="", kind="processing") -> timedelta:
        key = f"{activity}_{resource}_{kind}"
        if key in self.models_basic:
            if(np.random.rand() < self.models_basic[key]["0-proportion"]):
                return timedelta(minutes=0)
            return self.sample_distrib(self.models_basic[key]["distribution"], self.models_basic[key]["parameters"])
        else:
            return timedelta(0)
```

Vorschlag:
```python
    def sampleTime_basic(self, activity, resource="", kind="processing") -> timedelta:
        model = self.models_basic.get(f"{activity}_{resource}_{kind}") \
            or self.fallback_models_basic.get(f"{activity}_{kind}")
        if model is None:
            return timedelta(0)
        if np.random.rand() < model["0-proportion"]:
            return timedelta(0)
        return self.sample_distrib(model["distribution"], model["parameters"])
```

### 3. Training: format_data, Zeile 183 und 198-199

.seconds gibt nur den Anteil innerhalb eines Tages (0 bis 86399), ab einem Tag abgeschnitten, genau
ein Tag wird 0. Und die Wartezeit ist vorzeichenverkehrt (Suspend minus Resume statt Resume minus Suspend).

Aktuell:
```python
                    total_waiting_time = activity_waiting - log_entry["time:timestamp"]
```
```python
                        "processing_time": total_active_time.seconds,
                        "waiting_time": total_waiting_time.seconds,
```

Vorschlag:
```python
                    total_waiting_time = log_entry["time:timestamp"] - activity_waiting
```
```python
                        "processing_time": total_active_time.total_seconds(),
                        "waiting_time": total_waiting_time.total_seconds(),
```

