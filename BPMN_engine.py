import pm4py
from pm4py.objects.petri_net.utils import petri_utils
from pm4py.objects.petri_net.semantics import is_enabled, execute, enabled_transitions
from pm4py.objects.petri_net.obj import PetriNet, Marking

class BPMNEngine:
    def __init__(self):
        self.case_markings = {}
        self.model_filename = "models/v4_replay.bpmn"

        try:
            bpmn_graph = pm4py.read_bpmn(self.model_filename)
            self.net, self.init_marking, self.final_marking = pm4py.convert_to_petri_net(bpmn_graph)
            print("[BPMNEngine] Model successfully loaded and converted to Petri net.")
        except Exception as e:
            print(f"[BPMNEngine] Fallback: Failed to load model due to error: {e}")
            from pm4py.objects.petri_net.obj import PetriNet, Marking
            self.net = PetriNet("Safe Fallback")
            self.init_marking = Marking()
            self.final_marking = Marking()

    def initialize_case(self, case_id):
        from copy import copy
        self.case_markings[case_id] = copy(self.init_marking)

    def getStartActivity(self, data=None):
        for transition in self.net.transitions:
            if transition.label is not None:
                if is_enabled(transition, self.net, self.init_marking):
                    return transition.label
        return None
    
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
