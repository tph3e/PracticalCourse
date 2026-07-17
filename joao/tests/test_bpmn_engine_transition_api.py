from pm4py.objects.petri_net.obj import Marking, PetriNet
from pm4py.objects.petri_net.utils import petri_utils

from BPMN_engine import BPMNEngine
from Helper import Case


def make_engine():
    net = PetriNet("transition-api-test")
    p0 = PetriNet.Place("p0")
    p1 = PetriNet.Place("p1")
    p2 = PetriNet.Place("p2")
    p3 = PetriNet.Place("p3")
    pf = PetriNet.Place("pf")
    pdead = PetriNet.Place("pdead")
    net.places.update({p0, p1, p2, p3, pf, pdead})

    tau = PetriNet.Transition("tau_start", None)
    tau_loop = PetriNet.Transition("tau_loop", None)
    a_left = PetriNet.Transition("a_left", "A")
    a_right = PetriNet.Transition("a_right", "A")
    b = PetriNet.Transition("b", "B")
    c = PetriNet.Transition("c", "C")
    net.transitions.update({tau, tau_loop, a_left, a_right, b, c})

    petri_utils.add_arc_from_to(p0, tau, net)
    petri_utils.add_arc_from_to(tau, p1, net)
    petri_utils.add_arc_from_to(p1, tau_loop, net)
    petri_utils.add_arc_from_to(tau_loop, p1, net)
    petri_utils.add_arc_from_to(p1, a_left, net)
    petri_utils.add_arc_from_to(a_left, p2, net)
    petri_utils.add_arc_from_to(p1, a_right, net)
    petri_utils.add_arc_from_to(a_right, p3, net)
    petri_utils.add_arc_from_to(p2, b, net)
    petri_utils.add_arc_from_to(b, pf, net)
    petri_utils.add_arc_from_to(p3, c, net)
    petri_utils.add_arc_from_to(c, pf, net)

    engine = BPMNEngine.__new__(BPMNEngine)
    engine.case_markings = {}
    engine.model_filename = "in-memory"
    engine.net = net
    engine.init_marking = Marking({p0: 1})
    engine.final_marking = Marking({pf: 1})
    engine.diagnostics = {
        "duplicate_label_ambiguities": 0,
        "invalid_transition_fires": 0,
        "label_fire_ambiguities": 0,
    }
    engine.dead_marking = Marking({pdead: 1})
    return engine


def test_stable_case_id_normalization_uses_case_id_not_case_object_string():
    engine = make_engine()
    case = Case("C1")

    engine.initialize_case(case)

    assert set(engine.case_markings) == {"C1"}
    assert engine.current_marking_signature("C1") == engine.current_marking_signature(case)


def test_transition_candidates_include_identity_silent_path_and_duplicate_label_count():
    engine = make_engine()
    engine.initialize_case("C1")
    before = engine.current_marking_signature("C1")

    candidates = engine.get_enabled_transition_alternatives("C1")

    assert engine.current_marking_signature("C1") == before
    assert {candidate.transition_id for candidate in candidates} == {"a_left", "a_right"}
    assert {candidate.activity_label for candidate in candidates} == {"A"}
    assert all(candidate.silent_transition_path == ("tau_start",) for candidate in candidates)
    assert all(candidate.duplicate_label_count == 2 for candidate in candidates)
    assert all(candidate.source_marking == before for candidate in candidates)


def test_label_fire_reports_duplicate_ambiguity_but_transition_identity_fires():
    engine = make_engine()
    engine.initialize_case("C1")

    assert engine.fire_activity("A", "C1") is False
    assert engine.diagnostics["label_fire_ambiguities"] == 1

    assert engine.fire_transition("a_left", "C1") is True
    assert engine.getPossibleNextActivities("A", "C1") == ["B"]


def test_final_marking_and_deadlock_status_are_exposed():
    engine = make_engine()
    engine.initialize_case("C1")

    assert engine.fire_transition("a_left", "C1") is True
    assert engine.fire_transition("b", "C1") is True
    assert engine.is_final_marking("C1") is True
    assert engine.is_deadlocked("C1") is False

    engine.case_markings["D1"] = engine.dead_marking
    assert engine.is_deadlocked("D1") is True
