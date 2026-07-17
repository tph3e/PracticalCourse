from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pm4py
from pm4py.objects.petri_net.semantics import is_enabled, execute, enabled_transitions
from pm4py.objects.petri_net.obj import Marking, PetriNet


@dataclass(frozen=True)
class TransitionCandidate:
    transition_id: str
    activity_label: str
    source_marking: str
    pre_visible_marking: str
    silent_transition_path: tuple[str, ...]
    resulting_marking: str
    duplicate_label_count: int = 1

    @property
    def marking_before(self) -> str:
        return self.pre_visible_marking


class BPMNEngine:
    def __init__(self, model_filename: str | None = None):
        self.case_markings = {}
        self.diagnostics: dict[str, int] = {
            "duplicate_label_ambiguities": 0,
            "invalid_transition_fires": 0,
            "label_fire_ambiguities": 0,
        }
        self.model_filename = model_filename or self._default_model_filename()

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

    def _default_model_filename(self) -> str:
        return "models/v4_replay.bpmn"

    def normalize_case_id(self, case_id: Any) -> str:
        if hasattr(case_id, "caseId"):
            return str(case_id.caseId)
        return str(case_id)

    def initialize_case(self, case_id):
        self.case_markings[self.normalize_case_id(case_id)] = copy(self.init_marking)

    def getStartActivity(self, data=None):
        for candidate in self.get_enabled_transition_alternatives_from_marking(
            self.init_marking
        ):
            return candidate.activity_label
        return None

    def getStartTransitionCandidate(self) -> TransitionCandidate | None:
        candidates = self.get_enabled_transition_alternatives_from_marking(
            self.init_marking
        )
        return candidates[0] if candidates else None
    
    def getPossibleNextActivities(self, current_activity, case_id=None) -> list:
        return sorted(
            {
                candidate.activity_label
                for candidate in self.get_enabled_transition_alternatives(case_id)
            }
        )

    def get_enabled_transition_alternatives(
        self,
        case_id=None,
    ) -> list[TransitionCandidate]:
        if case_id is None:
            marking = self.init_marking
        else:
            normalized_case_id = self.normalize_case_id(case_id)
            if normalized_case_id not in self.case_markings:
                self.initialize_case(normalized_case_id)
            marking = self.case_markings[normalized_case_id]
        return self.get_enabled_transition_alternatives_from_marking(marking)

    def getPossibleNextTransitionCandidates(self, case_id=None) -> list[TransitionCandidate]:
        return self.get_enabled_transition_alternatives(case_id)

    def get_enabled_transition_alternatives_from_marking(
        self,
        marking: Marking,
    ) -> list[TransitionCandidate]:
        candidates: list[TransitionCandidate] = []
        source_signature = self.marking_signature(marking)
        queue: list[tuple[Marking, tuple[str, ...]]] = [(marking, ())]
        visited: set[str] = set()

        while queue:
            current_marking, silent_path = queue.pop(0)
            signature = self.marking_signature(current_marking)
            if signature in visited:
                continue
            visited.add(signature)

            for transition in self._sorted_enabled_transitions(current_marking):
                transition_id = self.transition_id(transition)
                if transition.label is not None:
                    resulting = execute(transition, self.net, current_marking)
                    candidates.append(
                        TransitionCandidate(
                            transition_id=transition_id,
                            activity_label=str(transition.label),
                            source_marking=source_signature,
                            pre_visible_marking=signature,
                            silent_transition_path=silent_path,
                            resulting_marking=self.marking_signature(resulting),
                        )
                    )
                    continue

                if transition_id in silent_path:
                    continue
                next_marking = execute(transition, self.net, current_marking)
                next_signature = self.marking_signature(next_marking)
                if next_signature not in visited:
                    queue.append((next_marking, (*silent_path, transition_id)))

        label_counts: dict[str, int] = {}
        for candidate in candidates:
            label_counts[candidate.activity_label] = (
                label_counts.get(candidate.activity_label, 0) + 1
            )
        return [
            TransitionCandidate(
                transition_id=candidate.transition_id,
                activity_label=candidate.activity_label,
                source_marking=candidate.source_marking,
                pre_visible_marking=candidate.pre_visible_marking,
                silent_transition_path=candidate.silent_transition_path,
                resulting_marking=candidate.resulting_marking,
                duplicate_label_count=label_counts[candidate.activity_label],
            )
            for candidate in sorted(
                candidates,
                key=lambda item: (
                    item.activity_label,
                    item.transition_id,
                    item.silent_transition_path,
                    item.resulting_marking,
                ),
            )
        ]
    
    def fire_activity(self, activity_name, case_id) -> bool:
        candidates = [
            candidate
            for candidate in self.get_enabled_transition_alternatives(case_id)
            if candidate.activity_label == activity_name
        ]
        if len(candidates) != 1:
            if len(candidates) > 1:
                self.diagnostics["label_fire_ambiguities"] += 1
            return False
        return self.fire_transition(candidates[0].transition_id, case_id)

    def fire_transition_candidate(self, case_id, candidate_or_transition_id) -> bool:
        transition_id = (
            candidate_or_transition_id.transition_id
            if hasattr(candidate_or_transition_id, "transition_id")
            else str(candidate_or_transition_id)
        )
        return self.fire_transition(transition_id, case_id)

    def fire_transition(self, transition_id: str, case_id) -> bool:
        normalized_case_id = self.normalize_case_id(case_id)
        if normalized_case_id not in self.case_markings:
            self.initialize_case(normalized_case_id)

        queue: list[tuple[Marking, tuple[str, ...]]] = [
            (self.case_markings[normalized_case_id], ())
        ]
        visited: set[str] = set()
        while queue:
            marking, silent_path = queue.pop(0)
            signature = self.marking_signature(marking)
            if signature in visited:
                continue
            visited.add(signature)
            for transition in self._sorted_enabled_transitions(marking):
                current_transition_id = self.transition_id(transition)
                if current_transition_id == transition_id and transition.label is not None:
                    self.case_markings[normalized_case_id] = execute(
                        transition,
                        self.net,
                        marking,
                    )
                    return True
                if transition.label is None and current_transition_id not in silent_path:
                    new_marking = execute(transition, self.net, marking)
                    new_signature = self.marking_signature(new_marking)
                    if new_signature not in visited:
                        queue.append((new_marking, (*silent_path, current_transition_id)))
        self.diagnostics["invalid_transition_fires"] += 1
        return False

    def current_marking_signature(self, case_id) -> str:
        normalized_case_id = self.normalize_case_id(case_id)
        if normalized_case_id not in self.case_markings:
            self.initialize_case(normalized_case_id)
        return self.marking_signature(self.case_markings[normalized_case_id])

    def get_marking_signature(self, case_id) -> str:
        return self.current_marking_signature(case_id)

    def is_final_marking(self, case_id) -> bool:
        normalized_case_id = self.normalize_case_id(case_id)
        marking = self.case_markings.get(normalized_case_id)
        return marking == self.final_marking

    def is_deadlocked(self, case_id) -> bool:
        return (
            not self.is_final_marking(case_id)
            and len(self.get_enabled_transition_alternatives(case_id)) == 0
        )

    def can_reach_final_by_silent_path(self, case_id) -> bool:
        normalized_case_id = self.normalize_case_id(case_id)
        marking = self.case_markings.get(normalized_case_id)
        if marking is None:
            return False
        if marking == self.final_marking:
            return True
        queue = [marking]
        visited: set[str] = set()
        while queue:
            current = queue.pop(0)
            signature = self.marking_signature(current)
            if signature in visited:
                continue
            visited.add(signature)
            for transition in self._sorted_enabled_transitions(current):
                if transition.label is not None:
                    continue
                next_marking = execute(transition, self.net, current)
                if next_marking == self.final_marking:
                    return True
                queue.append(next_marking)
        return False

    def transition_id(self, transition) -> str:
        return str(getattr(transition, "name", None) or id(transition))

    def marking_signature(self, marking: Marking | None) -> str:
        if marking is None:
            return "<none>"
        parts = []
        for place, count in marking.items():
            place_name = str(getattr(place, "name", place))
            parts.append(f"{place_name}:{count}")
        return "|".join(sorted(parts))

    def _sorted_enabled_transitions(self, marking: Marking):
        return sorted(
            enabled_transitions(self.net, marking),
            key=lambda transition: (
                "" if transition.label is None else str(transition.label),
                self.transition_id(transition),
            ),
        )
