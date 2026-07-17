from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import pickle


ARTIFACT_FORMAT = "joao_transition_aware_branching_v1"


@dataclass
class TransitionDisambiguationModel:
    marking_counts: dict[str, Counter[str]] = field(default_factory=dict)
    marking_label_counts: dict[tuple[str, str], Counter[str]] = field(default_factory=dict)
    state_counts: dict[tuple[str, str, int, int], Counter[str]] = field(default_factory=dict)
    activity_counts: dict[str, Counter[str]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def observe(
        self,
        transition_id: str,
        activity_label: str,
        marking_signature: str,
        previous_activity: str,
        current_activity: str,
        visit_count_bucket: int,
        repetition_bucket: int,
    ) -> None:
        self.marking_counts.setdefault(marking_signature, Counter())[transition_id] += 1
        self.marking_label_counts.setdefault(
            (marking_signature, activity_label),
            Counter(),
        )[transition_id] += 1
        self.state_counts.setdefault(
            (
                current_activity,
                previous_activity,
                visit_count_bucket,
                repetition_bucket,
            ),
            Counter(),
        )[transition_id] += 1
        self.activity_counts.setdefault(current_activity, Counter())[transition_id] += 1

    def choose_for_label(
        self,
        candidates: list[Any],
        activity_label: str,
        context: dict[str, Any],
    ) -> tuple[Any | None, str]:
        matches = [
            candidate
            for candidate in candidates
            if str(candidate.activity_label) == str(activity_label)
        ]
        if not matches:
            return None, "no_label_match"
        if len(matches) == 1:
            return matches[0], "unique_label_match"
        return self.choose(matches, context, required_label=activity_label)

    def choose(
        self,
        candidates: list[Any],
        context: dict[str, Any],
        required_label: str | None = None,
    ) -> tuple[Any | None, str]:
        if not candidates:
            return None, "no_candidates"
        valid_ids = {str(candidate.transition_id) for candidate in candidates}
        marking = str(context.get("marking_signature") or "")
        current_activity = str(context.get("current_activity") or "")
        previous_activity = str(context.get("previous_activity") or "START")
        visit_bucket = int(context.get("visit_count_bucket") or 0)
        repetition_bucket = int(context.get("repetition_bucket") or 0)

        sources: list[tuple[str, Counter[str] | None]] = []
        if required_label is not None:
            sources.append(
                (
                    "marking_label_probability",
                    self.marking_label_counts.get((marking, str(required_label))),
                )
            )
        sources.extend(
            [
                ("marking_probability", self.marking_counts.get(marking)),
                (
                    "state_probability",
                    self.state_counts.get(
                        (
                            current_activity,
                            previous_activity,
                            visit_bucket,
                            repetition_bucket,
                        )
                    ),
                ),
                ("activity_probability", self.activity_counts.get(current_activity)),
            ]
        )

        for source, counter in sources:
            selected = self._best_counter_candidate(candidates, valid_ids, counter)
            if selected is not None:
                return selected, source
        return self._deterministic_candidate(candidates), "deterministic_candidate_order"

    def _best_counter_candidate(
        self,
        candidates: list[Any],
        valid_ids: set[str],
        counter: Counter[str] | None,
    ) -> Any | None:
        if not counter:
            return None
        supported = {
            transition_id: count
            for transition_id, count in counter.items()
            if transition_id in valid_ids and count > 0
        }
        if not supported:
            return None
        best_id = sorted(supported, key=lambda tid: (-supported[tid], tid))[0]
        for candidate in candidates:
            if str(candidate.transition_id) == best_id:
                return candidate
        return None

    def _deterministic_candidate(self, candidates: list[Any]) -> Any:
        return sorted(
            candidates,
            key=lambda candidate: (
                str(getattr(candidate, "activity_label", "")),
                str(getattr(candidate, "pre_visible_marking", "")),
                str(getattr(candidate, "transition_id", "")),
            ),
        )[0]

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": ARTIFACT_FORMAT,
            "model": self,
            "metadata": self.metadata,
        }


def save_transition_model(model: TransitionDisambiguationModel, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        pickle.dump(model.to_payload(), handle)


def load_transition_model(path: str | Path) -> TransitionDisambiguationModel:
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, TransitionDisambiguationModel):
        return payload
    if not isinstance(payload, dict) or payload.get("format") != ARTIFACT_FORMAT:
        raise ValueError(f"Unsupported transition artifact format: {type(payload)!r}")
    model = payload.get("model")
    if not isinstance(model, TransitionDisambiguationModel):
        raise ValueError("Transition artifact does not contain a model.")
    model.metadata.update(payload.get("metadata") or {})
    return model
