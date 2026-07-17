from __future__ import annotations

from collections import Counter
from copy import copy
from typing import Any

import pandas as pd

from BPMN_engine import BPMNEngine, TransitionCandidate
from joao.src.resource_allocation.integration.TransitionAwareBranching import (
    TransitionDisambiguationModel,
)


CASE_COL = "case:concept:name"
ACTIVITY_COL = "concept:name"
TIME_COL = "time:timestamp"


def evaluate_transition_continuation(
    log: pd.DataFrame,
    model: TransitionDisambiguationModel,
    *,
    bpmn_model: str = "models/v4_replay.bpmn",
    case_ids: list[str] | None = None,
    case_limit: int = 500,
) -> dict[str, Any]:
    engine = BPMNEngine(model_filename=bpmn_model)
    work_log = log.copy()
    work_log[TIME_COL] = pd.to_datetime(work_log[TIME_COL], utc=True)
    work_log[CASE_COL] = work_log[CASE_COL].astype(str)
    if case_ids is not None:
        allowed = set(str(case_id) for case_id in case_ids)
        work_log = work_log[work_log[CASE_COL].isin(allowed)]
    selected_cases = (
        work_log.groupby(CASE_COL)[TIME_COL]
        .min()
        .sort_values()
        .index.astype(str)
        .tolist()
    )
    if case_limit > 0:
        selected_cases = selected_cases[:case_limit]

    metrics = Counter()
    source_counts = Counter()
    random_success = Counter()
    first_success = Counter()
    model_success = Counter()

    for case_id in selected_cases:
        case_events = work_log[work_log[CASE_COL] == case_id].sort_values(TIME_COL)
        activities = case_events[ACTIVITY_COL].astype(str).tolist()
        engine.initialize_case(case_id)
        history: list[str] = []
        for index, activity in enumerate(activities):
            candidates = engine.getPossibleNextTransitionCandidates(case_id)
            if not candidates:
                metrics["deadlocked_before_event"] += 1
                break
            matching = [candidate for candidate in candidates if candidate.activity_label == activity]
            if len(matching) > 1:
                metrics["ambiguous_decisions"] += 1
                context = _context(engine, case_id, activity, history)
                selected, source = model.choose_for_label(matching, activity, context)
                source_counts[source] += 1
                if selected is None:
                    metrics["fallback_count"] += 1
                    selected = _first_candidate(matching)
                metrics["model_invocations"] += 1
                _record_continuation(
                    engine=engine,
                    case_id=case_id,
                    activities=activities,
                    index=index,
                    candidate=selected,
                    counter=model_success,
                )
                _record_continuation(
                    engine=engine,
                    case_id=case_id,
                    activities=activities,
                    index=index,
                    candidate=_first_candidate(matching),
                    counter=first_success,
                )
                _record_continuation(
                    engine=engine,
                    case_id=case_id,
                    activities=activities,
                    index=index,
                    candidate=_deterministic_random_proxy(matching),
                    counter=random_success,
                )
                if not engine.fire_transition_candidate(case_id, selected):
                    metrics["invalid_model_selection"] += 1
                    break
            elif len(matching) == 1:
                if not engine.fire_transition_candidate(case_id, matching[0]):
                    metrics["fire_failures"] += 1
                    break
            else:
                metrics["nonconformant_events"] += 1
                break
            history.append(activity)

    ambiguous = metrics["ambiguous_decisions"]
    return {
        "case_count": len(selected_cases),
        "ambiguous_decisions": int(ambiguous),
        "model_invocations": int(metrics["model_invocations"]),
        "fallback_rate": metrics["fallback_count"] / ambiguous if ambiguous else 0.0,
        "invalid_model_selection": int(metrics["invalid_model_selection"]),
        "source_counts": dict(source_counts),
        "model": _rates(model_success, ambiguous),
        "first_candidate": _rates(first_success, ambiguous),
        "deterministic_random_proxy": _rates(random_success, ambiguous),
        "status": "evaluated" if ambiguous else "not_identifiable_on_this_log_bpmn",
    }


def _record_continuation(
    *,
    engine: BPMNEngine,
    case_id: str,
    activities: list[str],
    index: int,
    candidate: TransitionCandidate,
    counter: Counter,
) -> None:
    original = copy(engine.case_markings[engine.normalize_case_id(case_id)])
    try:
        if not engine.fire_transition_candidate(case_id, candidate):
            return
        for horizon in [1, 3]:
            if _can_replay_next(engine, case_id, activities, index + 1, horizon):
                counter[f"continuation_{horizon}_step"] += 1
        if _can_replay_next(engine, case_id, activities, index + 1, len(activities)):
            counter["continuation_full"] += 1
    finally:
        engine.case_markings[engine.normalize_case_id(case_id)] = original


def _can_replay_next(
    engine: BPMNEngine,
    case_id: str,
    activities: list[str],
    start: int,
    horizon: int,
) -> bool:
    original = copy(engine.case_markings[engine.normalize_case_id(case_id)])
    try:
        end = min(len(activities), start + horizon)
        for activity in activities[start:end]:
            candidates = [
                candidate
                for candidate in engine.getPossibleNextTransitionCandidates(case_id)
                if candidate.activity_label == activity
            ]
            if not candidates:
                return False
            if not engine.fire_transition_candidate(case_id, _first_candidate(candidates)):
                return False
        return True
    finally:
        engine.case_markings[engine.normalize_case_id(case_id)] = original


def _context(
    engine: BPMNEngine,
    case_id: str,
    current_activity: str,
    history: list[str],
) -> dict[str, Any]:
    visit_count = sum(1 for activity in history if activity == current_activity)
    consecutive = 0
    for activity in reversed(history):
        if activity != current_activity:
            break
        consecutive += 1
    return {
        "marking_signature": engine.current_marking_signature(case_id),
        "current_activity": current_activity,
        "previous_activity": history[-1] if history else "START",
        "visit_count_bucket": _bucket(visit_count),
        "repetition_bucket": _bucket(consecutive),
    }


def _bucket(value: int) -> int:
    if value <= 0:
        return 0
    if value == 1:
        return 1
    if value <= 3:
        return 3
    if value <= 10:
        return 10
    return 99


def _first_candidate(candidates: list[TransitionCandidate]) -> TransitionCandidate:
    return sorted(candidates, key=lambda item: (item.transition_id, item.resulting_marking))[0]


def _deterministic_random_proxy(candidates: list[TransitionCandidate]) -> TransitionCandidate:
    return sorted(candidates, key=lambda item: (item.resulting_marking, item.transition_id))[-1]


def _rates(counter: Counter, denominator: int) -> dict[str, float]:
    return {
        "continuation_1_step": counter["continuation_1_step"] / denominator if denominator else 0.0,
        "continuation_3_steps": counter["continuation_3_step"] / denominator if denominator else 0.0,
        "continuation_full": counter["continuation_full"] / denominator if denominator else 0.0,
    }
