from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from joao.src.branching.CompositeBranchingArtifact import load_artifact_payload


ROOT = Path(__file__).resolve().parents[2]

EXPECTED_HASHES = {
    "joao/models/branching/composite_branching_evaluation_train70.pkl": "ebb16554ecf0e42faa87c7334faef3ab81b742b8e29820df1ce032bd4109f1c4",
    "joao/models/branching/composite_branching_deployment_full.pkl": "6dcd01744f635d0fdd24008c3e7c4ae28bedb340c4f37aabbc1bc53fd7e7ab3e",
    "joao/models/branching/composite_branching_evaluation_train70_rfopt_v1.pkl": "490da841440cd019a0819892ab659e5b61f5ee55182f6db12a536a9c4d23ff82",
    "joao/models/branching/composite_branching_deployment_full_rfopt_v1.pkl": "14b61cd4e13762e584fc35dfbd657142d1bfb4868c0cff3daf2b9b83d75e47e3",
    "joao/models/branching/transition_aware_branching_v1_20260715.pkl": "79127e6b1cea6cb58fc1f1f19b1ce96564837a24aa5adf80efcbcc71fc183e54",
    "joao/models/process_time/final_process_time_coverage_v2.pkl": "c540304cdbb6f60159ad1023112e6d5e71aeecae68b885ee2e2b1ac3c826a886",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_corrected_canonical_artifact_hashes_and_scopes() -> None:
    for relative_path, expected_hash in EXPECTED_HASHES.items():
        path = ROOT / relative_path
        assert path.exists(), relative_path
        assert _sha256(path) == expected_hash, relative_path

    evaluation_payload = load_artifact_payload(
        ROOT / "joao/models/branching/composite_branching_evaluation_train70.pkl",
        expected_scope="evaluation",
        expected_sha256=EXPECTED_HASHES["joao/models/branching/composite_branching_evaluation_train70.pkl"],
    )
    deployment_payload = load_artifact_payload(
        ROOT / "joao/models/branching/composite_branching_deployment_full.pkl",
        expected_scope="deployment",
        expected_sha256=EXPECTED_HASHES["joao/models/branching/composite_branching_deployment_full.pkl"],
    )

    assert evaluation_payload["metadata"]["artifact_scope"] == "evaluation"
    assert deployment_payload["metadata"]["artifact_scope"] == "deployment"
    assert evaluation_payload["metadata"]["case_overlap"] == 0

    with pytest.raises(ValueError):
        load_artifact_payload(
            ROOT / "joao/models/branching/composite_branching_deployment_full.pkl",
            expected_scope="evaluation",
        )


def test_corrected_fixed_replay_uses_evaluation_artifact_without_case_overlap() -> None:
    corrected_dir = ROOT / "joao/results/final_canonical_branching_corrected_20260717/fixed_replay"
    config = json.loads((corrected_dir / "fixed_replay_config.json").read_text())
    artifact_hashes = json.loads((corrected_dir / "final_artifact_hashes.json").read_text())
    split_manifest = json.loads(
        (ROOT / "joao/results/branching_corrected_20260717/branching_split_manifest.json").read_text()
    )

    assert config["branching_artifact"].endswith(
        "joao/models/branching/composite_branching_evaluation_train70.pkl"
    )
    assert artifact_hashes["branching_sha256"] == EXPECTED_HASHES[
        "joao/models/branching/composite_branching_evaluation_train70.pkl"
    ]
    assert artifact_hashes["processing_time_sha256"] == EXPECTED_HASHES[
        "joao/models/process_time/final_process_time_coverage_v2.pkl"
    ]
    assert split_manifest["case_counts"]["fixed_replay"] == 76
    assert split_manifest["fixed_replay_cases_in_outer_test"] is True
    assert split_manifest["case_overlap"]["training_fixed_replay"] == 0
    assert split_manifest["case_overlap"]["outer_train_outer_test"] == 0


def test_rfopt_candidate_artifacts_and_final_core_scope() -> None:
    evaluation_payload = load_artifact_payload(
        ROOT / "joao/models/branching/composite_branching_evaluation_train70_rfopt_v1.pkl",
        expected_scope="evaluation",
        expected_sha256=EXPECTED_HASHES[
            "joao/models/branching/composite_branching_evaluation_train70_rfopt_v1.pkl"
        ],
    )
    deployment_payload = load_artifact_payload(
        ROOT / "joao/models/branching/composite_branching_deployment_full_rfopt_v1.pkl",
        expected_scope="deployment",
        expected_sha256=EXPECTED_HASHES[
            "joao/models/branching/composite_branching_deployment_full_rfopt_v1.pkl"
        ],
    )

    expected_hierarchy = ["PredictiveBranchingEngine", "ProbabilityBranchingEngine"]
    assert evaluation_payload["metadata"]["artifact_scope"] == "evaluation"
    assert deployment_payload["metadata"]["artifact_scope"] == "deployment"
    assert evaluation_payload["metadata"]["case_overlap"] == 0
    assert evaluation_payload["metadata"]["composite_hierarchy"] == expected_hierarchy

    with pytest.raises(ValueError):
        load_artifact_payload(
            ROOT / "joao/models/branching/composite_branching_deployment_full_rfopt_v1.pkl",
            expected_scope="evaluation",
        )

    import pandas as pd

    core = pd.read_csv(ROOT / "joao/results/rf_training_optimization_20260717/branching_final_core_metrics.csv")
    assert set(core["method"]) == {
        "MajorityBaseline",
        "RandomCandidateBaseline",
        "ProbabilityBranching",
        "PredictiveML-Raw",
        "PredictiveML-BPMNConstrained",
        "CompositeRuntime",
    }


def test_rfopt_fixed_replay_candidate_uses_candidate_artifact() -> None:
    candidate_dir = ROOT / "joao/results/final_canonical_rfopt_candidate_20260717/fixed_replay"
    config = json.loads((candidate_dir / "fixed_replay_config.json").read_text())
    artifact_hashes = json.loads((candidate_dir / "final_artifact_hashes.json").read_text())
    failures = (candidate_dir / "failures.csv").read_text().strip().splitlines()

    assert config["branching_artifact"].endswith(
        "joao/models/branching/composite_branching_evaluation_train70_rfopt_v1.pkl"
    )
    assert artifact_hashes["branching_sha256"] == EXPECTED_HASHES[
        "joao/models/branching/composite_branching_evaluation_train70_rfopt_v1.pkl"
    ]
    assert len(failures) == 1


def test_legacy_branching_engines_remain_importable() -> None:
    from joao.src.branching.AttributeBasedBranchingEngine import AttributeBasedBranchingEngine
    from joao.src.branching.AttributeSamplingBranchingEngine import AttributeSamplingBranchingEngine

    assert AttributeBasedBranchingEngine is not None
    assert AttributeSamplingBranchingEngine is not None


def test_historical_package_is_marked_pre_correction() -> None:
    historical_readme = (ROOT / "joao/results/final_canonical_20260716/README.md").read_text()
    assert "Historical Pre-Correction Results" in historical_readme
    assert "not the current" in historical_readme
    assert "must not be used to reconstruct the final" in historical_readme
    assert "final_composite_branching.pkl" in historical_readme
    assert "full-log/deployment-style artifact" in historical_readme
    assert "branching_corrected_20260717" in historical_readme
    assert "final_canonical_branching_corrected_20260717" in historical_readme


def test_current_readmes_do_not_call_historical_artifact_evaluation_artifact() -> None:
    current_paths = [
        ROOT / "joao/README.md",
        ROOT / "joao/results/branching_corrected_20260717/README.md",
        ROOT / "joao/results/final_canonical_branching_corrected_20260717/README.md",
        ROOT / "report/joao_branching_support/README.md",
        ROOT / "report/joao_evaluation_protocol_support/README.md",
    ]
    for path in current_paths:
        text = path.read_text()
        if "final_composite_branching.pkl" in text:
            lowered = text.lower()
            assert "historical" in lowered or "pre-correction" in lowered
            assert "evaluation artifact: `joao/models/branching/final_composite_branching.pkl`" not in lowered
