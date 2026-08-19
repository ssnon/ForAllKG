from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import pytest

from campaigns.sers_alpha4_epoch.alpha4.alpha4c5f1_sers_readiness import (
    REFREEZE_ELIGIBLE_REASONS,
)
from campaigns.sers_alpha4_epoch.readiness.canonical_readiness import (
    CANONICAL_READINESS_SEMANTICS_ID,
    CanonicalReadinessError,
    atomic_json,
    canonical_graph_snapshot,
    guarded_write_consumption_marker,
    make_readiness_lock,
    verify_readiness_lock,
)
from dac_her.measurement_merge_invariants import (
    MEASUREMENT_MERGE_INVARIANT_ID,
)


DOMAIN = "sers_au_ag"


def _write_graph(path: Path, *, numeric: str, text: str) -> None:
    graph = nx.MultiDiGraph()
    graph.graph["domain_profile_id"] = DOMAIN
    graph.graph["measurement_merge_invariant_id"] = (
        MEASUREMENT_MERGE_INVARIANT_ID
    )
    graph.add_node(
        "m1",
        type="Measurement",
        metric_id="lod",
        subject_id="sample",
        value_numeric=numeric,
        value_text=text,
        unit="M",
        source_expression="fixture",
    )
    nx.write_graphml(graph, path)


def _record(path: Path) -> dict:
    snap = canonical_graph_snapshot(
        path,
        expected_domain_profile_id=DOMAIN,
        include_issue_details=True,
    )
    return {
        "canonical": snap,
        "resolution_decisions": {
            "path": str(path.parent / "decisions.jsonl"),
            "present": False,
            "sha256": "",
        },
    }


def test_xor_gate_rejects_numeric_and_text_simultaneously(tmp_path: Path):
    path = tmp_path / "paper.graphml"
    _write_graph(path, numeric="1e-9", text="< 1 nM")
    snap = canonical_graph_snapshot(
        path,
        expected_domain_profile_id=DOMAIN,
        include_issue_details=True,
    )
    assert snap["ready"] is False
    assert snap["measurement_xor_issue_count"] == 1
    assert snap["measurement_xor_issues"][0]["id"] == "m1"


def test_xor_gate_rejects_measurement_with_neither_value(tmp_path: Path):
    path = tmp_path / "paper.graphml"
    _write_graph(path, numeric="", text="")
    snap = canonical_graph_snapshot(
        path,
        expected_domain_profile_id=DOMAIN,
    )
    assert snap["ready"] is False
    assert snap["measurement_xor_issue_count"] == 1


def test_ready_numeric_measurement_builds_non_disclosing_lock(tmp_path: Path):
    path = tmp_path / "paper.graphml"
    _write_graph(path, numeric="1e-9", text="")
    record = _record(path)
    lock = make_readiness_lock(
        root=tmp_path,
        paper_ids=["P1"],
        expected_domain_profile_id=DOMAIN,
        paper_records={"P1": record},
        source_label="unit-test",
    )
    assert lock["semantics_id"] == CANONICAL_READINESS_SEMANTICS_ID
    assert lock["all_ready"] is True
    assert lock["scientific_values_disclosed"] is False
    assert "measurement_xor_issues" not in lock["paper_records"]["P1"]["canonical"]
    assert verify_readiness_lock(
        root=tmp_path,
        lock=lock,
        expected_paper_ids=["P1"],
        expected_domain_profile_id=DOMAIN,
    ) == []


def test_lock_detects_canonical_sha_drift(tmp_path: Path):
    path = tmp_path / "paper.graphml"
    _write_graph(path, numeric="1e-9", text="")
    lock = make_readiness_lock(
        root=tmp_path,
        paper_ids=["P1"],
        expected_domain_profile_id=DOMAIN,
        paper_records={"P1": _record(path)},
        source_label="unit-test",
    )
    _write_graph(path, numeric="2e-9", text="")
    issues = verify_readiness_lock(
        root=tmp_path,
        lock=lock,
        expected_paper_ids=["P1"],
        expected_domain_profile_id=DOMAIN,
    )
    assert any("SHA256 drifted" in issue for issue in issues)


def test_consumption_marker_is_written_only_after_current_lock_verifies(
    tmp_path: Path,
):
    path = tmp_path / "paper.graphml"
    _write_graph(path, numeric="1e-9", text="")
    lock = make_readiness_lock(
        root=tmp_path,
        paper_ids=["P1"],
        expected_domain_profile_id=DOMAIN,
        paper_records={"P1": _record(path)},
        source_label="unit-test",
    )
    lock_path = tmp_path / "readiness.json"
    marker_path = tmp_path / "consumption.json"
    atomic_json(lock_path, lock)

    payload = guarded_write_consumption_marker(
        root=tmp_path,
        lock_path=lock_path,
        marker_path=marker_path,
        expected_paper_ids=["P1"],
        expected_domain_profile_id=DOMAIN,
        marker_payload={"reserve_consumed": True},
    )
    assert marker_path.exists()
    assert payload[
        "canonical_readiness_verified_immediately_before_consumption"
    ] is True

    with pytest.raises(CanonicalReadinessError):
        guarded_write_consumption_marker(
            root=tmp_path,
            lock_path=lock_path,
            marker_path=marker_path,
            expected_paper_ids=["P1"],
            expected_domain_profile_id=DOMAIN,
            marker_payload={"reserve_consumed": True},
        )


def test_consumption_guard_fails_if_graph_drifted_after_lock(tmp_path: Path):
    path = tmp_path / "paper.graphml"
    _write_graph(path, numeric="1e-9", text="")
    lock = make_readiness_lock(
        root=tmp_path,
        paper_ids=["P1"],
        expected_domain_profile_id=DOMAIN,
        paper_records={"P1": _record(path)},
        source_label="unit-test",
    )
    lock_path = tmp_path / "readiness.json"
    atomic_json(lock_path, lock)
    _write_graph(path, numeric="1e-9", text="< 1 nM")

    with pytest.raises(CanonicalReadinessError):
        guarded_write_consumption_marker(
            root=tmp_path,
            lock_path=lock_path,
            marker_path=tmp_path / "consumption.json",
            expected_paper_ids=["P1"],
            expected_domain_profile_id=DOMAIN,
            marker_payload={"reserve_consumed": True},
        )


def test_refreeze_policy_is_structural_only():
    assert REFREEZE_ELIGIBLE_REASONS == {
        "canonical_missing",
        "measurement_merge_invariant_mismatch",
        "measurement_numeric_text_xor_violation",
    }
    assert "domain_profile_mismatch" not in REFREEZE_ELIGIBLE_REASONS
