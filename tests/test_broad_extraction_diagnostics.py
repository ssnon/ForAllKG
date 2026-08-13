from __future__ import annotations

import json
from pathlib import Path

from dac_her.broad_extraction_diagnostics import (
    aggregate_broad_extraction_diagnostics,
    inspect_broad_paper,
    write_broad_extraction_diagnostics,
)


def _write_run(
    data_root: Path,
    paper_id: str,
    *,
    status: str,
    records: list[dict],
    projection_edges: int | None = None,
) -> Path:
    paper_root = data_root / "extracted" / paper_id
    run_dir = paper_root / "runs" / "r1"
    run_dir.mkdir(parents=True, exist_ok=True)
    active = [row for row in records if row.get("status") in {"success", "skipped"}]
    quarantined = [row for row in records if row.get("status") == "quarantined"]
    failed = [row for row in records if row.get("status") == "failed"]
    payload = {
        "paper_id": paper_id,
        "run_id": "r1",
        "run_fingerprint": "fp-r1",
        "complete": status == "complete",
        "paper_status": "complete" if status == "complete" else "quarantined",
        "graph_materialization_status": status,
        "active_chunk_count": len(active),
        "chunks": active,
        "quarantined_chunks": quarantined,
        "failed_chunks": failed,
        "quality": {
            "graph_materialization_status": status,
            "strict_complete": status == "complete",
            "source_token_coverage": 1.0 if status == "complete" else 0.0,
            "active_chunk_count": len(active),
            "quarantine_tier_counts": (
                {"Q2_RECOVERY_OR_COMPLEX": len(quarantined)}
                if quarantined else {}
            ),
        },
    }
    (run_dir / "active_chunks.json").write_text(json.dumps(payload), encoding="utf-8")
    (run_dir / "summary.json").write_text(
        json.dumps({
            "run_id": "r1",
            "run_fingerprint": "fp-r1",
            "graph_materialization_status": status,
        }),
        encoding="utf-8",
    )
    paper_root.mkdir(parents=True, exist_ok=True)
    (paper_root / "latest_run.json").write_text(
        json.dumps({
            "paper_id": paper_id,
            "run_id": "r1",
            "run_fingerprint": "fp-r1",
            "run_directory": str(run_dir),
        }),
        encoding="utf-8",
    )
    validation = run_dir / "validation"
    validation.mkdir(parents=True, exist_ok=True)
    (validation / "chunk__raw.json").write_text(
        json.dumps({
            "issues": [
                {
                    "code": "ISOLATED_NODE",
                    "node_id": "descriptor_1",
                    "node_collection": "entities",
                    "actual": {"type": "Descriptor"},
                },
                {
                    "code": "ISOLATED_NODE",
                    "node_id": "descriptor_2",
                    "node_collection": "entities",
                    "actual": {"type": "Descriptor"},
                },
                {"code": "MECHANISM_MISSING_SUPPORT"},
                {
                    "code": "RELATION_SOURCE_TYPE_MISMATCH",
                    "relation": "HAS_ACTIVE_SITE",
                    "actual": {"type": "ActiveSite"},
                    "expected": {"types": ["Catalyst", "CatalystModel"]},
                },
            ]
        }),
        encoding="utf-8",
    )
    if projection_edges is not None:
        projection_dir = paper_root / "graphagents" / "mechanism"
        projection_dir.mkdir(parents=True, exist_ok=True)
        (projection_dir / "summary.json").write_text(
            json.dumps({
                "paper_id": paper_id,
                "nodes": 8,
                "edges": 6,
                "direct_mechanism_edges": projection_edges,
                "source_extraction_run_id": "r1",
                "source_extraction_run_fingerprint": "fp-r1",
            }),
            encoding="utf-8",
        )
    return run_dir


def test_inspect_broad_paper_aggregates_attempts_and_issues(tmp_path: Path):
    data_root = tmp_path / "data_broad"
    _write_run(
        data_root,
        "broad_A",
        status="rejected",
        records=[{
            "status": "quarantined",
            "validation_issue_counts": {
                "ISOLATED_NODE": 2,
                "MECHANISM_MISSING_SUPPORT": 1,
            },
            "recovery_reason": "micro_reextract_exhausted",
            "quarantine_reason_class": "recovery_exhausted",
            "error_type": "UnsplittableRecovery",
            "attempt_usages": [
                {
                    "call_kind": "graph_generation",
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "total_tokens": 120,
                },
                {
                    "call_kind": "semantic_patch",
                    "input_tokens": 50,
                    "output_tokens": 10,
                    "total_tokens": 60,
                },
            ],
        }],
    )

    row = inspect_broad_paper(data_root=data_root, paper_id="broad_A")
    assert row["graph_materialization_status"] == "rejected"
    assert row["graph_usable"] is False
    assert row["llm_calls"] == 2
    assert row["total_tokens"] == 180
    assert row["call_kind_counts"] == {
        "graph_generation": 1,
        "semantic_patch": 1,
    }
    # Derived from attempt_usages even when the terminal record does not
    # persist a generation_attempts field.
    assert row["generation_attempts"] == 1
    assert row["relation_mismatch_patterns"] == [{
        "code": "RELATION_SOURCE_TYPE_MISMATCH",
        "relation": "HAS_ACTIVE_SITE",
        "side": "source",
        "actual_type": "ActiveSite",
        "expected_types": ["Catalyst", "CatalystModel"],
        "count": 1,
    }]
    assert row["terminal_validation_issue_counts"]["ISOLATED_NODE"] == 2
    assert row["observed_validation_issue_counts"]["ISOLATED_NODE"] == 2
    assert row["isolated_node_patterns"] == [{
        "node_collection": "entities",
        "actual_type": "Descriptor",
        "count": 2,
        "example_node_ids": ["descriptor_1", "descriptor_2"],
    }]
    assert row["recovery_reason_counts"][
        "quarantine_reason_class:recovery_exhausted"
    ] == 1


def test_aggregate_diagnostics_reports_efficiency_and_mechanism_yield(tmp_path: Path):
    data_root = tmp_path / "data_broad"
    _write_run(
        data_root,
        "broad_A",
        status="complete",
        records=[{
            "status": "success",
            "attempt_usages": [{
                "call_kind": "graph_generation",
                "input_tokens": 100,
                "output_tokens": 25,
                "total_tokens": 125,
            }],
        }],
        projection_edges=2,
    )
    _write_run(
        data_root,
        "broad_B",
        status="rejected",
        records=[{
            "status": "quarantined",
            "validation_issue_counts": {"ISOLATED_NODE": 1},
            "attempt_usages": [{
                "call_kind": "graph_generation",
                "input_tokens": 80,
                "output_tokens": 20,
                "total_tokens": 100,
            }],
        }],
    )
    rows = [
        inspect_broad_paper(data_root=data_root, paper_id="broad_A"),
        inspect_broad_paper(data_root=data_root, paper_id="broad_B"),
    ]
    report = aggregate_broad_extraction_diagnostics(rows)
    assert report["requested_paper_count"] == 2
    assert report["graph_usable_paper_count"] == 1
    assert report["graph_usable_paper_fraction"] == 0.5
    assert report["mechanism_bearing_paper_count"] == 1
    assert report["direct_mechanism_edges"] == 2
    assert report["llm_calls"] == 2
    assert report["total_tokens"] == 225
    assert report["generation_attempts"] == 2
    assert report["wasted_llm_calls"] == 1
    assert report["wasted_call_fraction"] == 0.5
    assert report["wasted_tokens"] == 100
    assert report["wasted_token_fraction"] == 100 / 225
    assert report["llm_calls_per_usable_paper"] == 2.0
    assert report["llm_calls_per_direct_mechanism_edge"] == 1.0
    assert report["terminal_validation_issue_counts"] == {"ISOLATED_NODE": 1}
    assert report["relation_mismatch_patterns"][0] == {
        "code": "RELATION_SOURCE_TYPE_MISMATCH",
        "relation": "HAS_ACTIVE_SITE",
        "side": "source",
        "actual_type": "ActiveSite",
        "expected_types": ["Catalyst", "CatalystModel"],
        "count": 2,
    }


def test_stale_projection_is_not_counted_as_current_mechanism_yield(tmp_path: Path):
    data_root = tmp_path / "data_broad"
    _write_run(
        data_root,
        "broad_A",
        status="complete",
        records=[{"status": "success", "attempt_usages": []}],
        projection_edges=3,
    )
    summary_path = (
        data_root
        / "extracted"
        / "broad_A"
        / "graphagents"
        / "mechanism"
        / "summary.json"
    )
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["source_extraction_run_id"] = "older-run"
    summary_path.write_text(json.dumps(payload), encoding="utf-8")

    row = inspect_broad_paper(data_root=data_root, paper_id="broad_A")
    assert row["projection_found"] is True
    assert row["projection_current"] is False
    assert row["stale_projection_found"] is True
    assert row["direct_mechanism_edges"] == 0

    report = aggregate_broad_extraction_diagnostics([row])
    assert report["projection_paper_count"] == 0
    assert report["stale_projection_count"] == 1
    assert report["direct_mechanism_edges"] == 0


def test_attempt_scoped_projection_requires_matching_attempt_identity(
    tmp_path: Path,
):
    data_root = tmp_path / "data_broad"
    family_dir = _write_run(
        data_root,
        "broad_A",
        status="complete",
        records=[{"status": "success", "attempt_usages": []}],
        projection_edges=3,
    )
    paper_root = data_root / "extracted" / "broad_A"
    attempt_id = "attempt-new"
    attempt_dir = family_dir / "attempts" / attempt_id
    attempt_dir.mkdir(parents=True, exist_ok=False)

    for filename in ("active_chunks.json", "summary.json"):
        payload = json.loads(
            (family_dir / filename).read_text(encoding="utf-8")
        )
        payload["attempt_id"] = attempt_id
        (attempt_dir / filename).write_text(
            json.dumps(payload),
            encoding="utf-8",
        )

    pointer_path = paper_root / "latest_run.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    pointer["attempt_id"] = attempt_id
    pointer["attempt_directory"] = str(attempt_dir)
    pointer_path.write_text(json.dumps(pointer), encoding="utf-8")

    projection_path = (
        paper_root / "graphagents" / "mechanism" / "summary.json"
    )

    # A run/fingerprint-only projection is legacy with respect to the new
    # concrete attempt. It must not be counted as current for this attempt.
    row = inspect_broad_paper(data_root=data_root, paper_id="broad_A")
    assert row["attempt_id"] == attempt_id
    assert row["projection_source_extraction_attempt_id"] == ""
    assert row["projection_current"] is False
    assert row["stale_projection_found"] is True
    assert row["direct_mechanism_edges"] == 0

    projection = json.loads(projection_path.read_text(encoding="utf-8"))
    projection["source_extraction_attempt_id"] = "attempt-old"
    projection_path.write_text(json.dumps(projection), encoding="utf-8")
    row = inspect_broad_paper(data_root=data_root, paper_id="broad_A")
    assert row["projection_source_extraction_attempt_id"] == "attempt-old"
    assert row["projection_current"] is False
    assert row["direct_mechanism_edges"] == 0

    projection["source_extraction_attempt_id"] = attempt_id
    projection_path.write_text(json.dumps(projection), encoding="utf-8")
    row = inspect_broad_paper(data_root=data_root, paper_id="broad_A")
    assert row["projection_source_extraction_attempt_id"] == attempt_id
    assert row["projection_current"] is True
    assert row["stale_projection_found"] is False
    assert row["direct_mechanism_edges"] == 3


def test_preflight_outlier_ignores_historical_run_cost(tmp_path: Path):
    data_root = tmp_path / "data_broad"
    _write_run(
        data_root,
        "broad_A",
        status="rejected",
        records=[{
            "status": "quarantined",
            "attempt_usages": [{
                "call_kind": "graph_generation",
                "total_tokens": 9999,
            }],
        }],
    )
    row = inspect_broad_paper(
        data_root=data_root,
        paper_id="broad_A",
        preflight_outlier=True,
    )
    assert row["input_guard_status"] == "ABSTRACT_LENGTH_OUTLIER"
    assert row["historical_run_found"] is True
    assert row["llm_calls"] == 0
    assert row["total_tokens"] == 0


def test_write_diagnostics_emits_three_audit_files(tmp_path: Path):
    data_root = tmp_path / "data_broad"
    _write_run(
        data_root,
        "broad_A",
        status="complete",
        records=[{"status": "success", "attempt_usages": []}],
        projection_edges=1,
    )
    report, rows, issues = write_broad_extraction_diagnostics(
        data_root=data_root,
        paper_ids=["broad_A", "broad_missing"],
        output_dir=tmp_path / "diagnostics",
    )
    assert report.exists()
    assert rows.exists()
    assert issues.exists()
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["requested_paper_count"] == 2
    assert payload["run_missing_count"] == 1
