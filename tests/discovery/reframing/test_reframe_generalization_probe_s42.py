from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pipeline_core.discovery.reframing.generalization_probe import (
    build_generalization_preflight,
)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _freeze(tmp_path: Path, *, domain: str = "sers", case_keys=None, task_ids=None) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    paths = [
        "pipeline_core/discovery/reframing/evidence_tension.py",
        "pipeline_core/discovery/reframing/trigger_detection.py",
    ]
    hashes = {}
    for relative in paths:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative.endswith("evidence_tension.py"):
            path.write_text("# SERS hotspot and Raman / LSPR / electromagnetic diagnostic\n", encoding="utf-8")
        else:
            path.write_text("# generic trigger\n", encoding="utf-8")
        hashes[relative] = _sha(path.read_bytes())
    payload = json.dumps(hashes, sort_keys=True, separators=(",", ":"))
    freeze = {
        "schema_version": "scientific-reframe-calibration-freeze-v1",
        "freeze_id": "freeze:test",
        "calibration_domain_label": domain,
        "semantics_fingerprint": _sha(payload.encode("utf-8")),
        "semantics_file_sha256": hashes,
        "case_keys": case_keys or ["CAL01"],
        "task_ids": task_ids or ["task:cal"],
    }
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    return repo, freeze_path


def _task(root: Path, benchmark: str, case: str, replicate: str, *, domain: str | None, task_id: str, missing=()) -> Path:
    canonical = root / benchmark / case / replicate / "canonical"
    canonical.mkdir(parents=True)
    values = {
        "explorer.packet.json": {"task_id": task_id, "nested": {"domain_profile_id": domain} if domain else {}},
        "explorer.report.json": {"task_id": task_id},
        "hypothesis.context.json": {"task_id": task_id},
    }
    for name, value in values.items():
        if name not in missing:
            (canonical / name).write_text(json.dumps(value), encoding="utf-8")
    return canonical


def test_discovers_complete_cross_domain_candidate(tmp_path: Path) -> None:
    repo, freeze = _freeze(tmp_path)
    evaluation = tmp_path / "evaluation"
    _task(evaluation, "dac_bench", "D01", "replicate_01", domain="dac_her", task_id="task:d1")
    report = build_generalization_preflight(
        freeze_path=freeze,
        search_root=evaluation,
        repository_root=repo,
    )
    assert report.complete_triplet_count == 1
    assert report.cross_domain_candidate_roots == [str((evaluation / "dac_bench").resolve())]
    assert report.benchmark_inventories[0].domain_classification == "cross_domain_candidate"


def test_source_domain_benchmark_is_not_cross_domain_candidate(tmp_path: Path) -> None:
    repo, freeze = _freeze(tmp_path)
    evaluation = tmp_path / "evaluation"
    _task(evaluation, "sers_holdout", "H01", "replicate_01", domain="sers", task_id="task:h1")
    report = build_generalization_preflight(freeze_path=freeze, search_root=evaluation, repository_root=repo)
    assert report.cross_domain_candidate_roots == []
    assert report.benchmark_inventories[0].domain_classification == "source_domain"


def test_domain_unknown_root_is_reported_unclassified(tmp_path: Path) -> None:
    repo, freeze = _freeze(tmp_path)
    evaluation = tmp_path / "evaluation"
    _task(evaluation, "mystery", "M01", "replicate_01", domain=None, task_id="task:m1")
    report = build_generalization_preflight(freeze_path=freeze, search_root=evaluation, repository_root=repo)
    assert report.unclassified_candidate_roots == [str((evaluation / "mystery").resolve())]
    assert report.benchmark_inventories[0].domain_classification == "domain_unknown"


def test_incomplete_materialization_is_counted(tmp_path: Path) -> None:
    repo, freeze = _freeze(tmp_path)
    evaluation = tmp_path / "evaluation"
    _task(
        evaluation,
        "partial",
        "P01",
        "replicate_01",
        domain="dac_her",
        task_id="task:p1",
        missing=("hypothesis.context.json",),
    )
    report = build_generalization_preflight(freeze_path=freeze, search_root=evaluation, repository_root=repo)
    assert report.complete_triplet_count == 0
    assert report.incomplete_materialization_count == 1
    assert report.cross_domain_candidate_roots == []


def test_calibration_overlap_detected_by_case_or_task_id(tmp_path: Path) -> None:
    repo, freeze = _freeze(tmp_path, case_keys=["DUP"], task_ids=["task:cal"])
    evaluation = tmp_path / "evaluation"
    _task(evaluation, "candidate", "DUP", "replicate_01", domain="dac_her", task_id="task:new")
    _task(evaluation, "candidate", "D02", "replicate_01", domain="dac_her", task_id="task:cal")
    report = build_generalization_preflight(freeze_path=freeze, search_root=evaluation, repository_root=repo)
    assert report.calibration_overlap_detected is True
    assert sum(row.any_frozen_overlap for row in report.task_rows) == 2
    assert report.cross_domain_candidate_roots == []


def test_marker_scan_is_static_diagnostic_only(tmp_path: Path) -> None:
    repo, freeze = _freeze(tmp_path)
    evaluation = tmp_path / "evaluation"
    report = build_generalization_preflight(
        freeze_path=freeze,
        search_root=evaluation,
        repository_root=repo,
        source_domain_markers=["sers", "raman", "hotspot"],
    )
    assert report.source_domain_marker_hit_count == 3
    assert report.marker_hit_counts == {"hotspot": 1, "raman": 1, "sers": 1}
    assert report.marker_hits_are_static_diagnostics_only is True
    assert report.marker_hits_prove_runtime_dependency is False


def test_semantics_drift_is_detected_without_modifying_files(tmp_path: Path) -> None:
    repo, freeze = _freeze(tmp_path)
    target = repo / "pipeline_core/discovery/reframing/trigger_detection.py"
    target.write_text("# changed after freeze\n", encoding="utf-8")
    report = build_generalization_preflight(freeze_path=freeze, search_root=tmp_path / "evaluation", repository_root=repo)
    assert report.semantics_unchanged is False
    assert report.frozen_semantics_modified is False


def test_preflight_is_deterministic_under_filesystem_order(tmp_path: Path) -> None:
    repo, freeze = _freeze(tmp_path)
    evaluation = tmp_path / "evaluation"
    _task(evaluation, "b", "B01", "replicate_01", domain="dac_her", task_id="task:b")
    _task(evaluation, "a", "A01", "replicate_01", domain="dac_her", task_id="task:a")
    first = build_generalization_preflight(freeze_path=freeze, search_root=evaluation, repository_root=repo)
    second = build_generalization_preflight(freeze_path=freeze, search_root=evaluation, repository_root=repo)
    assert first.preflight_id == second.preflight_id
    assert [row.task_key for row in first.task_rows] == [row.task_key for row in second.task_rows]
