from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pipeline_core.discovery.reframing.generalization_probe import build_generalization_preflight


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _freeze(tmp_path: Path, *, domain: str = "sers", case_keys=None, task_ids=None) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    relative = "pipeline_core/discovery/reframing/trigger_detection.py"
    target = repo / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("# frozen semantics\n", encoding="utf-8")
    hashes = {relative: _sha(target.read_bytes())}
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
    path = tmp_path / "freeze.json"
    path.write_text(json.dumps(freeze), encoding="utf-8")
    return repo, path


def _write_triplet(canonical: Path, *, domain: str | None, task_id: str, missing=()) -> None:
    canonical.mkdir(parents=True, exist_ok=True)
    values = {
        "explorer.packet.json": {"task_id": task_id, "nested": {"domain_profile_id": domain} if domain else {}},
        "explorer.report.json": {"task_id": task_id},
        "hypothesis.context.json": {"task_id": task_id},
    }
    for name, value in values.items():
        if name not in missing:
            (canonical / name).write_text(json.dumps(value), encoding="utf-8")


def _nested_task(root: Path, bench: str, case: str, *, domain: str, task_id: str) -> None:
    _write_triplet(root / bench / case / "replicate_01" / "canonical", domain=domain, task_id=task_id)


def _flat_task(root: Path, case: str, *, domain: str, task_id: str, missing=()) -> None:
    _write_triplet(root / case / "canonical", domain=domain, task_id=task_id, missing=missing)


def test_sers_profile_alias_matches_frozen_sers_family(tmp_path: Path) -> None:
    repo, freeze = _freeze(tmp_path, domain="sers")
    evaluation = tmp_path / "evaluation"
    _nested_task(evaluation, "holdout", "H01", domain="sers_au_ag", task_id="task:h1")
    report = build_generalization_preflight(freeze_path=freeze, search_root=evaluation, repository_root=repo)
    inventory = report.benchmark_inventories[0]
    assert inventory.inferred_domain_labels == ["sers_au_ag"]
    assert inventory.canonical_domain_labels == ["sers"]
    assert inventory.domain_classification == "source_domain"
    assert report.cross_domain_candidate_cohorts == []


def test_domain_alias_normalization_is_case_and_separator_stable(tmp_path: Path) -> None:
    repo, freeze = _freeze(tmp_path, domain="SERS")
    evaluation = tmp_path / "evaluation"
    _nested_task(evaluation, "holdout", "H01", domain="SERS-AU-AG", task_id="task:h1")
    report = build_generalization_preflight(freeze_path=freeze, search_root=evaluation, repository_root=repo)
    assert report.benchmark_inventories[0].canonical_domain_labels == ["sers"]
    assert report.benchmark_inventories[0].domain_classification == "source_domain"


def test_mixed_path_root_still_yields_path_independent_dac_cohort(tmp_path: Path) -> None:
    repo, freeze = _freeze(tmp_path, domain="sers", case_keys=["CAL01"], task_ids=["task:cal"])
    evaluation = tmp_path / "evaluation"
    _flat_task(evaluation, "CAL01", domain="sers_au_ag", task_id="task:cal")
    _flat_task(evaluation, "D01", domain="dac_her", task_id="task:d1")
    _flat_task(evaluation, "D02", domain="dac_her", task_id="task:d2")
    report = build_generalization_preflight(freeze_path=freeze, search_root=evaluation, repository_root=repo)
    assert report.benchmark_inventories[0].domain_classification == "mixed_or_ambiguous_domain"
    assert report.cross_domain_candidate_roots == []
    assert len(report.cross_domain_candidate_cohorts) == 1
    cohort = report.cross_domain_candidate_cohorts[0]
    assert cohort.canonical_domain_label == "dac_her"
    assert cohort.task_count == 2
    assert cohort.case_count == 2
    assert cohort.calibration_overlap_detected is False
    assert cohort.cohort_is_path_independent is True


def test_cross_domain_cohort_excludes_incomplete_materializations(tmp_path: Path) -> None:
    repo, freeze = _freeze(tmp_path)
    evaluation = tmp_path / "evaluation"
    _flat_task(evaluation, "D01", domain="dac_her", task_id="task:d1")
    _flat_task(
        evaluation,
        "D02",
        domain="dac_her",
        task_id="task:d2",
        missing=("hypothesis.context.json",),
    )
    report = build_generalization_preflight(freeze_path=freeze, search_root=evaluation, repository_root=repo)
    assert report.cross_domain_candidate_cohorts[0].task_count == 1
    assert report.cross_domain_candidate_cohorts[0].task_keys == ["D01"]


def test_cross_domain_cohort_excludes_frozen_overlap_even_inside_mixed_root(tmp_path: Path) -> None:
    repo, freeze = _freeze(tmp_path, case_keys=["DUP"], task_ids=["task:cal"])
    evaluation = tmp_path / "evaluation"
    _flat_task(evaluation, "DUP", domain="dac_her", task_id="task:new")
    _flat_task(evaluation, "D02", domain="dac_her", task_id="task:d2")
    report = build_generalization_preflight(freeze_path=freeze, search_root=evaluation, repository_root=repo)
    cohort = report.cross_domain_candidate_cohorts[0]
    assert cohort.task_count == 1
    assert cohort.task_keys == ["D02"]


def test_raw_domain_aliases_collapse_to_one_candidate_family(tmp_path: Path) -> None:
    repo, freeze = _freeze(tmp_path)
    evaluation = tmp_path / "evaluation"
    _nested_task(evaluation, "dac_a", "D01", domain="DAC-HER", task_id="task:d1")
    _nested_task(evaluation, "dac_b", "D02", domain="dac_her", task_id="task:d2")
    report = build_generalization_preflight(freeze_path=freeze, search_root=evaluation, repository_root=repo)
    assert len(report.cross_domain_candidate_cohorts) == 1
    cohort = report.cross_domain_candidate_cohorts[0]
    assert cohort.canonical_domain_label == "dac_her"
    assert cohort.task_count == 2
    assert cohort.raw_domain_labels == ["DAC-HER", "dac_her"]


def test_unknown_domain_does_not_enter_cross_domain_cohort(tmp_path: Path) -> None:
    repo, freeze = _freeze(tmp_path)
    evaluation = tmp_path / "evaluation"
    canonical = evaluation / "mystery" / "M01" / "replicate_01" / "canonical"
    _write_triplet(canonical, domain=None, task_id="task:m1")
    report = build_generalization_preflight(freeze_path=freeze, search_root=evaluation, repository_root=repo)
    assert report.cross_domain_candidate_cohorts == []
    assert report.unclassified_candidate_roots == [str((evaluation / "mystery").resolve())]


def test_domain_cohort_inventory_is_deterministic(tmp_path: Path) -> None:
    repo, freeze = _freeze(tmp_path)
    evaluation = tmp_path / "evaluation"
    _flat_task(evaluation, "D02", domain="dac_her", task_id="task:d2")
    _flat_task(evaluation, "D01", domain="dac_her", task_id="task:d1")
    first = build_generalization_preflight(freeze_path=freeze, search_root=evaluation, repository_root=repo)
    second = build_generalization_preflight(freeze_path=freeze, search_root=evaluation, repository_root=repo)
    assert first.preflight_id == second.preflight_id
    assert first.cross_domain_candidate_cohorts == second.cross_domain_candidate_cohorts
    assert first.cross_domain_candidate_cohorts[0].task_keys == ["D01", "D02"]


def test_root_level_flat_triplet_preserves_task_identity(tmp_path: Path) -> None:
    repo, freeze = _freeze(tmp_path)
    evaluation = tmp_path / "evaluation"
    flat = evaluation / "s28_dac_her106_canonical_parity_v1"
    _write_triplet(flat, domain="dac_her", task_id="task:dac-flat")

    report = build_generalization_preflight(
        freeze_path=freeze,
        search_root=evaluation,
        repository_root=repo,
    )

    row = next(item for item in report.task_rows if item.task_id == "task:dac-flat")
    assert row.task_key == "s28_dac_her106_canonical_parity_v1"
    assert row.case_key == "s28_dac_her106_canonical_parity_v1"
    assert row.replicate_key == "default"
    assert row.inferred_benchmark_root == str(evaluation.resolve())
    assert report.cross_domain_candidate_cohorts[0].task_keys == [
        "s28_dac_her106_canonical_parity_v1"
    ]
