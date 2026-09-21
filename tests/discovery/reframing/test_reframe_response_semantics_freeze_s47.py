from __future__ import annotations

import json
from pathlib import Path

from pipeline_core.discovery.reframing.response_semantics_probe import (
    ScientificResponseSemanticsProbeReport,
)
from pipeline_core.discovery.reframing.response_semantics_validation import (
    build_response_semantics_candidate_freeze,
    candidate_semantics_fingerprint,
    inspect_response_semantics_validation_eligibility,
)


def _repo(tmp_path: Path) -> Path:
    path = tmp_path / "repo"
    target = path / "pipeline_core/discovery/reframing/response_semantics_probe.py"
    target.parent.mkdir(parents=True)
    target.write_text("# candidate semantics\nVALUE = 1\n", encoding="utf-8")
    return path


def _probe() -> ScientificResponseSemanticsProbeReport:
    return ScientificResponseSemanticsProbeReport(
        probe_id="probe:1",
        source_diagnostics_pack_id="diag:1",
        source_task_id="task:adapt",
        validation_domain_label="dac_her",
        semantics_profile_id="scientific-response-semantics-v2-probe:dac_her",
        statements=[],
        premise_count=0,
        gap_count=0,
        v1_response_match_count=0,
        candidate_response_match_count=0,
        candidate_law_signal_counts={},
        candidate_latent_response_support_premise_count=0,
        candidate_latent_support_paper_count=0,
        candidate_latent_current_floor_met=False,
    )


def _freeze(tmp_path: Path):
    repo = _repo(tmp_path)
    probe = _probe()
    probe_path = tmp_path / "probe.json"
    probe_path.write_text(probe.model_dump_json(indent=2), encoding="utf-8")
    freeze = build_response_semantics_candidate_freeze(
        probe=probe,
        source_probe_path=probe_path,
        repository_root=repo,
    )
    return repo, freeze


def test_candidate_freeze_records_exact_adaptation_task(tmp_path: Path):
    _, freeze = _freeze(tmp_path)
    assert freeze.adaptation_task_ids == ["task:adapt"]
    assert freeze.adaptation_data_excluded_from_untouched_validation is True


def test_candidate_freeze_records_probe_file_hash(tmp_path: Path):
    repo, freeze = _freeze(tmp_path)
    fingerprint, file_hashes = candidate_semantics_fingerprint(repo)
    assert freeze.candidate_semantics_fingerprint == fingerprint
    assert freeze.candidate_semantics_file_sha256 == file_hashes


def test_adaptation_task_is_rejected_as_untouched_validation(tmp_path: Path):
    repo, freeze = _freeze(tmp_path)
    result = inspect_response_semantics_validation_eligibility(
        freeze=freeze,
        validation_task_id="task:adapt",
        validation_domain_label="dac_her",
        repository_root=repo,
    )
    assert result.status == "adaptation_overlap"
    assert result.eligible_as_untouched_validation is False


def test_new_task_is_eligible_when_candidate_semantics_unchanged(tmp_path: Path):
    repo, freeze = _freeze(tmp_path)
    result = inspect_response_semantics_validation_eligibility(
        freeze=freeze,
        validation_task_id="task:new",
        validation_domain_label="dac_her",
        repository_root=repo,
    )
    assert result.status == "eligible_untouched"
    assert result.eligible_as_untouched_validation is True


def test_candidate_semantic_drift_blocks_untouched_validation(tmp_path: Path):
    repo, freeze = _freeze(tmp_path)
    target = repo / "pipeline_core/discovery/reframing/response_semantics_probe.py"
    target.write_text("# changed semantics\nVALUE = 2\n", encoding="utf-8")
    result = inspect_response_semantics_validation_eligibility(
        freeze=freeze,
        validation_task_id="task:new",
        validation_domain_label="dac_her",
        repository_root=repo,
    )
    assert result.status == "candidate_semantic_drift"
    assert result.candidate_semantics_unchanged is False


def test_overlap_plus_semantic_drift_is_multiple_issues(tmp_path: Path):
    repo, freeze = _freeze(tmp_path)
    target = repo / "pipeline_core/discovery/reframing/response_semantics_probe.py"
    target.write_text("# changed semantics\nVALUE = 2\n", encoding="utf-8")
    result = inspect_response_semantics_validation_eligibility(
        freeze=freeze,
        validation_task_id="task:adapt",
        validation_domain_label="dac_her",
        repository_root=repo,
    )
    assert result.status == "multiple_issues"
    assert result.eligible_as_untouched_validation is False


def test_freeze_keeps_candidate_diagnostic_only(tmp_path: Path):
    _, freeze = _freeze(tmp_path)
    assert freeze.production_trigger_semantics_modified is False
    assert freeze.actual_trigger_authority is False
    assert freeze.scientific_authority is False
    assert freeze.llm_calls_performed == 0


def test_freeze_id_is_stable_for_same_inputs(tmp_path: Path):
    repo = _repo(tmp_path)
    probe = _probe()
    probe_path = tmp_path / "probe.json"
    payload = json.loads(probe.model_dump_json())
    probe_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    first = build_response_semantics_candidate_freeze(
        probe=probe,
        source_probe_path=probe_path,
        repository_root=repo,
    )
    second = build_response_semantics_candidate_freeze(
        probe=probe,
        source_probe_path=probe_path,
        repository_root=repo,
    )
    assert first.freeze_id == second.freeze_id
