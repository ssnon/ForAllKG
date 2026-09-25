from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pipeline_core.discovery.pre_n10_external_n10_shadow_v1 import (
    PreN10ExternalN10LineageResultV1,
    PreN10ExternalN10ShadowReportV1,
)
from pipeline_core.discovery.pre_n10_relational_binding_bridge_v1 import (
    build_pre_n10_relational_binding_bridge_v1,
)
from tests.discovery.test_pre_n10_external_n10_shadow_v1_s162 import (
    _initial_handoff,
    _regenerated_handoff,
)


def _canonical_sha(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _write(path: Path, value: object) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gate(row, *, selection: str):
    if selection == "ELIGIBLE":
        positive = True
        fallback = True
    else:
        positive = False
        fallback = False
    return {
        "schema_version": "scientific-novelty-fallback-gate-v2",
        "production_authority": True,
        "authority_scope": (
            "alpha6_original_fallback"
            if row.origin == "INITIAL_PRIMARY_READY"
            else "alpha6_post_generation_candidate"
        ),
        "authority_source": "n10_role_aware_nonobviousness_v2",
        "source_portfolio_id": row.portfolio_id,
        "source_query_plan_id": row.query_plan_id,
        "positive_authority_requires": (
            "ELIGIBLE_AND_ROLE_AWARE_POSITIVE_NONOBVIOUSNESS"
        ),
        "conditional_is_positive": False,
        "absence_is_novelty": False,
        "candidate_semantics_preserved": True,
        "gate_count": 1,
        "gates": [
            {
                "hypothesis_id": row.downstream_hypothesis_id,
                "selection_class": selection,
                "positive_nonobviousness_authority": positive,
                "fallback_allowed": fallback,
                "action": "KEEP" if fallback else "RESOLVE",
                "blocking_claim_ids": [],
                "unresolved_claim_ids": [],
                "resolution_requirements": [],
                "reason_codes": [],
            }
        ],
    }


def _shadow(tmp_path: Path, handoff, *, selection: str):
    hrow = handoff.lineages[0]
    external_path = tmp_path / "external.report.json"
    external_path.write_text('{"fixture":"external"}\n', encoding="utf-8")
    intake_path = tmp_path / "n9.intake.json"
    full_path = tmp_path / "n9.full.json"
    candidate_path = tmp_path / "n10.candidate.json"
    for path in (intake_path, full_path, candidate_path):
        path.write_text('{"fixture":true}\n', encoding="utf-8")

    production_path = tmp_path / "n10.production.json"
    _write(production_path, _gate(hrow, selection=selection))

    status = {
        "ELIGIBLE": "NOVELTY_CERTIFIED",
        "CONDITIONAL": "NOVELTY_UNRESOLVED",
        "INELIGIBLE": "NOVELTY_REJECTED",
    }[selection]
    row = PreN10ExternalN10LineageResultV1(
        lineage_id=hrow.lineage_id,
        origin=hrow.origin,
        source_hypothesis_id=hrow.source_hypothesis_id,
        downstream_hypothesis_id=hrow.downstream_hypothesis_id,
        portfolio_id=hrow.portfolio_id,
        query_plan_id=hrow.query_plan_id,
        external_report_path=str(external_path),
        external_report_file_sha256=_file_sha(external_path),
        n9_intake_path=str(intake_path),
        n9_intake_file_sha256=_file_sha(intake_path),
        n9_full_closure_path=str(full_path),
        n9_full_closure_file_sha256=_file_sha(full_path),
        n10_candidate_gate_path=str(candidate_path),
        n10_candidate_gate_file_sha256=_file_sha(candidate_path),
        n10_production_gate_path=str(production_path),
        n10_production_gate_file_sha256=_file_sha(production_path),
        n10_authority_scope=(
            "alpha6_original_fallback"
            if hrow.origin == "INITIAL_PRIMARY_READY"
            else "alpha6_post_generation_candidate"
        ),
        n10_selection_class=selection,
        positive_nonobviousness_authority=(selection == "ELIGIBLE"),
        fallback_allowed=(selection == "ELIGIBLE"),
        certification_status=status,
    )
    body = {
        "schema_version": "pre-n10-external-n9-n10-shadow-report-v1",
        "source_execution_plan_id": "plan:fixture",
        "source_execution_plan_sha256": "1" * 64,
        "source_handoff_report_id": handoff.report_id,
        "source_handoff_report_sha256": handoff.report_sha256,
        "lineages": [row.model_dump(mode="json")],
        "lineage_count": 1,
        "certified_count": int(status == "NOVELTY_CERTIFIED"),
        "unresolved_count": int(status == "NOVELTY_UNRESOLVED"),
        "rejected_count": int(status == "NOVELTY_REJECTED"),
        "exact_handoff_population_consumed": True,
        "pre_n10_ready_was_required": True,
        "semantic_disposition_pass_was_required": True,
        "frozen_query_plan_reused": True,
        "candidate_survival_authority": False,
        "novelty_certification_authority": True,
        "authority_source": "n10_role_aware_nonobviousness_v2",
        "authority_mode": "certification_only_shadow",
        "conditional_is_positive": False,
        "absence_is_novelty": False,
        "ineligible_deletes_scientific_candidate": False,
        "targeted_novelty_continuation_performed": False,
        "novelty_refinement_performed": False,
        "second_regeneration_performed": False,
        "endpoint_binding_performed": False,
        "verifier_performed": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
    }
    digest = _canonical_sha(body)
    report = PreN10ExternalN10ShadowReportV1(
        **body,
        report_id="pre_n10_external_n9_n10_shadow_report_v1:" + digest[:20],
        report_sha256=digest,
    )
    report_path = tmp_path / "external_n10.report.json"
    _write(report_path, report)
    return report_path, production_path


@pytest.mark.parametrize(
    ("selection", "expected_status"),
    [
        ("ELIGIBLE", "NOVELTY_CERTIFIED"),
        ("CONDITIONAL", "NOVELTY_UNRESOLVED"),
        ("INELIGIBLE", "NOVELTY_REJECTED"),
    ],
)
def test_n10_class_does_not_filter_strict_binding_reachability(
    tmp_path: Path,
    selection: str,
    expected_status: str,
) -> None:
    handoff = _initial_handoff(tmp_path)
    report_path, _ = _shadow(tmp_path, handoff, selection=selection)

    report = build_pre_n10_relational_binding_bridge_v1(
        handoff=handoff,
        external_n10_report_path=report_path,
        output_root=tmp_path / "binding",
    )

    assert report.lineage_count == 1
    row = report.lineages[0]
    assert row.n10_certification_status == expected_status
    assert row.binding_status == "READY_FOR_LITERAL_ENDPOINT_BINDING"
    assert row.binding_ready_claim_count >= 1
    assert row.novelty_bearing_binding_ready_claim_count >= 1
    assert row.n10_certification_status_filters_binding_reachability is False
    assert Path(row.binding_plan_path).is_file()


def test_regenerated_lineage_uses_exact_downstream_identity(tmp_path: Path) -> None:
    handoff = _regenerated_handoff(tmp_path)
    report_path, _ = _shadow(tmp_path, handoff, selection="CONDITIONAL")

    report = build_pre_n10_relational_binding_bridge_v1(
        handoff=handoff,
        external_n10_report_path=report_path,
        output_root=tmp_path / "binding",
    )

    row = report.lineages[0]
    payload = json.loads(Path(row.binding_plan_path).read_text(encoding="utf-8"))
    hypothesis = payload["hypotheses"][0]
    assert hypothesis["original_hypothesis_id"] == row.source_hypothesis_id
    assert hypothesis["candidate_hypothesis_id"] == row.downstream_hypothesis_id
    assert hypothesis["final_hypothesis_id"] == row.downstream_hypothesis_id
    assert hypothesis["candidate_final_authority_equivalent"] is True


def test_binding_bridge_rejects_mutated_n10_gate(tmp_path: Path) -> None:
    handoff = _initial_handoff(tmp_path)
    report_path, production_path = _shadow(
        tmp_path, handoff, selection="CONDITIONAL"
    )
    production_path.write_text('{"mutated":true}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="N10 production gate changed"):
        build_pre_n10_relational_binding_bridge_v1(
            handoff=handoff,
            external_n10_report_path=report_path,
            output_root=tmp_path / "binding",
        )


def test_binding_bridge_is_write_once_and_replay_exact(tmp_path: Path) -> None:
    handoff = _initial_handoff(tmp_path)
    report_path, _ = _shadow(tmp_path, handoff, selection="CONDITIONAL")
    root = tmp_path / "binding"

    first = build_pre_n10_relational_binding_bridge_v1(
        handoff=handoff,
        external_n10_report_path=report_path,
        output_root=root,
    )
    second = build_pre_n10_relational_binding_bridge_v1(
        handoff=handoff,
        external_n10_report_path=report_path,
        output_root=root,
    )
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
