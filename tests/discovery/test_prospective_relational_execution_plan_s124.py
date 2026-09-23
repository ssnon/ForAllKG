from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pipeline_core.discovery.prospective_relational_execution_plan import (
    ProspectiveRelationalExecutionSettings,
    build_prospective_relational_execution_plan,
    select_prospective_relational_hypothesis,
)
from pipeline_core.discovery.prospective_source_task_freeze import (
    build_prospective_source_task_campaign_freeze,
    ProspectiveSourceTaskCampaignSpec,
)
from pipeline_core.discovery.relational_atomic_binding_plan import (
    RelationalAtomicBindingClaimPlan,
    RelationalAtomicBindingHypothesisPlan,
    RelationalAtomicBindingPlan,
)


def _sha(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode()).hexdigest()


def _source_freeze():
    spec = ProspectiveSourceTaskCampaignSpec(
        campaign_name="test",
        campaign_root="/tmp/campaign",
        domain_profile_id="sers_au_ag",
        corpus_id="sers500_final_v2",
        data_root="/tmp/data",
        semantic_roots=["/tmp/semantic"],
        generation_model="openai/gpt-5.6-luna",
        critic_model="openai/gpt-5.6-luna",
        tasks=[
            {
                "case_id": f"P{index:02d}",
                "relation_family": f"family_{index}",
                "source": f"source {index}",
                "target": f"target {index}",
                "question": f"How does source {index} relate to target {index}?",
            }
            for index in range(6, 11)
        ],
    )
    return build_prospective_source_task_campaign_freeze(
        spec=spec,
        source_spec_sha256="a" * 64,
        repository_head_sha="1" * 40,
        repository_tracked_worktree_dirty=False,
    )


def _claim(
    *,
    candidate_id: str,
    final_id: str,
    claim_id: str,
) -> RelationalAtomicBindingClaimPlan:
    return RelationalAtomicBindingClaimPlan(
        candidate_hypothesis_id=candidate_id,
        final_hypothesis_id=final_id,
        claim_id=claim_id,
        claim_rank=1,
        kind="moderator_interaction",
        importance="core",
        novelty_selection_role="NOVELTY_BEARING",
        claim_text=(
            "Composition changes Raman intensity with particle spacing."
        ),
        rationale="test",
        prior_art_identity_terms=["Composition"],
        relation_nucleus_terms=["Raman intensity", "particle spacing"],
        required_bridge=(
            "Composition changes Raman intensity with particle spacing."
        ),
        predicted_observation=(
            "Composition changes Raman intensity with particle spacing."
        ),
        falsification_condition=(
            "Composition does not change Raman intensity "
            "with particle spacing."
        ),
        search_concepts=[],
        search_queries=["Raman intensity particle spacing"],
        source_claim_sha256="b" * 64,
        binding_status="READY_FOR_LITERAL_ENDPOINT_BINDING",
    )


def _hypothesis(
    tmp_path: Path,
    *,
    suffix: str,
    ready_claims: int,
    novelty_ready: int,
    claim_count: int,
    alpha6_decision: str = "accepted_refinement",
    certification_status: str = "NOVELTY_UNRESOLVED",
    n10_selection_class: str = "CONDITIONAL",
) -> RelationalAtomicBindingHypothesisPlan:
    candidate = "hypothesis:candidate:" + suffix
    final = "hypothesis:final:" + suffix
    claims = [
        _claim(
            candidate_id=candidate,
            final_id=final,
            claim_id=f"claim:{suffix}:{i}",
        )
        for i in range(claim_count)
    ]
    for i, claim in enumerate(claims):
        if i >= ready_claims:
            payload = claim.model_dump(mode="json")
            payload["binding_status"] = (
                "INELIGIBLE_INCOMPLETE_ATOMIC_SPECIFICATION"
            )
            payload["reason_codes"] = ["synthetic_not_ready"]
            claims[i] = RelationalAtomicBindingClaimPlan.model_validate(payload)
        elif i >= novelty_ready:
            payload = claim.model_dump(mode="json")
            payload["novelty_selection_role"] = "REQUIRED_ENABLING_RELATION"
            claims[i] = RelationalAtomicBindingClaimPlan.model_validate(payload)

    return RelationalAtomicBindingHypothesisPlan(
        original_hypothesis_id="hypothesis:original:" + suffix,
        candidate_hypothesis_id=candidate,
        final_hypothesis_id=final,
        alpha6_decision=alpha6_decision,
        certification_status=certification_status,
        n10_selection_class=n10_selection_class,
        source_candidate_portfolio=str(tmp_path / (suffix + ".portfolio.json")),
        source_candidate_portfolio_sha256="c" * 64,
        source_query_plan=str(tmp_path / (suffix + ".claims_queries.json")),
        source_query_plan_sha256="d" * 64,
        claim_count=claim_count,
        binding_ready_claim_count=ready_claims,
        novelty_bearing_binding_ready_claim_count=novelty_ready,
        binding_status=(
            "READY_FOR_LITERAL_ENDPOINT_BINDING"
            if novelty_ready >= 1
            else "NO_NOVELTY_BEARING_BINDABLE_CLAIM"
        ),
        claims=claims,
    )


def _plan(tmp_path: Path) -> RelationalAtomicBindingPlan:
    h1 = _hypothesis(
        tmp_path,
        suffix="a",
        ready_claims=1,
        novelty_ready=1,
        claim_count=1,
    )
    h2 = _hypothesis(
        tmp_path,
        suffix="b",
        ready_claims=2,
        novelty_ready=2,
        claim_count=2,
    )
    cert_path = tmp_path / "certification.json"
    cert_path.write_text(
        json.dumps(
            {
                "candidate_artifacts": [
                    {
                        "candidate_id": h1.candidate_hypothesis_id,
                        "final_hypothesis_id": h1.final_hypothesis_id,
                        "candidate_final_authority_equivalent": True,
                        "external_report": str(tmp_path / "a.report.json"),
                        "query_plan": h1.source_query_plan,
                    },
                    {
                        "candidate_id": h2.candidate_hypothesis_id,
                        "final_hypothesis_id": h2.final_hypothesis_id,
                        "candidate_final_authority_equivalent": True,
                        "external_report": str(tmp_path / "b.report.json"),
                        "query_plan": h2.source_query_plan,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    body = {
        "schema_version": "relational-atomic-binding-plan-v1",
        "run_dir": str(tmp_path),
        "source_alpha6_candidate_portfolio": str(tmp_path / "alpha6.json"),
        "source_alpha6_candidate_portfolio_sha256": "e" * 64,
        "source_certification_report": str(cert_path),
        "source_certification_report_sha256": "f" * 64,
        "hypotheses": [
            h1.model_dump(mode="json"),
            h2.model_dump(mode="json"),
        ],
        "hypothesis_count": 2,
        "ready_hypothesis_count": 2,
        "not_ready_hypothesis_count": 0,
        "claim_count": 3,
        "binding_ready_claim_count": 3,
        "novelty_bearing_binding_ready_claim_count": 3,
        "hypothesis_status_counts": {
            "READY_FOR_LITERAL_ENDPOINT_BINDING": 2
        },
        "claim_status_counts": {
            "READY_FOR_LITERAL_ENDPOINT_BINDING": 3
        },
        "certification_only_source_required": True,
        "candidate_final_authority_equivalence_required": True,
        "endpoint_binding_performed": False,
        "verifier_result_observed": False,
        "production_selection_authority": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
    }
    digest = _sha(body)
    return RelationalAtomicBindingPlan(
        **body,
        plan_id="relational_atomic_binding_plan:" + digest[:20],
        plan_sha256=digest,
    )


def test_execution_plan_freezes_identical_main_semantics_for_all_cases() -> None:
    plan = build_prospective_relational_execution_plan(
        source_freeze=_source_freeze(),
        source_freeze_sha256="2" * 64,
        execution_plan_repository_head_sha="3" * 40,
        repository_tracked_worktree_dirty=False,
        settings=ProspectiveRelationalExecutionSettings(),
    )

    assert plan.case_ids == ["P06", "P07", "P08", "P09", "P10"]
    assert plan.post_case_adaptation_allowed is False
    assert plan.case_replacement_allowed is False

    for row in plan.cases:
        argv = row.main_e2e_argv
        assert "--nonobviousness-original-fallback-enforce" in argv
        assert "--nonobviousness-post-generation-enforce" in argv
        idx = argv.index("--post-generation-n10-authority-mode")
        assert argv[idx + 1] == "certification_only"
        idx = argv.index("--providers")
        assert argv[idx + 1] == "auto"
        idx = argv.index("--results-per-query")
        assert argv[idx + 1] == "12"


def test_structural_selector_prefers_more_novelty_binding_ready_claims(
    tmp_path: Path,
) -> None:
    selection, selected = select_prospective_relational_hypothesis(
        binding_plan=_plan(tmp_path),
    )

    assert selection.status == "SELECTED_BINDING_READY_HYPOTHESIS"
    assert selection.selected_final_hypothesis_id == "hypothesis:final:b"
    assert selected is not None
    assert selected.hypothesis_count == 1
    assert selected.hypotheses[0].final_hypothesis_id == "hypothesis:final:b"
    assert selection.endpoint_binding_outcome_used_for_selection is False


def test_structural_selector_ignores_old_n10_outcome_metadata(
    tmp_path: Path,
) -> None:
    baseline = _plan(tmp_path)
    first, _ = select_prospective_relational_hypothesis(
        binding_plan=baseline,
    )

    payload = baseline.model_dump(mode="json")
    payload["hypotheses"][0]["alpha6_decision"] = "different_a"
    payload["hypotheses"][0]["certification_status"] = "different_b"
    payload["hypotheses"][0]["n10_selection_class"] = "different_c"
    payload["hypotheses"][1]["alpha6_decision"] = "different_d"
    payload["hypotheses"][1]["certification_status"] = "different_e"
    payload["hypotheses"][1]["n10_selection_class"] = "different_f"
    payload.pop("plan_id")
    payload.pop("plan_sha256")
    digest = _sha(payload)
    mutated = RelationalAtomicBindingPlan(
        **payload,
        plan_id="relational_atomic_binding_plan:" + digest[:20],
        plan_sha256=digest,
    )

    second, _ = select_prospective_relational_hypothesis(
        binding_plan=mutated,
    )
    assert first.selected_final_hypothesis_id == second.selected_final_hypothesis_id
    assert second.old_n10_status_used_for_selection is False
    assert second.old_n10_selection_class_used_for_selection is False


def test_selected_plan_is_hash_valid_and_single_hypothesis(
    tmp_path: Path,
) -> None:
    selection, selected = select_prospective_relational_hypothesis(
        binding_plan=_plan(tmp_path),
    )
    assert selected is not None
    reparsed = RelationalAtomicBindingPlan.model_validate(
        selected.model_dump(mode="json")
    )
    assert reparsed.plan_id == selection.selected_binding_plan_id
    assert reparsed.plan_sha256 == selection.selected_binding_plan_sha256


def test_selector_reports_no_ready_hypothesis_without_fallback(
    tmp_path: Path,
) -> None:
    source = _plan(tmp_path)
    payload = source.model_dump(mode="json")
    for row in payload["hypotheses"]:
        row["binding_status"] = "NO_NOVELTY_BEARING_BINDABLE_CLAIM"
        row["novelty_bearing_binding_ready_claim_count"] = 0
        for claim in row["claims"]:
            claim["novelty_selection_role"] = "REQUIRED_ENABLING_RELATION"
    payload["ready_hypothesis_count"] = 0
    payload["not_ready_hypothesis_count"] = 2
    payload["novelty_bearing_binding_ready_claim_count"] = 0
    payload["hypothesis_status_counts"] = {
        "NO_NOVELTY_BEARING_BINDABLE_CLAIM": 2
    }
    payload.pop("plan_id")
    payload.pop("plan_sha256")
    digest = _sha(payload)
    no_ready = RelationalAtomicBindingPlan(
        **payload,
        plan_id="relational_atomic_binding_plan:" + digest[:20],
        plan_sha256=digest,
    )

    selection, selected = select_prospective_relational_hypothesis(
        binding_plan=no_ready,
    )
    assert selection.status == "NO_BINDING_READY_HYPOTHESIS"
    assert selected is None
    assert selection.eligible_final_hypothesis_ids == []
