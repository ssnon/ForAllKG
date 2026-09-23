from __future__ import annotations

import hashlib
import json

import pytest

from pipeline_core.discovery.relational_atomic_binding_plan import (
    RelationalAtomicBindingClaimPlan,
    RelationalAtomicBindingHypothesisPlan,
    RelationalAtomicBindingPlan,
)
from pipeline_core.discovery.relational_atomic_endpoint_binding import (
    LiteralEndpointBindingBatchDraft,
    LiteralEndpointBindingDraft,
    build_endpoint_binding_prompt,
    compile_endpoint_bindings,
    selected_binding_claims,
)


def _sha(value: object) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode()).hexdigest()


def _claim(
    *,
    claim_id: str = "claim:1",
    role: str = "NOVELTY_BEARING",
) -> RelationalAtomicBindingClaimPlan:
    return RelationalAtomicBindingClaimPlan(
        candidate_hypothesis_id="hypothesis:candidate",
        final_hypothesis_id="hypothesis:final",
        claim_id=claim_id,
        claim_rank=1,
        kind="moderator_interaction",
        importance="core",
        novelty_selection_role=role,
        claim_text=(
            "Catalyst composition changes the Raman-intensity response "
            "to nanostructure spacing under matched conditions."
        ),
        rationale="frozen",
        prior_art_identity_terms=["Catalyst composition"],
        relation_nucleus_terms=[
            "Raman intensity",
            "nanostructure spacing",
            "response",
        ],
        required_bridge=(
            "Catalyst composition changes the Raman-intensity response "
            "to nanostructure spacing under matched conditions."
        ),
        predicted_observation=(
            "Catalyst composition changes the Raman-intensity response "
            "to nanostructure spacing."
        ),
        falsification_condition=(
            "Catalyst composition does not change the Raman-intensity "
            "response to nanostructure spacing."
        ),
        search_concepts=[],
        search_queries=[],
        source_claim_sha256="a" * 64,
        binding_status="READY_FOR_LITERAL_ENDPOINT_BINDING",
    )


def _plan() -> RelationalAtomicBindingPlan:
    claims = [
        _claim(),
        _claim(claim_id="claim:2", role="REQUIRED_ENABLING_RELATION"),
    ]
    hypothesis = RelationalAtomicBindingHypothesisPlan(
        original_hypothesis_id="hypothesis:original",
        candidate_hypothesis_id="hypothesis:candidate",
        final_hypothesis_id="hypothesis:final",
        alpha6_decision="accepted_refinement",
        certification_status="NOVELTY_UNRESOLVED",
        n10_selection_class="CONDITIONAL",
        source_candidate_portfolio="/tmp/candidate.json",
        source_candidate_portfolio_sha256="b" * 64,
        source_query_plan="/tmp/plan.json",
        source_query_plan_sha256="c" * 64,
        claim_count=2,
        binding_ready_claim_count=2,
        novelty_bearing_binding_ready_claim_count=1,
        binding_status="READY_FOR_LITERAL_ENDPOINT_BINDING",
        claims=claims,
    )
    body = {
        "schema_version": "relational-atomic-binding-plan-v1",
        "run_dir": "/tmp/run",
        "source_alpha6_candidate_portfolio": "/tmp/final.json",
        "source_alpha6_candidate_portfolio_sha256": "d" * 64,
        "source_certification_report": "/tmp/cert.json",
        "source_certification_report_sha256": "e" * 64,
        "hypotheses": [hypothesis.model_dump(mode="json")],
        "hypothesis_count": 1,
        "ready_hypothesis_count": 1,
        "not_ready_hypothesis_count": 0,
        "claim_count": 2,
        "binding_ready_claim_count": 2,
        "novelty_bearing_binding_ready_claim_count": 1,
        "hypothesis_status_counts": {
            "READY_FOR_LITERAL_ENDPOINT_BINDING": 1
        },
        "claim_status_counts": {
            "READY_FOR_LITERAL_ENDPOINT_BINDING": 2
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


def _draft_for(
    claim_id: str,
    *,
    endpoints: list[str] | None = None,
    abstention_reason: str | None = None,
) -> LiteralEndpointBindingDraft:
    return LiteralEndpointBindingDraft(
        claim_id=claim_id,
        relation_endpoint_anchors=(
            endpoints
            if endpoints is not None
            else ["Raman intensity", "nanostructure spacing"]
        ),
        abstention_reason=abstention_reason,
    )


def test_selected_population_includes_ready_enabling_claims() -> None:
    rows = selected_binding_claims(_plan())
    assert [row.claim_id for row in rows] == ["claim:1", "claim:2"]


def test_prompt_excludes_relation_nucleus_authority() -> None:
    prompt = build_endpoint_binding_prompt(_plan())
    assert '"relation_nucleus_terms"' not in prompt.user_prompt
    assert "Do not use relation_nucleus_terms" in prompt.system_prompt


def test_compile_accepts_exact_literal_endpoint_population() -> None:
    report = compile_endpoint_bindings(
        plan=_plan(),
        draft=LiteralEndpointBindingBatchDraft(
            bindings=[
                _draft_for("claim:1"),
                _draft_for("claim:2"),
            ]
        ),
        backend_name="test",
        model_name="test",
        llm_calls_performed=1,
    )
    assert report.selected_hypothesis_count == 1
    assert report.selected_claim_count == 2
    assert report.bound_claim_count == 2
    assert report.novelty_bearing_bound_claim_count == 1
    assert report.claim_content_mutated is False


def test_compile_rejects_nonliteral_endpoint() -> None:
    with pytest.raises(
        ValueError,
        match="absent from claim text",
    ):
        compile_endpoint_bindings(
            plan=_plan(),
            draft=LiteralEndpointBindingBatchDraft(
                bindings=[
                    _draft_for(
                        "claim:1",
                        endpoints=["Raman intensity", "spectral response"],
                    ),
                    _draft_for("claim:2"),
                ]
            ),
            backend_name="test",
            model_name="test",
            llm_calls_performed=1,
        )


def test_compile_rejects_generic_relation_operator_as_endpoint() -> None:
    with pytest.raises(
        ValueError,
        match="lacks scientific content",
    ):
        compile_endpoint_bindings(
            plan=_plan(),
            draft=LiteralEndpointBindingBatchDraft(
                bindings=[
                    _draft_for(
                        "claim:1",
                        endpoints=["changes", "Raman intensity"],
                    ),
                    _draft_for("claim:2"),
                ]
            ),
            backend_name="test",
            model_name="test",
            llm_calls_performed=1,
        )


def test_compile_rejects_branch_identity_as_endpoint() -> None:
    with pytest.raises(
        ValueError,
        match="retains complete branch identity",
    ):
        compile_endpoint_bindings(
            plan=_plan(),
            draft=LiteralEndpointBindingBatchDraft(
                bindings=[
                    _draft_for(
                        "claim:1",
                        endpoints=["Catalyst composition", "Raman intensity"],
                    ),
                    _draft_for(
                        "claim:2",
                        endpoints=["Catalyst composition", "Raman intensity"],
                    ),
                ]
            ),
            backend_name="test",
            model_name="test",
            llm_calls_performed=1,
        )


def test_compile_requires_exact_frozen_claim_population() -> None:
    with pytest.raises(
        ValueError,
        match="exact frozen claim population",
    ):
        compile_endpoint_bindings(
            plan=_plan(),
            draft=LiteralEndpointBindingBatchDraft(
                bindings=[_draft_for("claim:1")]
            ),
            backend_name="test",
            model_name="test",
            llm_calls_performed=1,
        )


def test_claim_level_abstention_is_fail_closed() -> None:
    report = compile_endpoint_bindings(
        plan=_plan(),
        draft=LiteralEndpointBindingBatchDraft(
            bindings=[
                _draft_for("claim:1"),
                _draft_for(
                    "claim:2",
                    endpoints=[],
                    abstention_reason=(
                        "No two literal distinct endpoints are available."
                    ),
                ),
            ]
        ),
        backend_name="test",
        model_name="test",
        llm_calls_performed=1,
    )
    assert report.bound_claim_count == 1
    assert report.abstained_claim_count == 1
    assert report.novelty_bearing_bound_claim_count == 1


def test_nested_endpoint_phrases_are_rejected() -> None:
    plan = _plan()
    claim = plan.hypotheses[0].claims[0]
    payload = claim.model_dump(mode="json")
    payload["claim_text"] = (
        "Catalyst composition changes Raman intensity relative "
        "to relative Raman intensity."
    )
    payload["required_bridge"] = (
        "Catalyst composition changes Raman intensity relative "
        "to relative Raman intensity."
    )
    payload["prior_art_identity_terms"] = ["Catalyst composition"]
    payload["predicted_observation"] = (
        "Catalyst composition changes Raman intensity relative "
        "to relative Raman intensity."
    )
    payload["falsification_condition"] = (
        "Catalyst composition does not change Raman intensity relative "
        "to relative Raman intensity."
    )
    plan_payload = plan.model_dump(mode="json")
    plan_payload["hypotheses"][0]["claims"][0] = payload

    # Recompute nested plan identities after the synthetic mutation.
    plan_payload.pop("plan_id")
    plan_payload.pop("plan_sha256")
    digest = _sha(plan_payload)
    mutated = RelationalAtomicBindingPlan(
        **plan_payload,
        plan_id="relational_atomic_binding_plan:" + digest[:20],
        plan_sha256=digest,
    )

    with pytest.raises(
        ValueError,
        match="nested/redundant",
    ):
        compile_endpoint_bindings(
            plan=mutated,
            draft=LiteralEndpointBindingBatchDraft(
                bindings=[
                    _draft_for(
                        "claim:1",
                        endpoints=[
                            "Raman intensity",
                            "relative Raman intensity",
                        ],
                    ),
                    _draft_for("claim:2"),
                ]
            ),
            backend_name="test",
            model_name="test",
            llm_calls_performed=1,
        )
