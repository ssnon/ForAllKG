from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pipeline_core.discovery.external_novelty_contracts import (
    HypothesisNoveltyClaims,
    LiteratureQueryPlan,
    NoveltyClaim,
)
from pipeline_core.discovery.relational_atomic_binding_plan import (
    RelationalAtomicBindingClaimPlan,
    RelationalAtomicBindingHypothesisPlan,
    RelationalAtomicBindingPlan,
)
from pipeline_core.discovery.relational_atomic_endpoint_binding import (
    CompiledLiteralEndpointBinding,
    RelationalAtomicEndpointBindingReport,
)
from pipeline_core.discovery.repaired_verifier_provenance_rebind import (
    _build_query_plan_sidecar,
    _rebind_endpoint_report,
    _rebuild_binding_plan,
)


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _novelty_claim() -> NoveltyClaim:
    return NoveltyClaim(
        claim_id="claim:1",
        hypothesis_id="hypothesis:candidate",
        claim_rank=1,
        kind="moderator_interaction",
        importance="core",
        novelty_selection_role="NOVELTY_BEARING",
        text=(
            "For substrate composition, decreasing interparticle gap "
            "increases Raman intensity."
        ),
        rationale="test",
        search_concepts=["gap", "Raman"],
        search_queries=["gap Raman"],
        prior_art_identity_terms=["substrate composition"],
        relation_nucleus_terms=["interparticle gap", "Raman intensity"],
        required_bridge=(
            "For substrate composition, gap reduction increases Raman intensity."
        ),
        predicted_observation=(
            "For substrate composition, decreasing interparticle gap "
            "increases Raman intensity."
        ),
        falsification_condition=(
            "For substrate composition, decreasing interparticle gap "
            "does not increase Raman intensity."
        ),
    )


def _source_query_plan() -> LiteratureQueryPlan:
    claim = _novelty_claim()
    body = {
        "schema_version": "literature-query-plan-v1",
        "source_portfolio_id": "portfolio:1",
        "queries": [],
        "claims": [
            HypothesisNoveltyClaims(
                hypothesis_id="hypothesis:candidate",
                title="title",
                claims=[claim],
            ).model_dump(mode="json")
        ],
        "policy_version": "external-novelty-query-policy-v1",
    }
    digest = _sha(body)
    return LiteratureQueryPlan(
        **body,
        plan_id="literature_query_plan:" + digest[:20],
        plan_sha256=digest,
    )


def _binding_claim() -> RelationalAtomicBindingClaimPlan:
    source = _novelty_claim()
    repaired = source.model_copy(
        update={
            "required_bridge": (
                "For substrate composition, decreasing interparticle gap "
                "increases Raman intensity."
            )
        }
    )
    return RelationalAtomicBindingClaimPlan(
        candidate_hypothesis_id="hypothesis:candidate",
        final_hypothesis_id="hypothesis:final",
        claim_id=repaired.claim_id,
        claim_rank=repaired.claim_rank,
        kind=repaired.kind,
        importance=repaired.importance,
        novelty_selection_role=repaired.novelty_selection_role,
        claim_text=repaired.text,
        rationale=repaired.rationale,
        prior_art_identity_terms=repaired.prior_art_identity_terms,
        relation_nucleus_terms=repaired.relation_nucleus_terms,
        required_bridge=repaired.required_bridge,
        predicted_observation=repaired.predicted_observation,
        falsification_condition=repaired.falsification_condition,
        search_concepts=repaired.search_concepts,
        search_queries=repaired.search_queries,
        source_claim_sha256="a" * 64,
        binding_status="READY_FOR_LITERAL_ENDPOINT_BINDING",
        reason_codes=[],
    )


def test_sidecar_hash_matches_actual_repaired_novelty_claim() -> None:
    claim = _binding_claim()
    sidecar, hashes = _build_query_plan_sidecar(
        source_plan=_source_query_plan(),
        candidate_hypothesis_id="hypothesis:candidate",
        repaired_claims={claim.claim_id: claim},
    )
    repaired_claim = sidecar.claims[0].claims[0]
    assert hashes[claim.claim_id] == _sha(
        repaired_claim.model_dump(mode="json")
    )
    assert repaired_claim.required_bridge == claim.required_bridge


def test_sidecar_preserves_queries_and_source_portfolio() -> None:
    source = _source_query_plan()
    claim = _binding_claim()
    sidecar, _ = _build_query_plan_sidecar(
        source_plan=source,
        candidate_hypothesis_id="hypothesis:candidate",
        repaired_claims={claim.claim_id: claim},
    )
    assert sidecar.source_portfolio_id == source.source_portfolio_id
    assert sidecar.queries == source.queries
    assert sidecar.policy_version == source.policy_version


def test_sidecar_rejects_nonpermitted_claim_change() -> None:
    claim = _binding_claim()
    payload = claim.model_dump(mode="json")
    payload["rationale"] = "mutated rationale"
    mutated = RelationalAtomicBindingClaimPlan.model_validate(payload)
    try:
        _build_query_plan_sidecar(
            source_plan=_source_query_plan(),
            candidate_hypothesis_id="hypothesis:candidate",
            repaired_claims={mutated.claim_id: mutated},
        )
    except ValueError as exc:
        assert "outside permitted specification fields" in str(exc)
    else:
        raise AssertionError("expected nonpermitted mutation rejection")


def _plan_with_paths(
    tmp_path: Path,
) -> tuple[RelationalAtomicBindingPlan, Path]:
    source_query = _source_query_plan()
    query_path = tmp_path / "source_query.json"
    query_path.write_text(
        json.dumps(source_query.model_dump(mode="json"), sort_keys=True),
        encoding="utf-8",
    )
    claim = _binding_claim()
    hyp = RelationalAtomicBindingHypothesisPlan(
        original_hypothesis_id="hypothesis:original",
        candidate_hypothesis_id="hypothesis:candidate",
        final_hypothesis_id="hypothesis:final",
        alpha6_decision="accepted",
        certification_status="UNRESOLVED",
        n10_selection_class="CONDITIONAL",
        source_candidate_portfolio="/tmp/portfolio.json",
        source_candidate_portfolio_sha256="b" * 64,
        source_query_plan=str(query_path),
        source_query_plan_sha256=hashlib.sha256(
            query_path.read_bytes()
        ).hexdigest(),
        claim_count=1,
        binding_ready_claim_count=1,
        novelty_bearing_binding_ready_claim_count=1,
        binding_status="READY_FOR_LITERAL_ENDPOINT_BINDING",
        claims=[claim],
    )
    body = {
        "schema_version": "relational-atomic-binding-plan-v1",
        "run_dir": "/tmp/run",
        "source_alpha6_candidate_portfolio": "/tmp/final.json",
        "source_alpha6_candidate_portfolio_sha256": "c" * 64,
        "source_certification_report": "/tmp/cert.json",
        "source_certification_report_sha256": "d" * 64,
        "hypotheses": [hyp.model_dump(mode="json")],
        "hypothesis_count": 1,
        "ready_hypothesis_count": 1,
        "not_ready_hypothesis_count": 0,
        "claim_count": 1,
        "binding_ready_claim_count": 1,
        "novelty_bearing_binding_ready_claim_count": 1,
        "hypothesis_status_counts": {
            "READY_FOR_LITERAL_ENDPOINT_BINDING": 1
        },
        "claim_status_counts": {
            "READY_FOR_LITERAL_ENDPOINT_BINDING": 1
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
    return (
        RelationalAtomicBindingPlan(
            **body,
            plan_id="relational_atomic_binding_plan:" + digest[:20],
            plan_sha256=digest,
        ),
        query_path,
    )


def test_rebuilt_plan_uses_actual_repaired_claim_hash(tmp_path: Path) -> None:
    plan, _ = _plan_with_paths(tmp_path)
    source_query = _source_query_plan()
    claim = plan.hypotheses[0].claims[0]
    sidecar, hashes = _build_query_plan_sidecar(
        source_plan=source_query,
        candidate_hypothesis_id="hypothesis:candidate",
        repaired_claims={claim.claim_id: claim},
    )
    sidecar_path = tmp_path / "sidecar.json"
    sidecar_path.write_text(
        json.dumps(sidecar.model_dump(mode="json"), sort_keys=True),
        encoding="utf-8",
    )
    rebuilt = _rebuild_binding_plan(
        source_plan=plan,
        sidecar_paths={"hypothesis:final": sidecar_path},
        sidecar_file_hashes={
            "hypothesis:final": hashlib.sha256(
                sidecar_path.read_bytes()
            ).hexdigest()
        },
        repaired_claim_hashes=hashes,
    )
    assert rebuilt.hypotheses[0].claims[0].source_claim_sha256 == (
        hashes["claim:1"]
    )


def _endpoint(plan: RelationalAtomicBindingPlan):
    claim = plan.hypotheses[0].claims[0]
    binding = CompiledLiteralEndpointBinding(
        candidate_hypothesis_id=claim.candidate_hypothesis_id,
        final_hypothesis_id=claim.final_hypothesis_id,
        claim_id=claim.claim_id,
        novelty_selection_role="NOVELTY_BEARING",
        source_claim_sha256=claim.source_claim_sha256,
        relation_endpoint_anchors=[
            "decreasing interparticle gap",
            "Raman intensity",
        ],
        outcome="BOUND_LITERAL_ENDPOINTS",
    )
    return RelationalAtomicEndpointBindingReport(
        report_id="endpoint:source",
        source_binding_plan_id=plan.plan_id,
        source_binding_plan_sha256=plan.plan_sha256,
        backend_name="test",
        model_name="test",
        bindings=[binding],
        selected_hypothesis_count=1,
        selected_claim_count=1,
        bound_claim_count=1,
        abstained_claim_count=0,
        novelty_bearing_bound_claim_count=1,
        llm_calls_performed=1,
    )


def test_endpoint_rebind_preserves_anchors_exactly(tmp_path: Path) -> None:
    plan, _ = _plan_with_paths(tmp_path)
    endpoint = _endpoint(plan)
    source_query = _source_query_plan()
    claim = plan.hypotheses[0].claims[0]
    sidecar, hashes = _build_query_plan_sidecar(
        source_plan=source_query,
        candidate_hypothesis_id="hypothesis:candidate",
        repaired_claims={claim.claim_id: claim},
    )
    sidecar_path = tmp_path / "sidecar.json"
    sidecar_path.write_text(
        json.dumps(sidecar.model_dump(mode="json"), sort_keys=True),
        encoding="utf-8",
    )
    rebound_plan = _rebuild_binding_plan(
        source_plan=plan,
        sidecar_paths={"hypothesis:final": sidecar_path},
        sidecar_file_hashes={
            "hypothesis:final": hashlib.sha256(
                sidecar_path.read_bytes()
            ).hexdigest()
        },
        repaired_claim_hashes=hashes,
    )
    rebound = _rebind_endpoint_report(
        source=endpoint,
        rebound_plan=rebound_plan,
    )
    assert (
        rebound.bindings[0].relation_endpoint_anchors
        == endpoint.bindings[0].relation_endpoint_anchors
    )
    assert rebound.llm_calls_performed == endpoint.llm_calls_performed
    assert rebound.source_binding_plan_id == rebound_plan.plan_id


def test_endpoint_rebind_rejects_anchor_not_literal_after_rebind(
    tmp_path: Path,
) -> None:
    plan, _ = _plan_with_paths(tmp_path)
    endpoint = _endpoint(plan)
    payload = plan.model_dump(mode="json")
    payload["hypotheses"][0]["claims"][0]["required_bridge"] = (
        "For substrate composition, a different relation appears."
    )
    body = dict(payload)
    body.pop("plan_id")
    body.pop("plan_sha256")
    digest = _sha(body)
    payload["plan_id"] = "relational_atomic_binding_plan:" + digest[:20]
    payload["plan_sha256"] = digest
    mutated = RelationalAtomicBindingPlan.model_validate(payload)

    try:
        _rebind_endpoint_report(
            source=endpoint,
            rebound_plan=mutated,
        )
    except ValueError as exc:
        assert "no longer literal in bridge" in str(exc)
    else:
        raise AssertionError("expected endpoint literalness rejection")
