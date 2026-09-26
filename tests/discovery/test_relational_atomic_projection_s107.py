from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from domains.registry import get_domain_profile
from pipeline_core.discovery.external_novelty_contracts import (
    HypothesisNoveltyClaims,
    LiteratureQueryPlan,
    NoveltyClaim,
    NoveltyClaimScientificStructure,
)
from pipeline_core.discovery.atomic_scientific_specification import (
    CompiledAtomicSpecification,
)
from pipeline_core.discovery.hypothesis_contracts import (
    FalsificationCriterion,
    HypothesisCard,
    HypothesisEvidenceProfile,
    HypothesisPortfolio,
    PredictedObservation,
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
from pipeline_core.discovery.relational_atomic_projection import (
    compile_relational_atomic_projection,
    compile_relational_atomic_projection_relation_ir,
)
from pipeline_core.discovery.scientific_relation_ir import (
    NullRelationTypingAdapter,
)


def _canonical_sha(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode()).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        value.model_dump(mode="json")
        if hasattr(value, "model_dump")
        else value
    )
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _claim(
    *,
    prediction: str = (
        "Substrate composition changes the Raman-intensity response "
        "to nanostructure spacing."
    ),
) -> NoveltyClaim:
    return NoveltyClaim(
        claim_id="external_novelty_claim:c1",
        hypothesis_id="hypothesis:candidate",
        claim_rank=1,
        kind="moderator_interaction",
        importance="core",
        novelty_selection_role="NOVELTY_BEARING",
        text=(
            "Substrate composition changes the Raman-intensity response "
            "to nanostructure spacing."
        ),
        rationale="Frozen canonical N10 claim.",
        prior_art_identity_terms=["Substrate composition"],
        relation_nucleus_terms=[
            "Raman intensity",
            "nanostructure spacing",
        ],
        required_bridge=(
            "Substrate composition changes the Raman-intensity response "
            "to nanostructure spacing."
        ),
        predicted_observation=prediction,
        falsification_condition=(
            "Substrate composition does not change the Raman-intensity "
            "response to nanostructure spacing."
        ),
        search_queries=["Raman intensity nanostructure spacing"],
    )


def _candidate_card(
    *,
    prediction_observable: str | None = None,
) -> HypothesisCard:
    prediction_observable = prediction_observable or (
        "Substrate composition changes the Raman-intensity response "
        "to nanostructure spacing."
    )
    return HypothesisCard(
        hypothesis_id="hypothesis:candidate",
        domain_profile_id="sers_au_ag",
        source_context_id="context:1",
        source_context_sha256="c" * 64,
        source_report_id="report:1",
        source_report_sha256="r" * 64,
        title="Title",
        hypothesis_statement="H",
        hypothesis_type="context_dependency",
        premise_statement_ids=["statement:1"],
        gap_statement_ids=[],
        inferential_bridge=(
            "Substrate composition changes the Raman-intensity response "
            "to nanostructure spacing."
        ),
        predicted_observations=[
            PredictedObservation(
                observation_id="prediction:1",
                observable=prediction_observable,
                expected_direction="unspecified",
                rationale=(
                    "The hypothesis predicts a composition-dependent "
                    "nanostructure-spacing response."
                ),
            )
        ],
        falsification_criteria=[
            FalsificationCriterion(
                criterion_id="falsifier:1",
                observable=prediction_observable,
                falsifying_outcome=(
                    "Substrate composition does not change the Raman-intensity "
                    "response to nanostructure spacing."
                ),
            )
        ],
        assumptions=[],
        source_paper_ids=["paper:1"],
        gap_paper_ids=[],
        cross_paper_synthesis=False,
        candidate_dependency="none",
        evidence_profile=HypothesisEvidenceProfile(
            premise_count=1,
            gap_count=0,
            source_paper_count=1,
            candidate_premise_count=0,
            reported_premise_count=1,
            synthesis_premise_count=0,
        ),
    )


def _fixture(
    tmp_path: Path,
    *,
    prediction_observable: str | None = None,
    binding_outcome: str = "BOUND_LITERAL_ENDPOINTS",
) -> tuple[
    RelationalAtomicBindingPlan,
    RelationalAtomicEndpointBindingReport,
]:
    candidate = _candidate_card(
        prediction_observable=prediction_observable
    )
    claim = _claim()
    portfolio = HypothesisPortfolio(
        portfolio_id="portfolio:candidate",
        domain_profile_id="sers_au_ag",
        source_context_id="context:1",
        source_context_sha256="c" * 64,
        source_report_id="report:1",
        source_report_sha256="r" * 64,
        hypotheses=[candidate],
    )
    query_plan = LiteratureQueryPlan(
        plan_id="plan:1",
        plan_sha256="p" * 64,
        source_portfolio_id=portfolio.portfolio_id,
        queries=[],
        claims=[
            HypothesisNoveltyClaims(
                hypothesis_id=candidate.hypothesis_id,
                title=candidate.title,
                claims=[claim],
            )
        ],
    )

    portfolio_path = tmp_path / "candidate.portfolio.json"
    query_path = tmp_path / "candidate.claims_queries.json"
    _write(portfolio_path, portfolio)
    _write(query_path, query_plan)

    claim_plan = RelationalAtomicBindingClaimPlan(
        candidate_hypothesis_id="hypothesis:candidate",
        final_hypothesis_id="hypothesis:final",
        claim_id=claim.claim_id,
        claim_rank=claim.claim_rank,
        kind=claim.kind,
        importance=claim.importance,
        novelty_selection_role=claim.novelty_selection_role,
        claim_text=claim.text,
        rationale=claim.rationale,
        prior_art_identity_terms=list(claim.prior_art_identity_terms),
        relation_nucleus_terms=list(claim.relation_nucleus_terms),
        required_bridge=claim.required_bridge,
        predicted_observation=claim.predicted_observation,
        falsification_condition=claim.falsification_condition,
        search_concepts=list(claim.search_concepts),
        search_queries=list(claim.search_queries),
        source_claim_sha256=_canonical_sha(claim),
        binding_status="READY_FOR_LITERAL_ENDPOINT_BINDING",
    )
    hypothesis_plan = RelationalAtomicBindingHypothesisPlan(
        original_hypothesis_id="hypothesis:original",
        candidate_hypothesis_id="hypothesis:candidate",
        final_hypothesis_id="hypothesis:final",
        alpha6_decision="accepted_refinement",
        certification_status="NOVELTY_UNRESOLVED",
        n10_selection_class="CONDITIONAL",
        source_candidate_portfolio=str(portfolio_path),
        source_candidate_portfolio_sha256=_file_sha(portfolio_path),
        source_query_plan=str(query_path),
        source_query_plan_sha256=_file_sha(query_path),
        claim_count=1,
        binding_ready_claim_count=1,
        novelty_bearing_binding_ready_claim_count=1,
        binding_status="READY_FOR_LITERAL_ENDPOINT_BINDING",
        claims=[claim_plan],
    )
    plan_body = {
        "schema_version": "relational-atomic-binding-plan-v1",
        "run_dir": str(tmp_path),
        "source_alpha6_candidate_portfolio": str(
            tmp_path / "final.portfolio.json"
        ),
        "source_alpha6_candidate_portfolio_sha256": "d" * 64,
        "source_certification_report": str(tmp_path / "cert.json"),
        "source_certification_report_sha256": "e" * 64,
        "hypotheses": [hypothesis_plan.model_dump(mode="json")],
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
    plan_digest = _canonical_sha(plan_body)
    plan = RelationalAtomicBindingPlan(
        **plan_body,
        plan_id="relational_atomic_binding_plan:" + plan_digest[:20],
        plan_sha256=plan_digest,
    )

    if binding_outcome == "BOUND_LITERAL_ENDPOINTS":
        binding = CompiledLiteralEndpointBinding(
            candidate_hypothesis_id="hypothesis:candidate",
            final_hypothesis_id="hypothesis:final",
            claim_id=claim.claim_id,
            novelty_selection_role="NOVELTY_BEARING",
            source_claim_sha256=claim_plan.source_claim_sha256,
            relation_endpoint_anchors=[
                "Raman-intensity",
                "nanostructure spacing",
            ],
            outcome="BOUND_LITERAL_ENDPOINTS",
        )
        bound = 1
        abstained = 0
        novelty_bound = 1
    else:
        binding = CompiledLiteralEndpointBinding(
            candidate_hypothesis_id="hypothesis:candidate",
            final_hypothesis_id="hypothesis:final",
            claim_id=claim.claim_id,
            novelty_selection_role="NOVELTY_BEARING",
            source_claim_sha256=claim_plan.source_claim_sha256,
            outcome="ABSTAINED_UNBINDABLE",
            abstention_reason="No exact endpoint pair.",
        )
        bound = 0
        abstained = 1
        novelty_bound = 0

    endpoint_report = RelationalAtomicEndpointBindingReport(
        report_id="endpoint_binding:r1",
        source_binding_plan_id=plan.plan_id,
        source_binding_plan_sha256=plan.plan_sha256,
        backend_name="test",
        model_name="test",
        bindings=[binding],
        selected_hypothesis_count=1,
        selected_claim_count=1,
        bound_claim_count=bound,
        abstained_claim_count=abstained,
        novelty_bearing_bound_claim_count=novelty_bound,
        llm_calls_performed=1,
    )
    return plan, endpoint_report


def test_projection_resolves_exact_observable_source_binding(
    tmp_path: Path,
) -> None:
    plan, endpoint = _fixture(tmp_path)
    report = compile_relational_atomic_projection(
        plan=plan,
        endpoint_report=endpoint,
    )
    assert report.projected_claim_count == 1
    row = report.rows[0]
    assert row.projection_status == "PROJECTED"
    assert row.specification is not None
    assert row.specification.observable == (
        "Substrate composition changes the Raman-intensity response "
        "to nanostructure spacing."
    )
    assert row.specification.prediction_observation_id == "prediction:1"
    assert row.specification.falsification_criterion_id == "falsifier:1"
    assert row.specification.relation_endpoint_anchors == [
        "Raman-intensity",
        "nanostructure spacing",
    ]
    assert row.specification.prior_art_identity_terms == [
        "Substrate composition"
    ]


def test_projection_abstains_when_prediction_observable_binding_is_not_exact(
    tmp_path: Path,
) -> None:
    plan, endpoint = _fixture(
        tmp_path,
        prediction_observable="Different prediction wording.",
    )
    report = compile_relational_atomic_projection(
        plan=plan,
        endpoint_report=endpoint,
    )
    assert report.projected_claim_count == 0
    assert report.source_binding_abstention_count == 1
    assert report.rows[0].projection_status == "ABSTAINED_SOURCE_BINDING"
    assert report.rows[0].reason_codes == [
        "prediction_exact_source_binding_cardinality:0"
    ]


def test_projection_preserves_endpoint_abstention(
    tmp_path: Path,
) -> None:
    plan, endpoint = _fixture(
        tmp_path,
        binding_outcome="ABSTAINED_UNBINDABLE",
    )
    report = compile_relational_atomic_projection(
        plan=plan,
        endpoint_report=endpoint,
    )
    assert report.projected_claim_count == 0
    assert report.skipped_endpoint_abstention_count == 1
    assert report.rows[0].projection_status == (
        "SKIPPED_ENDPOINT_ABSTENTION"
    )


def test_projection_rejects_source_artifact_mutation(
    tmp_path: Path,
) -> None:
    plan, endpoint = _fixture(tmp_path)
    source_path = Path(
        plan.hypotheses[0].source_candidate_portfolio
    )
    source_path.write_text(
        source_path.read_text(encoding="utf-8") + " ",
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError,
        match="changed after binding-plan freeze",
    ):
        compile_relational_atomic_projection(
            plan=plan,
            endpoint_report=endpoint,
        )


def test_relation_ir_records_relational_projection_source_contract(
    tmp_path: Path,
) -> None:
    plan, endpoint = _fixture(tmp_path)
    projection = compile_relational_atomic_projection(
        plan=plan,
        endpoint_report=endpoint,
    )
    relation_ir = compile_relational_atomic_projection_relation_ir(
        report=projection,
        domain_profile=get_domain_profile("sers_au_ag"),
        typing_adapter=NullRelationTypingAdapter(),
    )
    assert relation_ir.source_contract == (
        "relational-atomic-projection-report-v1"
    )
    assert relation_ir.source_atomic_report_id == projection.report_id
    assert relation_ir.relation_count == 1
    assert relation_ir.relations[0].source_contract == (
        "relational-atomic-projection-report-v1"
    )
    assert relation_ir.relations[0].hypothesis_id == "hypothesis:final"


def test_projection_report_hash_is_deterministic(
    tmp_path: Path,
) -> None:
    plan, endpoint = _fixture(tmp_path)
    first = compile_relational_atomic_projection(
        plan=plan,
        endpoint_report=endpoint,
    )
    second = compile_relational_atomic_projection(
        plan=plan,
        endpoint_report=endpoint,
    )
    assert first.report_id == second.report_id
    assert first.report_sha256 == second.report_sha256

def _canonical_specification(
    *,
    observable: str,
    prediction_id: str = "prediction:1",
    endpoints: list[str] | None = None,
) -> CompiledAtomicSpecification:
    claim = _claim()
    return CompiledAtomicSpecification(
        local_id="atomic:c1",
        claim_id=claim.claim_id,
        kind=claim.kind,
        importance=claim.importance,
        novelty_selection_role="NOVELTY_BEARING",
        text=claim.text,
        rationale=claim.rationale,
        source_candidate_ids=["candidate:source"],
        premise_statement_ids=["statement:1"],
        gap_statement_ids=[],
        prior_art_identity_terms=list(
            claim.prior_art_identity_terms
        ),
        relation_endpoint_anchors=(
            endpoints
            if endpoints is not None
            else ["Raman-intensity", "nanostructure spacing"]
        ),
        scope_qualifier_spans=[],
        directional_qualifier_spans=[],
        relation_nucleus_terms=list(claim.relation_nucleus_terms),
        distinguishing_terms=list(claim.distinguishing_terms),
        required_bridge=claim.required_bridge,
        observable=observable,
        predicted_observation=claim.predicted_observation,
        falsification_condition=claim.falsification_condition,
        prediction_observation_id=prediction_id,
        falsification_criterion_id="falsifier:1",
        search_concepts=list(claim.search_concepts),
        search_queries=list(claim.search_queries),
        scientific_structure=claim.scientific_structure,
        scientific_structure_reason_codes=list(
            claim.scientific_structure_reason_codes
        ),
    )


def test_canonical_source_ids_bypass_legacy_prediction_text_reverse_lookup(
    tmp_path: Path,
) -> None:
    source_observable = "Different prediction wording."
    plan, endpoint = _fixture(
        tmp_path,
        prediction_observable=source_observable,
    )

    legacy = compile_relational_atomic_projection(
        plan=plan,
        endpoint_report=endpoint,
    )
    assert legacy.projected_claim_count == 0
    assert legacy.rows[0].reason_codes == [
        "prediction_exact_source_binding_cardinality:0"
    ]

    canonical = _canonical_specification(
        observable=source_observable,
    )
    migrated = compile_relational_atomic_projection(
        plan=plan,
        endpoint_report=endpoint,
        canonical_specifications={
            canonical.claim_id: canonical,
        },
    )

    assert migrated.projected_claim_count == 1
    row = migrated.rows[0]
    assert row.projection_status == "PROJECTED"
    assert row.specification is not None
    assert row.specification.prediction_observation_id == "prediction:1"
    assert row.specification.falsification_criterion_id == "falsifier:1"
    assert row.specification.observable == source_observable
    assert row.specification.source_candidate_ids == [
        "candidate:source"
    ]


def test_canonical_source_id_path_fails_closed_on_unknown_prediction_id(
    tmp_path: Path,
) -> None:
    source_observable = "Different prediction wording."
    plan, endpoint = _fixture(
        tmp_path,
        prediction_observable=source_observable,
    )
    canonical = _canonical_specification(
        observable=source_observable,
        prediction_id="prediction:unknown",
    )

    report = compile_relational_atomic_projection(
        plan=plan,
        endpoint_report=endpoint,
        canonical_specifications={
            canonical.claim_id: canonical,
        },
    )

    assert report.projected_claim_count == 0
    assert report.rows[0].projection_status == (
        "ABSTAINED_SOURCE_BINDING"
    )
    assert report.rows[0].reason_codes == [
        "prediction_source_id_cardinality:0"
    ]


def test_canonical_path_rejects_endpoint_identity_drift(
    tmp_path: Path,
) -> None:
    source_observable = "Different prediction wording."
    plan, endpoint = _fixture(
        tmp_path,
        prediction_observable=source_observable,
    )
    canonical = _canonical_specification(
        observable=source_observable,
        endpoints=[
            "Substrate composition",
            "nanostructure spacing",
        ],
    )

    report = compile_relational_atomic_projection(
        plan=plan,
        endpoint_report=endpoint,
        canonical_specifications={
            canonical.claim_id: canonical,
        },
    )

    assert report.projected_claim_count == 0
    assert report.rows[0].reason_codes == [
        "canonical_specification_literal_binding_mismatch:"
        "relation_endpoint_anchors"
    ]


def test_canonical_mapping_rejects_unknown_claim_population(
    tmp_path: Path,
) -> None:
    plan, endpoint = _fixture(tmp_path)
    canonical = _canonical_specification(
        observable=(
            "Substrate composition changes the Raman-intensity response "
            "to nanostructure spacing."
        ),
    )

    with pytest.raises(
        ValueError,
        match="canonical specification population contains unknown claims",
    ):
        compile_relational_atomic_projection(
            plan=plan,
            endpoint_report=endpoint,
            canonical_specifications={
                "external_novelty_claim:unknown": canonical,
            },
        )
