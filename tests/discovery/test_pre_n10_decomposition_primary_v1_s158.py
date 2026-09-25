from __future__ import annotations

import json
from pathlib import Path

from pipeline_core.discovery.external_novelty_contracts import (
    HypothesisNoveltyClaims,
    LiteratureQuery,
    LiteratureQueryPlan,
    NoveltyClaim,
)
from pipeline_core.discovery.hypothesis_contracts import (
    FalsificationCriterion,
    HypothesisCard,
    HypothesisEvidenceProfile,
    HypothesisPortfolio,
    PredictedObservation,
)
from pipeline_core.discovery.pre_n10_decomposition_primary_v1 import (
    execute_pre_n10_decomposition_primary_v1,
)
from pipeline_core.discovery.pre_n10_scientific_contract_v1 import (
    build_pre_n10_scientific_contract_v1,
)


def _portfolio() -> HypothesisPortfolio:
    observable = "surface state changes optical signal"
    card = HypothesisCard(
        hypothesis_id="hypothesis:h1",
        domain_profile_id="sers_au_ag",
        source_context_id="context:1",
        source_context_sha256="a" * 64,
        source_report_id="report:1",
        source_report_sha256="b" * 64,
        title="Surface state and optical signal",
        hypothesis_statement="Surface state may change optical signal.",
        hypothesis_type="mechanistic_extension",
        premise_statement_ids=["stmt:1"],
        inferential_bridge=observable,
        predicted_observations=[
            PredictedObservation(
                observation_id="observation:1",
                observable=observable,
                expected_direction="qualitative_change",
                rationale="Tests the relation.",
            )
        ],
        falsification_criteria=[
            FalsificationCriterion(
                criterion_id="criterion:1",
                observable=observable,
                falsifying_outcome="surface state does not change optical signal",
            )
        ],
        source_paper_ids=["paper:1"],
        evidence_profile=HypothesisEvidenceProfile(
            premise_count=1,
            gap_count=0,
            source_paper_count=1,
            candidate_premise_count=0,
            reported_premise_count=1,
            synthesis_premise_count=0,
        ),
    )
    return HypothesisPortfolio(
        portfolio_id="hypothesis_portfolio:p1",
        domain_profile_id="sers_au_ag",
        source_context_id="context:1",
        source_context_sha256="a" * 64,
        source_report_id="report:1",
        source_report_sha256="b" * 64,
        hypotheses=[card],
    )


def _claim(
    claim_id: str,
    *,
    kind: str,
    role: str,
    rank: int,
    components: list[str] | None = None,
    basis: list[str] | None = None,
) -> NoveltyClaim:
    observable = "surface state changes optical signal"
    return NoveltyClaim(
        claim_id=claim_id,
        hypothesis_id="hypothesis:h1",
        claim_rank=rank,
        kind=kind,
        importance="core",
        novelty_selection_role=role,
        text=observable,
        rationale="Bounded relation.",
        search_concepts=["surface state", "optical signal"],
        search_queries=["surface state optical signal"],
        prior_art_identity_terms=["surface state"],
        relation_nucleus_terms=["surface state", "optical signal", "changes"],
        required_bridge=observable,
        predicted_observation=observable,
        falsification_condition="surface state does not change optical signal",
        higher_order_component_claim_ids=list(components or []),
        higher_order_relation_basis=list(basis or []),
    )


def _plan(*, available: bool) -> LiteratureQueryPlan:
    a = _claim(
        "claim:a",
        kind="mechanistic_link",
        role="NOVELTY_BEARING",
        rank=1,
    )
    b = _claim(
        "claim:b",
        kind="context_condition",
        role="REQUIRED_ENABLING_RELATION",
        rank=2,
    )
    composite = _claim(
        "claim:c",
        kind="composite",
        role="REQUIRED_ENABLING_RELATION",
        rank=3,
        components=(
            ["claim:a", "claim:b"]
            if available
            else ["claim:a"]
        ),
        basis=["Explicit higher-order relation."],
    )
    return LiteratureQueryPlan(
        plan_id="literature_query_plan:q1",
        plan_sha256="c" * 64,
        source_portfolio_id="hypothesis_portfolio:p1",
        queries=[
            LiteratureQuery(
                query_id="query:a",
                hypothesis_id="hypothesis:h1",
                claim_id="claim:a",
                query_kind="claim_primary",
                query_text="surface state optical signal a",
            ),
            LiteratureQuery(
                query_id="query:b",
                hypothesis_id="hypothesis:h1",
                claim_id="claim:b",
                query_kind="claim_primary",
                query_text="surface state optical signal b",
            ),
            LiteratureQuery(
                query_id="query:c",
                hypothesis_id="hypothesis:h1",
                claim_id="claim:c",
                query_kind="claim_primary",
                query_text="surface state optical signal composite",
            ),
        ],
        claims=[
            HypothesisNoveltyClaims(
                hypothesis_id="hypothesis:h1",
                title="Surface state and optical signal",
                claims=[a, b, composite],
            )
        ],
    )


def _write(path: Path, value: object) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _setup(tmp_path: Path, *, available: bool):
    portfolio_path = tmp_path / "portfolio.json"
    plan_path = tmp_path / "claims_queries.json"
    _write(portfolio_path, _portfolio())
    _write(plan_path, _plan(available=available))
    contract = build_pre_n10_scientific_contract_v1(
        portfolio_path=portfolio_path,
        query_plan_path=plan_path,
        claim_decomposition_request_count=1,
    )
    assert contract.disposition == "INTERVENTION_REQUIRED"
    rows = contract.hypotheses[0].claims
    composite = next(row for row in rows if row.claim_id == "claim:c")
    assert composite.router_hint == "DECOMPOSE_OR_REGENERATE_REVIEW"
    return portfolio_path, plan_path, contract


def test_deterministic_decomposition_recovers_pre_n10_contract(
    tmp_path: Path,
) -> None:
    portfolio_path, plan_path, contract = _setup(
        tmp_path,
        available=True,
    )

    decomposed, post, report = execute_pre_n10_decomposition_primary_v1(
        portfolio_path=portfolio_path,
        query_plan_path=plan_path,
        contract_report=contract,
        output_root=tmp_path / "primary",
    )

    assert report.materialized_decomposition_count == 1
    assert report.unavailable_decomposition_count == 0
    assert report.removed_composite_claim_count == 1
    assert report.recovered_for_n10_count == 1
    assert report.regeneration_fallback_required_count == 0
    assert post.disposition == "READY_FOR_N10"
    assert [row.claim_id for row in decomposed.claims[0].claims] == [
        "claim:a",
        "claim:b",
    ]
    assert [row.claim_id for row in decomposed.queries] == [
        "claim:a",
        "claim:b",
    ]
    assert report.decomposition_llm_calls == 0
    assert report.n10_performed is False


def test_unavailable_decomposition_preserves_claim_and_requires_regeneration(
    tmp_path: Path,
) -> None:
    portfolio_path, plan_path, contract = _setup(
        tmp_path,
        available=False,
    )

    decomposed, post, report = execute_pre_n10_decomposition_primary_v1(
        portfolio_path=portfolio_path,
        query_plan_path=plan_path,
        contract_report=contract,
        output_root=tmp_path / "primary",
    )

    assert report.materialized_decomposition_count == 0
    assert report.unavailable_decomposition_count == 1
    assert report.removed_composite_claim_count == 0
    assert report.recovered_for_n10_count == 0
    assert report.regeneration_fallback_required_count == 1
    assert post.disposition == "INTERVENTION_REQUIRED"
    assert [row.claim_id for row in decomposed.claims[0].claims] == [
        "claim:a",
        "claim:b",
        "claim:c",
    ]
    assert [row.claim_id for row in decomposed.queries] == [
        "claim:a",
        "claim:b",
        "claim:c",
    ]
    result = report.hypotheses[0].decomposition_results[0]
    assert (
        "insufficient_explicit_component_cardinality:1"
        in result.assessment.reason_codes
    )


def test_decomposition_primary_is_write_once_and_replay_exact(
    tmp_path: Path,
) -> None:
    portfolio_path, plan_path, contract = _setup(
        tmp_path,
        available=True,
    )
    root = tmp_path / "primary"

    first = execute_pre_n10_decomposition_primary_v1(
        portfolio_path=portfolio_path,
        query_plan_path=plan_path,
        contract_report=contract,
        output_root=root,
    )
    second = execute_pre_n10_decomposition_primary_v1(
        portfolio_path=portfolio_path,
        query_plan_path=plan_path,
        contract_report=contract,
        output_root=root,
    )

    assert first[0].model_dump(mode="json") == second[0].model_dump(mode="json")
    assert first[1].model_dump(mode="json") == second[1].model_dump(mode="json")
    assert first[2].model_dump(mode="json") == second[2].model_dump(mode="json")
