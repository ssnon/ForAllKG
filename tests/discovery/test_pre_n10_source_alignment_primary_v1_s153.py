from __future__ import annotations

import json
from pathlib import Path

from pipeline_core.discovery.external_novelty_contracts import (
    HypothesisNoveltyClaims,
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
from pipeline_core.discovery.pre_n10_scientific_contract_v1 import (
    build_pre_n10_scientific_contract_v1,
)
from pipeline_core.discovery.pre_n10_source_alignment_primary_v1 import (
    execute_pre_n10_source_alignment_primary_v1,
)
from pipeline_core.discovery.prospective_routed_primary_materializer_v2 import (
    SourceAlignmentAuditDraft,
)


class FakeAuditBackend:
    backend_name = "fake"
    model_name = "fake"

    def __init__(self, *, passes: bool = True) -> None:
        self.passes = passes
        self.calls = 0

    def audit(self, *, claim, candidate, candidate_card):
        self.calls += 1
        value = self.passes
        return (
            SourceAlignmentAuditDraft(
                claim_id=claim.claim_id,
                zero_scientific_delta=value,
                same_relation_commitment=value,
                same_scope=value,
                same_direction=value,
                no_new_mechanism=value,
                no_new_moderator=value,
                no_new_scientific_entity=value,
                rationale="test audit",
            ),
            object(),
        )


def _portfolio() -> HypothesisPortfolio:
    observable = (
        "spacing disorder increases spatial SERS intensity variance"
    )
    card = HypothesisCard(
        hypothesis_id="hypothesis:h1",
        domain_profile_id="sers_au_ag",
        source_context_id="context:1",
        source_context_sha256="a" * 64,
        source_report_id="report:1",
        source_report_sha256="b" * 64,
        title="Spacing disorder and SERS variance",
        hypothesis_statement=(
            "Spacing disorder may change spatial SERS intensity variance."
        ),
        hypothesis_type="context_dependency",
        premise_statement_ids=["stmt:1"],
        inferential_bridge=(
            "spacing disorder changes spatial SERS intensity variance"
        ),
        predicted_observations=[
            PredictedObservation(
                observation_id="observation:1",
                observable=observable,
                expected_direction="increase",
                rationale="Tests the relation.",
            )
        ],
        falsification_criteria=[
            FalsificationCriterion(
                criterion_id="criterion:1",
                observable=observable,
                falsifying_outcome=(
                    "spacing disorder does not alter "
                    "spatial SERS intensity variance"
                ),
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


def _plan() -> LiteratureQueryPlan:
    claim = NoveltyClaim(
        claim_id="claim:1",
        hypothesis_id="hypothesis:h1",
        claim_rank=1,
        kind="mechanistic_link",
        importance="core",
        novelty_selection_role="NOVELTY_BEARING",
        text="spacing disorder changes spatial SERS intensity variance",
        rationale="Bounded test claim.",
        search_concepts=["spacing disorder", "SERS variance"],
        search_queries=["spacing disorder SERS variance"],
        prior_art_identity_terms=["spacing disorder"],
        relation_nucleus_terms=[
            "spacing disorder",
            "spatial SERS intensity variance",
            "changes",
        ],
        required_bridge=(
            "spacing disorder changes spatial SERS intensity variance"
        ),
        predicted_observation=(
            "spacing disorder changes mean SERS intensity"
        ),
        falsification_condition=(
            "spacing disorder does not change mean SERS intensity"
        ),
    )
    return LiteratureQueryPlan(
        plan_id="literature_query_plan:q1",
        plan_sha256="c" * 64,
        source_portfolio_id="hypothesis_portfolio:p1",
        claims=[
            HypothesisNoveltyClaims(
                hypothesis_id="hypothesis:h1",
                title="Spacing disorder and SERS variance",
                claims=[claim],
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


def _setup(tmp_path: Path):
    portfolio_path = tmp_path / "portfolio.json"
    plan_path = tmp_path / "claims_queries.json"
    _write(portfolio_path, _portfolio())
    _write(plan_path, _plan())
    contract = build_pre_n10_scientific_contract_v1(
        portfolio_path=portfolio_path,
        query_plan_path=plan_path,
        claim_decomposition_request_count=1,
    )
    assert contract.disposition == "INTERVENTION_REQUIRED"
    assert (
        contract.hypotheses[0].claims[0].router_hint
        == "SOURCE_CONTRACT_ALIGNMENT_OR_REGENERATE_REVIEW"
    )
    return portfolio_path, plan_path, contract


def test_source_alignment_primary_recovers_pre_n10_contract(
    tmp_path: Path,
):
    portfolio_path, plan_path, contract = _setup(tmp_path)
    backend = FakeAuditBackend(passes=True)

    aligned, post, report = execute_pre_n10_source_alignment_primary_v1(
        portfolio_path=portfolio_path,
        query_plan_path=plan_path,
        contract_report=contract,
        audit_backend=backend,
        output_root=tmp_path / "primary",
    )

    assert backend.calls == 1
    assert report.materialized_alignment_count == 1
    assert report.recovered_for_n10_count == 1
    assert report.regeneration_fallback_required_count == 0
    assert post.disposition == "READY_FOR_N10"
    claim = aligned.claims[0].claims[0]
    assert claim.predicted_observation == (
        "spacing disorder increases spatial SERS intensity variance"
    )
    assert claim.falsification_condition == (
        "spacing disorder does not alter spatial SERS intensity variance"
    )
    assert report.retrieval_performed is False
    assert report.n10_performed is False


def test_source_alignment_semantic_reject_requires_regeneration_fallback(
    tmp_path: Path,
):
    portfolio_path, plan_path, contract = _setup(tmp_path)
    backend = FakeAuditBackend(passes=False)

    _aligned, post, report = execute_pre_n10_source_alignment_primary_v1(
        portfolio_path=portfolio_path,
        query_plan_path=plan_path,
        contract_report=contract,
        audit_backend=backend,
        output_root=tmp_path / "primary",
    )

    assert backend.calls == 1
    assert report.semantic_reject_count == 1
    assert report.recovered_for_n10_count == 0
    assert report.regeneration_fallback_required_count == 1
    assert post.disposition == "INTERVENTION_REQUIRED"


def test_source_alignment_primary_is_write_once_and_replay_exact(
    tmp_path: Path,
):
    portfolio_path, plan_path, contract = _setup(tmp_path)
    backend = FakeAuditBackend(passes=True)
    root = tmp_path / "primary"

    first = execute_pre_n10_source_alignment_primary_v1(
        portfolio_path=portfolio_path,
        query_plan_path=plan_path,
        contract_report=contract,
        audit_backend=backend,
        output_root=root,
    )
    second_backend = FakeAuditBackend(passes=True)
    second = execute_pre_n10_source_alignment_primary_v1(
        portfolio_path=portfolio_path,
        query_plan_path=plan_path,
        contract_report=contract,
        audit_backend=second_backend,
        output_root=root,
    )

    assert first[0].model_dump(mode="json") == second[0].model_dump(mode="json")
    assert first[1].model_dump(mode="json") == second[1].model_dump(mode="json")
    assert first[2].model_dump(mode="json") == second[2].model_dump(mode="json")
