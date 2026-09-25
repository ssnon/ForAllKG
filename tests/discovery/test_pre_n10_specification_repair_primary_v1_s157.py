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
from pipeline_core.discovery.pre_n10_specification_repair_primary_v1 import (
    execute_pre_n10_specification_repair_primary_v1,
)
from pipeline_core.discovery.preverifier_specification_repair_executor import (
    SpecificationRepairAuditDraft,
    SpecificationRepairDraft,
)


class FakeRepairBackend:
    def __init__(self, *, audit_passes: bool = True) -> None:
        self.audit_passes = audit_passes
        self.generation_calls = 0
        self.audit_calls = 0

    def generate(self, plan):
        self.generation_calls += 1
        return (
            SpecificationRepairDraft(
                claim_id=plan.claim_id,
                required_bridge=(
                    plan.source_claim_text
                    if "required_bridge" in plan.editable_fields
                    else plan.source_required_bridge
                ),
                predicted_observation=(
                    plan.source_claim_text
                    if "predicted_observation" in plan.editable_fields
                    else plan.source_predicted_observation
                ),
                falsification_condition=(
                    plan.source_claim_text
                    if "falsification_condition" in plan.editable_fields
                    else plan.source_falsification_condition
                ),
            ),
            object(),
        )

    def audit(self, *, plan, draft):
        self.audit_calls += 1
        value = self.audit_passes
        return (
            SpecificationRepairAuditDraft(
                claim_id=plan.claim_id,
                zero_scientific_delta=value,
                added_scientific_concepts=([] if value else ["new mechanism"]),
                new_mechanism_introduced=not value,
                rationale="fixture audit",
            ),
            object(),
        )


def _portfolio() -> HypothesisPortfolio:
    observable = "spacing disorder increases spatial SERS intensity variance"
    card = HypothesisCard(
        hypothesis_id="hypothesis:h1",
        domain_profile_id="sers_au_ag",
        source_context_id="context:1",
        source_context_sha256="a" * 64,
        source_report_id="report:1",
        source_report_sha256="b" * 64,
        title="Spacing disorder and SERS variance",
        hypothesis_statement=(
            "Spacing disorder may increase spatial SERS intensity variance."
        ),
        hypothesis_type="context_dependency",
        premise_statement_ids=["stmt:1"],
        inferential_bridge=observable,
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
                    "spacing disorder does not increase "
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


def _plan(*, missing_bridge: bool, role="NOVELTY_BEARING") -> LiteratureQueryPlan:
    observable = "spacing disorder increases spatial SERS intensity variance"
    claim = NoveltyClaim(
        claim_id="claim:1",
        hypothesis_id="hypothesis:h1",
        claim_rank=1,
        kind="mechanistic_link",
        importance="core",
        novelty_selection_role=role,
        text=observable,
        rationale="Bounded test claim.",
        search_concepts=["spacing disorder", "SERS variance"],
        search_queries=["spacing disorder SERS variance"],
        prior_art_identity_terms=["spacing disorder"],
        relation_nucleus_terms=[
            "spacing disorder",
            "spatial SERS intensity variance",
            "increases",
        ],
        required_bridge="" if missing_bridge else observable,
        predicted_observation=observable,
        falsification_condition=(
            "spacing disorder does not increase spatial SERS intensity variance"
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


def _setup(tmp_path: Path, *, missing_bridge: bool, role="NOVELTY_BEARING"):
    portfolio_path = tmp_path / "portfolio.json"
    plan_path = tmp_path / "claims_queries.json"
    _write(portfolio_path, _portfolio())
    _write(plan_path, _plan(missing_bridge=missing_bridge, role=role))
    contract = build_pre_n10_scientific_contract_v1(
        portfolio_path=portfolio_path,
        query_plan_path=plan_path,
        claim_decomposition_request_count=1,
    )
    assert contract.disposition == "INTERVENTION_REQUIRED"
    assert (
        contract.hypotheses[0].claims[0].router_hint
        == "SPECIFICATION_REPAIR_REVIEW"
    )
    return portfolio_path, plan_path, contract


def test_specification_repair_primary_recovers_missing_bridge(
    tmp_path: Path,
) -> None:
    portfolio_path, plan_path, contract = _setup(
        tmp_path,
        missing_bridge=True,
    )
    backend = FakeRepairBackend(audit_passes=True)

    repaired, post, report = execute_pre_n10_specification_repair_primary_v1(
        portfolio_path=portfolio_path,
        query_plan_path=plan_path,
        contract_report=contract,
        repair_backend=backend,
        output_root=tmp_path / "primary",
    )

    assert backend.generation_calls == 1
    assert backend.audit_calls == 1
    assert report.materialized_repair_count == 1
    assert report.recovered_for_n10_count == 1
    assert report.regeneration_fallback_required_count == 0
    assert post.disposition == "READY_FOR_N10"
    assert repaired.claims[0].claims[0].required_bridge == (
        "spacing disorder increases spatial SERS intensity variance"
    )
    assert report.n10_performed is False


def test_specification_repair_unsupported_metadata_fails_closed(
    tmp_path: Path,
) -> None:
    portfolio_path, plan_path, contract = _setup(
        tmp_path,
        missing_bridge=False,
        role=None,
    )
    backend = FakeRepairBackend(audit_passes=True)

    _repaired, post, report = execute_pre_n10_specification_repair_primary_v1(
        portfolio_path=portfolio_path,
        query_plan_path=plan_path,
        contract_report=contract,
        repair_backend=backend,
        output_root=tmp_path / "primary",
    )

    assert backend.generation_calls == 0
    assert backend.audit_calls == 0
    assert report.unsupported_automatic_repair_count == 1
    assert report.recovered_for_n10_count == 0
    assert report.regeneration_fallback_required_count == 1
    assert post.disposition == "INTERVENTION_REQUIRED"
    result = report.hypotheses[0].repair_results[0]
    assert "missing_novelty_selection_role" in result.unsupported_reason_codes


def test_specification_repair_semantic_delta_rejects_and_falls_back(
    tmp_path: Path,
) -> None:
    portfolio_path, plan_path, contract = _setup(
        tmp_path,
        missing_bridge=True,
    )
    backend = FakeRepairBackend(audit_passes=False)

    _repaired, post, report = execute_pre_n10_specification_repair_primary_v1(
        portfolio_path=portfolio_path,
        query_plan_path=plan_path,
        contract_report=contract,
        repair_backend=backend,
        output_root=tmp_path / "primary",
    )

    assert backend.generation_calls == 1
    assert backend.audit_calls == 1
    assert report.semantic_reject_count == 1
    assert report.materialized_repair_count == 0
    assert report.regeneration_fallback_required_count == 1
    assert post.disposition == "INTERVENTION_REQUIRED"
