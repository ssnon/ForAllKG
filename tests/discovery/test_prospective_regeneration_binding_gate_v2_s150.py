from pathlib import Path
import hashlib
import json

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
from pipeline_core.discovery.preverifier_contract_gate_v2 import (
    build_preverifier_contract_gate_v2,
)
from pipeline_core.discovery.prospective_regeneration_binding_gate_v2 import (
    build_compatibility_portfolio_v2,
    gate_ready_hypothesis_ids,
)
from pipeline_core.discovery.relational_atomic_binding_plan import (
    RelationalAtomicBindingHypothesisPlan,
    RelationalAtomicBindingPlan,
    assess_claim_binding_readiness,
)


def _card(hid="hypothesis:regen"):
    return HypothesisCard(
        hypothesis_id=hid,
        domain_profile_id="sers_au_ag",
        source_context_id="context:1",
        source_context_sha256="a" * 64,
        source_report_id="report:1",
        source_report_sha256="b" * 64,
        title="test",
        hypothesis_statement="signal changes",
        hypothesis_type="mechanistic_extension",
        premise_statement_ids=["statement:1"],
        inferential_bridge="signal bridge",
        predicted_observations=[
            PredictedObservation(
                observation_id="prediction:1",
                observable="signal",
                expected_direction="qualitative_change",
                rationale="signal rationale",
            )
        ],
        falsification_criteria=[
            FalsificationCriterion(
                criterion_id="falsifier:1",
                observable="signal",
                falsifying_outcome="signal",
            )
        ],
        evidence_profile=HypothesisEvidenceProfile(
            premise_count=1,
            gap_count=0,
            source_paper_count=1,
            candidate_premise_count=0,
            reported_premise_count=1,
            synthesis_premise_count=0,
        ),
    )


def _portfolio(hid="hypothesis:regen"):
    return HypothesisPortfolio(
        portfolio_id="portfolio:" + hid.split(":")[-1],
        domain_profile_id="sers_au_ag",
        source_context_id="context:1",
        source_context_sha256="a" * 64,
        source_report_id="report:1",
        source_report_sha256="b" * 64,
        hypotheses=[_card(hid)],
    )


def _claim(hid="hypothesis:regen"):
    return NoveltyClaim(
        claim_id="claim:1",
        hypothesis_id=hid,
        claim_rank=1,
        kind="mechanistic_link",
        importance="core",
        novelty_selection_role="NOVELTY_BEARING",
        text="signal relation",
        rationale="signal rationale",
        search_concepts=["signal"],
        search_queries=["signal"],
        prior_art_identity_terms=["signal"],
        relation_nucleus_terms=["signal"],
        required_bridge="signal bridge",
        predicted_observation="signal",
        falsification_condition="signal",
    )


def _sha_json(value):
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()


def test_compatibility_portfolio_preserves_cards_without_mutation():
    a = _portfolio("hypothesis:a")
    b = _portfolio("hypothesis:b")
    combined = build_compatibility_portfolio_v2(portfolios=[a, b])
    assert [row.hypothesis_id for row in combined.hypotheses] == [
        "hypothesis:a",
        "hypothesis:b",
    ]
    assert combined.hypotheses[0] == a.hypotheses[0]
    assert combined.hypotheses[1] == b.hypotheses[0]


def test_regenerated_unresolved_candidate_can_still_be_strict_gate_ready(tmp_path: Path):
    portfolio = _portfolio()
    claim = _claim()
    query_plan = LiteratureQueryPlan(
        plan_id="query-plan:1",
        plan_sha256="c" * 64,
        source_portfolio_id=portfolio.portfolio_id,
        claims=[
            HypothesisNoveltyClaims(
                hypothesis_id="hypothesis:regen",
                title="test",
                claims=[claim],
            )
        ],
    )
    portfolio_path = tmp_path / "portfolio.json"
    query_path = tmp_path / "query.json"
    portfolio_path.write_text(portfolio.model_dump_json(), encoding="utf-8")
    query_path.write_text(query_plan.model_dump_json(), encoding="utf-8")

    claim_plan = assess_claim_binding_readiness(
        claim=claim,
        candidate_hypothesis_id="hypothesis:regen",
        final_hypothesis_id="hypothesis:regen",
    )
    assert claim_plan.reason_codes == []

    hypothesis = RelationalAtomicBindingHypothesisPlan(
        original_hypothesis_id="hypothesis:old",
        candidate_hypothesis_id="hypothesis:regen",
        final_hypothesis_id="hypothesis:regen",
        alpha6_decision="regeneration_v2",
        certification_status="NOVELTY_UNRESOLVED",
        n10_selection_class="CONDITIONAL",
        source_candidate_portfolio=str(portfolio_path),
        source_candidate_portfolio_sha256=hashlib.sha256(
            portfolio_path.read_bytes()
        ).hexdigest(),
        source_query_plan=str(query_path),
        source_query_plan_sha256=hashlib.sha256(
            query_path.read_bytes()
        ).hexdigest(),
        claim_count=1,
        binding_ready_claim_count=1,
        novelty_bearing_binding_ready_claim_count=1,
        binding_status="READY_FOR_LITERAL_ENDPOINT_BINDING",
        claims=[claim_plan],
    )
    body = {
        "schema_version": "relational-atomic-binding-plan-v1",
        "run_dir": str(tmp_path),
        "source_alpha6_candidate_portfolio": str(portfolio_path),
        "source_alpha6_candidate_portfolio_sha256": hashlib.sha256(
            portfolio_path.read_bytes()
        ).hexdigest(),
        "source_certification_report": str(query_path),
        "source_certification_report_sha256": hashlib.sha256(
            query_path.read_bytes()
        ).hexdigest(),
        "hypotheses": [hypothesis.model_dump(mode="json")],
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
    digest = _sha_json(body)
    plan = RelationalAtomicBindingPlan(
        **body,
        plan_id="relational_atomic_binding_plan:" + digest[:20],
        plan_sha256=digest,
    )
    gate = build_preverifier_contract_gate_v2(plan=plan)
    assert gate.ready_claim_count == 1
    assert gate.novelty_bearing_ready_claim_count == 1
    assert gate_ready_hypothesis_ids(gate) == {"hypothesis:regen"}


def test_gate_reachability_helper_requires_novelty_bearing_role(tmp_path: Path):
    assert _claim().novelty_selection_role == "NOVELTY_BEARING"
