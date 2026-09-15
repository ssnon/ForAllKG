from __future__ import annotations

from collections import deque

from pipeline_core.discovery.higher_order_hypothesis_batch import (
    HigherOrderShadowBatchRuntime,
)
from pipeline_core.discovery.higher_order_shadow_synthesis import (
    select_shadow_synthesis_contexts,
)
from pipeline_core.discovery.higher_order_synthesis_context import (
    HigherOrderStructuralOpportunityView,
    HigherOrderSynthesisContext,
    HigherOrderSynthesisPremiseView,
)
from pipeline_core.discovery.higher_order_topology_carrier import (
    HigherOrderSynthesisLineageView,
)
from pipeline_core.discovery.hypothesis_contracts import (
    FalsificationCriterionDraft,
    HypothesisContext,
    HypothesisEvidenceStatement,
    HypothesisPortfolioDraft,
    HypothesisProposalDraft,
    PredictedObservationDraft,
)
from pipeline_core.discovery.hypothesis_llm import (
    HypothesisDraftGeneration,
)
from pipeline_core.discovery.relation_component_composition import (
    RelationComponentAuthority,
)


class QueueBackend:
    backend_name = "fake_higher_order_batch_backend"
    model_name = "fake-model"
    temperature = 0.0
    instructor_mode = "FAKE"
    base_url = "fake://local"
    parse_retries = 0

    def __init__(
        self,
        drafts: list[HypothesisPortfolioDraft],
    ) -> None:
        self.drafts = deque(drafts)
        self.generate_calls = 0
        self.prompts = []

    def generate(self, prompt):
        self.generate_calls += 1
        self.prompts.append(prompt)
        if not self.drafts:
            raise AssertionError("no queued generation draft")
        return HypothesisDraftGeneration(
            draft=self.drafts.popleft(),
            input_tokens=10,
            output_tokens=5,
        )

    def repair(self, prompt, previous_draft, feedback):
        raise AssertionError("unexpected repair call")


def _source_context() -> HypothesisContext:
    return HypothesisContext(
        context_id="hypothesis_context:source",
        context_sha256="source-context-sha",
        source_packet_id="packet:test",
        source_packet_sha256="packet-sha",
        source_report_id="report:test",
        source_report_sha256="report-sha",
        task_id="task:test",
        question=(
            "How does interparticle separation affect SERS intensity?"
        ),
        corpus_id="corpus:test",
        domain_profile_id="domain:test",
        evidence_statements=[
            HypothesisEvidenceStatement(
                statement_id="stmt:positive:1",
                text=(
                    "Changing interparticle separation changes optical "
                    "coupling and measured response."
                ),
                epistemic_role="reported",
                claim_kind="mechanistic_relation",
                paper_ids=["paper:1"],
                requires_verification=False,
                eligible_as_premise=True,
                eligible_as_gap=False,
            )
        ],
    )


def _context(
    *,
    context_id: str,
    modifier_component_id: str,
    modifier_text: str,
) -> HigherOrderSynthesisContext:
    topology_id = "higher_order_topology:" + context_id.split(":")[-1]
    carrier_id = "higher_order_synthesis_carrier:" + context_id.split(":")[-1]
    modifier_provenance = (
        "accepted_pattern:" + modifier_component_id.split(":")[-1]
    )

    lineage = HigherOrderSynthesisLineageView(
        higher_order_topology_id=topology_id,
        backbone_topology_id="relational_topology:shared",
        source_component_id="component:source",
        target_component_id="component:target",
        modifier_component_id=modifier_component_id,
        source_authority=RelationComponentAuthority.CONFIRMED_KNOWN,
        target_authority=RelationComponentAuthority.CONFIRMED_KNOWN,
        modifier_authority=RelationComponentAuthority.CONFIRMED_KNOWN,
        source_provenance_source_id="accepted_pattern:source",
        target_provenance_source_id="accepted_pattern:target",
        modifier_provenance_source_id=modifier_provenance,
        source_endpoint_binding_authority="exact",
        target_endpoint_binding_authority="exact",
        source_endpoint_equivalence_witness_id=None,
        target_endpoint_equivalence_witness_id=None,
        modifier_eligibility_witness_id=(
            "modifier_witness:" + modifier_component_id.split(":")[-1]
        ),
        modifier_anchor_role="mediator",
        modifier_anchor_slot="object",
        modifier_slot="subject",
        modifier_anchor_text="electric-field enhancement",
        modifier_text=modifier_text,
    )

    return HigherOrderSynthesisContext(
        context_id=context_id,
        carrier_id=carrier_id,
        higher_order_topology_id=topology_id,
        requested_source="interparticle separation",
        requested_target="SERS intensity",
        lineage=lineage,
        premises=[
            HigherOrderSynthesisPremiseView(
                premise_id="ho_premise:source:" + context_id,
                premise_role="source_backbone_relation",
                component_id="component:source",
                subject="interparticle separation",
                relation="VARIES_WITH",
                object="electric-field enhancement",
                authority=RelationComponentAuthority.CONFIRMED_KNOWN,
                provenance_source_id="accepted_pattern:source",
                epistemic_use="confirmed_known_component",
            ),
            HigherOrderSynthesisPremiseView(
                premise_id="ho_premise:target:" + context_id,
                premise_role="target_backbone_relation",
                component_id="component:target",
                subject="SERS intensity",
                relation="VARIES_WITH",
                object="electric-field enhancement",
                authority=RelationComponentAuthority.CONFIRMED_KNOWN,
                provenance_source_id="accepted_pattern:target",
                epistemic_use="confirmed_known_component",
            ),
            HigherOrderSynthesisPremiseView(
                premise_id="ho_premise:modifier:" + context_id,
                premise_role="modifier_relation",
                component_id=modifier_component_id,
                subject=modifier_text,
                relation="PROMOTES",
                object="electric-field enhancement",
                authority=RelationComponentAuthority.CONFIRMED_KNOWN,
                provenance_source_id=modifier_provenance,
                epistemic_use="confirmed_known_component",
            ),
        ],
        structural_opportunity=HigherOrderStructuralOpportunityView(
            requested_source="interparticle separation",
            requested_target="SERS intensity",
            source_side_mediator_text="electric-field enhancement",
            target_side_mediator_text="electric-field enhancement",
            modifier_text=modifier_text,
            modifier_anchor_role="mediator",
            modifier_anchor_text="electric-field enhancement",
        ),
    )


def _proposal(
    *,
    local_id: str = "h1",
    premise_id: str = "stmt:positive:1",
) -> HypothesisProposalDraft:
    observable = (
        "the qualitative separation-response relationship "
        "under the modifier condition"
    )
    return HypothesisProposalDraft(
        local_id=local_id,
        title="Modifier-conditioned separation response",
        hypothesis_statement=(
            "The modifier may condition how interparticle separation "
            "relates to SERS intensity."
        ),
        hypothesis_type="context_dependency",
        premise_statement_ids=[premise_id],
        gap_statement_ids=[],
        inferential_bridge=(
            "The higher-order structure motivates testing a bounded "
            "context dependency without treating it as established evidence."
        ),
        predicted_observations=[
            PredictedObservationDraft(
                local_id="p1",
                observable=observable,
                expected_direction="unspecified",
                rationale=(
                    "A modifier-dependent qualitative response would be "
                    "consistent with the proposed interaction."
                ),
            )
        ],
        falsification_criteria=[
            FalsificationCriterionDraft(
                local_id="f1",
                observable=observable,
                falsifying_outcome=(
                    "The qualitative separation-response relationship "
                    "does not vary with the modifier condition."
                ),
            )
        ],
        assumptions=[
            "The higher-order interaction remains unverified."
        ],
    )


def _valid_draft(
    *,
    local_id: str = "h1",
) -> HypothesisPortfolioDraft:
    return HypothesisPortfolioDraft(
        hypotheses=[_proposal(local_id=local_id)]
    )


def _abstain_draft() -> HypothesisPortfolioDraft:
    return HypothesisPortfolioDraft(
        hypotheses=[],
        abstention_reason=(
            "The supplied evidence is insufficient for a bounded "
            "higher-order hypothesis."
        ),
    )


def _pool():
    # Same modifier, duplicate backbone realization. Lexicographically smaller
    # context_id must be retained by the existing coverage selector.
    edge_b = _context(
        context_id="higher_order_synthesis_context:edge-b",
        modifier_component_id="component:edge",
        modifier_text="nanoparticle edge sharpness",
    )
    edge_a = _context(
        context_id="higher_order_synthesis_context:edge-a",
        modifier_component_id="component:edge",
        modifier_text="nanoparticle edge sharpness",
    )
    size = _context(
        context_id="higher_order_synthesis_context:size",
        modifier_component_id="component:size",
        modifier_text="particle size",
    )
    return [edge_b, size, edge_a]


def test_batch_reuses_deterministic_modifier_coverage_selection() -> None:
    contexts = _pool()
    selected = select_shadow_synthesis_contexts(
        contexts=contexts,
        max_contexts=100,
    )

    backend = QueueBackend(
        [_valid_draft(local_id="h1"), _valid_draft(local_id="h2")]
    )
    outcome = HigherOrderShadowBatchRuntime(
        backend,
        max_repairs=0,
    ).run(
        source_context=_source_context(),
        contexts=contexts,
        max_contexts=100,
    )

    expected_ids = [row.context_id for row in selected]
    actual_ids = outcome.record.selected_higher_order_context_ids

    assert actual_ids == expected_ids
    assert len(actual_ids) == 2
    assert "higher_order_synthesis_context:edge-a" in actual_ids
    assert "higher_order_synthesis_context:edge-b" not in actual_ids

    assert outcome.record.input_context_count == 3
    assert outcome.record.selected_context_count == 2
    assert outcome.record.unique_modifier_count == 2
    assert outcome.record.duplicate_or_over_quota_context_count == 1
    assert (
        outcome.record.selection_policy
        == "deterministic_modifier_coverage_no_quality_ranking"
    )
    assert outcome.record.scientific_quality_ranking_performed is False


def test_batch_runs_selected_arms_independently_without_portfolio_merge() -> None:
    contexts = _pool()
    backend = QueueBackend(
        [_valid_draft(local_id="h1"), _valid_draft(local_id="h2")]
    )

    outcome = HigherOrderShadowBatchRuntime(
        backend,
        max_repairs=0,
    ).run(
        source_context=_source_context(),
        contexts=contexts,
        max_contexts=100,
    )

    assert backend.generate_calls == 2
    assert outcome.record.proposed_count == 2
    assert outcome.record.abstained_count == 0
    assert outcome.record.canonical_rejected_count == 0
    assert outcome.record.shadow_contract_rejected_count == 0

    assert len(outcome.arms) == 2
    assert len(outcome.proposed_arms) == 2
    assert len(outcome.portfolios_for_downstream) == 2

    portfolio_ids = {
        row.portfolio_id
        for row in outcome.portfolios_for_downstream
    }
    assert len(portfolio_ids) == 2

    derived_context_ids = {
        row.source_context_id
        for row in outcome.portfolios_for_downstream
    }
    assert len(derived_context_ids) == 2
    assert "hypothesis_context:source" not in derived_context_ids

    assert outcome.record.legacy_portfolio_mutated is False
    assert outcome.record.production_selection_changed is False
    assert outcome.record.novelty_authority is False
    assert outcome.record.external_novelty_review_performed is False
    assert outcome.record.n10_review_performed is False


def test_batch_records_mixed_proposal_and_abstention() -> None:
    contexts = _pool()
    backend = QueueBackend(
        [_valid_draft(), _abstain_draft()]
    )

    outcome = HigherOrderShadowBatchRuntime(
        backend,
        max_repairs=0,
    ).run(
        source_context=_source_context(),
        contexts=contexts,
        max_contexts=100,
    )

    assert outcome.record.proposed_count == 1
    assert outcome.record.abstained_count == 1
    assert outcome.record.canonical_rejected_count == 0
    assert outcome.record.shadow_contract_rejected_count == 0
    assert len(outcome.portfolios_for_downstream) == 1

    statuses = sorted(row.status for row in outcome.record.arms)
    assert statuses == ["abstained", "proposed"]


def test_batch_records_canonical_rejection_per_arm() -> None:
    context = _context(
        context_id="higher_order_synthesis_context:only",
        modifier_component_id="component:only",
        modifier_text="modifier only",
    )
    backend = QueueBackend(
        [
            HypothesisPortfolioDraft(
                hypotheses=[
                    _proposal(
                        premise_id="stmt:unknown",
                    )
                ]
            )
        ]
    )

    outcome = HigherOrderShadowBatchRuntime(
        backend,
        max_repairs=0,
    ).run(
        source_context=_source_context(),
        contexts=[context],
        max_contexts=10,
    )

    assert outcome.record.proposed_count == 0
    assert outcome.record.canonical_rejected_count == 1
    assert len(outcome.portfolios_for_downstream) == 0

    arm = outcome.record.arms[0]
    assert arm.status == "canonical_rejected"
    assert arm.shadow_contract_passed is False
    assert arm.canonical_runtime_accepted is False
    assert "UNKNOWN_PREMISE_STATEMENT" in arm.canonical_compile_issue_codes


def test_batch_records_shadow_cardinality_rejection() -> None:
    context = _context(
        context_id="higher_order_synthesis_context:only",
        modifier_component_id="component:only",
        modifier_text="modifier only",
    )
    backend = QueueBackend(
        [
            HypothesisPortfolioDraft(
                hypotheses=[
                    _proposal(local_id="h1"),
                    _proposal(local_id="h2"),
                ]
            )
        ]
    )

    outcome = HigherOrderShadowBatchRuntime(
        backend,
        max_repairs=0,
    ).run(
        source_context=_source_context(),
        contexts=[context],
        max_contexts=10,
    )

    assert outcome.record.shadow_contract_rejected_count == 1
    assert len(outcome.portfolios_for_downstream) == 0

    arm = outcome.record.arms[0]
    assert arm.status == "shadow_contract_rejected"
    assert (
        arm.shadow_failure_code
        == "HIGHER_ORDER_SHADOW_CARDINALITY_VIOLATION"
    )
    assert arm.canonical_runtime_accepted is True


def test_batch_max_contexts_is_coverage_quota_not_quality_rank() -> None:
    contexts = _pool()
    expected = select_shadow_synthesis_contexts(
        contexts=contexts,
        max_contexts=1,
    )

    backend = QueueBackend([_valid_draft()])
    outcome = HigherOrderShadowBatchRuntime(
        backend,
        max_repairs=0,
    ).run(
        source_context=_source_context(),
        contexts=contexts,
        max_contexts=1,
    )

    assert outcome.record.selected_context_count == 1
    assert outcome.record.selected_higher_order_context_ids == [
        expected[0].context_id
    ]
    assert outcome.record.duplicate_or_over_quota_context_count == 2
    assert outcome.record.scientific_quality_ranking_performed is False
    assert outcome.record.production_selection_changed is False


def test_batch_rejects_mixed_requested_relation_before_generation() -> None:
    left = _context(
        context_id="higher_order_synthesis_context:left",
        modifier_component_id="component:left",
        modifier_text="left modifier",
    )
    right = _context(
        context_id="higher_order_synthesis_context:right",
        modifier_component_id="component:right",
        modifier_text="right modifier",
    ).model_copy(
        update={
            "requested_target": "different target",
        }
    )

    backend = QueueBackend([_valid_draft(), _valid_draft()])

    try:
        HigherOrderShadowBatchRuntime(
            backend,
            max_repairs=0,
        ).run(
            source_context=_source_context(),
            contexts=[left, right],
            max_contexts=10,
        )
    except ValueError as exc:
        assert "share one requested source-target relation" in str(exc)
    else:
        raise AssertionError("expected mixed requested relation rejection")

    assert backend.generate_calls == 0
