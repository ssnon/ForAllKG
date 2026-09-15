from __future__ import annotations

import pytest

from pipeline_core.discovery.higher_order_hypothesis_context import (
    materialize_higher_order_hypothesis_context,
)
from pipeline_core.discovery.higher_order_hypothesis_runtime import (
    HigherOrderShadowHypothesisRuntime,
    authorize_higher_order_shadow_generation,
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


class FakeBackend:
    backend_name = "fake_higher_order_shadow_backend"
    model_name = "fake-model"
    temperature = 0.0
    instructor_mode = "FAKE"
    base_url = "fake://local"
    parse_retries = 0

    def __init__(
        self,
        initial: HypothesisPortfolioDraft,
        repairs: list[HypothesisPortfolioDraft] | None = None,
    ) -> None:
        self.initial = initial
        self.repairs = list(repairs or [])
        self.feedback: list[str] = []
        self.generate_calls = 0
        self.last_prompt = None

    def generate(self, prompt):
        self.generate_calls += 1
        self.last_prompt = prompt
        return HypothesisDraftGeneration(
            draft=self.initial,
            input_tokens=20,
            output_tokens=10,
        )

    def repair(self, prompt, previous_draft, feedback):
        self.feedback.append(feedback)
        if not self.repairs:
            raise AssertionError("unexpected repair call")
        return HypothesisDraftGeneration(
            draft=self.repairs.pop(0),
            input_tokens=22,
            output_tokens=11,
        )


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
                    "coupling and the measured response."
                ),
                epistemic_role="reported",
                claim_kind="mechanistic_relation",
                paper_ids=["paper:1"],
                requires_verification=False,
                eligible_as_premise=True,
                eligible_as_gap=False,
            ),
            HypothesisEvidenceStatement(
                statement_id="stmt:gap:1",
                text=(
                    "The supplied context does not resolve whether another "
                    "structural variable changes this relationship."
                ),
                epistemic_role="unresolved",
                claim_kind="research_gap",
                paper_ids=[],
                requires_verification=True,
                eligible_as_premise=False,
                eligible_as_gap=True,
            ),
        ],
    )


def _higher_order_context() -> HigherOrderSynthesisContext:
    lineage = HigherOrderSynthesisLineageView(
        higher_order_topology_id="higher_order_topology:test",
        backbone_topology_id="relational_topology:test",
        source_component_id="component:source",
        target_component_id="component:target",
        modifier_component_id="component:modifier",
        source_authority=RelationComponentAuthority.CONFIRMED_KNOWN,
        target_authority=RelationComponentAuthority.CONFIRMED_KNOWN,
        modifier_authority=RelationComponentAuthority.CONFIRMED_KNOWN,
        source_provenance_source_id="accepted_pattern:source",
        target_provenance_source_id="accepted_pattern:target",
        modifier_provenance_source_id="accepted_pattern:modifier",
        source_endpoint_binding_authority="exact",
        target_endpoint_binding_authority="exact",
        source_endpoint_equivalence_witness_id=None,
        target_endpoint_equivalence_witness_id=None,
        modifier_eligibility_witness_id="modifier_witness:test",
        modifier_anchor_role="mediator",
        modifier_anchor_slot="object",
        modifier_slot="subject",
        modifier_anchor_text="electric-field enhancement",
        modifier_text="nanoparticle edge sharpness",
    )

    return HigherOrderSynthesisContext(
        context_id="higher_order_synthesis_context:test",
        carrier_id="higher_order_synthesis_carrier:test",
        higher_order_topology_id="higher_order_topology:test",
        requested_source="interparticle separation",
        requested_target="SERS intensity",
        lineage=lineage,
        premises=[
            HigherOrderSynthesisPremiseView(
                premise_id="ho_premise:source",
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
                premise_id="ho_premise:target",
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
                premise_id="ho_premise:modifier",
                premise_role="modifier_relation",
                component_id="component:modifier",
                subject="nanoparticle edge sharpness",
                relation="PROMOTES",
                object="electric-field enhancement",
                authority=RelationComponentAuthority.CONFIRMED_KNOWN,
                provenance_source_id="accepted_pattern:modifier",
                epistemic_use="confirmed_known_component",
            ),
        ],
        structural_opportunity=HigherOrderStructuralOpportunityView(
            requested_source="interparticle separation",
            requested_target="SERS intensity",
            source_side_mediator_text="electric-field enhancement",
            target_side_mediator_text="electric-field enhancement",
            modifier_text="nanoparticle edge sharpness",
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
        "the qualitative dependence of SERS intensity on "
        "interparticle separation across edge-sharpness conditions"
    )
    return HypothesisProposalDraft(
        local_id=local_id,
        title="Edge-sharpness-conditioned separation response",
        hypothesis_statement=(
            "Nanoparticle edge sharpness may condition how "
            "interparticle separation relates to SERS intensity."
        ),
        hypothesis_type="context_dependency",
        premise_statement_ids=[premise_id],
        gap_statement_ids=[],
        inferential_bridge=(
            "The higher-order composition motivates testing whether "
            "edge sharpness changes the separation-response relationship "
            "without treating that interaction as established evidence."
        ),
        predicted_observations=[
            PredictedObservationDraft(
                local_id="p1",
                observable=observable,
                expected_direction="unspecified",
                rationale=(
                    "A modifier-dependent response would be consistent "
                    "with the proposed context dependency."
                ),
            )
        ],
        falsification_criteria=[
            FalsificationCriterionDraft(
                local_id="f1",
                observable=observable,
                falsifying_outcome=(
                    "The qualitative dependence remains unchanged across "
                    "edge-sharpness conditions."
                ),
            )
        ],
        assumptions=[
            "The proposed modifier interaction remains unverified."
        ],
    )


def _draft(
    *,
    premise_id: str = "stmt:positive:1",
    local_id: str = "h1",
) -> HypothesisPortfolioDraft:
    return HypothesisPortfolioDraft(
        hypotheses=[
            _proposal(
                local_id=local_id,
                premise_id=premise_id,
            )
        ]
    )


def _prepared():
    source = _source_context()
    higher = _higher_order_context()
    projection = materialize_higher_order_hypothesis_context(
        source_context=source,
        higher_order_context=higher,
    )
    authorization = authorize_higher_order_shadow_generation(
        projection=projection,
        higher_order_context=higher,
    )
    return source, higher, projection, authorization


def test_authorization_is_separate_from_upstream_non_authority() -> None:
    _, higher, projection, authorization = _prepared()

    assert higher.guard.llm_call_authorized is False
    assert projection.materialization.llm_call_authorized is False

    assert authorization.llm_call_authorized is True
    assert authorization.canonical_hypothesis_runtime_only is True
    assert authorization.shadow_only is True
    assert authorization.max_hypotheses == 1

    assert authorization.novelty_authority is False
    assert authorization.external_novelty_bypass_authorized is False
    assert authorization.n10_bypass_authorized is False
    assert authorization.production_selection_changed is False


def test_runtime_delegates_to_canonical_hypothesis_runtime() -> None:
    _, higher, projection, authorization = _prepared()
    backend = FakeBackend(_draft())

    outcome = HigherOrderShadowHypothesisRuntime(
        backend,
        max_repairs=1,
    ).run(
        projection=projection,
        higher_order_context=higher,
        authorization=authorization,
    )

    assert outcome.status == "proposed"
    assert outcome.shadow_contract_passed is True
    assert outcome.shadow_failure_code is None
    assert outcome.portfolio_for_downstream is not None

    assert outcome.canonical_outcome.accepted is True
    assert (
        outcome.canonical_outcome.run_record.context_id
        == projection.context.context_id
    )
    assert (
        outcome.canonical_outcome.run_record.context_sha256
        == projection.context.context_sha256
    )
    assert outcome.canonical_outcome.run_record.generation_attempts == 1
    assert outcome.canonical_outcome.run_record.repair_attempts == 0

    card = outcome.portfolio_for_downstream.hypotheses[0]
    assert card.premise_statement_ids == ["stmt:positive:1"]
    assert card.gap_statement_ids == []

    assert backend.last_prompt is not None
    assert (
        "HIGHER-ORDER SHADOW SYNTHESIS FOCUS"
        in backend.last_prompt.user_prompt
    )
    assert (
        "nanoparticle edge sharpness"
        in backend.last_prompt.user_prompt
    )


def test_canonical_compiler_repairs_restricted_inspiration_premise() -> None:
    _, higher, projection, authorization = _prepared()
    restricted_id = (
        projection.materialization
        .generated_statement_lineage[0]
        .statement_id
    )

    backend = FakeBackend(
        _draft(premise_id=restricted_id),
        repairs=[_draft()],
    )

    outcome = HigherOrderShadowHypothesisRuntime(
        backend,
        max_repairs=1,
    ).run(
        projection=projection,
        higher_order_context=higher,
        authorization=authorization,
    )

    assert outcome.status == "proposed"
    assert outcome.shadow_contract_passed is True
    assert outcome.canonical_outcome.run_record.repair_attempts == 1
    assert backend.feedback
    assert "INELIGIBLE_POSITIVE_PREMISE" in backend.feedback[0]

    card = outcome.portfolio_for_downstream.hypotheses[0]
    assert restricted_id not in card.premise_statement_ids


def test_runtime_allows_explicit_abstention() -> None:
    _, higher, projection, authorization = _prepared()
    backend = FakeBackend(
        HypothesisPortfolioDraft(
            hypotheses=[],
            abstention_reason=(
                "The supplied evidence is insufficient for a bounded "
                "higher-order hypothesis."
            ),
        )
    )

    outcome = HigherOrderShadowHypothesisRuntime(
        backend,
        max_repairs=1,
    ).run(
        projection=projection,
        higher_order_context=higher,
        authorization=authorization,
    )

    assert outcome.status == "abstained"
    assert outcome.shadow_contract_passed is True
    assert outcome.abstained is True
    assert outcome.portfolio_for_downstream is None
    assert outcome.canonical_outcome.accepted is True
    assert (
        outcome.canonical_outcome.accepted_portfolio is not None
    )
    assert not outcome.canonical_outcome.accepted_portfolio.hypotheses


def test_runtime_rejects_multiple_hypotheses_after_canonical_validation() -> None:
    _, higher, projection, authorization = _prepared()

    backend = FakeBackend(
        HypothesisPortfolioDraft(
            hypotheses=[
                _proposal(local_id="h1"),
                _proposal(local_id="h2"),
            ]
        )
    )

    outcome = HigherOrderShadowHypothesisRuntime(
        backend,
        max_repairs=0,
    ).run(
        projection=projection,
        higher_order_context=higher,
        authorization=authorization,
    )

    # The generic runtime permits a multi-card portfolio. The higher-order
    # shadow boundary must fail closed rather than relying on prompt wording.
    assert outcome.canonical_outcome.accepted is True
    assert len(
        outcome.canonical_outcome.accepted_portfolio.hypotheses
    ) == 2

    assert outcome.status == "shadow_contract_rejected"
    assert outcome.shadow_contract_passed is False
    assert (
        outcome.shadow_failure_code
        == "HIGHER_ORDER_SHADOW_CARDINALITY_VIOLATION"
    )
    assert outcome.portfolio_for_downstream is None


def test_authorization_binding_mismatch_fails_before_llm_call() -> None:
    _, higher, projection, authorization = _prepared()
    backend = FakeBackend(_draft())

    tampered = authorization.model_copy(
        update={
            "derived_hypothesis_context_id": (
                "hypothesis_context_ho_shadow:wrong"
            )
        }
    )

    with pytest.raises(
        ValueError,
        match="authorization binding mismatch",
    ):
        HigherOrderShadowHypothesisRuntime(
            backend
        ).run(
            projection=projection,
            higher_order_context=higher,
            authorization=tampered,
        )

    assert backend.generate_calls == 0


def test_runtime_preserves_no_selection_and_no_novelty_authority() -> None:
    _, higher, projection, authorization = _prepared()
    backend = FakeBackend(_draft())

    outcome = HigherOrderShadowHypothesisRuntime(
        backend
    ).run(
        projection=projection,
        higher_order_context=higher,
        authorization=authorization,
    )

    assert outcome.status == "proposed"
    assert outcome.authorization.production_selection_changed is False
    assert outcome.authorization.novelty_authority is False
    assert (
        outcome.projection.materialization
        .production_selection_changed
        is False
    )
    assert outcome.projection.materialization.novelty_authority is False


def test_authorization_id_tamper_fails_before_llm_call() -> None:
    _, higher, projection, authorization = _prepared()
    backend = FakeBackend(_draft())

    tampered = authorization.model_copy(
        update={
            "authorization_id": (
                "higher_order_shadow_generation_authorization:wrong"
            )
        }
    )

    with pytest.raises(
        ValueError,
        match="authorization_id mismatch",
    ):
        HigherOrderShadowHypothesisRuntime(
            backend
        ).run(
            projection=projection,
            higher_order_context=higher,
            authorization=tampered,
        )

    assert backend.generate_calls == 0
