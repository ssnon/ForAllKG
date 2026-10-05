from types import SimpleNamespace

from pipeline_core.discovery.hypothesis_contracts import (
    FalsificationCriterionDraft,
    HypothesisPortfolioDraft,
    HypothesisProposalDraft,
    PredictedObservationDraft,
)
from pipeline_core.discovery.relation_validity_aware_generation import (
    IdentificationContract,
    RelationValidityAwareGenerationResponse,
    validate_response_against_control,
)


def control():
    return SimpleNamespace(
        premise_statement_ids=["s1", "s2"],
        gap_statement_ids=["g1"],
        hypothesis_type="cross_evidence_synthesis",
    )


def response(*, premises=None, gaps=None, assessment="SUPPORTED", empty=False):
    contract = IdentificationContract(
        proposed_relation="X is associated with Y under matched context.",
        independent_variable="X",
        dependent_observable="Y",
        comparison_context="matched context",
        required_controls=["C"],
        measurement_compatibility="same observable definition",
        measurement_compatibility_mode="DIRECTLY_COMPARABLE",
        measurement_support_statement_ids=["s1"],
        contextual_measurement_statement_ids=[],
        directional_evidence_basis="direction left unspecified unless supported",
        directionality_mode=(
            "NOT_IDENTIFIABLE"
            if assessment == "NOT_IDENTIFIABLE"
            else "NON_DIRECTIONAL"
        ),
        directional_support_statement_ids=[],
        potential_confounders=["Z"],
        task_alignment_rationale="addresses the same task",
        premise_statement_ids=premises or ["s1", "s2"],
        assessment=assessment,
        current_evidence_status="CONTEXT_ONLY" if assessment == "NOT_IDENTIFIABLE" else ("DIRECT_RELATION_SUPPORTED" if assessment == "SUPPORTED" else "PARTIAL_GROUNDING"),
        prospective_identifiability="NOT_OPERATIONALIZABLE" if assessment == "NOT_IDENTIFIABLE" else ("CURRENTLY_IDENTIFIED" if assessment == "SUPPORTED" else "PROSPECTIVELY_IDENTIFIABLE"),
        current_relation_support_statement_ids=([] if assessment != "SUPPORTED" else ["s1"]),
        grounded_bridge_statement_ids=(["s1"] if assessment == "PARTIALLY_IDENTIFIED" else []),
        prospective_test_design=(None if assessment != "PARTIALLY_IDENTIFIED" else "matched prospective test"),
        prospective_falsifier=(None if assessment != "PARTIALLY_IDENTIFIED" else "no predicted relation under the matched test"),
        ungrounded_required_concepts=[],
        limitation_reason=None,
    )
    if empty:
        draft = HypothesisPortfolioDraft(
            hypotheses=[],
            abstention_reason="not identifiable",
        )
    else:
        draft = HypothesisPortfolioDraft(
            hypotheses=[
                HypothesisProposalDraft(
                    local_id="t1",
                    title="Treatment",
                    hypothesis_statement="X is associated with Y under matched context.",
                    hypothesis_type="cross_evidence_synthesis",
                    premise_statement_ids=premises or ["s1", "s2"],
                    gap_statement_ids=gaps or ["g1"],
                    inferential_bridge="bounded association",
                    predicted_observations=[
                        PredictedObservationDraft(
                            local_id="p1",
                            observable="Y",
                            expected_direction="unspecified",
                            rationale="relation test",
                        )
                    ],
                    falsification_criteria=[
                        FalsificationCriterionDraft(
                            local_id="f1",
                            observable="Y",
                            falsifying_outcome="no association under matched context",
                        )
                    ],
                    assumptions=["matched context"],
                )
            ],
            abstention_reason=None,
        )
    return RelationValidityAwareGenerationResponse(
        identification_contract=contract,
        draft=draft,
    )


def test_exact_identity_passes():
    assert validate_response_against_control(control=control(), response=response()) == []


def test_premise_change_fails_closed():
    failures = validate_response_against_control(
        control=control(), response=response(premises=["s1"])
    )
    assert "IDENTIFICATION_CONTRACT_PREMISE_MISMATCH" in failures
    assert "TREATMENT_PREMISE_IDENTITY_VIOLATION" in failures


def test_not_identifiable_must_abstain():
    from pydantic import ValidationError

    try:
        response(assessment="NOT_IDENTIFIABLE")
    except ValidationError as exc:
        assert "NOT_IDENTIFIABLE requires an empty treatment draft" in str(exc)
    else:
        raise AssertionError(
            "expected NOT_IDENTIFIABLE/non-empty response to fail"
        )

    valid = response(
        assessment="NOT_IDENTIFIABLE",
        empty=True,
    )
    assert validate_response_against_control(
        control=control(),
        response=valid,
    ) == []


def test_empty_partial_identification_fails_closed():
    from pydantic import ValidationError

    try:
        response(
            assessment="PARTIALLY_IDENTIFIED",
            empty=True,
        )
    except ValidationError as exc:
        assert (
            "SUPPORTED/PARTIALLY_IDENTIFIED requires exactly one "
            "treatment hypothesis"
        ) in str(exc)
    else:
        raise AssertionError(
            "expected PARTIALLY_IDENTIFIED/empty response to fail"
        )


def test_response_model_rejects_unmatched_falsifier_observable():
    from pydantic import ValidationError

    contract = IdentificationContract(
        proposed_relation="X is associated with Y.",
        independent_variable="X",
        dependent_observable="Y",
        comparison_context="matched context",
        required_controls=[],
        measurement_compatibility="same Y definition",
        measurement_compatibility_mode="DIRECTLY_COMPARABLE",
        measurement_support_statement_ids=["s1"],
        contextual_measurement_statement_ids=[],
        directional_evidence_basis="no unsupported direction",
        directionality_mode="NON_DIRECTIONAL",
        directional_support_statement_ids=[],
        potential_confounders=[],
        task_alignment_rationale="same task",
        premise_statement_ids=["s1", "s2"],
        assessment="PARTIALLY_IDENTIFIED",
        current_evidence_status="PARTIAL_GROUNDING",
        prospective_identifiability="PROSPECTIVELY_IDENTIFIABLE",
        current_relation_support_statement_ids=[],
        grounded_bridge_statement_ids=["s1"],
        prospective_test_design="matched prospective test",
        prospective_falsifier="no predicted relation under the matched test",
        ungrounded_required_concepts=[],
        limitation_reason=None,
    )
    draft = HypothesisPortfolioDraft(
        hypotheses=[
            HypothesisProposalDraft(
                local_id="t1",
                title="Treatment",
                hypothesis_statement="X is associated with Y.",
                hypothesis_type="cross_evidence_synthesis",
                premise_statement_ids=["s1", "s2"],
                gap_statement_ids=["g1"],
                inferential_bridge="bounded association",
                predicted_observations=[
                    PredictedObservationDraft(
                        local_id="p1",
                        observable="Y",
                        expected_direction="unspecified",
                        rationale="relation test",
                    )
                ],
                falsification_criteria=[
                    FalsificationCriterionDraft(
                        local_id="f1",
                        observable="Different wording for Y",
                        falsifying_outcome="no association",
                    )
                ],
                assumptions=[],
            )
        ],
        abstention_reason=None,
    )

    try:
        RelationValidityAwareGenerationResponse(
            identification_contract=contract,
            draft=draft,
        )
    except ValidationError as exc:
        assert "falsifier.observable" in str(exc)
    else:
        raise AssertionError("expected validation failure")

def test_directionality_contract_requires_grounded_support_ids():
    from pydantic import ValidationError

    try:
        IdentificationContract(
            proposed_relation="X is higher than Y.",
            independent_variable="X",
            dependent_observable="Y",
            comparison_context="matched",
            required_controls=[],
            measurement_compatibility="same scale",
            measurement_compatibility_mode="DIRECTLY_COMPARABLE",
            measurement_support_statement_ids=["s1"],
            contextual_measurement_statement_ids=[],
            directional_evidence_basis="claimed direct support",
            directionality_mode="DIRECTLY_SUPPORTED",
            directional_support_statement_ids=[],
            potential_confounders=[],
            task_alignment_rationale="same task",
            premise_statement_ids=["s1"],
            assessment="SUPPORTED",
            current_evidence_status="DIRECT_RELATION_SUPPORTED",
            prospective_identifiability="CURRENTLY_IDENTIFIED",
            current_relation_support_statement_ids=["s1"],
            grounded_bridge_statement_ids=[],
            prospective_test_design=None,
            prospective_falsifier=None,
            ungrounded_required_concepts=[],
            limitation_reason=None,
        )
    except ValidationError as exc:
        assert "requires at least one directional support" in str(exc)
    else:
        raise AssertionError("expected missing directional support to fail")


def test_directional_support_ids_must_be_grounded_premises():
    from pydantic import ValidationError

    try:
        IdentificationContract(
            proposed_relation="X is higher than Y.",
            independent_variable="X",
            dependent_observable="Y",
            comparison_context="matched",
            required_controls=[],
            measurement_compatibility="same scale",
            measurement_compatibility_mode="DIRECTLY_COMPARABLE",
            measurement_support_statement_ids=["s1"],
            contextual_measurement_statement_ids=[],
            directional_evidence_basis="claimed direct support",
            directionality_mode="DIRECTLY_SUPPORTED",
            directional_support_statement_ids=["outside"],
            potential_confounders=[],
            task_alignment_rationale="same task",
            premise_statement_ids=["s1"],
            assessment="SUPPORTED",
            current_evidence_status="DIRECT_RELATION_SUPPORTED",
            prospective_identifiability="CURRENTLY_IDENTIFIED",
            current_relation_support_statement_ids=["s1"],
            grounded_bridge_statement_ids=[],
            prospective_test_design=None,
            prospective_falsifier=None,
            ungrounded_required_concepts=[],
            limitation_reason=None,
        )
    except ValidationError as exc:
        assert "subset of premise_statement_ids" in str(exc)
    else:
        raise AssertionError("expected outside support ID to fail")


def test_non_directional_response_rejects_directional_prediction():
    from pydantic import ValidationError

    contract = IdentificationContract(
        proposed_relation="X is associated with Y.",
        independent_variable="X",
        dependent_observable="Y",
        comparison_context="matched",
        required_controls=[],
        measurement_compatibility="same scale",
        measurement_compatibility_mode="DIRECTLY_COMPARABLE",
        measurement_support_statement_ids=["s1"],
        contextual_measurement_statement_ids=[],
        directional_evidence_basis="no supported direction",
        directionality_mode="NON_DIRECTIONAL",
        directional_support_statement_ids=[],
        potential_confounders=[],
        task_alignment_rationale="same task",
        premise_statement_ids=["s1", "s2"],
        assessment="PARTIALLY_IDENTIFIED",
        current_evidence_status="PARTIAL_GROUNDING",
        prospective_identifiability="PROSPECTIVELY_IDENTIFIABLE",
        current_relation_support_statement_ids=[],
        grounded_bridge_statement_ids=["s1"],
        prospective_test_design="matched prospective test",
        prospective_falsifier="no predicted relation under the matched test",
        ungrounded_required_concepts=[],
        limitation_reason=None,
    )
    draft = HypothesisPortfolioDraft(
        hypotheses=[
            HypothesisProposalDraft(
                local_id="t1",
                title="Treatment",
                hypothesis_statement="X is associated with Y.",
                hypothesis_type="cross_evidence_synthesis",
                premise_statement_ids=["s1", "s2"],
                gap_statement_ids=["g1"],
                inferential_bridge="bounded association",
                predicted_observations=[
                    PredictedObservationDraft(
                        local_id="p1",
                        observable="Y",
                        expected_direction="increase",
                        rationale="unsupported direction",
                    )
                ],
                falsification_criteria=[
                    FalsificationCriterionDraft(
                        local_id="f1",
                        observable="Y",
                        falsifying_outcome="no association",
                    )
                ],
                assumptions=[],
            )
        ],
        abstention_reason=None,
    )

    try:
        RelationValidityAwareGenerationResponse(
            identification_contract=contract,
            draft=draft,
        )
    except ValidationError as exc:
        assert "NON_DIRECTIONAL treatment requires" in str(exc)
    else:
        raise AssertionError(
            "expected directional prediction under NON_DIRECTIONAL to fail"
        )

def test_measurement_not_comparable_requires_abstention_assessment():
    from pydantic import ValidationError

    try:
        IdentificationContract(
            proposed_relation="Joint response and reproducibility.",
            independent_variable="architecture",
            dependent_observable="joint outcome",
            comparison_context="heterogeneous records",
            required_controls=[],
            measurement_compatibility="not established",
            measurement_compatibility_mode="NOT_COMPARABLE",
            measurement_support_statement_ids=[],
            contextual_measurement_statement_ids=["s1", "s2"],
            directional_evidence_basis="no supported direction",
            directionality_mode="NON_DIRECTIONAL",
            directional_support_statement_ids=[],
            potential_confounders=[],
            task_alignment_rationale="same task",
            premise_statement_ids=["s1", "s2"],
            assessment="PARTIALLY_IDENTIFIED",
            current_evidence_status="PARTIAL_GROUNDING",
            prospective_identifiability="PROSPECTIVELY_IDENTIFIABLE",
            current_relation_support_statement_ids=[],
            grounded_bridge_statement_ids=["s1"],
            prospective_test_design="matched prospective test",
            prospective_falsifier="no predicted relation under the matched test",
            ungrounded_required_concepts=[],
            limitation_reason="different measurement levels",
        )
    except ValidationError as exc:
        assert "NOT_COMPARABLE is terminal only when the relation is NOT_OPERATIONALIZABLE" in str(exc)
    else:
        raise AssertionError(
            "expected NOT_COMPARABLE non-abstention contract to fail"
        )


def test_contextual_measurements_require_primary_and_contextual_ids():
    from pydantic import ValidationError

    try:
        IdentificationContract(
            proposed_relation="Scoped within-system association.",
            independent_variable="X",
            dependent_observable="Y",
            comparison_context="one primary system",
            required_controls=[],
            measurement_compatibility="other record is contextual only",
            measurement_compatibility_mode="CONTEXTUAL_ONLY",
            measurement_support_statement_ids=["s1"],
            contextual_measurement_statement_ids=[],
            directional_evidence_basis="no supported direction",
            directionality_mode="NON_DIRECTIONAL",
            directional_support_statement_ids=[],
            potential_confounders=[],
            task_alignment_rationale="same task",
            premise_statement_ids=["s1", "s2"],
            assessment="PARTIALLY_IDENTIFIED",
            current_evidence_status="PARTIAL_GROUNDING",
            prospective_identifiability="PROSPECTIVELY_IDENTIFIABLE",
            current_relation_support_statement_ids=[],
            grounded_bridge_statement_ids=["s1"],
            prospective_test_design="matched prospective test",
            prospective_falsifier="no predicted relation under the matched test",
            ungrounded_required_concepts=[],
            limitation_reason=None,
        )
    except ValidationError as exc:
        assert "requires at least one contextual" in str(exc)
    else:
        raise AssertionError(
            "expected empty contextual-only measurement set to fail"
        )


def test_measurement_support_and_contextual_ids_must_be_disjoint():
    from pydantic import ValidationError

    try:
        IdentificationContract(
            proposed_relation="Scoped relation.",
            independent_variable="X",
            dependent_observable="Y",
            comparison_context="one primary system",
            required_controls=[],
            measurement_compatibility="context split",
            measurement_compatibility_mode="CONTEXTUAL_ONLY",
            measurement_support_statement_ids=["s1"],
            contextual_measurement_statement_ids=["s1"],
            directional_evidence_basis="no supported direction",
            directionality_mode="NON_DIRECTIONAL",
            directional_support_statement_ids=[],
            potential_confounders=[],
            task_alignment_rationale="same task",
            premise_statement_ids=["s1", "s2"],
            assessment="PARTIALLY_IDENTIFIED",
            current_evidence_status="PARTIAL_GROUNDING",
            prospective_identifiability="PROSPECTIVELY_IDENTIFIABLE",
            current_relation_support_statement_ids=[],
            grounded_bridge_statement_ids=["s1"],
            prospective_test_design="matched prospective test",
            prospective_falsifier="no predicted relation under the matched test",
            ungrounded_required_concepts=[],
            limitation_reason=None,
        )
    except ValidationError as exc:
        assert "must be disjoint" in str(exc)
    else:
        raise AssertionError(
            "expected overlapping measurement roles to fail"
        )

def test_prospective_match_required_can_generate_with_grounded_design():
    contract = IdentificationContract(
        proposed_relation="Geometry may shift the response-reproducibility balance.",
        independent_variable="geometry",
        dependent_observable="response and reproducibility",
        comparison_context="future matched geometry sweep",
        required_controls=["same analyte", "same acquisition"],
        measurement_compatibility="current records are not paired; future matched measurement is explicit",
        measurement_compatibility_mode="PROSPECTIVE_MATCH_REQUIRED",
        measurement_support_statement_ids=["s1"],
        contextual_measurement_statement_ids=["s2"],
        directional_evidence_basis="grounded prediction, not observed direction",
        directionality_mode="GROUNDED_PREDICTION",
        directional_support_statement_ids=["s1"],
        potential_confounders=["fabrication variation"],
        task_alignment_rationale="same scientific task",
        premise_statement_ids=["s1", "s2"],
        assessment="PARTIALLY_IDENTIFIED",
        limitation_reason="current evidence does not establish the full relation",
        current_evidence_status="PARTIAL_GROUNDING",
        prospective_identifiability="PROSPECTIVELY_IDENTIFIABLE",
        current_relation_support_statement_ids=[],
        grounded_bridge_statement_ids=["s1", "s2"],
        prospective_test_design="Sweep geometry while holding analyte and acquisition fixed; measure both outcomes on the same samples.",
        prospective_falsifier="No systematic relation between geometry and the joint response-reproducibility profile.",
        ungrounded_required_concepts=[],
    )
    assert contract.prospective_identifiability == "PROSPECTIVELY_IDENTIFIABLE"


def test_prospective_identifiable_rejects_ungrounded_required_concepts():
    from pydantic import ValidationError

    try:
        IdentificationContract(
            proposed_relation="Reporter chemistry controls response.",
            independent_variable="reporter chemistry",
            dependent_observable="SERS intensity",
            comparison_context="future matched assay",
            required_controls=[],
            measurement_compatibility="future assay could be defined",
            measurement_compatibility_mode="PROSPECTIVE_MATCH_REQUIRED",
            measurement_support_statement_ids=["s1"],
            contextual_measurement_statement_ids=[],
            directional_evidence_basis="grounded prediction",
            directionality_mode="GROUNDED_PREDICTION",
            directional_support_statement_ids=["s1"],
            potential_confounders=[],
            task_alignment_rationale="same task",
            premise_statement_ids=["s1"],
            assessment="PARTIALLY_IDENTIFIED",
            limitation_reason="reporter chemistry is absent from grounded evidence",
            current_evidence_status="PARTIAL_GROUNDING",
            prospective_identifiability="PROSPECTIVELY_IDENTIFIABLE",
            current_relation_support_statement_ids=[],
            grounded_bridge_statement_ids=["s1"],
            prospective_test_design="Compare reporters under matched conditions.",
            prospective_falsifier="No reporter-dependent difference.",
            ungrounded_required_concepts=["reporter-specific chemical affinity"],
        )
    except ValidationError as exc:
        assert "cannot depend on ungrounded required concepts" in str(exc)
    else:
        raise AssertionError("expected ungrounded prospective concept to fail")


def test_currently_identified_requires_direct_relation_support():
    from pydantic import ValidationError

    try:
        IdentificationContract(
            proposed_relation="X is associated with Y.",
            independent_variable="X",
            dependent_observable="Y",
            comparison_context="current study",
            required_controls=[],
            measurement_compatibility="same study",
            measurement_compatibility_mode="DIRECTLY_COMPARABLE",
            measurement_support_statement_ids=["s1"],
            contextual_measurement_statement_ids=[],
            directional_evidence_basis="no direction",
            directionality_mode="NON_DIRECTIONAL",
            directional_support_statement_ids=[],
            potential_confounders=[],
            task_alignment_rationale="same task",
            premise_statement_ids=["s1"],
            assessment="SUPPORTED",
            limitation_reason=None,
            current_evidence_status="DIRECT_RELATION_SUPPORTED",
            prospective_identifiability="CURRENTLY_IDENTIFIED",
            current_relation_support_statement_ids=[],
            grounded_bridge_statement_ids=[],
            prospective_test_design=None,
            prospective_falsifier=None,
            ungrounded_required_concepts=[],
        )
    except ValidationError as exc:
        assert "requires at least one current relation support" in str(exc)
    else:
        raise AssertionError("expected missing current relation support to fail")
