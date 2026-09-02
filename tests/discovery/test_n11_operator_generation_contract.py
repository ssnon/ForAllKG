import pytest

from pipeline_core.discovery.nonobviousness_operator_generation_contracts import (
    N11OperatorCandidateDraft,
    N11OperatorFalsificationDraft,
    N11OperatorGenerationDraft,
    N11OperatorPredictionDraft,
)
from pipeline_core.discovery.nonobviousness_operator_generation_validation import (
    N11OperatorGenerationAuthority,
    N11OperatorGenerationValidator,
)


def candidate(
    **overrides,
):
    body = dict(
        local_id="candidate:1",
        title=(
            "Spacing-dependent relative mechanism contribution"
        ),
        hypothesis_statement=(
            "Interparticle spacing may alter the relative "
            "contributions of electromagnetic and chemical "
            "enhancement to the observed SERS response."
        ),
        operator=(
            "RELATIVE_CONTRIBUTION_SHIFT"
        ),
        hypothesis_type=(
            "mechanistic_extension"
        ),
        baseline_premise_statement_ids=[
            "stmt:baseline"
        ],
        supplemental_mechanism_node_ids=[
            "node:supplemental"
        ],
        gap_statement_ids=[
            "stmt:gap"
        ],
        shared_component_ids=[
            "component:shared:0"
        ],
        supplemental_only_component_ids=[
            "component:supplemental:0"
        ],
        relative_contribution_claim=(
            "Changing interparticle spacing may change "
            "the relative contribution of electromagnetic "
            "and chemical enhancement."
        ),
        inferential_bridge=(
            "The grounded mechanisms are separately supported, "
            "while their spacing-dependent relative weighting "
            "is proposed here as an inference."
        ),
        predicted_observations=[
            N11OperatorPredictionDraft(
                local_id="prediction:1",
                observable=(
                    "relative electromagnetic-to-chemical "
                    "contribution proxy across spacing"
                ),
                expected_direction="shift",
                rationale=(
                    "A relative-contribution hypothesis predicts "
                    "a spacing-associated change in this balance."
                ),
            )
        ],
        discriminating_observation_local_id=(
            "prediction:1"
        ),
        falsification_criteria=[
            N11OperatorFalsificationDraft(
                local_id="falsifier:1",
                prediction_local_id="prediction:1",
                falsifying_outcome=(
                    "The relative contribution proxy remains "
                    "unchanged across spacing despite variation "
                    "in the total SERS response."
                ),
            )
        ],
        assumptions=[],
        generated_relation_status=(
            "INFERENCE_NOT_REPORTED"
        ),
        task_to_supplemental_relation_grounded=False,
    )

    body.update(
        overrides
    )

    return (
        N11OperatorCandidateDraft(
            **body
        )
    )


def authority(
    **overrides,
):
    body = dict(
        requested_operator=(
            "RELATIVE_CONTRIBUTION_SHIFT"
        ),
        eligible_operators=(
            "MECHANISM_AUGMENTATION",
            "RELATIVE_CONTRIBUTION_SHIFT",
        ),
        allowed_baseline_statement_ids=(
            "stmt:baseline",
        ),
        allowed_supplemental_node_ids=(
            "node:supplemental",
        ),
        allowed_gap_statement_ids=(
            "stmt:gap",
        ),
        allowed_shared_component_ids=(
            "component:shared:0",
        ),
        allowed_supplemental_only_component_ids=(
            "component:supplemental:0",
        ),
    )

    body.update(
        overrides
    )

    return (
        N11OperatorGenerationAuthority(
            **body
        )
    )


def validate(
    row,
    auth=None,
):
    return (
        N11OperatorGenerationValidator()
        .validate(
            authority=(
                auth
                or authority()
            ),
            draft=(
                N11OperatorGenerationDraft(
                    candidate=row,
                    abstention_reason=None,
                )
            ),
        )
    )


def test_valid_relative_contribution_candidate_passes():
    result = validate(
        candidate()
    )

    assert result.passes
    assert result.issues == []


def test_supplemental_node_cannot_be_invented():
    result = validate(
        candidate(
            supplemental_mechanism_node_ids=[
                "node:invented"
            ]
        )
    )

    assert not result.passes

    assert any(
        row.code
        == "UNKNOWN_SUPPLEMENTAL_MECHANISM_NODE"
        for row
        in result.issues
    )


def test_operator_must_be_deterministically_authorized():
    result = validate(
        candidate(
            operator="PATHWAY_COMPETITION"
        )
    )

    assert not result.passes

    assert any(
        row.code
        == "OPERATOR_MISMATCH"
        for row
        in result.issues
    )


def test_competition_semantics_are_blocked_in_relative_shift_candidate():
    result = validate(
        candidate(
            inferential_bridge=(
                "The two mechanisms compete as spacing varies."
            )
        )
    )

    assert not result.passes

    assert any(
        row.code
        == "UNAUTHORIZED_COMPETITION_CLAIM"
        for row
        in result.issues
    )


def test_relative_contribution_semantics_are_required():
    result = validate(
        candidate(
            relative_contribution_claim=(
                "Spacing changes the SERS response."
            )
        )
    )

    assert not result.passes

    assert any(
        row.code
        == "RELATIVE_CONTRIBUTION_SEMANTICS_MISSING"
        for row
        in result.issues
    )


def test_abstention_is_fail_closed_and_valid():
    draft = (
        N11OperatorGenerationDraft(
            candidate=None,
            abstention_reason=(
                "A discriminating prediction cannot be "
                "constructed from the supplied evidence."
            ),
        )
    )

    result = (
        N11OperatorGenerationValidator()
        .validate(
            authority=
                authority(),
            draft=
                draft,
        )
    )

    assert result.passes


def _assert_strict_schema(
    schema,
):
    if not isinstance(
        schema,
        dict,
    ):
        return

    if schema.get(
        "type"
    ) == "object":
        properties = set(
            schema.get(
                "properties",
                {}
            )
        )

        required = set(
            schema.get(
                "required",
                []
            )
        )

        assert (
            properties
            == required
        )

    for value in schema.values():
        if isinstance(
            value,
            dict,
        ):
            _assert_strict_schema(
                value
            )

        elif isinstance(
            value,
            list,
        ):
            for item in value:
                if isinstance(
                    item,
                    dict,
                ):
                    _assert_strict_schema(
                        item
                    )


def test_generation_draft_is_openai_strict_schema_compatible():
    schema = (
        N11OperatorGenerationDraft
        .model_json_schema()
    )

    _assert_strict_schema(
        schema
    )


def test_supplemental_lane_is_structurally_separate_from_baseline_premises():
    fields = set(
        N11OperatorCandidateDraft
        .model_fields
    )

    assert (
        "baseline_premise_statement_ids"
        in fields
    )

    assert (
        "supplemental_mechanism_node_ids"
        in fields
    )

    assert (
        "premise_statement_ids"
        not in fields
    )


def test_structural_prediction_id_digit_is_not_scientific_numeric_prediction():
    row = candidate()

    assert (
        row.falsification_criteria[0].prediction_local_id
        == "prediction:1"
    )

    result = validate(
        row
    )

    assert result.passes

    assert not any(
        issue.code
        == "NUMERIC_PREDICTION_NOT_ALLOWED_IN_C1"
        for issue
        in result.issues
    )


def test_numeric_value_in_scientific_prose_is_still_blocked():
    row = candidate(
        relative_contribution_claim=(
            "Changing interparticle spacing may change "
            "the relative contribution of electromagnetic "
            "and chemical enhancement by 2."
        )
    )

    result = validate(
        row
    )

    assert not result.passes

    assert any(
        issue.code
        == "NUMERIC_PREDICTION_NOT_ALLOWED_IN_C1"
        for issue
        in result.issues
    )


def test_explicit_competition_boundary_is_not_a_competition_claim():
    row = candidate(
        assumptions=[
            (
                "Any observed redistribution is interpreted as "
                "a relative-contribution inference rather than "
                "as evidence of pathway competition."
            )
        ]
    )

    result = validate(
        row
    )

    assert result.passes

    assert not any(
        issue.code
        == "UNAUTHORIZED_COMPETITION_CLAIM"
        for issue
        in result.issues
    )


def test_explicit_switch_boundary_is_not_a_switch_claim():
    row = candidate(
        assumptions=[
            (
                "The inference does not establish "
                "a mechanism switch."
            )
        ]
    )

    result = validate(
        row
    )

    assert result.passes

    assert not any(
        issue.code
        == "UNAUTHORIZED_SWITCH_CLAIM"
        for issue
        in result.issues
    )


def test_affirmative_mechanism_switch_remains_blocked():
    row = candidate(
        inferential_bridge=(
            "Changing spacing causes a mechanism switch "
            "from one enhancement pathway to another."
        )
    )

    result = validate(
        row
    )

    assert not result.passes

    assert any(
        issue.code
        == "UNAUTHORIZED_SWITCH_CLAIM"
        for issue
        in result.issues
    )
