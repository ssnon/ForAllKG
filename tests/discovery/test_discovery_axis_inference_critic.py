from __future__ import annotations

from types import SimpleNamespace

import pytest

from pipeline_core.discovery.discovery_axis_inference_contracts import (
    AxisInferenceAssertionDraft,
    AxisInferenceReviewDraft,
)
from pipeline_core.discovery.discovery_axis_inference_critic import (
    AxisInferenceReviewCompiler,
    AxisInferenceReviewValidationError,
    DiscoveryAxisInferenceCritic,
)
from pipeline_core.discovery.discovery_axis_inference_llm import (
    AxisInferenceGeneration,
)
from pipeline_core.discovery.discovery_axis_inference_prompt import (
    DiscoveryAxisInferencePromptAssembler,
    allowed_axis_basis,
)


class FakeStatement:
    def __init__(
        self,
        statement_id: str,
        text: str,
        *,
        eligible_as_premise: bool = True,
    ) -> None:
        self.statement_id = statement_id
        self.text = text
        self.eligible_as_premise = (
            eligible_as_premise
        )

    def model_dump(
        self,
        *,
        mode: str = "json",
    ) -> dict[str, object]:
        return {
            "statement_id":
                self.statement_id,
            "text":
                self.text,
            "eligible_as_premise":
                self.eligible_as_premise,
            "eligible_as_gap":
                False,
            "epistemic_role":
                "reported",
            "claim_kind":
                "mechanism",
            "paper_ids":
                ["paper:test"],
            "scientific_support_node_ids":
                [],
            "scientific_support_edge_ids":
                [],
            "support_path_ids":
                [],
            "alignment_path_ids":
                [],
            "requires_verification":
                False,
            "premise_restrictions":
                [],
        }


def context():
    return SimpleNamespace(
        context_id="context:test",
        context_sha256="ctxsha",
        evidence_statements=[
            FakeStatement(
                "stmt:1",
                "LSPR placement affects electromagnetic enhancement.",
            ),
            FakeStatement(
                "stmt:2",
                "Au-Ag structures show SERS enhancement.",
            ),
            FakeStatement(
                "stmt:other",
                "Other eligible statement not selected by this hypothesis.",
            ),
        ],
    )


def axis():
    return SimpleNamespace(
        axis_id="axis:test",
        inspiration_id="inspiration:test",
        candidate_unit_id="candidate:test",
        label=(
            "Copper-associated composition changes "
            "SERS performance"
        ),
        proposed_subject=
            "copper-associated composition",
        proposed_relation="PROMOTES",
        proposed_object="SERS performance",
        requires_verification=True,
    )


def prediction(
    observation_id: str,
    observable: str,
    direction: str = "qualitative_change",
):
    return SimpleNamespace(
        observation_id=observation_id,
        observable=observable,
        expected_direction=direction,
        rationale="Synthetic prediction rationale.",
    )


def card():
    return SimpleNamespace(
        source_context_id="context:test",
        source_context_sha256="ctxsha",
        hypothesis_id="hypothesis:test",
        title="Synthetic copper moderator hypothesis",
        hypothesis_statement=(
            "Copper-associated context may alter "
            "the Au-Ag SERS response."
        ),
        inferential_bridge=(
            "Grounded SERS factors are combined with "
            "the inspiration-only copper axis."
        ),
        premise_statement_ids=[
            "stmt:1",
            "stmt:2",
        ],
        predicted_observations=[
            prediction(
                "prediction:1",
                "The response differs qualitatively "
                "between copper-associated and comparison contexts.",
            ),
        ],
        assumptions=[],
    )


def central(
    *,
    source_class: str =
        "S_BOUNDED_SYNTHESIS",
    action: str = "KEEP",
    grounds: list[str] | None = None,
    basis: list[str] | None = None,
    text: str | None = None,
):
    c = card()
    a = axis()

    return AxisInferenceAssertionDraft(
        assertion_id=
            f"central:{c.hypothesis_id}",
        assertion_kind=
            "central_hypothesis",
        assertion_text=(
            c.hypothesis_statement
            if text is None
            else text
        ),
        source_class=source_class,
        action=action,
        grounded_statement_ids=(
            ["stmt:1", "stmt:2"]
            if grounds is None
            else grounds
        ),
        axis_basis=(
            [a.label]
            if basis is None
            else basis
        ),
        specificity_tags=[
            "moderator",
            "interaction",
        ],
        rationale="Synthetic central review.",
    )


def pred(
    *,
    source_class: str =
        "S_BOUNDED_SYNTHESIS",
    action: str = "KEEP",
    grounds: list[str] | None = None,
    basis: list[str] | None = None,
    assertion_id: str = "prediction:1",
    text: str | None = None,
):
    c = card()
    a = axis()

    observable = (
        c.predicted_observations[0]
        .observable
    )

    return AxisInferenceAssertionDraft(
        assertion_id=assertion_id,
        assertion_kind="prediction",
        assertion_text=(
            observable
            if text is None
            else text
        ),
        source_class=source_class,
        action=action,
        grounded_statement_ids=(
            ["stmt:1"]
            if grounds is None
            else grounds
        ),
        axis_basis=(
            [a.label]
            if basis is None
            else basis
        ),
        specificity_tags=[
            "interaction",
        ],
        rationale="Synthetic prediction review.",
    )


def draft(
    assertions,
    *,
    risk="moderate",
):
    return AxisInferenceReviewDraft(
        assertions=assertions,
        overall_risk=risk,
        interpretation="Synthetic review.",
    )


class FakeBackend:
    backend_name = "fake_axis_inference"
    model_name = "fake-model"

    def __init__(
        self,
        value: AxisInferenceReviewDraft,
    ) -> None:
        self.value = value
        self.calls = 0

    def review(self, prompt):
        self.calls += 1

        return AxisInferenceGeneration(
            draft=self.value,
        )


def compile_review(value):
    ctx = context()
    ax = axis()
    c = card()

    prompt = (
        DiscoveryAxisInferencePromptAssembler()
        .build(
            ctx,
            ax,
            c,
        )
    )

    return AxisInferenceReviewCompiler().compile(
        context=ctx,
        axis=ax,
        card=c,
        prompt=prompt,
        draft=value,
    )


def test_prompt_exposes_only_selected_grounded_ids() -> None:
    prompt = (
        DiscoveryAxisInferencePromptAssembler()
        .build(
            context(),
            axis(),
            card(),
        )
    )

    assert '"stmt:1"' in prompt.user_prompt
    assert '"stmt:2"' in prompt.user_prompt

    # The unselected statement is not supplied as a positive premise
    # and cannot become critic support.
    assert '"stmt:other"' not in prompt.user_prompt

    assert (
        f'"central:{card().hypothesis_id}"'
        in prompt.user_prompt
    )

    assert (
        '"prediction:1"'
        in prompt.user_prompt
    )


def test_allowed_axis_basis_is_explicit_and_bounded() -> None:
    rows = allowed_axis_basis(
        axis()
    )

    assert rows == (
        axis().label,
        (
            "copper-associated composition | "
            "PROMOTES | SERS performance"
        ),
    )


def test_bounded_h1_like_review_passes() -> None:
    review = compile_review(
        draft([
            central(),
            pred(),
        ])
    )

    assert review.status == "pass"
    assert review.reason_codes == []


def test_h2_like_unsupported_prediction_requires_reframe() -> None:
    review = compile_review(
        draft([
            central(),
            pred(
                source_class=
                    "X_UNSUPPORTED_SPECIFICITY",
                action="REFRAME",
            ),
        ], risk="high")
    )

    assert (
        review.status
        == "reframe_required"
    )

    assert (
        "unsupported_specificity"
        in review.reason_codes
    )

    assert (
        "reframe_required"
        in review.reason_codes
    )


def test_missing_prediction_review_is_rejected() -> None:
    with pytest.raises(
        AxisInferenceReviewValidationError,
        match="missing inference assertions",
    ):
        compile_review(
            draft([
                central(),
            ])
        )


def test_invented_assertion_is_rejected() -> None:
    with pytest.raises(
        AxisInferenceReviewValidationError,
        match="unknown inference assertions",
    ):
        compile_review(
            draft([
                central(),
                pred(
                    assertion_id=
                        "prediction:invented",
                ),
            ])
        )


def test_assertion_text_must_match_source_card() -> None:
    with pytest.raises(
        AxisInferenceReviewValidationError,
        match="assertion_text",
    ):
        compile_review(
            draft([
                central(),
                pred(
                    text="Invented stronger prediction.",
                ),
            ])
        )


def test_nonselected_grounded_reference_is_rejected() -> None:
    with pytest.raises(
        AxisInferenceReviewValidationError,
        match="non-selected grounded statement",
    ):
        compile_review(
            draft([
                central(),
                pred(
                    grounds=[
                        "stmt:1",
                        "stmt:other",
                    ],
                ),
            ])
        )


def test_invented_axis_basis_is_rejected() -> None:
    with pytest.raises(
        AxisInferenceReviewValidationError,
        match="unknown axis basis",
    ):
        compile_review(
            draft([
                central(),
                pred(
                    basis=[
                        "Invented axis mechanism"
                    ],
                ),
            ])
        )


def test_grounded_class_requires_grounded_reference() -> None:
    with pytest.raises(
        AxisInferenceReviewValidationError,
        match=(
            "G_GROUNDED requires "
            "grounded_statement_ids"
        ),
    ):
        compile_review(
            draft([
                central(
                    source_class=
                        "G_GROUNDED",
                    action="KEEP",
                    grounds=[],
                    basis=[],
                ),
                pred(),
            ])
        )


def test_axis_class_requires_axis_basis() -> None:
    with pytest.raises(
        AxisInferenceReviewValidationError,
        match="A_AXIS requires axis_basis",
    ):
        compile_review(
            draft([
                central(
                    source_class="A_AXIS",
                    action=
                        "KEEP_HYPOTHETICAL",
                    grounds=[],
                    basis=[],
                ),
                pred(),
            ])
        )


def test_synthesis_requires_both_ground_and_axis() -> None:
    with pytest.raises(
        AxisInferenceReviewValidationError,
        match=(
            "S_BOUNDED_SYNTHESIS requires "
            "axis_basis"
        ),
    ):
        compile_review(
            draft([
                central(
                    grounds=["stmt:1"],
                    basis=[],
                ),
                pred(),
            ])
        )


def test_standalone_critic_calls_backend_once() -> None:
    value = draft([
        central(),
        pred(),
    ])

    backend = FakeBackend(
        value
    )

    critic = (
        DiscoveryAxisInferenceCritic(
            backend
        )
    )

    outcome = critic.review(
        context(),
        axis(),
        card(),
    )

    assert backend.calls == 1
    assert outcome.review.status == "pass"
    assert (
        outcome.review.hypothesis_id
        == "hypothesis:test"
    )
    assert (
        outcome.review.axis_id
        == "axis:test"
    )


def test_review_id_is_deterministic() -> None:
    value = draft([
        central(),
        pred(),
    ])

    first = compile_review(value)
    second = compile_review(value)

    assert (
        first.review_id
        == second.review_id
    )


def test_prompt_treats_expected_direction_as_scientific_content() -> None:
    from pipeline_core.discovery.discovery_axis_inference_prompt import (
        SYSTEM_PROMPT,
    )

    assert (
        "Treat expected_direction as part of the scientific assertion."
        in SYSTEM_PROMPT
    )

    assert (
        "increase, decrease, shift, and non_monotonic each assert additional"
        in SYSTEM_PROMPT
    )

    assert (
        "OPEN_DIRECTION, REFRAME, or REMOVE"
        in SYSTEM_PROMPT
    )


def test_prompt_limits_synthesis_to_minimum_connection_or_direct_test() -> None:
    from pipeline_core.discovery.discovery_axis_inference_prompt import (
        SYSTEM_PROMPT,
    )

    assert (
        "limited to the minimum new relation needed to"
        in SYSTEM_PROMPT
    )

    assert (
        "or a prediction that directly tests that central"
        in SYSTEM_PROMPT
    )

    assert (
        "It is not a license to add a second downstream"
        in SYSTEM_PROMPT
    )

    assert (
        "secondary mechanistic or descriptor consequence"
        in SYSTEM_PROMPT
    )
