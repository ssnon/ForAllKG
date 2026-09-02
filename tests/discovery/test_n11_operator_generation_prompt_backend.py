from types import SimpleNamespace

import pytest

from pipeline_core.discovery.nonobviousness_mechanism_semantics_contracts import (
    MechanismOperatorPolicyResult,
    MechanismSemanticDraft,
)
from pipeline_core.discovery.nonobviousness_operator_generation_contracts import (
    N11OperatorCandidateDraft,
    N11OperatorFalsificationDraft,
    N11OperatorGenerationDraft,
    N11OperatorPredictionDraft,
)
from pipeline_core.discovery.nonobviousness_operator_generation_llm import (
    OpenRouterN11OperatorGenerationBackend,
)
from pipeline_core.discovery.nonobviousness_operator_generation_prompt import (
    N11OperatorGenerationPromptAssembler,
)


def semantic_draft():
    return MechanismSemanticDraft(
        classification=(
            "PARTIAL_OVERLAP_"
            "WITH_DISTINCT_COMPONENT"
        ),

        shared_mechanistic_components=[
            "electromagnetic enhancement",
        ],

        baseline_only_components=[
            "plasmon hybridization",
        ],

        supplemental_only_components=[
            "chemical enhancement",
        ],

        task_relation_grounded=False,

        reason_summary=(
            "The supplemental mechanism overlaps "
            "with the baseline EM component and "
            "adds chemical enhancement."
        ),

        epistemic_cautions=[
            (
                "The supplied evidence does not "
                "ground spacing-to-chemical modulation."
            )
        ],

        confidence="HIGH",
    )


def policy():
    return MechanismOperatorPolicyResult(
        supply_geometry=
            "COMMON_ANCHOR_CONTEXT",

        semantic_classification=(
            "PARTIAL_OVERLAP_"
            "WITH_DISTINCT_COMPONENT"
        ),

        task_relation_grounded=
            False,

        hypothesis_bound_gap_available=
            True,

        grounded_design_lever_available=
            True,

        explicit_competition_signal=
            False,

        explicit_switch_signal=
            False,

        eligible_operators=[
            "MECHANISM_AUGMENTATION",
            "RELATIVE_CONTRIBUTION_SHIFT",
        ],

        blocked_operators={
            "PATHWAY_COMPETITION": [
                "NO_EXPLICIT_COMPETITION_SIGNAL"
            ],

            "MECHANISM_SWITCH": [
                (
                    "NO_EXPLICIT_SWITCH_OR_"
                    "TRANSITION_SIGNAL"
                )
            ],
        },
    )


def build_prompt():
    return (
        N11OperatorGenerationPromptAssembler()
        .build(
            hypothesis_id=
                "hypothesis:test",

            scientific_task=(
                "How does spacing relate to SERS behavior?"
            ),

            requested_operator=(
                "RELATIVE_CONTRIBUTION_SHIFT"
            ),

            policy=
                policy(),

            semantic_draft=
                semantic_draft(),

            baseline_mechanism_statements=[
                {
                    "statement_id":
                        "stmt:baseline",

                    "text":
                        (
                            "Decreasing spacing was associated "
                            "with plasmon coupling."
                        ),

                    "paper_ids":
                        ["paper:A"],

                    "claim_kind":
                        "mechanism",

                    "epistemic_role":
                        "reported",
                }
            ],

            supplemental_mechanism_nodes=[
                {
                    "node_id":
                        "node:supplemental",

                    "label":
                        (
                            "Synergistic EM and chemical "
                            "enhancement"
                        ),

                    "node_text":
                        (
                            "SERS enhancement was attributed "
                            "to electromagnetic and chemical "
                            "enhancement."
                        ),

                    "source_paper_id":
                        "paper:B",
                }
            ],

            gap_statements=[
                {
                    "statement_id":
                        "stmt:gap",

                    "text":
                        (
                            "The supplied evidence does not "
                            "establish the spacing-to-SERS "
                            "mechanistic relation."
                        ),

                    "paper_ids":
                        ["paper:A"],

                    "claim_kind":
                        "scope_limit",

                    "epistemic_role":
                        "unresolved",
                }
            ],

            task_feature={
                "node_id":
                    "node:nanogap",

                "label":
                    "interparticle nanogap",
            },
        )
    )


def test_prompt_is_stable_and_builds_separate_authority_lanes():
    first = build_prompt()
    second = build_prompt()

    assert (
        first.prompt_sha256
        == second.prompt_sha256
    )

    assert (
        first.requested_operator
        == "RELATIVE_CONTRIBUTION_SHIFT"
    )

    assert (
        first
        .authority
        .allowed_baseline_statement_ids
        == ("stmt:baseline",)
    )

    assert (
        first
        .authority
        .allowed_supplemental_node_ids
        == ("node:supplemental",)
    )

    assert (
        first
        .authority
        .allowed_gap_statement_ids
        == ("stmt:gap",)
    )

    assert (
        first
        .authority
        .allowed_shared_component_ids
    )

    assert (
        first
        .authority
        .allowed_supplemental_only_component_ids
    )

    assert (
        "supplemental_nodes_are_baseline_premises"
        in first.user_prompt
    )


def test_component_ids_are_stable_and_lane_specific():
    prompt = build_prompt()

    shared_ids = set(
        prompt
        .shared_component_text_by_id
    )

    supplemental_ids = set(
        prompt
        .supplemental_only_component_text_by_id
    )

    assert shared_ids

    assert supplemental_ids

    assert (
        shared_ids
        .isdisjoint(
            supplemental_ids
        )
    )


def test_prompt_rejects_operator_not_in_policy():
    row = policy().model_copy(
        update={
            "eligible_operators": [
                "MECHANISM_AUGMENTATION"
            ]
        }
    )

    with pytest.raises(
        ValueError,
        match="not deterministically authorized",
    ):
        N11OperatorGenerationPromptAssembler().build(
            hypothesis_id="hypothesis:test",
            scientific_task="task",
            requested_operator=(
                "RELATIVE_CONTRIBUTION_SHIFT"
            ),
            policy=row,
            semantic_draft=
                semantic_draft(),
            baseline_mechanism_statements=[
                {
                    "statement_id":
                        "stmt:baseline",
                    "text":
                        "baseline mechanism",
                }
            ],
            supplemental_mechanism_nodes=[
                {
                    "node_id":
                        "node:supplemental",
                    "node_text":
                        "supplemental mechanism",
                }
            ],
            gap_statements=[
                {
                    "statement_id":
                        "stmt:gap",
                    "text":
                        "unresolved relation",
                }
            ],
            task_feature={},
        )


def test_prompt_rejects_grounded_task_relation():
    draft = semantic_draft().model_copy(
        update={
            "task_relation_grounded": True
        }
    )

    with pytest.raises(
        ValueError,
        match="unresolved task-to-supplemental",
    ):
        N11OperatorGenerationPromptAssembler().build(
            hypothesis_id="hypothesis:test",
            scientific_task="task",
            requested_operator=(
                "RELATIVE_CONTRIBUTION_SHIFT"
            ),
            policy=policy(),
            semantic_draft=draft,
            baseline_mechanism_statements=[
                {
                    "statement_id":
                        "stmt:baseline",
                    "text":
                        "baseline mechanism",
                }
            ],
            supplemental_mechanism_nodes=[
                {
                    "node_id":
                        "node:supplemental",
                    "node_text":
                        "supplemental mechanism",
                }
            ],
            gap_statements=[
                {
                    "statement_id":
                        "stmt:gap",
                    "text":
                        "gap",
                }
            ],
            task_feature={},
        )


class FakeLLM:
    def __init__(self):
        self.calls = []

        self.last_usage = SimpleNamespace(
            served_model=
                "served/test-model",

            input_tokens=
                120,

            output_tokens=
                80,

            total_tokens=
                200,
        )

    def generate_structured(
        self,
        **kwargs,
    ):
        self.calls.append(
            kwargs
        )

        prompt = build_prompt()

        shared_id = (
            prompt
            .authority
            .allowed_shared_component_ids[0]
        )

        supplemental_component_id = (
            prompt
            .authority
            .allowed_supplemental_only_component_ids[0]
        )

        return (
            N11OperatorGenerationDraft(
                candidate=
                    N11OperatorCandidateDraft(
                        local_id=
                            "candidate:1",

                        title=(
                            "Spacing-dependent "
                            "relative mechanism balance"
                        ),

                        hypothesis_statement=(
                            "Interparticle spacing may alter "
                            "the relative contribution of "
                            "electromagnetic and chemical "
                            "enhancement to SERS behavior."
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
                            shared_id
                        ],

                        supplemental_only_component_ids=[
                            supplemental_component_id
                        ],

                        relative_contribution_claim=(
                            "Spacing may shift the relative "
                            "contribution of electromagnetic "
                            "and chemical enhancement."
                        ),

                        inferential_bridge=(
                            "The mechanisms are separately "
                            "grounded; their spacing-dependent "
                            "relative weighting is inferred."
                        ),

                        predicted_observations=[
                            N11OperatorPredictionDraft(
                                local_id=
                                    "prediction:1",

                                observable=(
                                    "relative electromagnetic-"
                                    "to-chemical contribution "
                                    "proxy across spacing"
                                ),

                                expected_direction=
                                    "shift",

                                rationale=(
                                    "The hypothesis predicts "
                                    "a change in mechanistic "
                                    "balance."
                                ),
                            )
                        ],

                        discriminating_observation_local_id=(
                            "prediction:1"
                        ),

                        falsification_criteria=[
                            N11OperatorFalsificationDraft(
                                local_id=
                                    "falsifier:1",

                                prediction_local_id=
                                    "prediction:1",

                                falsifying_outcome=(
                                    "The relative balance "
                                    "remains unchanged while "
                                    "total SERS varies."
                                ),
                            )
                        ],

                        assumptions=[],

                        generated_relation_status=(
                            "INFERENCE_NOT_REPORTED"
                        ),

                        task_to_supplemental_relation_grounded=
                            False,
                    ),

                abstention_reason=None,
            )
        )


def test_backend_requests_operator_generation_draft_only():
    backend = object.__new__(
        OpenRouterN11OperatorGenerationBackend
    )

    backend.model_name = (
        "requested/test-model"
    )

    backend.temperature = 0.0

    backend.reasoning_effort = (
        "medium"
    )

    backend.llm = FakeLLM()

    prompt = build_prompt()

    generation = backend.generate(
        prompt,
        generation_pass_index=1,
    )

    assert (
        generation.requested_model
        == "requested/test-model"
    )

    assert (
        generation.served_model
        == "served/test-model"
    )

    assert (
        generation.total_tokens
        == 200
    )

    call = (
        backend
        .llm
        .calls[0]
    )

    assert (
        call["response_model"]
        is N11OperatorGenerationDraft
    )

    assert (
        call["system_prompt"]
        == prompt.system_prompt
    )

    assert (
        call["prompt"]
        == prompt.user_prompt
    )


def test_backend_rejects_invalid_generation_pass():
    backend = object.__new__(
        OpenRouterN11OperatorGenerationBackend
    )

    backend.model_name = "model"
    backend.temperature = 0.0
    backend.reasoning_effort = "medium"
    backend.llm = FakeLLM()

    with pytest.raises(
        ValueError,
        match="generation_pass_index",
    ):
        backend.generate(
            build_prompt(),
            generation_pass_index=0,
        )
