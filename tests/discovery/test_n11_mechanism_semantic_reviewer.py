from types import SimpleNamespace

import pytest

from pipeline_core.discovery.nonobviousness_mechanism_semantic_llm import (
    OpenRouterMechanismSemanticBackend,
)
from pipeline_core.discovery.nonobviousness_mechanism_semantic_prompt import (
    MechanismSemanticPromptAssembler,
)
from pipeline_core.discovery.nonobviousness_mechanism_semantics_contracts import (
    MechanismSemanticDraft,
)


def build_prompt():
    return (
        MechanismSemanticPromptAssembler()
        .build(
            hypothesis_id="hypothesis:test",

            scientific_task=(
                "How does spacing relate to SERS behavior?"
            ),

            supply_geometry=
                "COMMON_ANCHOR_CONTEXT",

            baseline_mechanism_statements=[
                {
                    "statement_id":
                        "stmt:baseline",
                    "text":
                        (
                            "Decreasing spacing was associated "
                            "with near-field plasmon coupling."
                        ),
                    "paper_ids":
                        ["paper:A"],
                    "claim_kind":
                        "mechanism",
                    "epistemic_role":
                        "reported",
                }
            ],

            task_feature={
                "node_id":
                    "node:nanogap",
                "label":
                    "Interparticle nanogap",
            },

            supplemental_mechanism_nodes=[
                {
                    "node_id":
                        "node:em_cm",
                    "label":
                        "Synergistic EM and chemical enhancement",
                    "node_text":
                        (
                            "SERS enhancement was attributed "
                            "to synergistic electromagnetic "
                            "and chemical mechanisms."
                        ),
                    "source_paper_id":
                        "paper:B",
                }
            ],

            scientific_steps=[
                {
                    "scientific_source":
                        "node:substrate",
                    "relation":
                        "HAS_STRUCTURAL_MOTIF",
                    "scientific_target":
                        "node:nanogap",
                    "traversal_direction":
                        "reverse",
                },
                {
                    "scientific_source":
                        "node:substrate",
                    "relation":
                        "SUPPORTED_MECHANISM_INTERPRETATION",
                    "scientific_target":
                        "node:em_cm",
                    "traversal_direction":
                        "forward",
                },
            ],
        )
    )


def test_prompt_is_stable_and_operator_free_contract():
    prompt_a = build_prompt()
    prompt_b = build_prompt()

    assert (
        prompt_a.prompt_sha256
        == prompt_b.prompt_sha256
    )

    assert (
        prompt_a.supply_geometry
        == "COMMON_ANCHOR_CONTEXT"
    )

    assert (
        prompt_a.baseline_support_statement_ids
        == ("stmt:baseline",)
    )

    assert (
        prompt_a.supplemental_mechanism_node_ids
        == ("node:em_cm",)
    )

    assert (
        "reviewer_has_operator_authority"
        in prompt_a.user_prompt
    )

    fields = set(
        MechanismSemanticDraft.model_fields
    )

    assert (
        "pathway_competition_search_eligible"
        not in fields
    )

    assert (
        "mechanism_switch_search_eligible"
        not in fields
    )

    assert (
        "recommended_operator_family"
        not in fields
    )


def test_prompt_requires_grounded_components():
    assembler = (
        MechanismSemanticPromptAssembler()
    )

    with pytest.raises(
        ValueError,
        match="baseline mechanism evidence",
    ):
        assembler.build(
            hypothesis_id="hypothesis:test",
            scientific_task="task",
            supply_geometry=
                "COMMON_ANCHOR_CONTEXT",
            baseline_mechanism_statements=[],
            task_feature={},
            supplemental_mechanism_nodes=[
                {
                    "node_id": "node:x",
                    "node_text": "mechanism",
                }
            ],
            scientific_steps=[],
        )

    with pytest.raises(
        ValueError,
        match="supplemental mechanism evidence",
    ):
        assembler.build(
            hypothesis_id="hypothesis:test",
            scientific_task="task",
            supply_geometry=
                "COMMON_ANCHOR_CONTEXT",
            baseline_mechanism_statements=[
                {
                    "statement_id": "stmt:x",
                    "text": "mechanism",
                }
            ],
            task_feature={},
            supplemental_mechanism_nodes=[],
            scientific_steps=[],
        )


class FakeLLM:
    def __init__(self):
        self.calls = []

        self.last_usage = SimpleNamespace(
            served_model=
                "served/test-model",
            input_tokens=
                100,
            output_tokens=
                50,
            total_tokens=
                150,
        )

    def generate_structured(
        self,
        **kwargs,
    ):
        self.calls.append(
            kwargs
        )

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
            reason_summary="test",
            epistemic_cautions=[
                (
                    "Common anchor does not "
                    "establish modulation."
                )
            ],
            confidence="HIGH",
        )


def test_backend_requests_only_semantic_draft():
    backend = object.__new__(
        OpenRouterMechanismSemanticBackend
    )

    backend.model_name = "requested/test-model"
    backend.temperature = 0.0
    backend.reasoning_effort = "medium"
    backend.llm = FakeLLM()

    prompt = build_prompt()

    generation = backend.review(
        prompt,
        review_pass_index=1,
    )

    assert (
        generation.draft.classification
        == (
            "PARTIAL_OVERLAP_"
            "WITH_DISTINCT_COMPONENT"
        )
    )

    assert (
        generation.requested_model
        == "requested/test-model"
    )

    assert (
        generation.served_model
        == "served/test-model"
    )

    assert generation.total_tokens == 150

    call = backend.llm.calls[0]

    assert (
        call["response_model"]
        is MechanismSemanticDraft
    )

    assert (
        call["system_prompt"]
        == prompt.system_prompt
    )

    assert (
        call["prompt"]
        == prompt.user_prompt
    )


def test_backend_rejects_invalid_pass_index():
    backend = object.__new__(
        OpenRouterMechanismSemanticBackend
    )

    backend.model_name = "model"
    backend.temperature = 0.0
    backend.reasoning_effort = "medium"
    backend.llm = FakeLLM()

    with pytest.raises(
        ValueError,
        match="review_pass_index",
    ):
        backend.review(
            build_prompt(),
            review_pass_index=0,
        )


def test_mechanism_semantic_draft_is_openai_strict_schema_compatible():
    schema = (
        MechanismSemanticDraft
        .model_json_schema()
    )

    assert set(
        schema["properties"]
    ) == set(
        schema["required"]
    )
