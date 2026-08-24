from __future__ import annotations

from types import SimpleNamespace

import pytest

from domains.sers.context_contracts import (
    SERSContextBinding,
    SERSContextFact,
    SERSContextProvenance,
    SERSContextSignature,
)
from domains.sers.hypothesis_context_contracts import (
    HypothesisContextAssertionDraft,
    HypothesisContextInterpretationDraft,
    HypothesisContextMentionDraft,
    expected_hypothesis_context_assertions,
)
from domains.sers.hypothesis_context_interpreter import (
    SERSHypothesisContextInterpreter,
)
from domains.sers.hypothesis_context_llm import (
    HypothesisContextGeneration,
    InstructorOpenAICompatibleHypothesisContextBackend,
)
from domains.sers.hypothesis_context_prompt import (
    HYPOTHESIS_CONTEXT_PROMPT_VERSION,
    SERSHypothesisContextPromptAssembler,
)
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisCard,
    HypothesisEvidenceProfile,
    PredictedObservation,
)


def _profile() -> HypothesisEvidenceProfile:
    return HypothesisEvidenceProfile(
        premise_count=1,
        gap_count=0,
        source_paper_count=1,
        candidate_premise_count=0,
        reported_premise_count=1,
        synthesis_premise_count=0,
    )


def _card() -> HypothesisCard:
    return HypothesisCard(
        hypothesis_id="hypothesis:test",
        domain_profile_id="sers_au_ag",
        source_context_id="context:test",
        source_context_sha256="sha-context",
        source_report_id="report:test",
        source_report_sha256="sha-report",
        title="test hypothesis",
        hypothesis_statement=(
            "Copper-associated context moderates "
            "nanogap response."
        ),
        hypothesis_type="context_dependency",
        premise_statement_ids=[
            "stmt:1"
        ],
        inferential_bridge=(
            "Copper context is treated as "
            "a comparison variable."
        ),
        predicted_observations=[
            PredictedObservation(
                observation_id="prediction:1",
                observable=(
                    "Gap response differs between "
                    "copper-conditioned and "
                    "copper-free structures"
                ),
                expected_direction=
                    "qualitative_change",
                rationale=(
                    "Copper is proposed as "
                    "a moderator."
                ),
            )
        ],
        falsification_criteria=[],
        assumptions=[
            (
                "Morphology is held comparable "
                "across copper conditions."
            )
        ],
        evidence_profile=_profile(),
    )


def _signature() -> SERSContextSignature:
    return SERSContextSignature(
        signature_id="sig:axis",
        domain_profile_id="sers_au_ag",
        scope="axis_inspiration",
        source_ref_id="axis:test",
        facts=[
            SERSContextFact(
                fact_id="fact:cu",
                dimension="material_identity",
                scientific_role="component",
                knowledge_state="explicit",
                value="Copper",
                binding=SERSContextBinding(
                    basis="direct_edge",
                    owner_ref_id="node:substrate",
                    owner_label=(
                        "Copper substrate with "
                        "nanostructured gold"
                    ),
                    owner_type="PlasmonicSubstrate",
                    relation="HAS_COMPONENT",
                ),
                provenance=[
                    SERSContextProvenance(
                        kind="axis_structural_edge",
                        node_ids=[
                            "node:substrate",
                            "node:cu",
                        ],
                        edge_ids=[
                            "edge:cu"
                        ],
                    )
                ],
            )
        ],
    )


def _valid_draft(
    card: HypothesisCard,
    signature: SERSContextSignature,
) -> HypothesisContextInterpretationDraft:
    assertions = [
        HypothesisContextAssertionDraft(
            assertion_id=
                row["assertion_id"],
            assertion_kind=
                row["assertion_kind"],
            assertion_text=
                row["assertion_text"],
            mentions=[],
        )
        for row in (
            expected_hypothesis_context_assertions(
                card
            )
        )
    ]

    assertions[0].mentions = [
        HypothesisContextMentionDraft(
            mention_id="mention:cu",
            mention_text=(
                "Copper-associated context"
            ),
            source_fact_ids=[
                "fact:cu"
            ],
            asserted_dimension=
                "material_identity",
            asserted_role="component",
            asserted_owner_label=(
                "experimental copper condition"
            ),
            asserted_owner_type=(
                "comparison_context"
            ),
            treatment="intentionally_vary",
            experimental_role="moderator",
            rationale=(
                "Copper is explicitly varied "
                "between compared conditions."
            ),
        )
    ]

    return HypothesisContextInterpretationDraft(
        hypothesis_id=
            card.hypothesis_id,
        source_signature_ids=[
            signature.signature_id
        ],
        assertions=assertions,
    )


def test_prompt_contains_stable_assertions_and_source_facts() -> None:
    card = _card()
    signature = _signature()

    prompt = (
        SERSHypothesisContextPromptAssembler()
        .build(
            card=card,
            source_signatures=[
                signature
            ],
        )
    )

    assert (
        prompt.prompt_version
        == HYPOTHESIS_CONTEXT_PROMPT_VERSION
    )

    assert (
        "central:hypothesis:test"
        in prompt.user_prompt
    )

    assert (
        '"fact_id": "fact:cu"'
        in prompt.user_prompt
    )

    assert (
        '"owner_label": "Copper substrate with '
        'nanostructured gold"'
        in prompt.user_prompt
    )

    assert (
        "You do NOT decide whether the hypothesis "
        "passes"
        in prompt.system_prompt
    )


def test_prompt_is_deterministic_across_signature_order() -> None:
    card = _card()
    first = _signature()

    second = first.model_copy(
        update={
            "signature_id":
                "sig:grounded",
            "scope":
                "grounded_premise",
            "source_ref_id":
                "stmt:1",
        }
    )

    assembler = (
        SERSHypothesisContextPromptAssembler()
    )

    a = assembler.build(
        card=card,
        source_signatures=[
            first,
            second,
        ],
    )

    b = assembler.build(
        card=card,
        source_signatures=[
            second,
            first,
        ],
    )

    assert a == b


def test_interpreter_compiles_fake_backend_output() -> None:
    card = _card()
    signature = _signature()

    draft = _valid_draft(
        card,
        signature,
    )

    class FakeBackend:
        backend_name = "fake"
        model_name = "fake-model"

        def interpret(
            self,
            prompt,
        ):
            return HypothesisContextGeneration(
                draft=draft,
                input_tokens=10,
                output_tokens=20,
                response_id="resp:test",
                elapsed_seconds=0.1,
            )

    outcome = (
        SERSHypothesisContextInterpreter(
            FakeBackend()
        ).interpret(
            card=card,
            source_signatures=[
                signature
            ],
        )
    )

    assert (
        outcome.interpretation.hypothesis_id
        == card.hypothesis_id
    )

    assert (
        outcome.interpretation.assertions[
            0
        ].mentions[
            0
        ].treatment
        == "intentionally_vary"
    )

    assert (
        outcome.generation.response_id
        == "resp:test"
    )


def test_openai_backend_uses_s1_response_model_and_telemetry(
    monkeypatch,
) -> None:
    from domains.sers import (
        hypothesis_context_llm as module,
    )

    card = _card()
    signature = _signature()

    draft = _valid_draft(
        card,
        signature,
    )

    captured = {}

    def fake_structured_call(
        completions_api,
        **kwargs,
    ):
        captured[
            "completions_api"
        ] = completions_api

        captured.update(
            kwargs
        )

        event = SimpleNamespace(
            provider_input_tokens=11,
            provider_output_tokens=22,
            response_id="resp:structured",
            elapsed_seconds=0.2,
        )

        return (
            draft.model_dump(
                mode="json"
            ),
            event,
        )

    monkeypatch.setattr(
        module,
        "run_instructor_structured_call",
        fake_structured_call,
    )

    completions = object()

    backend = (
        InstructorOpenAICompatibleHypothesisContextBackend(
            model="test-model",
            api_key="not-used",
        )
    )

    backend._client = (
        SimpleNamespace(
            chat=SimpleNamespace(
                completions=completions
            )
        )
    )

    prompt = (
        SERSHypothesisContextPromptAssembler()
        .build(
            card=card,
            source_signatures=[
                signature
            ],
        )
    )

    generation = backend.interpret(
        prompt
    )

    assert (
        captured["response_model"]
        is HypothesisContextInterpretationDraft
    )

    assert (
        captured["telemetry_context"][
            "pipeline"
        ]
        == "sers_hypothesis_context"
    )

    assert (
        captured["telemetry_context"][
            "stage"
        ]
        == "hypothesis_context_interpreter"
    )

    assert (
        captured["semantic_components"][
            "prompt_version"
        ]
        == HYPOTHESIS_CONTEXT_PROMPT_VERSION
    )

    assert generation.input_tokens == 11
    assert generation.output_tokens == 22

    assert (
        generation.response_id
        == "resp:structured"
    )


def test_prompt_rejects_domain_mismatch() -> None:
    card = _card()

    signature = _signature().model_copy(
        update={
            "domain_profile_id":
                "other-domain"
        }
    )

    with pytest.raises(
        ValueError,
        match="domain mismatch",
    ):
        (
            SERSHypothesisContextPromptAssembler()
            .build(
                card=card,
                source_signatures=[
                    signature
                ],
            )
        )


def test_interpreter_canonicalizes_model_source_signature_ids() -> None:
    card = _card()
    signature = _signature()

    draft = _valid_draft(
        card,
        signature,
    ).model_copy(
        update={
            # Deliberately wrong model-produced bookkeeping.
            "source_signature_ids": [
                "sig:model-invented"
            ]
        }
    )

    class FakeBackend:
        backend_name = "fake"
        model_name = "fake-model"

        def interpret(
            self,
            prompt,
        ):
            return HypothesisContextGeneration(
                draft=draft,
            )

    outcome = (
        SERSHypothesisContextInterpreter(
            FakeBackend()
        ).interpret(
            card=card,
            source_signatures=[
                signature
            ],
        )
    )

    # Raw model output is preserved for audit.
    assert (
        outcome.generation.draft.source_signature_ids
        == [
            "sig:model-invented"
        ]
    )

    # Validated interpretation uses authoritative supplied provenance.
    assert (
        outcome.interpretation.source_signature_ids
        == [
            signature.signature_id
        ]
    )


def test_prompt_explicitly_excludes_response_as_context() -> None:
    prompt = (
        SERSHypothesisContextPromptAssembler()
        .build(
            card=_card(),
            source_signatures=[
                _signature()
            ],
        )
    )

    assert (
        "local electric-field intensity/distribution "
        "and SERS intensity are"
        in prompt.system_prompt
    )

    assert (
        "not optical_condition"
        in prompt.system_prompt
    )

    normalized_system_prompt = " ".join(
        prompt.system_prompt.split()
    )

    assert (
        "cite only source facts whose dimension "
        "equals asserted_dimension"
        in normalized_system_prompt
    )


def test_prompt_defines_introduce_as_context_coverage_only() -> None:
    prompt = (
        SERSHypothesisContextPromptAssembler()
        .build(
            card=_card(),
            source_signatures=[
                _signature()
            ],
        )
    )

    normalized = " ".join(
        prompt.system_prompt.split()
    )

    assert (
        'The treatment "introduce" has a narrow meaning'
        in normalized
    )

    assert (
        "It does NOT mean that the broader scientific evidence "
        "or grounded premises fail to support"
        in normalized
    )


def test_interpreter_projects_known_mixed_dimension_citation_noise() -> None:
    from domains.sers.hypothesis_context_interpreter import (
        SERSHypothesisContextInterpreter,
    )

    card = _card()

    base = _signature()

    substrate_fact = SERSContextFact(
        fact_id="fact:substrate",
        dimension="substrate",
        scientific_role="plasmonic_substrate",
        knowledge_state="explicit",
        value=(
            "Copper substrate with "
            "nanostructured gold"
        ),
        binding=SERSContextBinding(
            basis="node",
            owner_ref_id="node:substrate",
            owner_label=(
                "Copper substrate with "
                "nanostructured gold"
            ),
            owner_type="PlasmonicSubstrate",
        ),
        provenance=[
            SERSContextProvenance(
                kind="axis_anchor",
                node_ids=[
                    "node:substrate"
                ],
            )
        ],
    )

    signature = base.model_copy(
        update={
            "facts": [
                *base.facts,
                substrate_fact,
            ]
        }
    )

    assertions = [
        HypothesisContextAssertionDraft(
            assertion_id=
                row["assertion_id"],
            assertion_kind=
                row["assertion_kind"],
            assertion_text=
                row["assertion_text"],
            mentions=[],
        )
        for row in (
            expected_hypothesis_context_assertions(
                card
            )
        )
    ]

    assertions[0].mentions = [
        HypothesisContextMentionDraft(
            mention_id="mention:mixed",
            mention_text=(
                "Copper-associated context"
            ),
            source_fact_ids=[
                "fact:cu",
                "fact:substrate",
            ],
            asserted_dimension=
                "material_identity",
            asserted_role=
                "component",
            asserted_owner_label=
                "Copper-associated context",
            asserted_owner_type=
                "comparison_context",
            treatment=
                "preserve",
            experimental_role=
                "comparison_context",
            rationale=(
                "Copper identity is preserved; "
                "the substrate citation is adjacent "
                "citation noise."
            ),
        )
    ]

    draft = HypothesisContextInterpretationDraft(
        hypothesis_id=
            card.hypothesis_id,
        source_signature_ids=[
            "sig:model-noise"
        ],
        assertions=assertions,
    )

    class FakeBackend:
        backend_name = "fake"
        model_name = "fake-model"

        def interpret(
            self,
            prompt,
        ):
            return HypothesisContextGeneration(
                draft=draft,
            )

    outcome = (
        SERSHypothesisContextInterpreter(
            FakeBackend()
        ).interpret(
            card=card,
            source_signatures=[
                signature
            ],
        )
    )

    raw_ids = (
        outcome.generation
        .draft.assertions[0]
        .mentions[0]
        .source_fact_ids
    )

    canonical_ids = (
        outcome.canonical_draft
        .assertions[0]
        .mentions[0]
        .source_fact_ids
    )

    assert raw_ids == [
        "fact:cu",
        "fact:substrate",
    ]

    assert canonical_ids == [
        "fact:cu"
    ]


def test_interpreter_does_not_hide_true_dimension_mismatch() -> None:
    from domains.sers.hypothesis_context_interpreter import (
        HypothesisContextInterpreterValidationError,
        SERSHypothesisContextInterpreter,
    )

    card = _card()
    signature = _signature()

    assertions = [
        HypothesisContextAssertionDraft(
            assertion_id=
                row["assertion_id"],
            assertion_kind=
                row["assertion_kind"],
            assertion_text=
                row["assertion_text"],
            mentions=[],
        )
        for row in (
            expected_hypothesis_context_assertions(
                card
            )
        )
    ]

    assertions[0].mentions = [
        HypothesisContextMentionDraft(
            mention_id="mention:true-mismatch",
            mention_text=(
                "Copper-associated context"
            ),
            source_fact_ids=[
                "fact:cu"
            ],
            # Source fact is material_identity, but model
            # claims this is substrate preservation.
            asserted_dimension=
                "substrate",
            asserted_role=
                "plasmonic_substrate",
            treatment=
                "preserve",
            rationale=(
                "Deliberately invalid dimension."
            ),
        )
    ]

    draft = HypothesisContextInterpretationDraft(
        hypothesis_id=
            card.hypothesis_id,
        source_signature_ids=[
            signature.signature_id
        ],
        assertions=assertions,
    )

    class FakeBackend:
        backend_name = "fake"
        model_name = "fake-model"

        def interpret(
            self,
            prompt,
        ):
            return HypothesisContextGeneration(
                draft=draft,
            )

    with pytest.raises(
        HypothesisContextInterpreterValidationError
    ) as caught:
        (
            SERSHypothesisContextInterpreter(
                FakeBackend()
            ).interpret(
                card=card,
                source_signatures=[
                    signature
                ],
            )
        )

    error = caught.value

    assert (
        error.generation.draft
        is draft
    )

    assert (
        error.canonical_draft
        .assertions[0]
        .mentions[0]
        .source_fact_ids
        == [
            "fact:cu"
        ]
    )

    assert any(
        "cannot silently change context dimension"
        in issue
        for issue in error.issues
    )


def test_interpreter_preserves_unknown_source_fact_for_rejection() -> None:
    from domains.sers.hypothesis_context_interpreter import (
        HypothesisContextInterpreterValidationError,
        SERSHypothesisContextInterpreter,
    )

    card = _card()
    signature = _signature()

    draft = _valid_draft(
        card,
        signature,
    )

    first = draft.assertions[0]
    mention = first.mentions[0]

    noisy_mention = mention.model_copy(
        update={
            "source_fact_ids": [
                "fact:cu",
                "fact:not-real",
            ]
        }
    )

    noisy_assertion = first.model_copy(
        update={
            "mentions": [
                noisy_mention
            ]
        }
    )

    noisy_draft = draft.model_copy(
        update={
            "assertions": [
                noisy_assertion,
                *draft.assertions[1:],
            ]
        }
    )

    class FakeBackend:
        backend_name = "fake"
        model_name = "fake-model"

        def interpret(
            self,
            prompt,
        ):
            return HypothesisContextGeneration(
                draft=noisy_draft,
            )

    with pytest.raises(
        HypothesisContextInterpreterValidationError
    ) as caught:
        (
            SERSHypothesisContextInterpreter(
                FakeBackend()
            ).interpret(
                card=card,
                source_signatures=[
                    signature
                ],
            )
        )

    canonical_ids = (
        caught.value
        .canonical_draft
        .assertions[0]
        .mentions[0]
        .source_fact_ids
    )

    assert (
        "fact:not-real"
        in canonical_ids
    )

    assert any(
        "unknown source fact"
        in issue
        for issue in caught.value.issues
    )
