from __future__ import annotations

from domains.sers.hypothesis_context_contracts import (
    HypothesisContextInterpretationDraft,
)
from domains.sers.hypothesis_context_interpreter import (
    _canonicalize_interpretation_draft,
    _is_reference_integrity_issue,
)
from domains.sers.hypothesis_context_prompt import (
    HYPOTHESIS_CONTEXT_PROMPT_VERSION,
    SYSTEM_PROMPT,
)


def _draft() -> HypothesisContextInterpretationDraft:
    return (
        HypothesisContextInterpretationDraft
        .model_validate(
            {
                "hypothesis_id":
                    "model-reproduced-id",

                "source_signature_ids": [
                    "model-reproduced-signature",
                ],

                "assertions": [
                    {
                        "assertion_id":
                            "central:h1",

                        "assertion_kind":
                            "central",

                        "assertion_text":
                            "Gold substrate response varies.",

                        "mentions": [
                            {
                                "mention_id":
                                    "m1",

                                "mention_text":
                                    "Gold substrate",

                                "source_fact_ids":
                                    [],

                                "asserted_dimension":
                                    "substrate",

                                "asserted_role":
                                    "plasmonic_substrate",

                                "asserted_owner_label":
                                    None,

                                "asserted_owner_type":
                                    None,

                                "asserted_relation":
                                    None,

                                "treatment":
                                    "introduce",

                                "experimental_role":
                                    "unspecified",

                                "rationale":
                                    "Explicit substrate mention.",
                            },
                        ],
                    },

                    {
                        "assertion_id":
                            "prediction:p1",

                        "assertion_kind":
                            "prediction",

                        "assertion_text":
                            "Gold morphology may vary.",

                        # Deliberately repeats m1 across assertions.
                        "mentions": [
                            {
                                "mention_id":
                                    "m1",

                                "mention_text":
                                    "Gold morphology",

                                "source_fact_ids":
                                    [],

                                "asserted_dimension":
                                    "morphology",

                                "asserted_role":
                                    "morphology",

                                "asserted_owner_label":
                                    None,

                                "asserted_owner_type":
                                    None,

                                "asserted_relation":
                                    None,

                                "treatment":
                                    "introduce",

                                "experimental_role":
                                    "unspecified",

                                "rationale":
                                    "Explicit morphology mention.",
                            },
                        ],
                    },
                ],
            }
        )
    )


def test_canonicalization_namespaces_mention_ids_globally() -> None:
    draft = _draft()

    canonical = (
        _canonicalize_interpretation_draft(
            draft=draft,
            authoritative_hypothesis_id="h1",
            source_signatures=[],
        )
    )

    mentions = [
        mention
        for assertion in canonical.assertions
        for mention in assertion.mentions
    ]

    ids = [
        mention.mention_id
        for mention in mentions
    ]

    assert ids == [
        "central:h1:mention:0",
        "prediction:p1:mention:0",
    ]

    assert len(ids) == len(set(ids))

    # Canonicalization must not change semantic content.
    assert (
        mentions[0].mention_text
        == "Gold substrate"
    )
    assert (
        mentions[1].mention_text
        == "Gold morphology"
    )
    assert (
        mentions[0].treatment
        == "introduce"
    )
    assert (
        mentions[1].treatment
        == "introduce"
    )


def test_reference_integrity_classifier_is_narrow() -> None:
    assert _is_reference_integrity_issue(
        "m1: unknown source fact fact:hallucinated"
    )

    assert _is_reference_integrity_issue(
        "duplicate global mention_id: m1"
    )

    assert _is_reference_integrity_issue(
        "m2: mention_text is not an exact assertion "
        "span after whitespace normalization"
    )

    # Scientific/context-semantic contract violations must stay
    # fail-closed and must not enter reference repair.
    assert not _is_reference_integrity_issue(
        "m3: generalize cannot silently "
        "change context dimension"
    )

    assert not _is_reference_integrity_issue(
        "m4: reattach requires a source fact "
        "with attachment binding"
    )


def test_prompt_exposes_global_mention_id_contract() -> None:
    assert (
        HYPOTHESIS_CONTEXT_PROMPT_VERSION
        == (
            "sers-hypothesis-context-prompt-"
            "v1.3-bounded-contract-repair"
        )
    )

    assert (
        "mention_id values must be globally unique "
        "across the complete"
        in SYSTEM_PROMPT
    )
