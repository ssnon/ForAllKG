from __future__ import annotations

import pytest

from domains.sers.context_comparator import (
    SERSContextComparatorError,
    SERSHypothesisContextComparator,
)
from domains.sers.context_contracts import (
    SERSContextBinding,
    SERSContextFact,
    SERSContextProvenance,
    SERSContextSignature,
)
from domains.sers.hypothesis_context_contracts import (
    HypothesisContextAssertionDraft,
    HypothesisContextInterpretation,
    HypothesisContextMentionDraft,
)


HID = "hypothesis:test"
SID = "sig:source"


def _source_fact(
    *,
    fact_id: str,
    dimension: str,
    role: str,
    value: str,
    owner: str,
    owner_type: str = "PlasmonicSubstrate",
    knowledge_state: str = "explicit",
) -> SERSContextFact:
    return SERSContextFact(
        fact_id=fact_id,
        dimension=dimension,
        scientific_role=role,
        knowledge_state=
            knowledge_state,
        value=(
            value
            if knowledge_state
            == "explicit"
            else None
        ),
        normalized_value=(
            value.lower()
            if knowledge_state
            == "explicit"
            else None
        ),
        binding=SERSContextBinding(
            basis="node",
            owner_ref_id=(
                "node:"
                + fact_id
            ),
            owner_label=owner,
            owner_type=
                owner_type,
        ),
        provenance=[
            SERSContextProvenance(
                kind="axis_anchor",
                node_ids=[
                    "node:"
                    + fact_id
                ],
            )
        ],
        tags=[],
    )


def _signature(
    *facts: SERSContextFact,
) -> SERSContextSignature:
    return SERSContextSignature(
        signature_id=SID,
        domain_profile_id=
            "sers_au_ag",
        scope="axis_inspiration",
        source_ref_id="axis:test",
        facts=list(facts),
    )


def _mention(
    *,
    mention_id: str,
    text: str,
    source_fact_ids: list[str],
    dimension: str,
    role: str,
    treatment: str,
    owner: str | None,
    experimental_role: str = "unspecified",
) -> HypothesisContextMentionDraft:
    return HypothesisContextMentionDraft(
        mention_id=
            mention_id,
        mention_text=
            text,
        source_fact_ids=
            source_fact_ids,
        asserted_dimension=
            dimension,
        asserted_role=
            role,
        asserted_owner_label=
            owner,
        asserted_owner_type=(
            "HypothesisContextOwner"
            if owner is not None
            else None
        ),
        treatment=
            treatment,
        experimental_role=
            experimental_role,
        rationale=
            "Synthetic comparator test.",
    )


def _interpretation(
    *mentions:
        HypothesisContextMentionDraft,
    assertion_id: str = "central:hypothesis:test",
    assertion_text: str | None = None,
) -> HypothesisContextInterpretation:
    if assertion_text is None:
        assertion_text = " ".join(
            row.mention_text
            for row in mentions
        )

    return HypothesisContextInterpretation(
        hypothesis_id=HID,
        source_signature_ids=[
            SID
        ],
        assertions=[
            HypothesisContextAssertionDraft(
                assertion_id=
                    assertion_id,
                assertion_kind=
                    "central",
                assertion_text=
                    assertion_text,
                mentions=list(
                    mentions
                ),
            )
        ],
    )


def _compare(
    signature,
    interpretation,
):
    return (
        SERSHypothesisContextComparator()
        .compare(
            interpretation=
                interpretation,
            source_signatures=[
                signature
            ],
            domain_profile_id=
                "sers_au_ag",
        )
    )


def test_preserve_becomes_match() -> None:
    fact = _source_fact(
        fact_id="fact:gap",
        dimension="gap_regime",
        role="gap_regime",
        value="10 nm gap",
        owner="Au@Ag nanoparticle",
    )

    mention = _mention(
        mention_id="m1",
        text="nanogap",
        source_fact_ids=[
            fact.fact_id
        ],
        dimension="gap_regime",
        role="gap_regime",
        treatment="preserve",
        owner="Au@Ag nanoparticle",
    )

    review = _compare(
        _signature(fact),
        _interpretation(
            mention
        ),
    )

    assert review.status == "pass"
    assert len(review.findings) == 1
    assert (
        review.findings[0].status
        == "match"
    )
    assert (
        review.findings[0].severity
        == "info"
    )


def test_generalize_becomes_compatible_extension() -> None:
    fact = _source_fact(
        fact_id="fact:substrate",
        dimension="substrate",
        role="plasmonic_substrate",
        value="Au@Ag substrate",
        owner="Au@Ag substrate",
    )

    mention = _mention(
        mention_id="m1",
        text="Au–Ag structures",
        source_fact_ids=[
            fact.fact_id
        ],
        dimension="substrate",
        role="plasmonic_substrate",
        treatment="generalize",
        owner="Au–Ag structures",
    )

    review = _compare(
        _signature(fact),
        _interpretation(
            mention
        ),
    )

    assert (
        review.findings[0].status
        == "compatible_extension"
    )
    assert review.status == "pass"


def test_intentional_variation_is_non_actionable() -> None:
    fact = _source_fact(
        fact_id="fact:gap",
        dimension="gap_regime",
        role="gap_regime",
        value="10 nm gap",
        owner="nanoparticle",
    )

    mention = _mention(
        mention_id="m1",
        text="nanogap size",
        source_fact_ids=[
            fact.fact_id
        ],
        dimension="gap_regime",
        role="gap_regime",
        treatment=
            "intentionally_vary",
        owner="Au–Ag structure",
        experimental_role=
            "experimental_variable",
    )

    review = _compare(
        _signature(fact),
        _interpretation(
            mention
        ),
    )

    assert (
        review.findings[0].status
        == "intentional_variation"
    )
    assert review.status == "pass"


def test_introduce_is_typed_context_unknown_not_unsupported() -> None:
    fact = _source_fact(
        fact_id="fact:metal",
        dimension="material_identity",
        role="component",
        value="Gold",
        owner="Au substrate",
    )

    mention = _mention(
        mention_id="m1",
        text="LSPR positioning",
        source_fact_ids=[],
        dimension="optical_condition",
        role="optical_condition",
        treatment="introduce",
        owner="plasmonic structure",
        experimental_role=
            "controlled_constant",
    )

    review = _compare(
        _signature(fact),
        _interpretation(
            mention
        ),
    )

    assert review.status == "pass_with_unknowns"

    unknown = [
        row
        for row in review.findings
        if row.status == "unknown"
    ]

    assert len(unknown) == 1
    assert unknown[0].severity == "warning"

    coverage = [
        row
        for row in review.signatures
        if row.scope == "question"
    ]

    assert len(coverage) == 1
    assert (
        coverage[0].facts[0].knowledge_state
        == "unknown"
    )


def test_h1_like_same_role_active_reattach_is_context_conflation() -> None:
    fact = _source_fact(
        fact_id="fact:pyramid",
        dimension="morphology",
        role="morphology",
        value=(
            "3D-Si with inserted "
            "small pyramid"
        ),
        owner="3D-Si substrate",
    )

    mention = _mention(
        mention_id="m1",
        text=(
            "inserted-pyramid "
            "geometry"
        ),
        source_fact_ids=[
            fact.fact_id
        ],
        dimension="morphology",
        role="morphology",
        treatment="reattach",
        owner="nanogap",
        experimental_role=
            "experimental_variable",
    )

    review = _compare(
        _signature(fact),
        _interpretation(
            mention
        ),
    )

    finding = review.findings[0]

    assert (
        finding.status
        == "context_conflation"
    )
    assert finding.severity == "actionable"
    assert review.status == "reframe_required"


def test_h2_like_dimension_role_shift_is_role_mismatch() -> None:
    substrate = _source_fact(
        fact_id="fact:cu-substrate",
        dimension="substrate",
        role="plasmonic_substrate",
        value=(
            "Copper substrate with "
            "nanostructured silver"
        ),
        owner=(
            "Copper substrate with "
            "nanostructured silver"
        ),
    )

    copper = _source_fact(
        fact_id="fact:cu",
        dimension="material_identity",
        role="component",
        value="Copper",
        owner=(
            "Copper substrate with "
            "nanostructured silver"
        ),
    )

    mention = _mention(
        mention_id="m1",
        text=(
            "copper-containing "
            "interfacial environment"
        ),
        source_fact_ids=[
            substrate.fact_id,
            copper.fact_id,
        ],
        dimension="environment",
        role="environment",
        treatment="reattach",
        owner="Au–Ag structures",
        experimental_role="moderator",
    )

    review = _compare(
        _signature(
            substrate,
            copper,
        ),
        _interpretation(
            mention
        ),
    )

    assert len(review.findings) == 1

    finding = review.findings[0]

    assert (
        finding.status
        == "role_mismatch"
    )
    assert finding.severity == "actionable"
    assert review.status == "reframe_required"


def test_same_role_controlled_constant_reattach_is_conservative_unknown() -> None:
    fact = _source_fact(
        fact_id="fact:morph",
        dimension="morphology",
        role="morphology",
        value="Nanostar morphology",
        owner="AuNSt@Ag",
    )

    mention = _mention(
        mention_id="m1",
        text=(
            "Au–Ag nanostructure morphology"
        ),
        source_fact_ids=[
            fact.fact_id
        ],
        dimension="morphology",
        role="morphology",
        treatment="reattach",
        owner="Au–Ag nanostructure",
        experimental_role=
            "controlled_constant",
    )

    review = _compare(
        _signature(fact),
        _interpretation(
            mention
        ),
    )

    assert (
        review.findings[0].status
        == "unknown"
    )
    assert (
        review.status
        == "pass_with_unknowns"
    )


def test_combine_same_dimension_and_role_is_compatible_extension() -> None:
    gold = _source_fact(
        fact_id="fact:au",
        dimension="material_identity",
        role="component",
        value="Gold",
        owner="Au–Ag substrate",
    )

    silver = _source_fact(
        fact_id="fact:ag",
        dimension="material_identity",
        role="component",
        value="Silver",
        owner="Au–Ag substrate",
    )

    mention = _mention(
        mention_id="m1",
        text="Au–Ag composition",
        source_fact_ids=[
            gold.fact_id,
            silver.fact_id,
        ],
        dimension="material_identity",
        role="component",
        treatment="combine",
        owner="Au–Ag structure",
        experimental_role=
            "controlled_constant",
    )

    review = _compare(
        _signature(
            gold,
            silver,
        ),
        _interpretation(
            mention
        ),
    )

    assert (
        review.findings[0].status
        == "compatible_extension"
    )
    assert review.status == "pass"


def test_repeated_same_transformation_is_one_semantic_finding_family() -> None:
    fact = _source_fact(
        fact_id="fact:pyramid",
        dimension="morphology",
        role="morphology",
        value="Inserted pyramid",
        owner="3D-Si substrate",
    )

    first = _mention(
        mention_id="central_m1",
        text="inserted-pyramid geometry",
        source_fact_ids=[
            fact.fact_id
        ],
        dimension="morphology",
        role="morphology",
        treatment="reattach",
        owner="nanogap",
        experimental_role=
            "experimental_variable",
    )

    second = _mention(
        mention_id="prediction_m1",
        text="inserted-pyramid geometry",
        source_fact_ids=[
            fact.fact_id
        ],
        dimension="morphology",
        role="morphology",
        treatment="reattach",
        owner="nanogap",
        experimental_role=
            "experimental_variable",
    )

    interpretation = (
        HypothesisContextInterpretation(
            hypothesis_id=HID,
            source_signature_ids=[
                SID
            ],
            assertions=[
                HypothesisContextAssertionDraft(
                    assertion_id=
                        "central:hypothesis:test",
                    assertion_kind="central",
                    assertion_text=(
                        "inserted-pyramid geometry "
                        "changes response"
                    ),
                    mentions=[first],
                ),
                HypothesisContextAssertionDraft(
                    assertion_id=
                        "prediction:test",
                    assertion_kind="prediction",
                    assertion_text=(
                        "inserted-pyramid geometry "
                        "changes field"
                    ),
                    mentions=[second],
                ),
            ],
        )
    )

    review = _compare(
        _signature(fact),
        interpretation,
    )

    actionable = [
        row
        for row in review.findings
        if (
            row.status
            == "context_conflation"
        )
    ]

    assert len(actionable) == 1
    assert (
        len(
            actionable[0].right_fact_ids
        )
        == 2
    )

    assertion_tags = [
        tag
        for tag in actionable[0].tags
        if tag.startswith(
            "assertion:"
        )
    ]

    assert len(assertion_tags) == 2


def test_unknown_source_knowledge_remains_unknown() -> None:
    fact = _source_fact(
        fact_id="fact:state",
        dimension="material_state",
        role="material_state",
        value="ignored",
        owner="Copper",
        knowledge_state="unknown",
    )

    mention = _mention(
        mention_id="m1",
        text="copper material state",
        source_fact_ids=[
            fact.fact_id
        ],
        dimension="material_state",
        role="material_state",
        treatment="preserve",
        owner="Copper",
    )

    review = _compare(
        _signature(fact),
        _interpretation(
            mention
        ),
    )

    assert (
        review.findings[0].status
        == "unknown"
    )
    assert (
        review.status
        == "pass_with_unknowns"
    )


def test_source_signature_set_mismatch_fails_closed() -> None:
    fact = _source_fact(
        fact_id="fact:gap",
        dimension="gap_regime",
        role="gap_regime",
        value="10 nm",
        owner="particle",
    )

    signature = _signature(
        fact
    )

    interpretation = (
        _interpretation(
            _mention(
                mention_id="m1",
                text="nanogap",
                source_fact_ids=[
                    fact.fact_id
                ],
                dimension="gap_regime",
                role="gap_regime",
                treatment="preserve",
                owner="particle",
            )
        ).model_copy(
            update={
                "source_signature_ids": [
                    "sig:not-supplied"
                ]
            }
        )
    )

    with pytest.raises(
        SERSContextComparatorError,
        match="do not match supplied",
    ):
        _compare(
            signature,
            interpretation,
        )


def test_repeated_role_mismatch_across_owner_wording_is_one_semantic_family() -> None:
    substrate = _source_fact(
        fact_id="fact:cu-substrate-repeat",
        dimension="substrate",
        role="plasmonic_substrate",
        value=(
            "Copper substrate with "
            "nanostructured silver"
        ),
        owner=(
            "Copper substrate with "
            "nanostructured silver"
        ),
    )

    copper = _source_fact(
        fact_id="fact:cu-repeat",
        dimension="material_identity",
        role="component",
        value="Copper",
        owner=(
            "Copper substrate with "
            "nanostructured silver"
        ),
    )

    source = _signature(
        substrate,
        copper,
    )

    owners = [
        "Au–Ag structures",
        "Au–Ag environment",
        "Au–Ag substrate",
    ]

    assertions = []

    for index, owner in enumerate(
        owners
    ):
        mention = _mention(
            mention_id=f"cu_m{index}",
            text=(
                "copper-associated "
                "interfacial environment"
            ),
            source_fact_ids=[
                substrate.fact_id,
                copper.fact_id,
            ],
            dimension="environment",
            role="environment",
            treatment="reattach",
            owner=owner,
            experimental_role=(
                "moderator"
                if index != 1
                else "experimental_variable"
            ),
        )

        assertions.append(
            HypothesisContextAssertionDraft(
                assertion_id=(
                    f"assertion:{index}"
                ),
                assertion_kind=(
                    "central"
                    if index == 0
                    else "prediction"
                ),
                assertion_text=(
                    "Copper-associated context "
                    "moderates Au–Ag response."
                ),
                mentions=[
                    mention
                ],
            )
        )

    interpretation = (
        HypothesisContextInterpretation(
            hypothesis_id=HID,
            source_signature_ids=[
                SID
            ],
            assertions=
                assertions,
        )
    )

    review = _compare(
        source,
        interpretation,
    )

    role_mismatches = [
        row
        for row in review.findings
        if (
            row.status
            == "role_mismatch"
        )
    ]

    assert len(
        role_mismatches
    ) == 1

    finding = role_mismatches[0]

    assert (
        finding.severity
        == "actionable"
    )

    # All hypothesis-side occurrences remain preserved
    # as right-side provenance.
    assert len(
        finding.right_fact_ids
    ) == 3

    assert {
        tag
        for tag in finding.tags
        if tag.startswith(
            "assertion:"
        )
    } == {
        "assertion:assertion:0",
        "assertion:assertion:1",
        "assertion:assertion:2",
    }

    assert {
        tag
        for tag in finding.tags
        if tag.startswith(
            "experimental_role:"
        )
    } == {
        "experimental_role:moderator",
        (
            "experimental_role:"
            "experimental_variable"
        ),
    }

    assert (
        review.status
        == "reframe_required"
    )
