from __future__ import annotations

from dac_her.hypothesis_prompt import HypothesisPrompt
from dac_her.synthesis_constituent_guidance import (
    GUIDED_PROMPT_VERSION,
    SynthesisConstituentGroup,
    SynthesisConstituentHierarchy,
    SynthesisConstituentMember,
    SynthesisConstituentPromptAugmenter,
    SynthesisConstituentSelectionPolicy,
    render_synthesis_constituent_guidance,
)


def _hierarchy() -> SynthesisConstituentHierarchy:
    return SynthesisConstituentHierarchy(
        hierarchy_id="hier:test",
        hierarchy_sha256="a" * 64,
        source_resolution_report_id="resolution:test",
        source_resolution_report_sha256="b" * 64,
        source_context_id="context:test",
        source_context_sha256="c" * 64,
        domain_profile_id="dac_her",
        groups=[
            SynthesisConstituentGroup(
                parent_statement_id="stmt:parent",
                constituent_statement_ids=[
                    "stmt:observation",
                    "stmt:mechanism",
                ],
                members=[
                    SynthesisConstituentMember(
                        family_id="family:obs",
                        family_claim_kind="observation",
                        family_paper_ids=["P1", "P2"],
                        constituent_statement_id="stmt:observation",
                        resolution_status=(
                            "resolved_to_existing_statement"
                        ),
                        resolution_basis=(
                            "contained_existing_scientific_support"
                        ),
                        exact_equivalence=False,
                    ),
                    SynthesisConstituentMember(
                        family_id="family:mech",
                        family_claim_kind="mechanism",
                        family_paper_ids=["P3"],
                        constituent_statement_id="stmt:mechanism",
                        resolution_status=(
                            "resolved_to_existing_statement"
                        ),
                        resolution_basis=(
                            "exact_existing_scientific_support"
                        ),
                        exact_equivalence=True,
                    ),
                ],
            )
        ],
    )


def test_ec2d2_policy_is_guidance_not_forcing():
    policy = SynthesisConstituentSelectionPolicy()
    assert policy.constituent_use_forced is False
    assert policy.parent_use_forbidden is False
    assert (
        policy.prefer_existing_covering_constituent_when_sufficient
        is True
    )
    assert policy.parent_allowed_for_cross_family_synthesis is True
    assert policy.contained_relation_is_exact_equivalence is False


def test_ec2d2_guidance_distinguishes_contained_from_exact():
    text = render_synthesis_constituent_guidance(
        _hierarchy()
    )

    assert "MINIMALLY-SUFFICIENT PREMISE PRINCIPLE" in text
    assert "stmt:parent" in text
    assert "stmt:observation" in text
    assert "coverage_relation=contained" in text
    assert "stmt:mechanism" in text
    assert "coverage_relation=exact" in text
    assert "It is NOT exact" in text


def test_ec2d2_prompt_augmentation_is_deterministic_and_nonmutating():
    original = HypothesisPrompt(
        prompt_version="base-version",
        system_prompt="system",
        user_prompt="user",
        prompt_sha256="base-sha",
    )

    augmenter = SynthesisConstituentPromptAugmenter(
        _hierarchy()
    )
    first = augmenter.augment(original)
    second = augmenter.augment(original)

    assert original.user_prompt == "user"
    assert original.prompt_version == "base-version"

    assert first.prompt_version == GUIDED_PROMPT_VERSION
    assert first.user_prompt == second.user_prompt
    assert first.prompt_sha256 == second.prompt_sha256
    assert "SYNTHESIS–CONSTITUENT LINEAGE" in first.user_prompt
    assert "stmt:observation" in first.user_prompt
