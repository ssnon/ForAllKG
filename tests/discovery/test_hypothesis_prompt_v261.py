from pipeline_core.discovery.hypothesis_compiler import HypothesisCompileIssue
from pipeline_core.discovery.hypothesis_prompt import HypothesisPromptAssembler, PROMPT_VERSION

from tests.support._hypothesis_v261_fixtures import make_context, make_valid_draft


def test_prompt_is_deterministic_and_separates_epistemic_roles():
    context = make_context()
    assembler = HypothesisPromptAssembler()

    p1 = assembler.build(context)
    p2 = assembler.build(context)

    assert p1.prompt_version == PROMPT_VERSION
    assert p1.prompt_sha256 == p2.prompt_sha256
    assert p1.system_prompt == p2.system_prompt
    assert p1.user_prompt == p2.user_prompt

    assert "A hypothesis is not evidence." in p1.system_prompt
    assert "ELIGIBLE POSITIVE PREMISES" in p1.user_prompt
    assert "s:reported" in p1.user_prompt
    assert "s:candidate" in p1.user_prompt
    assert "requires_verification=YES" in p1.user_prompt

    assert "RESEARCH GAPS" in p1.user_prompt
    assert "s:gap" in p1.user_prompt
    assert "NOT A POSITIVE PREMISE" in p1.user_prompt

    assert "RESTRICTED / NON-PREMISE STATEMENTS" in p1.user_prompt
    assert "s:restricted" in p1.user_prompt
    assert "scope_limit_not_positive_premise" in p1.user_prompt
    assert "Kiwook_10" in p1.user_prompt


def test_repair_feedback_is_bounded_and_contains_exact_issue():
    assembler = HypothesisPromptAssembler()
    draft = make_valid_draft()
    issue = HypothesisCompileIssue(
        code="INELIGIBLE_POSITIVE_PREMISE",
        location="draft.hypotheses[0].premise_statement_ids",
        message="s:gap is not eligible",
    )

    feedback = assembler.repair_feedback(previous_draft=draft, issues=[issue])

    assert "REPAIR REQUEST" in feedback
    assert "INELIGIBLE_POSITIVE_PREMISE" in feedback
    assert "s:gap is not eligible" in feedback
    assert "Do not introduce new evidence IDs" in feedback
    assert "PREVIOUS DRAFT" in feedback
