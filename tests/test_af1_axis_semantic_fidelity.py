from __future__ import annotations

from types import SimpleNamespace

from dac_her.af1_axis_semantic_fidelity import (
    AF1AxisComponentReview,
    AF1AxisSemanticFidelityAudit,
    af1_audit_passes,
    validate_af1_audit,
)


def _axis():
    return SimpleNamespace(
        axis_id="axis:1",
        proposed_subject="bonding-antibonding balance",
        proposed_relation="promotes",
        proposed_object="adsorption-desorption compatibility",
        label=(
            "bonding-antibonding balance promotes "
            "adsorption-desorption compatibility"
        ),
    )


def _hypothesis():
    return SimpleNamespace(
        hypothesis_id="h1",
        hypothesis_statement=(
            "bonding-antibonding balance is associated with "
            "adsorption-desorption compatibility"
        ),
    )


def _review(
    component,
    axis_text,
    status,
    expression,
):
    return AF1AxisComponentReview(
        component=component,
        axis_text=axis_text,
        status=status,
        hypothesis_expression=expression,
        axis_evidence_status="grounded_in_selected_premises",
        explanation="test",
    )


def _audit(reviews, *, overall, preserved):
    return AF1AxisSemanticFidelityAudit(
        hypothesis_id="h1",
        axis_id="axis:1",
        hypothesis_statement=_hypothesis().hypothesis_statement,
        component_reviews=list(reviews),
        component_coverage_complete=True,
        all_essential_axis_semantics_preserved=preserved,
        overall_status=overall,
        explanation="test",
    )


def test_exact_or_equivalent_components_pass():
    audit = _audit(
        [
            _review(
                "subject",
                "bonding-antibonding balance",
                "preserved",
                "bonding-antibonding balance",
            ),
            _review(
                "relation",
                "promotes",
                "epistemically_weakened_but_faithful",
                "is associated with",
            ),
            _review(
                "object",
                "adsorption-desorption compatibility",
                "preserved",
                "adsorption-desorption compatibility",
            ),
        ],
        overall="faithful_with_narrowing_or_epistemic_weakening",
        preserved=True,
    )
    assert af1_audit_passes(audit)


def test_subject_substitution_fails():
    hypothesis = SimpleNamespace(
        hypothesis_id="h1",
        hypothesis_statement=(
            "strong bonding is predicted to promote "
            "adsorption-desorption compatibility"
        ),
    )
    audit = AF1AxisSemanticFidelityAudit(
        hypothesis_id="h1",
        axis_id="axis:1",
        hypothesis_statement=hypothesis.hypothesis_statement,
        component_reviews=[
            _review(
                "subject",
                "bonding-antibonding balance",
                "substituted",
                "strong bonding",
            ),
            _review(
                "relation",
                "promotes",
                "preserved",
                "promote",
            ),
            _review(
                "object",
                "adsorption-desorption compatibility",
                "preserved",
                "adsorption-desorption compatibility",
            ),
        ],
        component_coverage_complete=True,
        all_essential_axis_semantics_preserved=False,
        overall_status="axis_fidelity_lost",
        explanation="test",
    )
    assert not af1_audit_passes(audit)


def test_object_substitution_fails():
    hypothesis = SimpleNamespace(
        hypothesis_id="h1",
        hypothesis_statement=(
            "bonding-antibonding balance promotes "
            "differentiated H* and OH* adsorption roles"
        ),
    )
    audit = AF1AxisSemanticFidelityAudit(
        hypothesis_id="h1",
        axis_id="axis:1",
        hypothesis_statement=hypothesis.hypothesis_statement,
        component_reviews=[
            _review(
                "subject",
                "bonding-antibonding balance",
                "preserved",
                "bonding-antibonding balance",
            ),
            _review(
                "relation",
                "promotes",
                "preserved",
                "promotes",
            ),
            _review(
                "object",
                "adsorption-desorption compatibility",
                "substituted",
                "differentiated H* and OH* adsorption roles",
            ),
        ],
        component_coverage_complete=True,
        all_essential_axis_semantics_preserved=False,
        overall_status="axis_fidelity_lost",
        explanation="test",
    )
    assert not af1_audit_passes(audit)


def test_epistemic_weakening_only_allowed_for_relation():
    audit = _audit(
        [
            _review(
                "subject",
                "bonding-antibonding balance",
                "epistemically_weakened_but_faithful",
                "bonding-antibonding balance",
            ),
            _review(
                "relation",
                "promotes",
                "preserved",
                "is associated with",
            ),
            _review(
                "object",
                "adsorption-desorption compatibility",
                "preserved",
                "adsorption-desorption compatibility",
            ),
        ],
        overall="faithful_with_narrowing_or_epistemic_weakening",
        preserved=True,
    )
    issues = validate_af1_audit(
        audit,
        hypothesis=_hypothesis(),
        axis=_axis(),
    )
    assert "epistemic_weakening_nonrelation_component" in {
        row.code for row in issues
    }


def test_component_expression_must_be_exact_hypothesis_substring():
    audit = _audit(
        [
            _review(
                "subject",
                "bonding-antibonding balance",
                "preserved",
                "not present in hypothesis",
            ),
            _review(
                "relation",
                "promotes",
                "epistemically_weakened_but_faithful",
                "is associated with",
            ),
            _review(
                "object",
                "adsorption-desorption compatibility",
                "preserved",
                "adsorption-desorption compatibility",
            ),
        ],
        overall="faithful_with_narrowing_or_epistemic_weakening",
        preserved=True,
    )
    issues = validate_af1_audit(
        audit,
        hypothesis=_hypothesis(),
        axis=_axis(),
    )
    assert "hypothesis_expression_not_exact_substring" in {
        row.code for row in issues
    }


def test_component_set_must_match_axis():
    audit = AF1AxisSemanticFidelityAudit(
        hypothesis_id="h1",
        axis_id="axis:1",
        hypothesis_statement=_hypothesis().hypothesis_statement,
        component_reviews=[
            _review(
                "subject",
                "bonding-antibonding balance",
                "preserved",
                "bonding-antibonding balance",
            ),
            _review(
                "relation",
                "promotes",
                "epistemically_weakened_but_faithful",
                "is associated with",
            ),
        ],
        component_coverage_complete=False,
        all_essential_axis_semantics_preserved=False,
        overall_status="axis_fidelity_lost",
        explanation="test",
    )
    issues = validate_af1_audit(
        audit,
        hypothesis=_hypothesis(),
        axis=_axis(),
    )
    assert "component_set_mismatch" in {
        row.code for row in issues
    }


def test_overall_status_is_recomputed():
    audit = _audit(
        [
            _review(
                "subject",
                "bonding-antibonding balance",
                "substituted",
                "bonding-antibonding balance",
            ),
            _review(
                "relation",
                "promotes",
                "epistemically_weakened_but_faithful",
                "is associated with",
            ),
            _review(
                "object",
                "adsorption-desorption compatibility",
                "preserved",
                "adsorption-desorption compatibility",
            ),
        ],
        overall="faithful_with_narrowing_or_epistemic_weakening",
        preserved=True,
    )
    issues = validate_af1_audit(
        audit,
        hypothesis=_hypothesis(),
        axis=_axis(),
    )
    codes = {row.code for row in issues}
    assert "all_semantics_preserved_self_audit_mismatch" in codes
    assert "overall_status_self_audit_mismatch" in codes


def test_axis_component_text_is_immutable():
    audit = _audit(
        [
            _review(
                "subject",
                "different subject",
                "preserved",
                "bonding-antibonding balance",
            ),
            _review(
                "relation",
                "promotes",
                "epistemically_weakened_but_faithful",
                "is associated with",
            ),
            _review(
                "object",
                "adsorption-desorption compatibility",
                "preserved",
                "adsorption-desorption compatibility",
            ),
        ],
        overall="faithful_with_narrowing_or_epistemic_weakening",
        preserved=True,
    )
    issues = validate_af1_audit(
        audit,
        hypothesis=_hypothesis(),
        axis=_axis(),
    )
    assert "axis_component_text_mismatch" in {
        row.code for row in issues
    }
