from __future__ import annotations

from pipeline_core.discovery.hypothesis_semantic_contracts import (
    SEMANTIC_DIMENSIONS,
    HypothesisSemanticReviewDraft,
)
from pipeline_core.discovery.hypothesis_semantic_reference import (
    HypothesisSemanticReferenceSanitizer,
)


def _draft(*, dimension: str, hypothesis_ids=None, statement_ids=None, verdict="pass"):
    rows = []
    for name in SEMANTIC_DIMENSIONS:
        rows.append(
            {
                "dimension": name,
                "verdict": verdict if name == dimension else "pass",
                "rationale": "bounded review",
                "hypothesis_ids": hypothesis_ids if name == dimension else [],
                "statement_ids": statement_ids if name == dimension else [],
            }
        )
    return HypothesisSemanticReviewDraft.model_validate(
        {"dimensions": rows, "overall_summary": "bounded"}
    )


def test_directional_specificity_unknown_statement_id_is_optional_and_nonfatal():
    draft = _draft(
        dimension="directional_specificity",
        hypothesis_ids=["hypothesis:1"],
        statement_ids=["stmt:hallucinated"],
    )
    result = HypothesisSemanticReferenceSanitizer().sanitize(
        draft,
        valid_hypothesis_ids={"hypothesis:1"},
        valid_statement_ids={"stmt:valid"},
    )
    row = next(x for x in result.draft.dimensions if x.dimension == "directional_specificity")
    assert row.hypothesis_ids == ["hypothesis:1"]
    assert row.statement_ids == []
    assert result.audit.applied is True
    assert result.audit.fatal is False


def test_premise_fidelity_all_unknown_statement_ids_remains_fatal():
    draft = _draft(
        dimension="premise_fidelity",
        hypothesis_ids=["hypothesis:1"],
        statement_ids=["stmt:hallucinated"],
    )
    result = HypothesisSemanticReferenceSanitizer().sanitize(
        draft,
        valid_hypothesis_ids={"hypothesis:1"},
        valid_statement_ids={"stmt:valid"},
    )
    assert result.audit.fatal is True
    assert any("premise_fidelity" in reason for reason in result.audit.fatal_reasons)


def test_not_applicable_all_unknown_references_are_safe_dropped():
    draft = _draft(
        dimension="candidate_calibration",
        hypothesis_ids=["hypothesis:hallucinated"],
        statement_ids=["stmt:hallucinated"],
        verdict="not_applicable",
    )
    result = HypothesisSemanticReferenceSanitizer().sanitize(
        draft,
        valid_hypothesis_ids={"hypothesis:1"},
        valid_statement_ids={"stmt:valid"},
    )
    assert result.audit.fatal is False
