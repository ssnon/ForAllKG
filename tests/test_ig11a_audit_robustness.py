from __future__ import annotations

from types import SimpleNamespace

from dac_her.ig11_endpoint_scope import (
    IG11AxisEntityAudit,
    IG11AxisEvidenceAudit,
    IG11StatementAxisAudit,
    validate_axis_audit,
)


def _stmt(statement_id: str, text: str):
    return SimpleNamespace(
        statement_id=statement_id,
        text=text,
        eligible_as_premise=True,
    )


def _context():
    return SimpleNamespace(
        evidence_statements=[
            _stmt(
                "stmt:direct",
                "Hydrogen adsorption energetics vary across modeled sites.",
            ),
            _stmt(
                "stmt:adjacent",
                "Structural stability was reported for a supported model.",
            ),
            _stmt(
                "stmt:scope",
                "The packet does not establish a universal structural rule.",
            ),
        ]
    )


def _review(
    statement_id: str,
    *,
    axis_support: str,
    endpoint_candidate: bool,
    grounding_excerpt: str | None,
):
    role = (
        "axis_variable"
        if axis_support == "direct_axis_grounding"
        else "structural_context"
    )
    return IG11StatementAxisAudit(
        statement_id=statement_id,
        axis_support=axis_support,
        endpoint_role=role,
        grounding_excerpt=grounding_excerpt,
        scope_basis_excerpt=None,
        scope_breadth="unclear",
        scope_summary="test scope",
        endpoint_candidate=endpoint_candidate,
        reason="test",
    )


def _audit(reviews, *, entities=None):
    return IG11AxisEvidenceAudit(
        axis_id="axis:test",
        statement_reviews=reviews,
        axis_entities=entities or [],
        direct_axis_statement_ids=[
            row.statement_id
            for row in reviews
            if row.axis_support == "direct_axis_grounding"
        ],
        endpoint_candidate_statement_ids=[
            row.statement_id
            for row in reviews
            if row.endpoint_candidate
        ],
        audit_summary="test",
    )


def test_ig11a_adjacent_noncandidate_may_omit_grounding_excerpt():
    audit = _audit(
        [
            _review(
                "stmt:direct",
                axis_support="direct_axis_grounding",
                endpoint_candidate=True,
                grounding_excerpt="Hydrogen adsorption energetics",
            ),
            _review(
                "stmt:adjacent",
                axis_support="adjacent_context",
                endpoint_candidate=False,
                grounding_excerpt=None,
            ),
            _review(
                "stmt:scope",
                axis_support="scope_only",
                endpoint_candidate=False,
                grounding_excerpt=None,
            ),
        ]
    )
    issues = validate_axis_audit(
        audit,
        context=_context(),
        expected_axis_id="axis:test",
    )
    assert issues == []


def test_ig11a_direct_axis_still_requires_grounding_excerpt():
    audit = _audit(
        [
            _review(
                "stmt:direct",
                axis_support="direct_axis_grounding",
                endpoint_candidate=True,
                grounding_excerpt=None,
            ),
            _review(
                "stmt:adjacent",
                axis_support="adjacent_context",
                endpoint_candidate=False,
                grounding_excerpt=None,
            ),
            _review(
                "stmt:scope",
                axis_support="scope_only",
                endpoint_candidate=False,
                grounding_excerpt=None,
            ),
        ]
    )
    issues = validate_axis_audit(
        audit,
        context=_context(),
        expected_axis_id="axis:test",
    )
    assert "missing_grounding_excerpt" in {
        row.code for row in issues
    }


def test_ig11a_adjacent_endpoint_candidate_requires_excerpt():
    audit = _audit(
        [
            _review(
                "stmt:direct",
                axis_support="direct_axis_grounding",
                endpoint_candidate=True,
                grounding_excerpt="Hydrogen adsorption energetics",
            ),
            _review(
                "stmt:adjacent",
                axis_support="adjacent_context",
                endpoint_candidate=True,
                grounding_excerpt=None,
            ),
            _review(
                "stmt:scope",
                axis_support="scope_only",
                endpoint_candidate=False,
                grounding_excerpt=None,
            ),
        ]
    )
    issues = validate_axis_audit(
        audit,
        context=_context(),
        expected_axis_id="axis:test",
    )
    assert "missing_grounding_excerpt" in {
        row.code for row in issues
    }


def test_ig11a_optional_excerpt_is_still_provenance_checked():
    audit = _audit(
        [
            _review(
                "stmt:direct",
                axis_support="direct_axis_grounding",
                endpoint_candidate=True,
                grounding_excerpt="Hydrogen adsorption energetics",
            ),
            _review(
                "stmt:adjacent",
                axis_support="adjacent_context",
                endpoint_candidate=False,
                grounding_excerpt="not actually present",
            ),
            _review(
                "stmt:scope",
                axis_support="scope_only",
                endpoint_candidate=False,
                grounding_excerpt=None,
            ),
        ]
    )
    issues = validate_axis_audit(
        audit,
        context=_context(),
        expected_axis_id="axis:test",
    )
    assert "nonverbatim_optional_grounding_excerpt" in {
        row.code for row in issues
    }


def test_ig11a_ungrounded_entity_with_grounding_ids_is_invalid():
    audit = _audit(
        [
            _review(
                "stmt:direct",
                axis_support="direct_axis_grounding",
                endpoint_candidate=True,
                grounding_excerpt="Hydrogen adsorption energetics",
            ),
            _review(
                "stmt:adjacent",
                axis_support="adjacent_context",
                endpoint_candidate=False,
                grounding_excerpt=None,
            ),
            _review(
                "stmt:scope",
                axis_support="scope_only",
                endpoint_candidate=False,
                grounding_excerpt=None,
            ),
        ],
        entities=[
            IG11AxisEntityAudit(
                entity_text="ExamplePair",
                entity_kind="concrete_material_or_system",
                grounding_status="ungrounded",
                grounding_statement_ids=["stmt:direct"],
                entity_specific_claim_required_for_axis_fidelity=False,
                reason="self-inconsistent test entity",
            )
        ],
    )
    issues = validate_axis_audit(
        audit,
        context=_context(),
        expected_axis_id="axis:test",
    )
    assert "ungrounded_entity_has_grounding_ids" in {
        row.code for row in issues
    }
