from __future__ import annotations

import networkx as nx
import pytest

from dac_her.comparison_context import (
    audit_comparison_outputs,
    compare_contexts,
    dimension_from_values,
)
from dac_her.comparison_domain import (
    ComparisonContext,
    ComparisonDomainAdapter,
    ObservableComparisonPolicy,
)


DIMENSIONS = ("analyte", "wavelength")


def _adapter():
    return ComparisonDomainAdapter(
        adapter_id="demo",
        domain_profile_id="demo",
        semantics_id="demo-v1",
        dimensions=DIMENSIONS,
        required_for_numeric_ranking=frozenset(DIMENSIONS),
        extract_contexts_fn=lambda graph, paper_id: [],
        observable_policies=(
            ObservableComparisonPolicy(
                policy_id="demo-policy",
                family="demo",
                observable_keys=frozenset({"sers intensity"}),
                applicable_dimensions=DIMENSIONS,
                ranking_required_dimensions=frozenset(DIMENSIONS),
                numeric_ranking_mode="allowed_if_complete",
                ranking_direction="higher_better",
            ),
        ),
    )


def _context(
    context_id: str,
    paper_id: str,
    *,
    analyte: str | None,
    wavelength: str | None,
    numeric: float | None = 1.0,
    unit: str = "a.u.",
):
    def dim(name, value):
        if value is None:
            return dimension_from_values(name, [])
        return dimension_from_values(name, [(value, "m")])

    return ComparisonContext(
        context_id=context_id,
        domain_profile_id="demo",
        comparison_semantics_id="demo-v1",
        paper_id=paper_id,
        measurement_id="m",
        observable_key="sers intensity",
        observable_label="SERS intensity",
        value_numeric=numeric,
        value_text="" if numeric is not None else "strong",
        unit=unit,
        source_expression="source",
        subject_ids=("sub",),
        dimensions=(
            dim("analyte", analyte),
            dim("wavelength", wavelength),
        ),
        source_node_ids=("m",),
    )


def test_alpha4b3b_unknown_is_not_compatible():
    result = compare_contexts(
        _context("a", "P1", analyte=None, wavelength=None),
        _context("b", "P2", analyte=None, wavelength=None),
        adapter=_adapter(),
    )
    assert result.compatibility == "unknown"
    assert result.numeric_ranking_allowed is False


def test_alpha4b3b_partial_context_never_allows_numeric_ranking():
    result = compare_contexts(
        _context("a", "P1", analyte="R6G", wavelength=None),
        _context("b", "P2", analyte="R6G", wavelength=None),
        adapter=_adapter(),
    )
    assert result.compatibility == "partially_compatible"
    assert result.matched_dimensions == ("analyte",)
    assert result.unknown_dimensions == ("wavelength",)
    assert result.numeric_ranking_allowed is False


def test_alpha4b3b_explicit_mismatch_is_incompatible():
    result = compare_contexts(
        _context("a", "P1", analyte="R6G", wavelength="785 nm"),
        _context("b", "P2", analyte="MB", wavelength="785 nm"),
        adapter=_adapter(),
    )
    assert result.compatibility == "incompatible"
    assert result.mismatched_dimensions == ("analyte",)
    assert result.numeric_ranking_allowed is False


def test_alpha4b3b_numeric_ranking_requires_complete_context_and_units():
    allowed = compare_contexts(
        _context("a", "P1", analyte="R6G", wavelength="785 nm"),
        _context("b", "P2", analyte="R6G", wavelength="785 nm"),
        adapter=_adapter(),
    )
    assert allowed.compatibility == "compatible"
    assert allowed.measurement_unit_status == "matched"
    assert allowed.numeric_ranking_allowed is True

    blocked = compare_contexts(
        _context("c", "P1", analyte="R6G", wavelength="785 nm", unit=""),
        _context("d", "P2", analyte="R6G", wavelength="785 nm", unit=""),
        adapter=_adapter(),
    )
    assert blocked.compatibility == "compatible"
    assert blocked.measurement_unit_status == "unknown"
    assert blocked.numeric_ranking_allowed is False


def test_alpha4b3b_ambiguous_dimension_is_fail_closed():
    ambiguous = dimension_from_values(
        "analyte",
        [("R6G", "a"), ("Methylene blue", "b")],
    )
    assert ambiguous.status == "ambiguous"
    assert ambiguous.normalized_value == ""


def test_alpha4b3b_audit_does_not_quarantine_missing_context():
    graph = nx.MultiDiGraph()
    graph.add_node("m", type="Measurement")
    left = _context("a", "P1", analyte=None, wavelength=None)
    right = _context("b", "P2", analyte=None, wavelength=None)
    assessment = compare_contexts(left, right, adapter=_adapter())
    audit = audit_comparison_outputs(
        contexts=[left, right],
        assessments=[assessment],
        source_graphs={"P1": graph, "P2": graph.copy()},
        adapter=_adapter(),
    )
    assert audit["passes_structural_gate"] is True
    assert audit["missing_context_is_not_quarantine"] is True
    assert audit["numeric_ranking_allowed_count"] == 0


def test_alpha4b3b_adapter_contract_rejects_bad_required_dimensions():
    with pytest.raises(ValueError, match="subset"):
        ComparisonDomainAdapter(
            adapter_id="bad",
            domain_profile_id="bad",
            semantics_id="bad",
            dimensions=("x",),
            required_for_numeric_ranking=frozenset({"y"}),
            extract_contexts_fn=lambda graph, paper_id: [],
        )
