from __future__ import annotations

from domains.extraction_registry import get_extraction_adapter
from pipeline_core.corpus.extraction.evidence_relation_constraints import (
    COMMON_EVIDENCE_STRICT_RELATION_CONSTRAINTS,
)


def _sers_adapter():
    return get_extraction_adapter("sers_au_ag")


def _uses_precursor_constraint():
    matches = [
        constraint
        for constraint in _sers_adapter().strict_relation_constraints
        if constraint.relation == "USES_PRECURSOR"
    ]

    assert len(matches) == 1
    return matches[0]


def test_sers_strict_contract_adds_only_uses_precursor_to_common_evidence():
    adapter = _sers_adapter()

    common_relations = [
        constraint.relation
        for constraint in COMMON_EVIDENCE_STRICT_RELATION_CONSTRAINTS
    ]

    sers_relations = [
        constraint.relation
        for constraint in adapter.strict_relation_constraints
    ]

    assert len(adapter.strict_relation_constraints) == (
        len(COMMON_EVIDENCE_STRICT_RELATION_CONSTRAINTS) + 1
    )

    assert sers_relations[:-1] == common_relations
    assert sers_relations[-1] == "USES_PRECURSOR"


def test_sers_uses_precursor_strict_endpoint_contract():
    constraint = _uses_precursor_constraint()

    assert constraint.source_types == frozenset({"SynthesisMethod"})
    assert constraint.target_types == frozenset({"Precursor"})

    assert constraint.check(
        source_id="method",
        source_type="SynthesisMethod",
        target_id="precursor",
        target_type="Precursor",
        edge_key="e",
    ) == []


def test_sers_uses_precursor_rejects_material_target():
    constraint = _uses_precursor_constraint()

    issues = constraint.check(
        source_id="method",
        source_type="SynthesisMethod",
        target_id="material",
        target_type="Material",
        edge_key="e",
    )

    assert len(issues) == 1
    assert issues[0].code == "relation_target_type_mismatch"


def test_sers_uses_precursor_rejects_metal_target():
    constraint = _uses_precursor_constraint()

    issues = constraint.check(
        source_id="method",
        source_type="SynthesisMethod",
        target_id="metal",
        target_type="Metal",
        edge_key="e",
    )

    assert len(issues) == 1
    assert issues[0].code == "relation_target_type_mismatch"


def test_sers_uses_precursor_rejects_non_method_source():
    constraint = _uses_precursor_constraint()

    for source_type in ("Nanostructure", "Experiment"):
        issues = constraint.check(
            source_id="source",
            source_type=source_type,
            target_id="precursor",
            target_type="Precursor",
            edge_key="e",
        )

        assert len(issues) == 1
        assert issues[0].code == "relation_source_type_mismatch"
