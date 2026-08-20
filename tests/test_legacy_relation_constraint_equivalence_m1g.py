from __future__ import annotations

import pytest

from domains.dac_her.relation_constraints import DAC_LEGACY_STRICT_RELATION_CONSTRAINTS as DAC_HER_STRICT_RELATION_CONSTRAINTS
from pipeline_core.draft_schema import KnowledgeGraphDraft
from pipeline_core.graph_validation import collect_graph_issues
from pipeline_core.knowledge_graph_legacy_relation_compat import (
    validate_legacy_relation_semantics_compat,
)
from pipeline_core.knowledge_graph_schema import KnowledgeGraph
from pipeline_core.validation_issues import IssueCode


def _pointer():
    return {
        "document_id": "main",
        "document_role": "main",
        "page_id": None,
        "asset_ids": [],
        "locator_text": None,
    }


def _edge(source, relation, target):
    return {
        "source": source,
        "relation": relation,
        "target": target,
        "evidence_type": "structural_characterization",
        "evidence_strength": "direct",
        "evidence_text": "Evidence.",
        "confidence": "high",
        "evidence_pointers": [_pointer()],
        "subsection": None,
    }


def _entity(node_id, node_type):
    return {
        "id": node_id,
        "type": node_type,
        "label": node_id,
        "description": None,
    }


def _experiment(node_id):
    return {
        "id": node_id,
        "name": node_id,
        "experiment_type": "custom_method",
        "experiment_family": "spectroscopy",
        "method_label": "Custom",
        "raw_method_name": None,
        "conditions": [],
        "description": None,
    }


def _calculation(node_id):
    return {
        "id": node_id,
        "name": node_id,
        "calculation_type": "custom",
        "conditions": [],
        "method_details": "custom",
    }


def _measurement(node_id):
    return {
        "id": node_id,
        "metric_id": "example_metric",
        "metric": "Example metric",
        "subject_id": "subject",
        "source_expression": "1.0",
        "group_id": None,
        "value_numeric": 1.0,
        "value_text": None,
        "unit": None,
        "uncertainty": None,
        "qualifier": None,
        "basis": None,
        "conditions": [],
        "description": None,
    }


def _measurement_group(node_id):
    return {
        "id": node_id,
        "group_type": "comparison",
        "label": node_id,
        "member_measurement_ids": [],
        "description": None,
    }


def _observation_claim(node_id):
    return {
        "id": node_id,
        "claim_type": "other",
        "statement": "Observation.",
        "basis": "experimental",
        "description": None,
    }


def _mechanism_claim(node_id):
    return {
        "id": node_id,
        "claim_type": "other",
        "statement": "Mechanism.",
        "basis": "experimental",
        "description": None,
    }


def _empty_payload():
    return {
        "paper_id": "paper",
        "chunk_id": "chunk",
        "section": "Results",
        "document_id": "main",
        "document_role": "main",
        "page_ids": [],
        "asset_ids": [],
        "entities": [],
        "experiments": [],
        "calculations": [],
        "measurements": [],
        "measurement_groups": [],
        "observation_claims": [],
        "mechanism_claims": [],
        "edges": [],
    }


def _add_node(payload, node_id, semantic_type):
    if semantic_type == "Experiment":
        payload["experiments"].append(
            _experiment(node_id)
        )
        return

    if semantic_type == "Calculation":
        payload["calculations"].append(
            _calculation(node_id)
        )
        return

    if semantic_type == "Measurement":
        payload["measurements"].append(
            _measurement(node_id)
        )
        return

    if semantic_type == "MeasurementGroup":
        payload["measurement_groups"].append(
            _measurement_group(node_id)
        )
        return

    if semantic_type == "ObservationClaim":
        payload["observation_claims"].append(
            _observation_claim(node_id)
        )
        return

    if semantic_type == "MechanismClaim":
        payload["mechanism_claims"].append(
            _mechanism_claim(node_id)
        )
        return

    if semantic_type == "Entity":
        payload["entities"].append(
            _entity(node_id, "Material")
        )
        return

    payload["entities"].append(
        _entity(node_id, semantic_type)
    )


def _payload_for(
    relation,
    source_type,
    target_type,
):
    payload = _empty_payload()

    _add_node(
        payload,
        "source",
        source_type,
    )
    _add_node(
        payload,
        "target",
        target_type,
    )

    payload["edges"].append(
        _edge(
            "source",
            relation,
            "target",
        )
    )

    return payload


def _relation_constraint_valid(payload):
    draft = KnowledgeGraphDraft.model_validate(
        payload
    )

    report = collect_graph_issues(
        draft,
        relation_constraints=(
            DAC_HER_STRICT_RELATION_CONSTRAINTS
        ),
    )

    relation_codes = {
        IssueCode.RELATION_SOURCE_TYPE_MISMATCH,
        IssueCode.RELATION_TARGET_TYPE_MISMATCH,
    }

    return not bool(
        relation_codes & report.codes()
    )


def _legacy_relation_valid(payload):
    draft = KnowledgeGraphDraft.model_validate(
        payload
    )

    graph = KnowledgeGraph.model_construct(
        **{
            **draft.model_dump(),
            "entities": draft.entities,
            "experiments": draft.experiments,
            "calculations": draft.calculations,
            "measurements": draft.measurements,
            "measurement_groups": (
                draft.measurement_groups
            ),
            "observation_claims": (
                draft.observation_claims
            ),
            "mechanism_claims": (
                draft.mechanism_claims
            ),
            "edges": draft.edges,
        }
    )

    try:
        validate_legacy_relation_semantics_compat(
            graph
        )
    except ValueError:
        return False

    return True


# relation, valid source, valid target,
# invalid side, invalid semantic type
RELATION_CASES = (
    (
        "EVALUATED_IN",
        "Catalyst",
        "Experiment",
        "source",
        "Metal",
    ),
    (
        "CHARACTERIZED_BY",
        "Catalyst",
        "Experiment",
        "source",
        "Precursor",
    ),
    (
        "MODELED_BY",
        "CatalystModel",
        "Calculation",
        "source",
        "Catalyst",
    ),
    (
        "SYNTHESIZED_BY",
        "Catalyst",
        "SynthesisMethod",
        "source",
        "Material",
    ),
    (
        "USES_PRECURSOR",
        "SynthesisMethod",
        "Precursor",
        "source",
        "Catalyst",
    ),
    (
        "HAS_MEASUREMENT",
        "Experiment",
        "Measurement",
        "source",
        "Catalyst",
    ),
    (
        "MEASURED_FOR",
        "Measurement",
        "Entity",
        "source",
        "Experiment",
    ),
    (
        "IN_MEASUREMENT_GROUP",
        "Measurement",
        "MeasurementGroup",
        "source",
        "Experiment",
    ),
    (
        "SUPPORTS_CLAIM",
        "Measurement",
        "ObservationClaim",
        "source",
        "Entity",
    ),
    (
        "INTERPRETED_AS",
        "ObservationClaim",
        "MechanismClaim",
        "source",
        "MechanismClaim",
    ),
    (
        "APPLIES_TO",
        "ObservationClaim",
        "Entity",
        "source",
        "Measurement",
    ),
    (
        "MODEL_OF",
        "CatalystModel",
        "Catalyst",
        "source",
        "Catalyst",
    ),
    (
        "HAS_METAL",
        "Catalyst",
        "Metal",
        "source",
        "Support",
    ),
    (
        "SUPPORTED_ON",
        "Catalyst",
        "Support",
        "target",
        "Metal",
    ),
    (
        "CATALYZES",
        "Catalyst",
        "Reaction",
        "target",
        "Metal",
    ),
)


@pytest.mark.parametrize(
    (
        "relation",
        "source_type",
        "target_type",
        "invalid_side",
        "invalid_type",
    ),
    RELATION_CASES,
)
def test_valid_relation_semantics_are_equivalent(
    relation,
    source_type,
    target_type,
    invalid_side,
    invalid_type,
):
    del invalid_side, invalid_type

    payload = _payload_for(
        relation,
        source_type,
        target_type,
    )

    assert _legacy_relation_valid(payload)
    assert _relation_constraint_valid(payload)


@pytest.mark.parametrize(
    (
        "relation",
        "source_type",
        "target_type",
        "invalid_side",
        "invalid_type",
    ),
    RELATION_CASES,
)
def test_invalid_relation_semantics_are_equivalent(
    relation,
    source_type,
    target_type,
    invalid_side,
    invalid_type,
):
    if invalid_side == "source":
        source_type = invalid_type
    elif invalid_side == "target":
        target_type = invalid_type
    else:
        raise AssertionError(
            f"unknown invalid side: {invalid_side}"
        )

    payload = _payload_for(
        relation,
        source_type,
        target_type,
    )

    assert not _legacy_relation_valid(payload)
    assert not _relation_constraint_valid(payload)


def test_matrix_covers_every_dac_relation_constraint():
    matrix_relations = {
        relation
        for (
            relation,
            _source,
            _target,
            _invalid_side,
            _invalid_type,
        ) in RELATION_CASES
    }

    constraint_relations = {
        constraint.relation
        for constraint in (
            DAC_HER_STRICT_RELATION_CONSTRAINTS
        )
    }

    assert matrix_relations == constraint_relations

# ------------------------------------------------------------------
# M1g.5.2
#
# Triangulate the second historical relation implementation:
#
#   strict KnowledgeGraph legacy compatibility
#                ==
#   explicit DAC RelationConstraint table
#                ==
#   collect_graph_issues no-contract legacy compatibility
#
# This characterization compares endpoint acceptance semantics only.
# Historical error-message formatting remains a separate contract.
# ------------------------------------------------------------------


_RELATION_MISMATCH_CODES = {
    IssueCode.RELATION_SOURCE_TYPE_MISMATCH,
    IssueCode.RELATION_TARGET_TYPE_MISMATCH,
}


def _explicit_relation_constraint_codes(payload):
    draft = KnowledgeGraphDraft.model_validate(
        payload
    )

    report = collect_graph_issues(
        draft,
        relation_constraints=(
            DAC_HER_STRICT_RELATION_CONSTRAINTS
        ),
    )

    return (
        report.codes()
        & _RELATION_MISMATCH_CODES
    )


def _legacy_graph_validation_relation_codes(payload):
    draft = KnowledgeGraphDraft.model_validate(
        payload
    )

    report = collect_graph_issues(
        draft
    )

    return (
        report.codes()
        & _RELATION_MISMATCH_CODES
    )


@pytest.mark.parametrize(
    (
        "relation",
        "source_type",
        "target_type",
        "invalid_side",
        "invalid_type",
    ),
    RELATION_CASES,
)
def test_no_contract_graph_validation_matches_explicit_contract_for_valid_cases(
    relation,
    source_type,
    target_type,
    invalid_side,
    invalid_type,
):
    del invalid_side, invalid_type

    payload = _payload_for(
        relation,
        source_type,
        target_type,
    )

    legacy_codes = (
        _legacy_graph_validation_relation_codes(
            payload
        )
    )
    explicit_codes = (
        _explicit_relation_constraint_codes(
            payload
        )
    )

    assert legacy_codes == explicit_codes == set()

    # Existing M1g.3 strict-model side of the triangle.
    assert _legacy_relation_valid(payload)


@pytest.mark.parametrize(
    (
        "relation",
        "source_type",
        "target_type",
        "invalid_side",
        "invalid_type",
    ),
    RELATION_CASES,
)
def test_no_contract_graph_validation_matches_explicit_contract_for_invalid_cases(
    relation,
    source_type,
    target_type,
    invalid_side,
    invalid_type,
):
    if invalid_side == "source":
        source_type = invalid_type
    elif invalid_side == "target":
        target_type = invalid_type
    else:
        raise AssertionError(
            f"unknown invalid side: {invalid_side}"
        )

    payload = _payload_for(
        relation,
        source_type,
        target_type,
    )

    legacy_codes = (
        _legacy_graph_validation_relation_codes(
            payload
        )
    )
    explicit_codes = (
        _explicit_relation_constraint_codes(
            payload
        )
    )

    # Equality here is stronger than merely "both reject":
    # it also preserves which endpoint side is rejected.
    assert legacy_codes == explicit_codes
    assert legacy_codes

    # Existing M1g.3 strict-model side of the triangle.
    assert not _legacy_relation_valid(payload)
