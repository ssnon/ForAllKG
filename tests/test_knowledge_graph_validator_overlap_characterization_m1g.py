from __future__ import annotations

import pytest

from dac_her.domains.sers_au_ag_graph import (
    SERS_RELATION_CONSTRAINTS,
)
from dac_her.domains.strict_relation_contracts import (
    DAC_HER_STRICT_RELATION_CONSTRAINTS,
)
from pipeline_core.draft_schema import KnowledgeGraphDraft
from pipeline_core.graph_validation import collect_graph_issues
from pipeline_core.knowledge_graph_schema import KnowledgeGraph
from pipeline_core.validation_issues import IssueCode


def _pointer(*, document_id: str = "main") -> dict:
    return {
        "document_id": document_id,
        "document_role": "main",
        "page_id": None,
        "asset_ids": [],
        "locator_text": None,
    }


def _edge(
    source: str,
    relation: str,
    target: str,
    *,
    document_id: str = "main",
) -> dict:
    return {
        "source": source,
        "relation": relation,
        "target": target,
        "evidence_type": "structural_characterization",
        "evidence_strength": "direct",
        "evidence_text": "Source-grounded evidence.",
        "confidence": "high",
        "evidence_pointers": [
            _pointer(document_id=document_id),
        ],
        "subsection": None,
    }


def _entity(
    node_id: str,
    node_type: str,
) -> dict:
    return {
        "id": node_id,
        "type": node_type,
        "label": node_id,
        "description": None,
    }


def _experiment(node_id: str) -> dict:
    return {
        "id": node_id,
        "name": "Experiment",
        "experiment_type": "custom_method",
        "experiment_family": "spectroscopy",
        "method_label": "Custom method",
        "raw_method_name": None,
        "conditions": [],
        "description": None,
    }


def _measurement(
    node_id: str,
    subject_id: str,
) -> dict:
    return {
        "id": node_id,
        "metric_id": "example_metric",
        "metric": "Example metric",
        "subject_id": subject_id,
        "source_expression": "1.0 example unit",
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


def _payload(
    *,
    entities=None,
    experiments=None,
    calculations=None,
    measurements=None,
    measurement_groups=None,
    observation_claims=None,
    mechanism_claims=None,
    edges=None,
) -> dict:
    return {
        "paper_id": "paper",
        "chunk_id": "chunk",
        "section": "Results",
        "document_id": "main",
        "document_role": "main",
        "page_ids": [],
        "asset_ids": [],
        "entities": list(entities or []),
        "experiments": list(experiments or []),
        "calculations": list(calculations or []),
        "measurements": list(measurements or []),
        "measurement_groups": list(
            measurement_groups or []
        ),
        "observation_claims": list(
            observation_claims or []
        ),
        "mechanism_claims": list(
            mechanism_claims or []
        ),
        "edges": list(edges or []),
    }


def _draft_report(
    payload: dict,
    *,
    relation_constraints=None,
):
    draft = KnowledgeGraphDraft.model_validate(payload)
    return collect_graph_issues(
        draft,
        relation_constraints=relation_constraints,
    )


def test_provenance_failure_exists_in_both_validation_paths():
    payload = _payload(
        entities=[
            _entity("catalyst", "Catalyst"),
            _entity("metal", "Metal"),
        ],
        edges=[
            _edge(
                "catalyst",
                "HAS_METAL",
                "metal",
                document_id="wrong-document",
            ),
        ],
    )

    with pytest.raises(
        ValueError,
        match="Graph provenance validation failed",
    ):
        KnowledgeGraph.model_validate(payload)

    report = _draft_report(payload)

    assert IssueCode.POINTER_DOCUMENT_ID_MISMATCH in report.codes()


def test_undefined_endpoint_exists_in_both_validation_paths():
    payload = _payload(
        entities=[
            _entity("metal", "Metal"),
        ],
        edges=[
            _edge(
                "missing",
                "HAS_METAL",
                "metal",
            ),
        ],
    )

    with pytest.raises(
        ValueError,
        match="undefined source",
    ):
        KnowledgeGraph.model_validate(payload)

    report = _draft_report(payload)

    assert IssueCode.UNDEFINED_EDGE_SOURCE in report.codes()


def test_measurement_producer_failure_exists_in_both_paths():
    payload = _payload(
        entities=[
            _entity("subject", "Catalyst"),
        ],
        measurements=[
            _measurement("measurement", "subject"),
        ],
        edges=[
            _edge(
                "measurement",
                "MEASURED_FOR",
                "subject",
            ),
        ],
    )

    with pytest.raises(
        ValueError,
        match="no incoming HAS_MEASUREMENT",
    ):
        KnowledgeGraph.model_validate(payload)

    report = _draft_report(payload)

    assert IssueCode.MISSING_MEASUREMENT_PRODUCER in report.codes()


def test_dac_relation_failure_exists_in_direct_and_constraint_paths():
    payload = _payload(
        entities=[
            _entity("metal", "Metal"),
            _entity("catalyst", "Catalyst"),
        ],
        edges=[
            _edge(
                "metal",
                "HAS_METAL",
                "catalyst",
            ),
        ],
    )

    with pytest.raises(ValueError):
        KnowledgeGraph.model_validate(payload)

    report = _draft_report(
        payload,
        relation_constraints=(
            DAC_HER_STRICT_RELATION_CONSTRAINTS
        ),
    )

    assert (
        IssueCode.RELATION_SOURCE_TYPE_MISMATCH
        in report.codes()
    )
    assert (
        IssueCode.RELATION_TARGET_TYPE_MISMATCH
        in report.codes()
    )


def test_graph_validation_legacy_fallback_preserves_dac_semantics():
    payload = _payload(
        entities=[
            _entity("metal", "Metal"),
            _entity("catalyst", "Catalyst"),
        ],
        edges=[
            _edge(
                "metal",
                "HAS_METAL",
                "catalyst",
            ),
        ],
    )

    report = _draft_report(payload)

    assert (
        IssueCode.RELATION_SOURCE_TYPE_MISMATCH
        in report.codes()
    )
    assert (
        IssueCode.RELATION_TARGET_TYPE_MISMATCH
        in report.codes()
    )


def test_valid_sers_relation_is_accepted_by_both_layers():
    payload = _payload(
        entities=[
            _entity(
                "substrate",
                "PlasmonicSubstrate",
            ),
        ],
        experiments=[
            _experiment("experiment"),
        ],
        edges=[
            _edge(
                "substrate",
                "TESTED_IN",
                "experiment",
            ),
        ],
    )

    graph = KnowledgeGraph.model_validate(payload)
    assert graph.edges[0].relation == "TESTED_IN"

    report = _draft_report(
        payload,
        relation_constraints=SERS_RELATION_CONSTRAINTS,
    )

    assert report.valid


def test_invalid_sers_relation_requires_domain_constraint_layer():
    payload = _payload(
        entities=[
            _entity(
                "substrate",
                "PlasmonicSubstrate",
            ),
        ],
        experiments=[
            _experiment("experiment"),
        ],
        edges=[
            _edge(
                "experiment",
                "TESTED_IN",
                "substrate",
            ),
        ],
    )

    # The wire container does not own SERS-specific relation semantics.
    graph = KnowledgeGraph.model_validate(payload)
    assert graph.edges[0].relation == "TESTED_IN"

    report = _draft_report(
        payload,
        relation_constraints=SERS_RELATION_CONSTRAINTS,
    )

    assert not report.valid
    assert (
        IssueCode.RELATION_SOURCE_TYPE_MISMATCH
        in report.codes()
    )
    assert (
        IssueCode.RELATION_TARGET_TYPE_MISMATCH
        in report.codes()
    )
