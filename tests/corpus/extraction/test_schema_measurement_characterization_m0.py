from __future__ import annotations

import hashlib
import json

import pytest
from pydantic import ValidationError

from pipeline_core.corpus.schemas import (
    CalculationNode,
    Condition,
    EntityNode,
    EvidencePointer,
    ExperimentNode,
    KGEdge,
    KnowledgeGraph,
    MeasurementGroupNode,
    MeasurementNode,
)
from pipeline_core.corpus.extraction.draft_schema import (
    KnowledgeGraphDraft,
    MeasurementGroupDraft,
)


EXPECTED_SCHEMA_SHA256 = {
    "Condition":
        "c8ec2b1708a76e6c1ed72bee75b9c2ff3db12cfb6e37d660d71e9293d9f6b0b9",
    "ExperimentNode":
        "ce3bb76b86de69906ac0add7280677151a5ababb6e39f0d3f6d8ec40327aa8b1",
    "CalculationNode":
        "4e37f05d92d8479e5fd10a9c498c915ecc27a8f7b56d374fa5cb308b9a995844",
    "MeasurementNode":
        "7984501f15dc9f7550946d8ac8b221e8aee1151feac2d63e61ce4f9dad9885bb",
    "MeasurementGroupNode":
        "e1e60af0b27936cb307ac719a7e698bef71aafa0dde77f09c5761a62668613a7",
    "EvidencePointer":
        "80f71fae389c8f5663adb4ff00035f240deb9d50e0d2391c8188bc95e4476593",
    "KGEdge":
        "0973126cb2dfa62abd16bf57ff325bfb5e7055237387a39470c99ee7e2a6bd4b",
    "KnowledgeGraph":
        "c6936df0d2173a0698402646d4f2ae5b05cba47e53b8c670a17bea5a75a2fac7",
    "KnowledgeGraphDraft":
        "21edd941b3c82abb761c7bca3a58d24fc1208e69152156e9d01d89a30ee0a3bc",
}


SCHEMA_MODELS = (
    Condition,
    ExperimentNode,
    CalculationNode,
    MeasurementNode,
    MeasurementGroupNode,
    EvidencePointer,
    KGEdge,
    KnowledgeGraph,
    KnowledgeGraphDraft,
)


def _schema_sha256(model) -> str:
    payload = json.dumps(
        model.model_json_schema(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize("model", SCHEMA_MODELS)
def test_wire_schema_fingerprint_is_frozen_for_refactor(model):
    assert _schema_sha256(model) == EXPECTED_SCHEMA_SHA256[model.__name__]


def _condition_numeric() -> Condition:
    return Condition(
        name="temperature",
        value_numeric=298.15,
        value_text=None,
        unit="K",
        reference=None,
    )


def _measurement(
    *,
    measurement_id: str = "m1",
    subject_id: str = "subject",
    value_numeric=1.23,
    value_text=None,
    group_id=None,
) -> MeasurementNode:
    return MeasurementNode(
        id=measurement_id,
        metric_id="example_metric",
        metric="Example metric",
        subject_id=subject_id,
        source_expression="Example metric was 1.23.",
        group_id=group_id,
        value_numeric=value_numeric,
        value_text=value_text,
        unit="a.u." if value_numeric is not None else None,
        uncertainty=None,
        qualifier=None,
        basis=None,
        conditions=[],
        description=None,
    )


def _pointer() -> EvidencePointer:
    return EvidencePointer(
        document_id="main",
        document_role="main",
        page_id=1,
        asset_ids=[],
        locator_text=None,
    )


def _edge(source: str, relation: str, target: str) -> KGEdge:
    return KGEdge(
        source=source,
        relation=relation,
        target=target,
        evidence_type="experimental_observation",
        evidence_strength="direct",
        evidence_text=f"{source} {relation} {target}",
        confidence="high",
        evidence_pointers=[_pointer()],
        subsection=None,
    )


def _minimal_graph(*, subject_type: str) -> KnowledgeGraph:
    return KnowledgeGraph(
        paper_id="paper",
        chunk_id="chunk",
        section="Results",
        document_id="main",
        document_role="main",
        page_ids=[1],
        asset_ids=[],
        entities=[
            EntityNode(
                id="subject",
                type=subject_type,
                label="Scientific subject",
                description=None,
            ),
        ],
        experiments=[
            ExperimentNode(
                id="exp",
                name="Example experiment",
                experiment_type="unregistered_example_experiment",
                experiment_family="other",
                method_label="Example experiment",
                raw_method_name=None,
                conditions=[_condition_numeric()],
                description=None,
            ),
        ],
        calculations=[],
        measurements=[_measurement()],
        measurement_groups=[],
        observation_claims=[],
        mechanism_claims=[],
        edges=[
            _edge("exp", "HAS_MEASUREMENT", "m1"),
            _edge("m1", "MEASURED_FOR", "subject"),
        ],
    )


def test_condition_requires_a_value_but_does_not_use_measurement_xor():
    with pytest.raises(ValidationError):
        Condition(
            name="temperature",
            value_numeric=None,
            value_text=None,
            unit=None,
            reference=None,
        )

    condition = Condition(
        name="temperature",
        value_numeric=298.15,
        value_text="room temperature",
        unit="K",
        reference=None,
    )
    assert condition.value_numeric == 298.15
    assert condition.value_text == "room temperature"


@pytest.mark.parametrize(
    ("value_numeric", "value_text"),
    [
        (None, None),
        (1.0, "approximately one"),
    ],
)
def test_measurement_requires_exact_numeric_text_xor(
    value_numeric,
    value_text,
):
    with pytest.raises(
        ValidationError,
        match="exactly one of value_numeric or value_text",
    ):
        _measurement(
            value_numeric=value_numeric,
            value_text=value_text,
        )


def test_measurement_accepts_numeric_or_textual_payload_individually():
    numeric = _measurement(
        value_numeric=1.23,
        value_text=None,
    )
    textual = _measurement(
        value_numeric=None,
        value_text="below detection limit",
    )

    assert numeric.value_numeric == 1.23
    assert numeric.value_text is None
    assert textual.value_numeric is None
    assert textual.value_text == "below detection limit"


def test_strict_group_rejects_singleton_but_draft_preserves_it():
    with pytest.raises(
        ValidationError,
        match="at least two scalar measurements",
    ):
        MeasurementGroupNode(
            id="g1",
            group_type="comparison",
            label="Comparison",
            member_measurement_ids=["m1"],
            description=None,
        )

    draft = MeasurementGroupDraft(
        id="g1",
        group_type="comparison",
        label="Comparison",
        member_measurement_ids=["m1"],
        description=None,
    )
    assert draft.member_measurement_ids == ["m1"]


@pytest.mark.parametrize(
    "subject_type",
    [
        "Catalyst",
        "PlasmonicSubstrate",
    ],
)
def test_shared_measurement_structure_accepts_dac_and_sers_subject_types(
    subject_type,
):
    graph = _minimal_graph(subject_type=subject_type)

    assert graph.measurements[0].subject_id == "subject"
    assert graph.entities[0].type == subject_type


def test_measurement_requires_experiment_or_calculation_producer():
    graph = _minimal_graph(subject_type="Catalyst")
    payload = graph.model_dump()
    payload["edges"] = [
        edge
        for edge in payload["edges"]
        if edge["relation"] != "HAS_MEASUREMENT"
    ]

    with pytest.raises(
        ValidationError,
        match="no incoming HAS_MEASUREMENT",
    ):
        KnowledgeGraph.model_validate(payload)


def test_measurement_requires_exactly_one_measured_for_edge():
    graph = _minimal_graph(subject_type="Catalyst")
    payload = graph.model_dump()
    payload["edges"].append(
        _edge("m1", "MEASURED_FOR", "subject").model_dump()
    )

    with pytest.raises(
        ValidationError,
        match="exactly one MEASURED_FOR",
    ):
        KnowledgeGraph.model_validate(payload)


def test_measurement_subject_id_must_match_measured_for_target():
    graph = _minimal_graph(subject_type="Catalyst")
    payload = graph.model_dump()
    payload["entities"].append(
        EntityNode(
            id="other",
            type="Material",
            label="Other subject",
            description=None,
        ).model_dump()
    )

    # Keep "other" non-isolated while deliberately violating
    # subject_id <-> MEASURED_FOR target agreement.
    payload["edges"][-1] = _edge(
        "m1",
        "MEASURED_FOR",
        "other",
    ).model_dump()

    with pytest.raises(
        ValidationError,
        match="subject_id does not match MEASURED_FOR target",
    ):
        KnowledgeGraph.model_validate(payload)
