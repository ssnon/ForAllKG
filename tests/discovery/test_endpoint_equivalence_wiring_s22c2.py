from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.discovery.build_task_conditioned_axis_plan import (
    _endpoint_fidelity_counts,
    _load_endpoint_equivalences,
    _used_endpoint_equivalence_witness_ids,
)
from pipeline_core.discovery.relation_component_composition import (
    compose_relation_component_topologies,
    confirmed_known_component_from_mapping,
)


def _accepted(
    *,
    node_id: str,
    subject: str,
    relation: str,
    object_: str,
) -> dict[str, object]:
    source_phrase = f"{subject} {relation.lower()} {object_}"
    return {
        "node_id": node_id,
        "source_local_id": node_id,
        "concept_type": "RelationPattern",
        "label": source_phrase,
        "source_phrase": source_phrase,
        "description": "",
        "retention_lane": "accepted_pattern",
        "evidence_scope": "paper_result",
        "pattern_subject": subject,
        "pattern_relation": relation,
        "pattern_object": object_,
        "relation_strength": "causal_interpretive",
        "qualifiers_json": "[]",
        "pattern_support_mode": "explicit_single_span",
        "supporting_phrases_json": json.dumps(
            [source_phrase]
        ),
        "subject_evidence_phrase": subject,
        "relation_evidence_phrase": relation.lower(),
        "object_evidence_phrase": object_,
        "comparison_items_json": "[]",
        "paper_id": "paper:test",
        "chunk_id": "chunk:test",
        "document_id": "document:test",
    }


def _write_set(
    path: Path,
    *,
    source: str = "interparticle separation",
    target: str = "SERS intensity",
    witnesses: list[dict[str, object]] | None = None,
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version":
                    "endpoint-equivalence-set-v1",
                "scope": {
                    "requested_source": source,
                    "requested_target": target,
                },
                "witnesses": (
                    witnesses
                    if witnesses is not None
                    else []
                ),
            }
        ),
        encoding="utf-8",
    )


def test_task_scoped_witness_loader_accepts_valid_file(tmp_path: Path):
    path = tmp_path / "eq.json"
    _write_set(
        path,
        witnesses=[
            {
                "witness_id": "q1:source:distance",
                "left_endpoint": "interparticle separation",
                "right_endpoint": "interparticle distance",
                "witness_kind": "task_supplied",
                "provenance_ids": ["review:s22c2:q1"],
            }
        ],
    )

    rows = _load_endpoint_equivalences(
        path,
        requested_source="interparticle separation",
        requested_target="SERS intensity",
    )

    assert len(rows) == 1
    assert rows[0].witness_id == "q1:source:distance"


def test_task_scope_mismatch_fails_closed(tmp_path: Path):
    path = tmp_path / "eq.json"
    _write_set(
        path,
        source="different source",
    )

    with pytest.raises(
        ValueError,
        match="source scope does not match task",
    ):
        _load_endpoint_equivalences(
            path,
            requested_source="interparticle separation",
            requested_target="SERS intensity",
        )


def test_witness_provenance_is_required(tmp_path: Path):
    path = tmp_path / "eq.json"
    _write_set(
        path,
        witnesses=[
            {
                "witness_id": "q1:no-provenance",
                "left_endpoint": "interparticle separation",
                "right_endpoint": "interparticle distance",
                "witness_kind": "task_supplied",
                "provenance_ids": [],
            }
        ],
    )

    with pytest.raises(
        ValueError,
        match="requires provenance_ids",
    ):
        _load_endpoint_equivalences(
            path,
            requested_source="interparticle separation",
            requested_target="SERS intensity",
        )


def test_duplicate_witness_id_fails_closed(tmp_path: Path):
    path = tmp_path / "eq.json"
    row = {
        "witness_id": "q1:duplicate",
        "left_endpoint": "interparticle separation",
        "right_endpoint": "interparticle distance",
        "witness_kind": "task_supplied",
        "provenance_ids": ["review:s22c2:q1"],
    }
    _write_set(
        path,
        witnesses=[row, dict(row)],
    )

    with pytest.raises(
        ValueError,
        match="duplicate endpoint-equivalence witness_id",
    ):
        _load_endpoint_equivalences(
            path,
            requested_source="interparticle separation",
            requested_target="SERS intensity",
        )


def test_q1_minimal_witnesses_promote_good_backbone_shape(tmp_path: Path):
    path = tmp_path / "eq.json"
    _write_set(
        path,
        witnesses=[
            {
                "witness_id": "q1:source:distance",
                "left_endpoint": "interparticle separation",
                "right_endpoint": "interparticle distance",
                "witness_kind": "task_supplied",
                "provenance_ids": ["review:s22c2:q1"],
            },
            {
                "witness_id": "q1:target:signal",
                "left_endpoint": "SERS intensity",
                "right_endpoint": "SERS signal",
                "witness_kind": "task_supplied",
                "provenance_ids": ["review:s22c2:q1"],
            },
        ],
    )
    witnesses = _load_endpoint_equivalences(
        path,
        requested_source="interparticle separation",
        requested_target="SERS intensity",
    )

    source = confirmed_known_component_from_mapping(
        _accepted(
            node_id="known:distance-field",
            subject="electric field enhancement",
            relation="VARIES_WITH",
            object_="interparticle distance",
        )
    )
    target = confirmed_known_component_from_mapping(
        _accepted(
            node_id="known:field-signal",
            subject="electric field enhancement",
            relation="PROMOTES",
            object_="SERS signal",
        )
    )

    rows = compose_relation_component_topologies(
        components=[source, target],
        requested_source="interparticle separation",
        requested_target="SERS intensity",
        endpoint_equivalences=witnesses,
        require_endpoint_fidelity=True,
    )

    assert len(rows) == 1
    assert (
        rows[0].source_binding.binding_authority
        == "equivalent"
    )
    assert (
        rows[0].target_binding.binding_authority
        == "equivalent"
    )
    assert _endpoint_fidelity_counts(rows) == (0, 1)
    assert _used_endpoint_equivalence_witness_ids(rows) == [
        "q1:source:distance",
        "q1:target:signal",
    ]
