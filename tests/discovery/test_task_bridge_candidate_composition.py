from __future__ import annotations

from pipeline_core.discovery.task_bridge_candidate_composition import (
    CandidateRelationView,
    compose_task_bridge_candidates,
    lexical_tokens,
)


def _candidate(
    unit_id: str,
    subject: str,
    relation: str,
    object_: str,
) -> CandidateRelationView:
    return CandidateRelationView(
        unit_id=unit_id,
        label=unit_id,
        proposed_subject=subject,
        proposed_relation=relation,
        proposed_object=object_,
    )


def test_plural_normalization_supports_size_and_sizes():
    assert "size" in lexical_tokens(
        "particle sizes"
    )


def test_requires_source_and_target_side_candidates():
    source = _candidate(
        "source",
        "superlattice structural order",
        "VARIES_WITH",
        "size and shape uniformity",
    )

    unrelated = _candidate(
        "other",
        "particle concentration",
        "VARIES_WITH",
        "temperature",
    )

    rows = compose_task_bridge_candidates(
        candidates=[
            source,
            unrelated,
        ],
        requested_source=(
            "structural motif"
        ),
        requested_target=(
            "SERS plasmonic behavior"
        ),
    )

    assert rows == ()


def test_shared_mediator_composes_source_and_target_relations():
    source = _candidate(
        "source",
        "superlattice structural order",
        "VARIES_WITH",
        "size and shape uniformity",
    )

    target = _candidate(
        "target",
        "plasmonic properties",
        "VARIES_WITH",
        "nanostructure size shape composition arrangement",
    )

    rows = compose_task_bridge_candidates(
        candidates=[
            source,
            target,
        ],
        requested_source=(
            "structural motif"
        ),
        requested_target=(
            "SERS plasmonic behavior"
        ),
    )

    assert len(rows) == 1

    assert rows[0].source_unit_id == "source"
    assert rows[0].target_unit_id == "target"

    assert rows[0].shared_mediator_tokens == [
        "shape",
        "size",
    ]

    assert (
        rows[0].epistemic_status
        == "inspiration_only"
    )

    assert rows[0].requires_verification


def test_no_shared_mediator_means_no_composition():
    source = _candidate(
        "source",
        "structural order",
        "VARIES_WITH",
        "building block uniformity",
    )

    target = _candidate(
        "target",
        "SERS enhancement",
        "VARIES_WITH",
        "laser power",
    )

    rows = compose_task_bridge_candidates(
        candidates=[
            source,
            target,
        ],
        requested_source=(
            "structural motif"
        ),
        requested_target=(
            "SERS behavior"
        ),
    )

    assert rows == ()


def test_two_shared_mediators_rank_above_one():
    source = _candidate(
        "source",
        "structural order",
        "VARIES_WITH",
        "particle size and shape uniformity",
    )

    target_two = _candidate(
        "target_two",
        "plasmonic behavior",
        "VARIES_WITH",
        "particle size and shape",
    )

    target_one = _candidate(
        "target_one",
        "SERS behavior",
        "VARIES_WITH",
        "particle size",
    )

    rows = compose_task_bridge_candidates(
        candidates=[
            source,
            target_one,
            target_two,
        ],
        requested_source=(
            "structural motif"
        ),
        requested_target=(
            "SERS plasmonic behavior"
        ),
    )

    assert len(rows) == 2

    assert (
        rows[0].target_unit_id
        == "target_two"
    )

    assert rows[0].shared_mediator_tokens == [
        "particle",
        "shape",
        "size",
    ]


def test_semantic_novelty_is_not_an_input():
    source = _candidate(
        "source",
        "structural morphology",
        "VARIES_WITH",
        "particle geometry",
    )

    target = _candidate(
        "target",
        "SERS response",
        "VARIES_WITH",
        "particle geometry",
    )

    rows = compose_task_bridge_candidates(
        candidates=[
            source,
            target,
        ],
        requested_source="structural morphology",
        requested_target="SERS response",
    )

    assert len(rows) == 1

    assert all(
        "novel" not in key.lower()
        and "semantic" not in key.lower()
        for key in type(rows[0]).model_fields
    )
