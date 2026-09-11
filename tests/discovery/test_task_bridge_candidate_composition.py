from __future__ import annotations

from pipeline_core.discovery.task_bridge_candidate_composition import (
    CandidateRelationView,
    _target_endpoint_atom_matches,
    _target_endpoint_atoms,
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


def test_shared_mediator_composition_records_argument_role_bindings():
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
        (
            "nanostructure size shape "
            "composition arrangement"
        ),
    )

    rows = compose_task_bridge_candidates(
        candidates=[
            source,
            target,
        ],
        requested_source="structural motif",
        requested_target="SERS plasmonic behavior",
    )

    assert len(rows) == 1

    row = rows[0]

    assert row.source_role_binding.task_subject_tokens == [
        "structural"
    ]
    assert row.source_role_binding.task_predicate_tokens == []
    assert row.source_role_binding.task_object_tokens == []
    assert row.source_role_binding.mediator_subject_tokens == []
    assert row.source_role_binding.mediator_predicate_tokens == []
    assert row.source_role_binding.mediator_object_tokens == [
        "shape",
        "size",
    ]

    assert row.target_role_binding.task_subject_tokens == [
        "plasmonic"
    ]
    assert row.target_role_binding.task_predicate_tokens == []
    assert row.target_role_binding.task_object_tokens == []
    assert row.target_role_binding.mediator_subject_tokens == []
    assert row.target_role_binding.mediator_predicate_tokens == []
    assert row.target_role_binding.mediator_object_tokens == [
        "shape",
        "size",
    ]

def test_q3_bad_fixture_is_rejected_by_target_atom_fidelity():
    source = _candidate(
        "q3_source",
        "plasmonic properties",
        "VARIES_WITH",
        (
            "nanostructure size, shape, "
            "composition, and arrangement"
        ),
    )

    target = _candidate(
        "q3_target",
        "nanoparticle size",
        "VARIES_WITH",
        "laser wavelength",
    )

    rows = compose_task_bridge_candidates(
        candidates=[
            source,
            target,
        ],
        requested_source=(
            "Au nanostructure plasmonic response"
        ),
        requested_target=(
            "excitation wavelength and Raman "
            "reporter or dye selection"
        ),
    )

    assert rows == ()


def test_target_endpoint_atoms_split_explicit_coordination():
    atoms = _target_endpoint_atoms(
        (
            "excitation wavelength and Raman "
            "reporter or dye selection"
        )
    )

    assert atoms == (
        frozenset(
            {
                "excitation",
                "wavelength",
            }
        ),
        frozenset(
            {
                "raman",
                "reporter",
            }
        ),
        frozenset(
            {
                "dye",
                "selection",
            }
        ),
    )


def test_compound_target_requires_complete_atom_in_single_argument():
    relation = _candidate(
        "target",
        "nanoparticle size",
        "VARIES_WITH",
        "laser wavelength",
    )

    atoms = _target_endpoint_atoms(
        (
            "excitation wavelength and Raman "
            "reporter or dye selection"
        )
    )

    assert not _target_endpoint_atom_matches(
        relation=relation,
        target_atoms=atoms,
    )


def test_compound_target_accepts_complete_atom_in_subject():
    relation = _candidate(
        "target",
        "SERS intensity",
        "VARIES_WITH",
        "interparticle separation",
    )

    atoms = _target_endpoint_atoms(
        (
            "electromagnetic hotspot number, "
            "intensity, and spatial distribution"
        )
    )

    assert _target_endpoint_atom_matches(
        relation=relation,
        target_atoms=atoms,
    )


def test_single_atom_target_preserves_legacy_overlap_semantics():
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
        (
            "nanostructure size shape "
            "composition arrangement"
        ),
    )

    rows = compose_task_bridge_candidates(
        candidates=[
            source,
            target,
        ],
        requested_source="structural motif",
        requested_target="SERS plasmonic behavior",
    )

    assert len(rows) == 1
    assert (
        rows[0].composite_id
        == "task_bridge_composite:5ccdb4182f7dc352daa2"
    )
    assert rows[0].compatibility_score == 4.25
