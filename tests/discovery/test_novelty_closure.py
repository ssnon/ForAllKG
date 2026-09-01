from types import SimpleNamespace

from pipeline_core.discovery.novelty_closure import (
    compile_closure_slot,
)


def review(status, relationships=()):
    return SimpleNamespace(
        status=status,
        matches=[
            SimpleNamespace(
                relationship=relationship,
                work_id=f"work:{i}",
            )
            for i, relationship
            in enumerate(
                relationships,
                1,
            )
        ],
    )


def test_direct_review_establishes_slot():
    result = compile_closure_slot(
        "BASE_RELATION",
        [
            review(
                "DIRECT_PRIOR_ART",
                ("DIRECT_PRIOR_ART",),
            )
        ],
    )

    assert result.evidence_state == "ESTABLISHED"


def test_partial_review_establishes_atomic_slot():
    result = compile_closure_slot(
        "BRIDGE_RELATION",
        [
            review(
                "PARTIAL_PRIOR_ART",
                ("PARTIAL_PRIOR_ART",),
            )
        ],
    )

    assert result.evidence_state == "ESTABLISHED"


def test_components_only_does_not_establish_relation():
    result = compile_closure_slot(
        "BRIDGE_RELATION",
        [
            review(
                "COMPONENTS_ONLY",
                ("COMPONENT_ONLY",),
            )
        ],
    )

    assert result.evidence_state == "NOT_FOUND"


def test_no_reviews_is_unassessed():
    result = compile_closure_slot(
        "BRIDGE_RELATION",
        [],
    )

    assert result.evidence_state == "UNASSESSED"


def test_multiple_bridge_probes_need_only_one_positive_relation():
    result = compile_closure_slot(
        "BRIDGE_RELATION",
        [
            review(
                "COMPONENTS_ONLY",
                ("COMPONENT_ONLY",),
            ),
            review(
                "DIRECT_PRIOR_ART",
                ("DIRECT_PRIOR_ART",),
            ),
        ],
    )

    assert result.evidence_state == "ESTABLISHED"
