from __future__ import annotations

from types import SimpleNamespace

from pipeline_core.discovery.projection_relation_adjudication import _downgrade


def test_explicit_domain_mismatch_cannot_create_contextual_conflict():
    candidate = SimpleNamespace(
        abstract="Endpoint A is directly associated with endpoint B.",
        typed_compatibility_state="EXPLICIT_DOMAIN_MISMATCH",
        counterevidence_modes=[],
        second_pass_roles=[],
    )
    relationship, reasons = _downgrade(
        "CONTEXTUAL_CONFLICT",
        candidate=candidate,
        basis_projection_ids=[],
        counterevidence_modes=[],
        second_pass_roles=[],
        evidence_span="Endpoint A is directly associated with endpoint B",
        projection_by_id={},
    )

    assert relationship == "UNRELATED"
    assert (
        "strong_relation_downgraded_for_explicit_domain_mismatch"
        in reasons
    )
