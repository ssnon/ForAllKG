from scripts.discovery.run_s233_blind_relationship_adjudication import (
    evidence_spans_valid,
)


def test_grounded_span_required_for_substantive_label():
    abstract = "A changes B under C."
    assert evidence_spans_valid(
        abstract=abstract,
        spans=["A changes B"],
        relationship="PARTIAL_PRIOR_ART",
    )
    assert not evidence_spans_valid(
        abstract=abstract,
        spans=["A changes D"],
        relationship="PARTIAL_PRIOR_ART",
    )


def test_unrelated_does_not_require_span():
    assert evidence_spans_valid(
        abstract="",
        spans=[],
        relationship="UNRELATED",
    )


def test_insufficient_metadata_does_not_require_span():
    assert evidence_spans_valid(
        abstract="",
        spans=[],
        relationship="INSUFFICIENT_METADATA",
    )
