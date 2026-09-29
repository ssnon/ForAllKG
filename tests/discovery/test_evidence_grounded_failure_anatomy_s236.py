from scripts.discovery.run_s236_evidence_grounded_failure_anatomy import (
    classify_span,
)


def test_exact_span():
    assert (
        classify_span(
            span="A changes B.",
            abstract="We show A changes B. End.",
            title="Other",
        )
        == "EXACT_ABSTRACT_MATCH"
    )


def test_title_only_span():
    assert (
        classify_span(
            span="A changes B",
            abstract="No relation here.",
            title="A changes B",
        )
        == "TITLE_ONLY_MATCH"
    )


def test_whitespace_normalization():
    assert (
        classify_span(
            span="A  changes   B",
            abstract="A changes B",
            title="",
        )
        == "WHITESPACE_OR_UNICODE_NORMALIZATION_ONLY"
    )


def test_truncation_ellipsis():
    assert (
        classify_span(
            span="A changes B…",
            abstract="A changes B under condition C.",
            title="",
        )
        == "TRUNCATION_ELLIPSIS_ONLY"
    )


def test_true_ungrounded():
    assert (
        classify_span(
            span="A causes Z",
            abstract="A changes B",
            title="",
        )
        == "UNGROUNDED_OR_PARAPHRASED"
    )
