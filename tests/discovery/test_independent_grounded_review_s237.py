from pipeline_core.discovery.evidence_span_grounding import (
    ground_evidence_spans,
)


def test_normalized_span_is_grounded():
    result = ground_evidence_spans(
        ["A  CHANGES–B"],
        "a changes-B under condition C.",
    )
    assert result.grounded_spans == [
        "A  CHANGES–B"
    ]
    assert result.invalid_spans == []
    assert result.match_kinds == [
        "NORMALIZED"
    ]


def test_truncated_prompt_prefix_is_grounded():
    result = ground_evidence_spans(
        ["A changes B…"],
        "A changes B under condition C.",
    )
    assert result.grounded_spans == [
        "A changes B…"
    ]
    assert result.match_kinds == [
        "TRUNCATED_PREFIX"
    ]


def test_true_paraphrase_remains_invalid():
    result = ground_evidence_spans(
        ["A causes Z"],
        "A changes B under condition C.",
    )
    assert result.grounded_spans == []
    assert result.invalid_spans == [
        "A causes Z"
    ]
