from pathlib import Path


RUNNER = Path(
    "scripts/discovery/"
    "run_nonobviousness_full_shadow.py"
)


def _source() -> str:
    return RUNNER.read_text(
        encoding="utf-8"
    )


def test_full_runner_accepts_hypothesis_context():
    text = _source()

    assert "HypothesisContext" in text
    assert '"--hypothesis-context"' in text
    assert (
        "HypothesisContext.model_validate_json"
        in text
    )


def test_full_runner_initializes_positive_only_internal_reviewer():
    text = _source()

    assert (
        "InstructorOpenAICompatibleInternalClosureBackend"
        in text
    )
    assert (
        "review_and_compile_internal_base_target"
        in text
    )
    assert (
        'targets_by_slot.get(\n'
        '                "BASE_RELATION"'
        in text
    )


def test_full_runner_passes_internal_reviews_to_compiler():
    text = _source()

    assert (
        "internal_reviews=internal_reviews"
        in text
    )
    assert (
        '"internal_slot_reviews.json"'
        in text
    )


def test_full_runner_preserves_internal_provenance_separately():
    text = _source()

    assert (
        '"internal_positive_statement_ids"'
        in text
    )
    assert (
        "base_internal_statement_ids"
        in text
    )

    # Internal statement IDs must not be inserted into
    # external work-ID fields.
    assert (
        'positive_work_ids=internal_reviews'
        not in text
    )
