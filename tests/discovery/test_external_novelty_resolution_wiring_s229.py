from pathlib import Path


def test_run_external_novelty_has_s229_pre_review_resolution_wiring():
    text = Path(
        "scripts/discovery/run_external_novelty.py"
    ).read_text(encoding="utf-8")

    assert "--pre-review-metadata-resolution" in text
    assert "--metadata-resolution-lookup-limit" in text
    assert "select_pre_review_resolution_targets" in text
    assert "PriorArtMetadataResolver" in text
    assert ".metadata_resolution.json" in text
    assert ".prior_art.resolved.json" in text


def test_resolution_occurs_after_ranker_before_pre_review_probe():
    text = Path(
        "scripts/discovery/run_external_novelty.py"
    ).read_text(encoding="utf-8")

    ranker = text.index("ranker = PriorArtRanker(")
    resolution = text.index(
        "if args.pre_review_metadata_resolution:"
    )
    coverage = text.index(
        "pre_review_coverage = None"
    )

    assert ranker < resolution < coverage
