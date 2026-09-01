from pathlib import Path


PATH = Path(
    "scripts/discovery/"
    "run_dac_discovery_e2e.py"
)


def _text():
    return PATH.read_text()


def test_post_generation_n10_runs_after_alpha6():
    text = _text()

    alpha6 = text.index(
        '"scripts.discovery.run_novelty_refinement"'
    )

    post = text.index(
        '"enforce_alpha6_nonobviousness"'
    )

    assert post > alpha6


def test_post_generation_n10_rebinds_downstream_portfolio():
    text = _text()

    assert (
        "refined_portfolio = (\n"
        "            post_n10_portfolio"
        in text
    )

    assert (
        "N10-filtered"
        in text
    )


def test_post_generation_requires_original_n10_authority():
    text = _text()

    assert (
        "args.nonobviousness_post_generation_enforce"
        in text
    )

    assert (
        "not args.nonobviousness_original_fallback_enforce"
        in text
    )


def test_legacy_and_n10_post_generation_authorities_are_exclusive():
    text = _text()

    assert (
        "args.post_generation_scientific_novelty_enforce"
        in text
    )

    assert (
        "N10 and legacy post-generation scientific novelty "
        in text
    )
    assert (
        "authorities are mutually exclusive"
        in text
    )


def test_unified_nonobviousness_flag_is_now_exposed():
    text = _text()

    assert (
        'parser.add_argument(\n'
        '        "--nonobviousness-enforce"'
        in text
    )

    assert (
        "args.nonobviousness_post_generation_enforce = True"
        in text
    )


def test_post_generation_n10_does_not_require_device_namespace():
    text = _text()

    start = text.index(
        '"[11N10/13] Fresh Alpha6 candidate "'
    )

    end = text.index(
        "final_hypotheses = _hypothesis_count",
        start,
    )

    block = text[start:end]

    # The top-level E2E parser does not expose --device.
    # Optional downstream propagation therefore must not
    # assume Namespace.device exists.
    assert (
        'getattr(args, "device", None)'
        in block
    )

    assert (
        "if args.device"
        not in block
    )
