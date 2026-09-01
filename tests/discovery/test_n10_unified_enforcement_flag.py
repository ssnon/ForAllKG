import ast
from pathlib import Path


PATH = Path(
    "scripts/discovery/"
    "run_dac_discovery_e2e.py"
)


def _text():
    return PATH.read_text()


def test_unified_nonobviousness_flag_is_exposed():
    text = _text()

    assert (
        '"--nonobviousness-enforce"'
        in text
    )

    tree = ast.parse(text)

    constants = {
        node.value
        for node in ast.walk(tree)
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
        )
    }

    assert (
        "--nonobviousness-enforce"
        in constants
    )


def test_unified_flag_activates_both_n10_authority_halves():
    text = _text()

    assert (
        "if args.nonobviousness_enforce:"
        in text
    )

    assert (
        "args.nonobviousness_original_fallback_enforce = True"
        in text
    )

    assert (
        "args.nonobviousness_post_generation_enforce = True"
        in text
    )


def test_original_fallback_stage_still_uses_n10_gate():
    text = _text()

    assert (
        "build_nonobviousness_production_gate"
        in text
    )

    assert (
        "nonobviousness_n10."
        in text
    )


def test_generated_candidates_still_receive_fresh_n10():
    text = _text()

    assert (
        "enforce_alpha6_nonobviousness"
        in text
    )

    assert (
        "Fresh Alpha6 candidate"
        in text
    )


def test_unified_mode_inherits_legacy_authority_exclusion():
    text = _text()

    # Unified mode becomes the staged flags before these guards,
    # so existing mutual-exclusion contracts remain authoritative.
    assert (
        "args.scientific_novelty_action_enforce"
        in text
    )

    assert (
        "args.post_generation_scientific_novelty_enforce"
        in text
    )

    assert (
        "are mutually exclusive"
        in text
    )
