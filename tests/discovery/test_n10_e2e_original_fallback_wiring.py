import ast
from pathlib import Path


PATH = Path(
    "scripts/discovery/"
    "run_dac_discovery_e2e.py"
)


def _text():
    return PATH.read_text()


def test_n10_original_fallback_flag_materializes_both_shadow_stages():
    text = _text()

    assert (
        "args.nonobviousness_original_fallback_enforce"
        in text
    )

    intake = text.split(
        "nonobviousness_shadow = None",
        1,
    )[1].split(
        "scientific_novelty_action_batch = None",
        1,
    )[0]

    assert (
        "args.nonobviousness_original_fallback_enforce"
        in intake
    )

    assert (
        "scripts.discovery.run_nonobviousness_full_shadow"
        in intake
    )


def test_n10_enforcement_expands_ready_claim_cap():
    text = _text()

    assert "nonobviousness_ready_count" in text

    assert '"--max-ready-claims"' in text

    assert (
        "max("
        in text
    )


def test_n10_and_legacy_action_authorities_are_mutually_exclusive():
    text = _text()

    assert (
        "args.nonobviousness_original_fallback_enforce"
        in text
    )

    assert (
        "args.scientific_novelty_action_enforce"
        in text
    )

    assert (
        "are mutually exclusive"
        in text
    )


def test_n10_compiles_into_existing_alpha6_gate_contract():
    text = _text()

    # Adjacent Python string literals may be split across physical
    # source lines. Inspect the parsed constant rather than depending
    # on source formatting.
    tree = ast.parse(text)

    string_constants = {
        node.value
        for node in ast.walk(tree)
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
        )
    }

    assert (
        "scripts.discovery."
        "build_nonobviousness_production_gate"
        in string_constants
    )

    assert (
        "nonobviousness_n10."
        "fallback_gate.production.json"
        in string_constants
    )

    # Alpha6 still has exactly one gate input variable.
    alpha6 = text.split(
        "# 11. Alpha6 targeted novelty refinement",
        1,
    )[1]

    assert (
        '"--scientific-novelty-gate"'
        in alpha6
    )

    assert (
        "str(scientific_novelty_gate)"
        in alpha6
    )


def test_full_nonobviousness_enforce_name_is_now_exposed():
    text = _text()

    assert (
        'parser.add_argument(\n'
        '        "--nonobviousness-enforce"'
        in text
    )

    assert (
        "args.nonobviousness_original_fallback_enforce = True"
        in text
    )
