from __future__ import annotations

from pathlib import Path


TOP = Path(
    "scripts/discovery/"
    "run_dac_discovery_e2e.py"
)

ALPHA6 = Path(
    "scripts/discovery/"
    "run_novelty_refinement.py"
)


def test_top_level_declares_post_generation_scientific_novelty_flag():
    text = TOP.read_text(
        encoding="utf-8"
    )

    assert (
        '"--post-generation-scientific-novelty-enforce"'
        in text
    )

    assert (
        "args.post_generation_scientific_novelty_enforce"
        in text
    )


def test_top_level_forwards_post_generation_flag_to_alpha6_stage():
    text = TOP.read_text(
        encoding="utf-8"
    )

    compact = "".join(
        text.split()
    )

    expected = (
        '*('
        '["--post-generation-scientific-novelty-enforce",]'
        'ifargs.post_generation_scientific_novelty_enforce'
        'else[]'
        '),'
    )

    assert expected in compact


def test_alpha6_accepts_forwarded_post_generation_flag():
    text = ALPHA6.read_text(
        encoding="utf-8"
    )

    assert (
        "--post-generation-scientific-novelty-enforce"
        in text
    )


def test_top_level_post_generation_enforcement_is_opt_in():
    import ast

    tree = ast.parse(
        TOP.read_text(
            encoding="utf-8"
        )
    )

    matches = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        if not (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            continue

        positional = [
            arg.value
            for arg in node.args
            if (
                isinstance(arg, ast.Constant)
                and isinstance(arg.value, str)
            )
        ]

        if (
            "--post-generation-scientific-novelty-enforce"
            not in positional
        ):
            continue

        keywords = {
            kw.arg: (
                kw.value.value
                if isinstance(
                    kw.value,
                    ast.Constant,
                )
                else None
            )
            for kw in node.keywords
            if kw.arg is not None
        }

        matches.append(
            keywords
        )

    assert len(matches) == 1

    # argparse store_true defaults to False, so historical/default
    # behavior remains unchanged unless the flag is explicitly supplied.
    assert (
        matches[0].get("action")
        == "store_true"
    )
