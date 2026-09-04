from pathlib import Path


E2E = Path(
    "scripts/discovery/"
    "run_dac_discovery_e2e.py"
)

PROD_BUILDER = Path(
    "scripts/discovery/"
    "build_nonobviousness_production_gate.py"
)

PROD_CORE = Path(
    "pipeline_core/discovery/"
    "nonobviousness_production_gate.py"
)


def _text():
    return E2E.read_text(
        encoding="utf-8"
    )


def test_comparison_stage_follows_v1_gate():
    text = _text()

    production = (
        '"build_nonobviousness_production_gate"'
    )

    comparison = (
        '"build_nonobviousness_dual_run_comparison"'
    )

    alpha6 = (
        '"scripts.discovery.run_novelty_refinement"'
    )

    assert text.count(production) == 1
    assert text.count(comparison) == 1

    production_pos = text.index(
        production
    )

    comparison_pos = text.index(
        comparison
    )

    alpha6_pos = text.index(
        alpha6,
        comparison_pos,
    )

    assert (
        production_pos
        < comparison_pos
        < alpha6_pos
    )


def test_comparison_uses_same_n9_inputs_and_query_plan():
    text = _text()

    start = text.index(
        '"build_nonobviousness_dual_run_comparison"'
    )

    end = text.index(
        "if (\n"
        "        args.nonobviousness_post_generation_enforce",
        start,
    )

    block = text[
        start:end
    ]

    assert '"--query-plan"' in block
    assert "str(external_plan)" in block

    assert '"--intake-shadow"' in block
    assert "str(nonobviousness_shadow)" in block

    assert '"--full-shadow"' in block
    assert "str(nonobviousness_full_shadow)" in block

    assert '"--output"' in block

    assert (
        "nonobviousness_dual_run_comparison"
        in block
    )


def test_comparison_artifact_name_is_not_production_named():
    text = _text()

    assert (
        '"dual_run_comparison.shadow.json"'
        in text
    )

    assert (
        "dual_run_comparison.production"
        not in text
    )


def test_alpha6_still_consumes_only_v1_scientific_gate():
    text = _text()

    comparison_pos = text.index(
        '"build_nonobviousness_dual_run_comparison"'
    )

    # There is an earlier occurrence inside
    # _check_alpha6_available(). Search only after the E1c
    # comparison stage so this resolves the actual Alpha6
    # runner.run_stage invocation.
    start = text.index(
        '"scripts.discovery.run_novelty_refinement"',
        comparison_pos,
    )

    end = text.index(
        "# ------------------------------------------------------------------",
        start + 1,
    )

    block = text[
        start:end
    ]

    assert (
        '"--scientific-novelty-gate"'
        in block
    )

    assert (
        "str(scientific_novelty_gate)"
        in block
    )

    assert (
        "nonobviousness_dual_run_comparison"
        not in block
    )


def test_authoritative_builder_does_not_consume_v2():
    text = PROD_BUILDER.read_text(
        encoding="utf-8"
    )

    assert (
        "build_nonobviousness_dual_run_comparison"
        not in text
    )

    assert (
        "hypothesis_selection_shadow_v2"
        not in text
    )


def test_authoritative_core_does_not_consume_v2():
    text = PROD_CORE.read_text(
        encoding="utf-8"
    )

    assert (
        "nonobviousness_dual_run_comparison"
        not in text
    )

    assert (
        "hypothesis_selection_shadow_v2"
        not in text
    )
