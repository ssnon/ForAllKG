from pathlib import Path


E2E = Path(
    "scripts/discovery/run_dac_discovery_e2e.py"
)

REFINEMENT_RUNTIME = Path(
    "pipeline_core/discovery/"
    "novelty_refinement_runtime.py"
)

REFINEMENT_RUNNER = Path(
    "scripts/discovery/run_novelty_refinement.py"
)

ALPHA6_ENFORCER = Path(
    "scripts/discovery/"
    "enforce_alpha6_nonobviousness.py"
)


def _source(path: Path) -> str:
    return path.read_text(
        encoding="utf-8"
    )


def _section(
    text: str,
    start: str,
    end: str,
) -> str:
    lo = text.index(start)
    hi = text.index(end, lo)
    return text[lo:hi]


def test_original_alpha4_n9_receives_exact_portfolio_and_context():
    text = _source(E2E)

    block = _section(
        text,
        '"[10N9-b/13] Non-obviousness full closure shadow"',
        "scientific_novelty_action_batch = None",
    )

    assert '"--portfolio"' in block
    assert "str(axis_portfolio)" in block

    assert '"--hypothesis-context"' in block
    assert "str(context)" in block


def test_fresh_external_artifact_retains_exact_source_portfolio():
    text = _source(
        REFINEMENT_RUNTIME
    )

    assert (
        "source_portfolio: "
        "HypothesisPortfolio | None = None"
        in text
    )

    fresh_block = _section(
        text,
        "def _fresh_external(",
        "\n    def run(",
    )

    assert "source_portfolio=portfolio" in fresh_block


def test_fresh_source_portfolio_is_persisted_with_external_bundle():
    text = _source(
        REFINEMENT_RUNNER
    )

    block = _section(
        text,
        "for i, row in enumerate(outcome.final_external_artifacts, 1):",
        'print()',
    )

    assert '".portfolio.json"' in block
    assert "row.source_portfolio" in block


def test_alpha6_n10_requires_canonical_context_and_exact_source_portfolio():
    text = _source(
        ALPHA6_ENFORCER
    )

    assert (
        "def find_final_external_bundle("
        in text
    )

    assert (
        '"--hypothesis-context"'
        in text
    )

    assert (
        "parsed_plan.source_portfolio_id"
        in text
    )

    assert (
        "parsed_prior.source_portfolio_id"
        in text
    )

    assert (
        "parsed_report.source_portfolio_id"
        in text
    )

    full_block = _section(
        text,
        "full_args = [",
        "\n        if args.base_url:",
    )

    assert '"--portfolio"' in full_block
    assert "str(source_portfolio)" in full_block

    assert '"--hypothesis-context"' in full_block
    assert "str(args.hypothesis_context)" in full_block


def test_main_e2e_passes_context_to_alpha6_n10_enforcer():
    text = _source(E2E)

    block = _section(
        text,
        '"[11N10/13] Fresh Alpha6 candidate "',
        "# Stage 12/13 now consume",
    )

    assert '"--hypothesis-context"' in block
    assert "str(context)" in block
