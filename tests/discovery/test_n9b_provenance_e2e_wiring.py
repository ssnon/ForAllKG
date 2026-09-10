import ast

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
    for stage_label in (
        (
            "[11N10/13] Fresh Alpha6 candidate "
            "non-obviousness enforcement"
        ),
        (
            "[11N10/13] Fresh Alpha6 candidate "
            "novelty certification"
        ),
    ):
        argv = _argv_for_run_stage(
            E2E,
            stage_label=stage_label,
            module_name=(
                "scripts.discovery."
                "enforce_alpha6_nonobviousness"
            ),
        )

        assert (
            _flag_value_expression(
                argv,
                "--hypothesis-context",
            )
            == "str(context)"
        )


# ------------------------------------------------------------------
# E3e8-V1 canonical source-portfolio invocation regression
#
# These tests intentionally inspect the exact argv list of the
# specific production invocation. A file/function-wide substring
# check is insufficient because other stages also pass --portfolio.
# ------------------------------------------------------------------


def _constant_string(
    node: ast.AST,
) -> str | None:
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
    ):
        return node.value

    return None


def _argv_for_run_stage(
    path: Path,
    *,
    stage_label: str,
    module_name: str,
) -> ast.List:
    tree = ast.parse(
        _source(path)
    )

    matches: list[ast.List] = []

    for node in ast.walk(tree):
        if not isinstance(
            node,
            ast.Call,
        ):
            continue

        if not (
            isinstance(
                node.func,
                ast.Attribute,
            )
            and node.func.attr
            == "run_stage"
        ):
            continue

        if len(node.args) < 3:
            continue

        if (
            _constant_string(
                node.args[0]
            )
            != stage_label
        ):
            continue

        if (
            _constant_string(
                node.args[1]
            )
            != module_name
        ):
            continue

        argv = node.args[2]

        if not isinstance(
            argv,
            ast.List,
        ):
            raise AssertionError(
                "target run_stage argv "
                "must be a literal list"
            )

        matches.append(
            argv
        )

    assert len(matches) == 1

    return matches[0]


def _argv_for_run_helper(
    path: Path,
    *,
    module_name: str,
) -> ast.List:
    tree = ast.parse(
        _source(path)
    )

    matches: list[ast.List] = []

    for node in ast.walk(tree):
        if not isinstance(
            node,
            ast.Call,
        ):
            continue

        if not (
            isinstance(
                node.func,
                ast.Name,
            )
            and node.func.id == "_run"
        ):
            continue

        if len(node.args) < 2:
            continue

        if (
            _constant_string(
                node.args[0]
            )
            != module_name
        ):
            continue

        argv = node.args[1]

        if not isinstance(
            argv,
            ast.List,
        ):
            raise AssertionError(
                "target _run argv "
                "must be a literal list"
            )

        matches.append(
            argv
        )

    assert len(matches) == 1

    return matches[0]


def _flag_value_expression(
    argv: ast.List,
    flag: str,
) -> str:
    positions = [
        index
        for index, node
        in enumerate(argv.elts)
        if (
            _constant_string(node)
            == flag
        )
    ]

    assert len(positions) == 1

    index = positions[0]

    assert (
        index + 1
        < len(argv.elts)
    )

    return ast.unparse(
        argv.elts[
            index + 1
        ]
    )


def test_original_n9_shadow_intake_passes_authoritative_source_portfolio():
    argv = _argv_for_run_stage(
        E2E,
        stage_label=(
            "[10N9-a/13] "
            "Non-obviousness shadow intake"
        ),
        module_name=(
            "scripts.discovery."
            "build_nonobviousness_shadow"
        ),
    )

    assert (
        _flag_value_expression(
            argv,
            "--query-plan",
        )
        == "str(external_plan)"
    )

    assert (
        _flag_value_expression(
            argv,
            "--external-report",
        )
        == "str(external_report)"
    )

    assert (
        _flag_value_expression(
            argv,
            "--portfolio",
        )
        == "str(axis_portfolio)"
    )

    assert (
        _flag_value_expression(
            argv,
            "--output",
        )
        == "str(nonobviousness_shadow)"
    )


def test_fresh_alpha6_n10_intake_passes_candidate_source_portfolio():
    argv = _argv_for_run_helper(
        ALPHA6_ENFORCER,
        module_name=(
            "scripts.discovery."
            "build_nonobviousness_shadow"
        ),
    )

    assert (
        _flag_value_expression(
            argv,
            "--query-plan",
        )
        == "str(query_plan)"
    )

    assert (
        _flag_value_expression(
            argv,
            "--external-report",
        )
        == "str(external_report)"
    )

    assert (
        _flag_value_expression(
            argv,
            "--portfolio",
        )
        == "str(source_portfolio)"
    )

    assert (
        _flag_value_expression(
            argv,
            "--output",
        )
        == "str(intake)"
    )
