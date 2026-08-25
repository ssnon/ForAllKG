from __future__ import annotations

from pathlib import Path

import pytest

from pipeline_core.discovery.discovery_axis_runtime import (
    DiscoveryAxisSynthesisRuntime,
)


class _Dummy:
    pass


def test_two_inference_repairs_are_allowed() -> None:
    runtime = DiscoveryAxisSynthesisRuntime(
        _Dummy(),
        _Dummy(),
        max_inference_repairs=2,
    )

    assert runtime.max_inference_repairs == 2


def test_other_repair_budgets_remain_one_max() -> None:
    for field in (
        "max_compile_repairs",
        "max_fidelity_repairs",
        "max_novelty_repairs",
    ):
        with pytest.raises(
            ValueError,
            match=field,
        ):
            DiscoveryAxisSynthesisRuntime(
                _Dummy(),
                _Dummy(),
                **{field: 2},
            )


def test_inference_budget_above_two_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="max_inference_repairs",
    ):
        DiscoveryAxisSynthesisRuntime(
            _Dummy(),
            _Dummy(),
            max_inference_repairs=3,
        )


def test_second_bounded_repair_is_fail_closed() -> None:
    text = Path(
        "pipeline_core/discovery/discovery_axis_runtime.py"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "and self.max_inference_repairs >= 2"
        in text
    )

    # These messages are written as adjacent source literals in the
    # runtime. Assert the source fragments rather than the Python-runtime
    # concatenated value.
    assert (
        '"second bounded inference-strength "'
        in text
    )
    assert (
        '"repair chose abstention"'
        in text
    )

    assert (
        '"second inference repair lost "'
        in text
    )
    assert (
        '"assigned-axis fidelity"'
        in text
    )

    # Final rejection remains after the optional second repair.
    second = text.index(
        "and self.max_inference_repairs >= 2"
    )

    final_reject = text.index(
        'if inference.status != "pass":',
        second,
    )

    assert second < final_reject


def test_stage8_cli_defaults_to_two_inference_repairs() -> None:
    text = Path(
        "scripts/discovery/run_discovery_axis_hypothesis_maker.py"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        'parser.add_argument("--max-inference-repairs", '
        'type=int, choices=(0, 1, 2), default=2)'
        in text
    )

def test_accepted_attempt_records_inference_repair_stage() -> None:
    text = Path(
        "pipeline_core/discovery/discovery_axis_runtime.py"
    ).read_text(
        encoding="utf-8"
    )

    expected = """                    stage=(
                        "novelty_repair"
                        if novelty_repaired
                        else "inference_repair"
                        if inference_repaired
                        else "fidelity_repair"
                        if fidelity_repaired
                        else "initial"
                    ),
"""

    assert expected in text

