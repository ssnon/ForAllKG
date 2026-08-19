from pathlib import Path
import ast

import campaigns.sers_novelty_gap.sers_targeted_retrieval_t1_live_guard as cg
import campaigns.sers_novelty_gap.sers_targeted_retrieval_t1_live_recovery_v2 as cr

import dac_her.sers_targeted_retrieval_t1_live_guard as lg
import dac_her.sers_targeted_retrieval_t1_live_recovery_v2 as lr


ROOT = Path(__file__).resolve().parents[1]


def test_guard_facade() -> None:
    assert (
        lg.validate_t1_pre_network_guard
        is cg.validate_t1_pre_network_guard
    )
    assert (
        lg.RUNTIME_TRACKED_FILES
        is cg.RUNTIME_TRACKED_FILES
    )


def test_guard_retains_historical_paths() -> None:
    assert (
        cg.RUNTIME_TRACKED_FILES[0]
        == "dac_her/sers_targeted_retrieval_t1_live_guard.py"
    )
    assert (
        cg.RUNTIME_TRACKED_FILES[1]
        == "dac_her/sers_targeted_retrieval_t1_live_validation.py"
    )


def test_recovery_facade() -> None:
    for name in (
        "build_v2_report",
        "load_frozen_context",
        "load_v1_gap1_models",
        "recover_v1_gap1_audit",
        "validate_v1_failure_evidence",
    ):
        assert (
            getattr(lr, name)
            is getattr(cr, name)
        )


def test_recovery_root_contract() -> None:
    assert cr.ROOT == ROOT

    assert (
        cr.SPEC_ROOT
        == ROOT
        / "evaluation"
        / "sers_novelty_gap"
        / "t1_live_targeted_retrieval_spec_v1"
    )

    assert (
        cr.V1_RUN_ROOT
        == ROOT
        / "evaluation"
        / "sers_novelty_gap"
        / "t1_live_targeted_retrieval_run_v1"
    )

    assert (
        cr.V2_RUN_ROOT
        == ROOT
        / "evaluation"
        / "sers_novelty_gap"
        / "t1_live_targeted_retrieval_run_v2"
    )


def test_recovery_uses_canonical_validation_v2() -> None:
    path = (
        ROOT
        / "campaigns"
        / "sers_novelty_gap"
        / "sers_targeted_retrieval_t1_live_recovery_v2.py"
    )

    tree = ast.parse(
        path.read_text(
            encoding="utf-8"
        ),
        filename=str(path),
    )

    modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }

    assert (
        "campaigns.sers_novelty_gap."
        "sers_targeted_retrieval_t1_live_validation_v2"
        in modules
    )

    assert (
        "dac_her."
        "sers_targeted_retrieval_t1_live_validation_v2"
        not in modules
    )
