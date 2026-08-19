from __future__ import annotations

import ast
from pathlib import Path

import campaigns.sers_novelty_gap.scripts.preflight_sers_targeted_retrieval_t1_live_v2 as canonical_preflight
import campaigns.sers_novelty_gap.scripts.run_sers_targeted_retrieval_t1_live_v2 as canonical_run

import scripts.preflight_sers_targeted_retrieval_t1_live_v2 as legacy_preflight
import scripts.run_sers_targeted_retrieval_t1_live_v2 as legacy_run


ROOT = Path(__file__).resolve().parents[1]


def imported_modules(
    path: Path,
) -> list[str]:
    tree = ast.parse(
        path.read_text(
            encoding="utf-8"
        ),
        filename=str(path),
    )

    result = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            result.append(
                node.module or ""
            )

        elif isinstance(node, ast.Import):
            result.extend(
                alias.name
                for alias in node.names
            )

    return result


def test_preflight_wrapper_contract() -> None:
    assert (
        legacy_preflight.main
        is canonical_preflight.main
    )

    assert (
        legacy_preflight.validate_v2_preflight
        is canonical_preflight.validate_v2_preflight
    )

    assert (
        legacy_preflight.V2_RUNTIME_FILES
        is canonical_preflight.V2_RUNTIME_FILES
    )


def test_run_wrapper_contract() -> None:
    assert (
        legacy_run.main
        is canonical_run.main
    )


def test_preflight_preserves_historical_runtime_file_contract() -> None:
    assert (
        "dac_her/sers_targeted_retrieval_t1_live_validation_v2.py"
        in canonical_preflight.V2_RUNTIME_FILES
    )

    assert (
        "dac_her/sers_targeted_retrieval_t1_live_recovery_v2.py"
        in canonical_preflight.V2_RUNTIME_FILES
    )

    assert (
        "scripts/preflight_sers_targeted_retrieval_t1_live_v2.py"
        in canonical_preflight.V2_RUNTIME_FILES
    )

    assert (
        "scripts/run_sers_targeted_retrieval_t1_live_v2.py"
        in canonical_preflight.V2_RUNTIME_FILES
    )

    assert (
        "scripts/verify_sers_targeted_retrieval_t1_live_v2.py"
        in canonical_preflight.V2_RUNTIME_FILES
    )

    assert (
        "scripts/verify_sers_targeted_retrieval_t1_v1_failure_evidence.py"
        in canonical_preflight.V2_RUNTIME_FILES
    )


def test_preflight_uses_canonical_campaign_modules() -> None:
    path = (
        ROOT
        / "campaigns"
        / "sers_novelty_gap"
        / "scripts"
        / "preflight_sers_targeted_retrieval_t1_live_v2.py"
    )

    modules = imported_modules(path)

    assert (
        "campaigns.sers_novelty_gap."
        "sers_targeted_retrieval_t1_live_guard"
        in modules
    )

    assert (
        "campaigns.sers_novelty_gap."
        "sers_targeted_retrieval_t1_live_recovery_v2"
        in modules
    )

    assert not any(
        module.startswith(
            "dac_her.sers_targeted_retrieval_"
        )
        for module in modules
    )


def test_run_uses_canonical_campaign_dependencies() -> None:
    path = (
        ROOT
        / "campaigns"
        / "sers_novelty_gap"
        / "scripts"
        / "run_sers_targeted_retrieval_t1_live_v2.py"
    )

    modules = imported_modules(path)

    required = {
        (
            "campaigns.sers_novelty_gap."
            "sers_targeted_retrieval_t1_live_recovery_v2"
        ),
        (
            "campaigns.sers_novelty_gap."
            "sers_targeted_retrieval_t1_live_validation_v2"
        ),
        (
            "campaigns.sers_novelty_gap.scripts."
            "preflight_sers_targeted_retrieval_t1_live_v2"
        ),
    }

    assert required.issubset(
        set(modules)
    )

    assert not any(
        module.startswith(
            "dac_her.sers_targeted_retrieval_"
        )
        for module in modules
    )

    assert (
        "scripts."
        "preflight_sers_targeted_retrieval_t1_live_v2"
        not in modules
    )
