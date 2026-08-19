from __future__ import annotations

import ast
from pathlib import Path

import campaigns.sers_novelty_gap.scripts.run_sers_targeted_retrieval_t0_offline as canonical_run
import campaigns.sers_novelty_gap.scripts.verify_sers_targeted_retrieval_t0_offline as canonical_verify

import scripts.run_sers_targeted_retrieval_t0_offline as legacy_run
import scripts.verify_sers_targeted_retrieval_t0_offline as legacy_verify


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


def test_run_wrapper_preserves_main_and_default_root() -> None:
    assert (
        legacy_run.main
        is canonical_run.main
    )

    assert (
        legacy_run.DEFAULT_RUN_ROOT
        == canonical_run.DEFAULT_RUN_ROOT
    )


def test_verify_wrapper_delegates_main() -> None:
    assert (
        legacy_verify.main
        is canonical_verify.main
    )


def test_canonical_run_preserves_repo_root() -> None:
    assert (
        canonical_run.ROOT
        == ROOT
    )


def test_canonical_run_uses_campaign_validation() -> None:
    path = (
        ROOT
        / "campaigns"
        / "sers_novelty_gap"
        / "scripts"
        / "run_sers_targeted_retrieval_t0_offline.py"
    )

    modules = imported_modules(path)

    assert (
        "campaigns.sers_novelty_gap."
        "sers_targeted_retrieval_t0_offline_validation"
        in modules
    )

    assert (
        "dac_her."
        "sers_targeted_retrieval_t0_offline_validation"
        not in modules
    )


def test_canonical_verify_uses_campaign_dependencies() -> None:
    path = (
        ROOT
        / "campaigns"
        / "sers_novelty_gap"
        / "scripts"
        / "verify_sers_targeted_retrieval_t0_offline.py"
    )

    modules = imported_modules(path)

    assert (
        "campaigns.sers_novelty_gap."
        "sers_targeted_retrieval_t0_offline_validation"
        in modules
    )

    assert (
        "campaigns.sers_novelty_gap.scripts."
        "run_sers_targeted_retrieval_t0_offline"
        in modules
    )

    assert not any(
        module.startswith(
            "dac_her.sers_targeted_retrieval_"
        )
        for module in modules
    )

    assert (
        "scripts."
        "run_sers_targeted_retrieval_t0_offline"
        not in modules
    )
