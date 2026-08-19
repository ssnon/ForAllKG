from __future__ import annotations

import ast
from pathlib import Path

import campaigns.sers_novelty_gap.scripts.verify_sers_targeted_retrieval_t1_v1_failure_evidence as canonical_failure
import campaigns.sers_novelty_gap.scripts.verify_sers_targeted_retrieval_t1_live_v2 as canonical_live

import scripts.verify_sers_targeted_retrieval_t1_v1_failure_evidence as legacy_failure
import scripts.verify_sers_targeted_retrieval_t1_live_v2 as legacy_live


ROOT = Path(__file__).resolve().parents[1]


def test_failure_wrapper_delegates_main() -> None:
    assert (
        legacy_failure.main
        is canonical_failure.main
    )


def test_live_wrapper_delegates_main() -> None:
    assert (
        legacy_live.main
        is canonical_live.main
    )


def imported_modules(path: Path) -> list[str]:
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


def test_failure_uses_canonical_recovery() -> None:
    path = (
        ROOT
        / "campaigns"
        / "sers_novelty_gap"
        / "scripts"
        / "verify_sers_targeted_retrieval_t1_v1_failure_evidence.py"
    )

    modules = imported_modules(path)

    assert (
        "campaigns.sers_novelty_gap."
        "sers_targeted_retrieval_t1_live_recovery_v2"
        in modules
    )

    assert (
        "dac_her."
        "sers_targeted_retrieval_t1_live_recovery_v2"
        not in modules
    )


def test_live_uses_only_canonical_campaign_dependencies() -> None:
    path = (
        ROOT
        / "campaigns"
        / "sers_novelty_gap"
        / "scripts"
        / "verify_sers_targeted_retrieval_t1_live_v2.py"
    )

    modules = imported_modules(path)

    assert (
        "campaigns.sers_novelty_gap."
        "sers_targeted_retrieval_t1_live_recovery_v2"
        in modules
    )

    assert (
        "campaigns.sers_novelty_gap."
        "sers_targeted_retrieval_t1_live_validation_v2"
        in modules
    )

    assert not any(
        module.startswith(
            "dac_her.sers_targeted_retrieval_"
        )
        for module in modules
    )


def test_live_subprocess_targets_canonical_failure_verifier() -> None:
    path = (
        ROOT
        / "campaigns"
        / "sers_novelty_gap"
        / "scripts"
        / "verify_sers_targeted_retrieval_t1_live_v2.py"
    )

    source = path.read_text(
        encoding="utf-8"
    )

    tree = ast.parse(
        source,
        filename=str(path),
    )

    literals = {
        node.value
        for node in ast.walk(tree)
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
        )
    }

    assert (
        "campaigns.sers_novelty_gap.scripts."
        "verify_sers_targeted_retrieval_t1_v1_failure_evidence"
        in literals
    )

    assert (
        "scripts."
        "verify_sers_targeted_retrieval_t1_v1_failure_evidence"
        not in literals
    )
