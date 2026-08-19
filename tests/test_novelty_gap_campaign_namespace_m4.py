from __future__ import annotations

import ast
from pathlib import Path

import campaigns.sers_novelty_gap.sers_targeted_retrieval_t0_offline_validation as canonical_t0
import campaigns.sers_novelty_gap.sers_targeted_retrieval_t1_live_validation as canonical_t1
import campaigns.sers_novelty_gap.sers_targeted_retrieval_t1_live_validation_v2 as canonical_t1_v2

import dac_her.sers_targeted_retrieval_t0_offline_validation as legacy_t0
import dac_her.sers_targeted_retrieval_t1_live_validation as legacy_t1
import dac_her.sers_targeted_retrieval_t1_live_validation_v2 as legacy_t1_v2


ROOT = Path(__file__).resolve().parents[1]


def test_legacy_t0_surface_delegates_to_campaign_namespace() -> None:
    assert (
        legacy_t0.build_t0_offline_report
        is canonical_t0.build_t0_offline_report
    )


def test_legacy_t1_surface_delegates_to_campaign_namespace() -> None:
    assert (
        legacy_t1.aggregate_t1_report
        is canonical_t1.aggregate_t1_report
    )

    assert (
        legacy_t1.audit_live_gap_outcome
        is canonical_t1.audit_live_gap_outcome
    )


def test_legacy_t1_v2_surface_preserves_private_test_contract() -> None:
    assert (
        legacy_t1_v2.aggregate_t1_report
        is canonical_t1_v2.aggregate_t1_report
    )

    assert (
        legacy_t1_v2.audit_live_gap_outcome
        is canonical_t1_v2.audit_live_gap_outcome
    )

    assert (
        legacy_t1_v2._canonical_json
        is canonical_t1_v2._canonical_json
    )


def test_campaign_modules_do_not_depend_on_legacy_targeted_retrieval_modules() -> None:
    paths = (
        ROOT
        / "campaigns"
        / "sers_novelty_gap"
        / "sers_targeted_retrieval_t0_offline_validation.py",

        ROOT
        / "campaigns"
        / "sers_novelty_gap"
        / "sers_targeted_retrieval_t1_live_validation.py",

        ROOT
        / "campaigns"
        / "sers_novelty_gap"
        / "sers_targeted_retrieval_t1_live_validation_v2.py",
    )

    forbidden_prefix = (
        "dac_her.sers_targeted_retrieval_"
    )

    violations = []

    for path in paths:
        tree = ast.parse(
            path.read_text(
                encoding="utf-8"
            ),
            filename=str(path),
        )

        for node in ast.walk(tree):
            if isinstance(
                node,
                ast.ImportFrom,
            ):
                module = (
                    node.module
                    or ""
                )

                if module.startswith(
                    forbidden_prefix
                ):
                    violations.append(
                        (
                            path.name,
                            node.lineno,
                            module,
                        )
                    )

            elif isinstance(
                node,
                ast.Import,
            ):
                for alias in node.names:
                    if alias.name.startswith(
                        forbidden_prefix
                    ):
                        violations.append(
                            (
                                path.name,
                                node.lineno,
                                alias.name,
                            )
                        )

    assert violations == []
