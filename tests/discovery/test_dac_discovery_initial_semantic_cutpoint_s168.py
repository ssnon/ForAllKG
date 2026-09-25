from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from scripts.discovery.run_dac_discovery_e2e import (
    _finish_after_initial_semantic_if_requested,
    _sha256_file,
)


class _Runner:
    def __init__(self, root: Path) -> None:
        self.manifest_path = root / "e2e_runner.manifest.json"
        self.manifest = {
            "schema_version": "dac-discovery-e2e-runner-v1",
            "status": "running",
            "stages": [],
        }
        self.save_calls = 0

    def _save_manifest(self) -> None:
        self.save_calls += 1
        self.manifest_path.write_text(
            json.dumps(
                self.manifest,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


def _artifact(path: Path, value: str) -> Path:
    path.write_text(value + "\n", encoding="utf-8")
    return path


def _paths(tmp_path: Path):
    return {
        "portfolio": _artifact(tmp_path / "portfolio.json", "portfolio"),
        "context": _artifact(tmp_path / "context.json", "context"),
        "run": _artifact(tmp_path / "semantic.run.json", "semantic-run"),
        "review": _artifact(
            tmp_path / "semantic.review.json",
            "semantic-review",
        ),
        "provider": _artifact(tmp_path / "provider.json", "provider"),
    }


def test_cutpoint_disabled_is_noop(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    runner = _Runner(tmp_path)
    args = argparse.Namespace(stop_after_initial_semantic=False)

    stopped = _finish_after_initial_semantic_if_requested(
        runner=runner,
        args=args,
        portfolio_path=paths["portfolio"],
        context_path=paths["context"],
        semantic_run_path=paths["run"],
        semantic_review_path=paths["review"],
        provider_plan_path=paths["provider"],
    )

    assert stopped is False
    assert runner.save_calls == 0
    assert "prospective_initial_semantic_cutpoint" not in runner.manifest


def test_cutpoint_freezes_exact_initial_artifacts_without_downstream(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    runner = _Runner(tmp_path)
    args = argparse.Namespace(stop_after_initial_semantic=True)

    stopped = _finish_after_initial_semantic_if_requested(
        runner=runner,
        args=args,
        portfolio_path=paths["portfolio"],
        context_path=paths["context"],
        semantic_run_path=paths["run"],
        semantic_review_path=paths["review"],
        provider_plan_path=paths["provider"],
    )

    assert stopped is True
    assert runner.manifest["status"] == "complete_after_initial_semantic"
    cut = runner.manifest["prospective_initial_semantic_cutpoint"]
    assert cut["external_novelty_performed"] is False
    assert cut["n9_performed"] is False
    assert cut["n10_performed"] is False
    assert cut["refinement_performed"] is False
    assert cut["production_selection_changed"] is False
    assert cut["canonical_graph_mutated"] is False
    for key, path in (
        ("portfolio", paths["portfolio"]),
        ("hypothesis_context", paths["context"]),
        ("semantic_run", paths["run"]),
        ("semantic_review", paths["review"]),
        ("provider_plan", paths["provider"]),
    ):
        row = cut["artifacts"][key]
        assert row["path"] == str(path.resolve())
        assert row["file_sha256"] == _sha256_file(path)
    assert runner.save_calls == 1


def test_cutpoint_allows_missing_review_after_semantic_hard_gate(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    paths["review"].unlink()
    runner = _Runner(tmp_path)
    args = argparse.Namespace(stop_after_initial_semantic=True)

    assert _finish_after_initial_semantic_if_requested(
        runner=runner,
        args=args,
        portfolio_path=paths["portfolio"],
        context_path=paths["context"],
        semantic_run_path=paths["run"],
        semantic_review_path=paths["review"],
        provider_plan_path=paths["provider"],
    )
    cut = runner.manifest["prospective_initial_semantic_cutpoint"]
    assert cut["semantic_review_optional_after_hard_gate_failure"] is True
    assert cut["artifacts"]["semantic_review"] is None


def test_cutpoint_requires_semantic_run_and_pre_novelty_sources(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    paths["run"].unlink()
    runner = _Runner(tmp_path)
    args = argparse.Namespace(stop_after_initial_semantic=True)

    with pytest.raises(RuntimeError, match="semantic_run"):
        _finish_after_initial_semantic_if_requested(
            runner=runner,
            args=args,
            portfolio_path=paths["portfolio"],
            context_path=paths["context"],
            semantic_run_path=paths["run"],
            semantic_review_path=paths["review"],
            provider_plan_path=paths["provider"],
        )

