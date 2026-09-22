from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core.discovery.reframing.authoritative_final_binding import (
    AuthoritativeFinalPortfolioResolution,
    resolve_authoritative_final_portfolio,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ScientificShadowCompanionPlan(StrictModel):
    schema_version: Literal[
        "scientific-e2e-shadow-companion-plan-v1"
    ] = "scientific-e2e-shadow-companion-plan-v1"

    run_dir: str
    e2e_manifest_path: str
    e2e_status: str
    reframing_portfolio_path: str
    reframe_shadow_path: str
    proxy_shadow_path: str | None = None
    contradiction_shadow_path: str | None = None
    authoritative_final: AuthoritativeFinalPortfolioResolution | None = None
    output_dir: str
    integrated_shadow_output: str
    companion_manifest_output: str
    max_syntheses: int = Field(ge=1)

    disposition: Literal[
        "ready",
        "abstained_no_authoritative_hypotheses",
    ]
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False
    legacy_e2e_manifest_mutated: Literal[False] = False


def _load_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _require_file(path: Path, *, label: str) -> Path:
    if not path.is_file():
        raise ValueError(f"{label} does not exist: {path}")
    return path.resolve()


def require_completed_e2e_run(run_dir: str | Path) -> tuple[Path, dict]:
    run = Path(run_dir).expanduser().resolve()
    if not run.is_dir():
        raise ValueError(f"E2E run directory does not exist: {run}")
    manifest = _require_file(
        run / "e2e_runner.manifest.json",
        label="E2E manifest",
    )
    payload = _load_object(manifest)
    status = str(payload.get("status") or "")
    if status not in {
        "complete",
        "complete_no_hypotheses_after_alpha4",
    }:
        raise ValueError(
            "scientific shadow companion requires a completed legacy E2E run; "
            f"manifest status={status!r}"
        )
    return run, payload


def build_scientific_shadow_companion_plan(
    *,
    run_dir: str | Path,
    reframing_portfolio_path: str | Path,
    reframe_shadow_path: str | Path,
    proxy_shadow_path: str | Path | None = None,
    contradiction_shadow_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    max_syntheses: int = 3,
) -> ScientificShadowCompanionPlan:
    if max_syntheses < 1:
        raise ValueError("max_syntheses must be at least 1")

    run, manifest = require_completed_e2e_run(run_dir)
    output = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else run
    )
    reframing = _require_file(
        Path(reframing_portfolio_path).expanduser(),
        label="scientific reframing reasoning portfolio",
    )
    shadow = _require_file(
        Path(reframe_shadow_path).expanduser(),
        label="scientific reframing shadow",
    )
    proxy = (
        _require_file(
            Path(proxy_shadow_path).expanduser(),
            label="proxy challenge shadow",
        )
        if proxy_shadow_path is not None
        else None
    )
    contradiction = (
        _require_file(
            Path(contradiction_shadow_path).expanduser(),
            label="contradiction-resolution shadow",
        )
        if contradiction_shadow_path is not None
        else None
    )

    integrated = output / "scientific_authoritative_integrated_hypothesis_shadow.json"
    companion_manifest = output / "scientific_e2e_shadow_companion_manifest.json"

    if manifest.get("status") == "complete_no_hypotheses_after_alpha4":
        return ScientificShadowCompanionPlan(
            run_dir=str(run),
            e2e_manifest_path=str((run / "e2e_runner.manifest.json").resolve()),
            e2e_status=str(manifest["status"]),
            reframing_portfolio_path=str(reframing),
            reframe_shadow_path=str(shadow),
            proxy_shadow_path=str(proxy) if proxy else None,
            contradiction_shadow_path=str(contradiction) if contradiction else None,
            authoritative_final=None,
            output_dir=str(output),
            integrated_shadow_output=str(integrated),
            companion_manifest_output=str(companion_manifest),
            max_syntheses=max_syntheses,
            disposition="abstained_no_authoritative_hypotheses",
        )

    resolution = resolve_authoritative_final_portfolio(run_dir=run)
    disposition = (
        "ready"
        if resolution.hypothesis_count > 0
        else "abstained_no_authoritative_hypotheses"
    )

    return ScientificShadowCompanionPlan(
        run_dir=str(run),
        e2e_manifest_path=str((run / "e2e_runner.manifest.json").resolve()),
        e2e_status=str(manifest["status"]),
        reframing_portfolio_path=str(reframing),
        reframe_shadow_path=str(shadow),
        proxy_shadow_path=str(proxy) if proxy else None,
        contradiction_shadow_path=str(contradiction) if contradiction else None,
        authoritative_final=resolution,
        output_dir=str(output),
        integrated_shadow_output=str(integrated),
        companion_manifest_output=str(companion_manifest),
        max_syntheses=max_syntheses,
        disposition=disposition,
    )


def build_attach_command(
    plan: ScientificShadowCompanionPlan,
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key_env: str = "OPENAI_API_KEY",
) -> list[str]:
    if plan.disposition != "ready":
        return []

    argv = [
        "-m",
        "scripts.discovery.build_authoritative_integrated_scientific_shadow",
        "--run-dir",
        plan.run_dir,
        "--reframing-portfolio",
        plan.reframing_portfolio_path,
        "--reframe-shadow",
        plan.reframe_shadow_path,
        "--output-dir",
        plan.output_dir,
        "--max-syntheses",
        str(plan.max_syntheses),
    ]
    if plan.proxy_shadow_path is not None:
        argv += ["--proxy-shadow", plan.proxy_shadow_path]
    if plan.contradiction_shadow_path is not None:
        argv += ["--contradiction-shadow", plan.contradiction_shadow_path]
    if model:
        argv += ["--model", model]
    if base_url:
        argv += ["--base-url", base_url]
    if api_key_env:
        argv += ["--api-key-env", api_key_env]
    return argv


__all__ = [
    "ScientificShadowCompanionPlan",
    "build_attach_command",
    "build_scientific_shadow_companion_plan",
    "require_completed_e2e_run",
]
