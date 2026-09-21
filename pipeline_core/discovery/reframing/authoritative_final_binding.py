from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core.discovery.hypothesis_contracts import HypothesisContext, HypothesisPortfolio


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


AuthorityKind = Literal[
    "explicit",
    "manifest_bounded_n10",
    "manifest_post_n10",
    "manifest_alpha6_refined",
    "filesystem_bounded_n10",
    "filesystem_n10_candidate",
    "filesystem_n10_hard_filter",
    "filesystem_alpha6_refined",
]


class AuthoritativeFinalPortfolioResolution(StrictModel):
    schema_version: Literal[
        "authoritative-final-hypothesis-portfolio-resolution-v1"
    ] = "authoritative-final-hypothesis-portfolio-resolution-v1"

    run_dir: str = Field(min_length=1)
    portfolio_path: str = Field(min_length=1)
    portfolio_sha256: str = Field(min_length=64, max_length=64)
    authority_kind: AuthorityKind
    source_manifest_path: str | None = None
    source_manifest_status: str | None = None
    hypothesis_count: int = Field(ge=0)

    pre_refinement_fallback_allowed: Literal[False] = False
    alpha4_portfolio_used_as_authoritative_final: Literal[False] = False
    production_selection_changed: Literal[False] = False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _portfolio(path: Path) -> HypothesisPortfolio:
    return HypothesisPortfolio.model_validate_json(path.read_text(encoding="utf-8"))


def _count(path: Path) -> int:
    return len(_portfolio(path).hypotheses)


def _require_file(path: Path, *, label: str) -> Path:
    if not path.is_file():
        raise ValueError(f"{label} does not exist: {path}")
    return path.resolve()


def _manifest_selected_path(run_dir: Path, manifest: dict) -> tuple[Path, AuthorityKind] | None:
    bounded = manifest.get("n10_bounded_continuation")
    if isinstance(bounded, dict) and bounded.get("enabled") is True:
        value = bounded.get("final_portfolio")
        if value:
            return Path(str(value)), "manifest_bounded_n10"

    post = manifest.get("post_generation_n10_authority")
    if isinstance(post, dict):
        value = post.get("downstream_portfolio")
        if value:
            return Path(str(value)), "manifest_post_n10"

    alpha6 = run_dir / "novelty_refinement_a6.portfolio.json"
    if alpha6.is_file():
        return alpha6, "manifest_alpha6_refined"
    return None


def resolve_authoritative_final_portfolio(
    *,
    run_dir: str | Path,
    explicit_portfolio: str | Path | None = None,
) -> AuthoritativeFinalPortfolioResolution:
    run = Path(run_dir).expanduser().resolve()
    if not run.is_dir():
        raise ValueError(f"run directory does not exist: {run}")

    manifest_path = run / "e2e_runner.manifest.json"
    manifest: dict | None = None
    selected: Path
    authority_kind: AuthorityKind

    if explicit_portfolio is not None:
        selected = _require_file(
            Path(explicit_portfolio).expanduser(),
            label="explicit authoritative portfolio",
        )
        authority_kind = "explicit"
    else:
        if manifest_path.is_file():
            manifest = _json(manifest_path)
            resolved = _manifest_selected_path(run, manifest)
            if resolved is not None:
                selected, authority_kind = resolved
                selected = _require_file(
                    selected,
                    label=f"manifest-selected {authority_kind} portfolio",
                )
            else:
                selected = Path()
                authority_kind = "filesystem_alpha6_refined"
        else:
            selected = Path()
            authority_kind = "filesystem_alpha6_refined"

        if not selected.is_file():
            filesystem_candidates: list[tuple[str, AuthorityKind]] = [
                (
                    "novelty_refinement_a6.n10.bounded.portfolio.json",
                    "filesystem_bounded_n10",
                ),
                (
                    "novelty_refinement_a6.n10.candidate.portfolio.json",
                    "filesystem_n10_candidate",
                ),
                (
                    "novelty_refinement_a6.n10.portfolio.json",
                    "filesystem_n10_hard_filter",
                ),
                (
                    "novelty_refinement_a6.portfolio.json",
                    "filesystem_alpha6_refined",
                ),
            ]
            chosen = next(
                (
                    (run / filename, kind)
                    for filename, kind in filesystem_candidates
                    if (run / filename).is_file()
                ),
                None,
            )
            if chosen is None:
                alpha4 = run / "hypothesis_axis_a4.portfolio.json"
                if alpha4.is_file():
                    raise ValueError(
                        "only pre-refinement hypothesis_axis_a4.portfolio.json is present; "
                        "it is not an authoritative final E2E portfolio"
                    )
                raise ValueError(
                    "no authoritative final hypothesis portfolio found; expected bounded/post-N10 "
                    "or novelty_refinement_a6.portfolio.json"
                )
            selected, authority_kind = chosen
            selected = selected.resolve()

    return AuthoritativeFinalPortfolioResolution(
        run_dir=str(run),
        portfolio_path=str(selected),
        portfolio_sha256=_sha256(selected),
        authority_kind=authority_kind,
        source_manifest_path=str(manifest_path.resolve()) if manifest_path.is_file() else None,
        source_manifest_status=(
            str(manifest.get("status"))
            if isinstance(manifest, dict) and manifest.get("status") is not None
            else None
        ),
        hypothesis_count=_count(selected),
    )


def validate_authoritative_final_context(
    *,
    resolution: AuthoritativeFinalPortfolioResolution,
    context: HypothesisContext,
) -> HypothesisPortfolio:
    portfolio_path = Path(resolution.portfolio_path)
    portfolio = _portfolio(portfolio_path)
    if portfolio.source_context_id != context.context_id:
        raise ValueError(
            "authoritative final portfolio/context mismatch: "
            f"{portfolio.source_context_id} != {context.context_id}"
        )
    if resolution.hypothesis_count != len(portfolio.hypotheses):
        raise ValueError("authoritative resolution hypothesis_count drifted")
    return portfolio


def build_authoritative_shadow_commands(
    *,
    context_path: Path,
    authoritative_portfolio_path: Path,
    reframing_portfolio_path: Path,
    reframe_shadow_path: Path,
    output_dir: Path,
    proxy_shadow_path: Path | None = None,
    contradiction_shadow_path: Path | None = None,
    max_syntheses: int = 3,
    model: str | None = None,
    base_url: str | None = None,
    api_key_env: str = "OPENAI_API_KEY",
) -> list[tuple[str, list[str], Path]]:
    if max_syntheses < 1:
        raise ValueError("max_syntheses must be at least 1")

    cross_lane = output_dir / "scientific_authoritative_cross_lane_reasoning_portfolio.json"
    candidate = output_dir / "scientific_authoritative_candidate_portfolio.json"
    synthesis = output_dir / "scientific_authoritative_cross_lane_synthesis_shadow.json"
    prompt = output_dir / "scientific_authoritative_cross_lane_synthesis_prompt.txt"
    integrated = output_dir / "scientific_authoritative_integrated_hypothesis_shadow.json"

    optional_operator_args: list[str] = []
    if proxy_shadow_path is not None:
        optional_operator_args += ["--proxy-shadow", str(proxy_shadow_path)]
    if contradiction_shadow_path is not None:
        optional_operator_args += ["--contradiction-shadow", str(contradiction_shadow_path)]

    synthesis_model_args: list[str] = []
    if model:
        synthesis_model_args += ["--model", model]
    if base_url:
        synthesis_model_args += ["--base-url", base_url]
    if api_key_env:
        synthesis_model_args += ["--api-key-env", api_key_env]

    return [
        (
            "authoritative_cross_lane",
            [
                "-m",
                "scripts.discovery.build_cross_lane_scientific_reasoning_portfolio",
                "--context",
                str(context_path),
                "--relational-portfolio",
                str(authoritative_portfolio_path),
                "--reframing-portfolio",
                str(reframing_portfolio_path),
                "--reframe-shadow",
                str(reframe_shadow_path),
                *optional_operator_args,
                "--output",
                str(cross_lane),
            ],
            cross_lane,
        ),
        (
            "authoritative_candidate_contract",
            [
                "-m",
                "scripts.discovery.build_production_facing_scientific_candidate_portfolio",
                "--cross-lane-portfolio",
                str(cross_lane),
                "--relational-portfolio",
                str(authoritative_portfolio_path),
                "--reframe-shadow",
                str(reframe_shadow_path),
                *optional_operator_args,
                "--output",
                str(candidate),
            ],
            candidate,
        ),
        (
            "authoritative_cross_lane_synthesis",
            [
                "-m",
                "scripts.discovery.run_cross_lane_scientific_synthesis_shadow",
                "--portfolio",
                str(candidate),
                "--max-syntheses",
                str(max_syntheses),
                "--prompt-output",
                str(prompt),
                *synthesis_model_args,
                "--output",
                str(synthesis),
            ],
            synthesis,
        ),
        (
            "authoritative_integrated_shadow",
            [
                "-m",
                "scripts.discovery.build_integrated_scientific_hypothesis_shadow",
                "--candidate-portfolio",
                str(candidate),
                "--synthesis-report",
                str(synthesis),
                "--output",
                str(integrated),
            ],
            integrated,
        ),
    ]


__all__ = [
    "AuthoritativeFinalPortfolioResolution",
    "build_authoritative_shadow_commands",
    "resolve_authoritative_final_portfolio",
    "validate_authoritative_final_context",
]
