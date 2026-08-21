from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core.corpus.extraction.asset_index import AssetRecord
from pipeline_core.corpus.extraction.document_config import FigureProcessingConfig
from pipeline_core.llm.openrouter_llm import OpenRouterLLM


class FigureValue(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str
    value: str
    unit: str | None
    context: str | None


class FigureAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")
    asset_id: str
    summary: str
    observed_elements: list[str]
    quantitative_values: list[FigureValue]
    scientific_relevance: str
    limitations: str
    confidence: Literal["high", "medium", "low"]


FIGURE_SYSTEM_PROMPT_TEMPLATE = """
You analyze a scientific figure for a provenance-preserving literature
knowledge graph in the following research domain:

{domain_context}

Rules:
1. Use only information visible in the supplied image and the supplied
   official caption/context.
2. Do not treat Marker-generated alt text as authoritative.
3. Do not infer precise numerical values from plotted curves unless the
   value is explicitly printed or unambiguous.
4. Distinguish observations from mechanistic interpretation.
5. State limitations when labels, axes, resolution, or panels are unclear.
6. Keep the output concise and source-grounded.
""".strip()


def build_figure_system_prompt(domain_context: str) -> str:
    normalized = " ".join(str(domain_context).split())
    if not normalized:
        raise ValueError("domain_context must not be empty.")
    return FIGURE_SYSTEM_PROMPT_TEMPLATE.format(
        domain_context=normalized,
    )


def _matches_requested(asset: AssetRecord, requested: tuple[str, ...]) -> bool:
    values = {
        asset.asset_id,
        asset.relative_path,
        Path(asset.relative_path).name,
    }
    return any(value in values for value in requested)


def should_analyze_asset(
    asset: AssetRecord,
    config: FigureProcessingConfig,
    *,
    force_all: bool = False,
    requested_assets: tuple[str, ...] = (),
) -> bool:
    if not asset.exists:
        return False
    if force_all:
        return True
    if requested_assets and _matches_requested(asset, requested_assets):
        return True
    if config.mode == "always_vision":
        return True
    if config.mode == "caption_first":
        return _matches_requested(asset, config.vision_assets)
    return False


def analysis_path(output_dir: str | Path, asset_id: str) -> Path:
    safe_id = asset_id.replace(":", "__")
    return Path(output_dir) / f"{safe_id}.json"


def load_figure_analysis(
    output_dir: str | Path,
    asset_id: str,
) -> FigureAnalysis | None:
    path = analysis_path(output_dir, asset_id)
    if not path.exists():
        return None
    return FigureAnalysis.model_validate_json(path.read_text(encoding="utf-8"))


def analyze_figure(
    *,
    asset: AssetRecord,
    model: str,
    provider: str | None,
    output_dir: str | Path,
    domain_context: str,
    force: bool = False,
) -> FigureAnalysis:
    path = analysis_path(output_dir, asset.asset_id)
    if path.exists() and not force:
        return FigureAnalysis.model_validate_json(path.read_text(encoding="utf-8"))

    llm = OpenRouterLLM(
        model=model,
        provider=provider,
        reproducible=False,
        zdr=True,
        application_title="GraphAgents Figure Extraction",
        default_debug_path=(
            Path(output_dir)
            / "last_invalid_structured_response.json"
        ),
    )
    prompt = f"""
ASSET_ID:
{asset.asset_id}

DOCUMENT_ROLE:
{asset.document_role}

SECTION:
{asset.section or 'unknown'}

OFFICIAL_CAPTION:
{asset.caption or 'not available'}

MARKER_ALT_TEXT (potentially noisy; never sole evidence):
{asset.marker_alt_text or 'not available'}
""".strip()

    result = llm.generate_structured_with_images(
        system_prompt=build_figure_system_prompt(
            domain_context
        ),
        prompt=prompt,
        image_paths=[asset.absolute_path],
        response_model=FigureAnalysis,
        temperature=0.0,
        max_tokens=2500,
        debug_path=path.with_suffix(".invalid.json"),
    )
    if result.asset_id != asset.asset_id:
        raise ValueError(
            f"Vision model returned incorrect asset_id: {result.asset_id!r}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def resolve_vision_model(
    config: FigureProcessingConfig,
    fallback_model: str | None,
) -> str:
    model = (
        config.vision_model
        or os.getenv("OPENROUTER_VISION_MODEL")
        or fallback_model
    )
    if not model:
        raise RuntimeError(
            "No vision model configured. Set OPENROUTER_VISION_MODEL, "
            "figure_processing.vision_model, or pass a fallback model."
        )
    return model


def format_asset_context(
    assets: list[AssetRecord],
    analyses: dict[str, FigureAnalysis],
) -> str:
    if not assets:
        return "No linked figure or table assets."

    blocks: list[str] = []
    for asset in assets:
        lines = [
            f"ASSET_ID: {asset.asset_id}",
            f"TYPE: {asset.asset_type}",
            f"PATH: {asset.relative_path}",
            f"PAGE_ID: {asset.page_id if asset.page_id is not None else 'unknown'}",
            f"CAPTION: {asset.caption or 'not available'}",
            (
                "MARKER_ALT_TEXT (noisy, not sufficient by itself): "
                f"{asset.marker_alt_text or 'not available'}"
            ),
        ]
        analysis = analyses.get(asset.asset_id)
        if analysis is not None:
            lines.extend([
                f"VISION_SUMMARY: {analysis.summary}",
                f"VISION_RELEVANCE: {analysis.scientific_relevance}",
                f"VISION_LIMITATIONS: {analysis.limitations}",
                "VISION_VALUES: " + json.dumps(
                    [value.model_dump() for value in analysis.quantitative_values],
                    ensure_ascii=False,
                ),
            ])
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
