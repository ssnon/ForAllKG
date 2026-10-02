
from __future__ import annotations

import json
from pathlib import Path

from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisContext,
    HypothesisPortfolio,
)


def resolve_grounded_hypothesis_context(
    *,
    source_path: Path,
    portfolio_path: Path,
) -> tuple[HypothesisContext, str]:
    payload = json.loads(
        source_path.read_text(encoding="utf-8")
    )
    if not isinstance(payload, dict):
        raise ValueError(
            f"context source must be a JSON object: {source_path}"
        )

    schema = str(payload.get("schema_version") or "")
    if schema == "hypothesis-context-v1":
        context_payload = payload
        resolution_mode = "DIRECT_HYPOTHESIS_CONTEXT"
    elif isinstance(payload.get("grounded_context"), dict):
        context_payload = payload["grounded_context"]
        resolution_mode = "DUAL_CONTEXT_GROUNDED_CONTEXT"
    else:
        raise ValueError(
            "context source is neither HypothesisContext nor a wrapper "
            "containing grounded_context"
        )

    context = HypothesisContext.model_validate(
        context_payload
    )
    portfolio = HypothesisPortfolio.model_validate_json(
        portfolio_path.read_text(encoding="utf-8")
    )

    if portfolio.source_context_id != context.context_id:
        raise ValueError(
            "effective portfolio/context ID mismatch"
        )
    if portfolio.source_context_sha256 != context.context_sha256:
        raise ValueError(
            "effective portfolio/context SHA mismatch"
        )
    if portfolio.domain_profile_id != context.domain_profile_id:
        raise ValueError(
            "effective portfolio/context domain mismatch"
        )
    if portfolio.source_report_id != context.source_report_id:
        raise ValueError(
            "effective portfolio/context source report ID mismatch"
        )
    if portfolio.source_report_sha256 != context.source_report_sha256:
        raise ValueError(
            "effective portfolio/context source report SHA mismatch"
        )

    return context, resolution_mode


def write_grounded_hypothesis_context(
    *,
    source_path: Path,
    portfolio_path: Path,
    output_path: Path,
) -> dict[str, str]:
    context, mode = resolve_grounded_hypothesis_context(
        source_path=source_path,
        portfolio_path=portfolio_path,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        context.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    validated = HypothesisContext.model_validate_json(
        output_path.read_text(encoding="utf-8")
    )
    if validated.context_id != context.context_id:
        raise ValueError(
            "emitted HypothesisContext changed context_id"
        )
    if validated.context_sha256 != context.context_sha256:
        raise ValueError(
            "emitted HypothesisContext changed context_sha256"
        )

    return {
        "resolution_mode": mode,
        "context_id": context.context_id,
        "context_sha256": context.context_sha256,
        "domain_profile_id": context.domain_profile_id,
        "output_path": str(output_path.resolve()),
    }
