from __future__ import annotations

from pathlib import Path

from pipeline_core.discovery.hypothesis_contracts import HypothesisContext
from pipeline_core.discovery.hypothesis_llm import (
    InstructorOpenAICompatibleHypothesisBackend,
)
from pipeline_core.discovery.pre_n10_primary_router_v1 import (
    PreN10PrimaryRouterReportV1,
)
from pipeline_core.discovery.pre_n10_regeneration_router_adapter_v1 import (
    execute_pre_n10_regeneration_from_router_v1,
)
from pipeline_core.discovery.pre_n10_regeneration_v1 import (
    PreN10RegenerationExecutionReportV1,
)
from pipeline_core.discovery.prospective_regeneration_unit_v2 import (
    ProspectiveRegenerationUnitV2Freeze,
    ProspectiveRegenerationUnitV2Result,
)


def execute_pre_n10_regeneration_router_runtime_v1(
    *,
    context_path: Path,
    primary_router_report_path: Path,
    regeneration_unit_freeze_path: Path,
    output_root: Path,
    model: str,
    api_key_env: str = "OPENAI_API_KEY",
    base_url: str | None = None,
    instructor_mode: str = "JSON",
    temperature: float = 0.0,
    parse_retries: int = 1,
    timeout_seconds: float = 180.0,
) -> tuple[
    PreN10RegenerationExecutionReportV1,
    dict[str, ProspectiveRegenerationUnitV2Result],
]:
    """Execute exactly one fresh regeneration generation unit per router fallback.

    This is a runtime/CLI adapter only.  The frozen regeneration-v2 policy and
    the existing pre-N10 regeneration executor remain authoritative:
    - only router rows with regeneration_fallback_required=True are consumed;
    - exactly one structured generation call is allowed per fallback lineage;
    - no repair, semantic critic, decomposition, retrieval, N9, N10, or second
      regeneration is performed here.
    """

    context_file = context_path.expanduser().resolve()
    primary_file = primary_router_report_path.expanduser().resolve()
    freeze_file = regeneration_unit_freeze_path.expanduser().resolve()
    root = output_root.expanduser().resolve()

    for path, label in (
        (context_file, "hypothesis context"),
        (primary_file, "primary-router report"),
        (freeze_file, "regeneration-unit freeze"),
    ):
        if not path.is_file():
            raise ValueError(
                "missing pre-N10 router-regeneration "
                + label
                + ": "
                + str(path)
            )

    resolved_model = str(model).strip()
    if not resolved_model:
        raise ValueError("pre-N10 router regeneration requires --model")

    context = HypothesisContext.model_validate_json(
        context_file.read_text(encoding="utf-8")
    )
    primary = PreN10PrimaryRouterReportV1.model_validate_json(
        primary_file.read_text(encoding="utf-8")
    )
    freeze = ProspectiveRegenerationUnitV2Freeze.model_validate_json(
        freeze_file.read_text(encoding="utf-8")
    )

    # Fail before constructing any LLM backend when there is no authorized
    # fallback population.
    if primary.regeneration_fallback_required_count < 1:
        raise ValueError(
            "pre-N10 router regeneration requires at least one fallback lineage"
        )

    def backend_factory(
        source_hypothesis_id: str,
        lineage_dir: Path,
    ) -> InstructorOpenAICompatibleHypothesisBackend:
        return InstructorOpenAICompatibleHypothesisBackend(
            model=resolved_model,
            api_key_env=str(api_key_env),
            base_url=base_url,
            instructor_mode=str(instructor_mode),
            temperature=float(temperature),
            parse_retries=int(parse_retries),
            timeout=float(timeout_seconds),
            telemetry_path=lineage_dir / "regeneration.telemetry.jsonl",
            telemetry_context={
                "pipeline": "pre_n10_regeneration_router_runtime_v1",
                "source_hypothesis_id": source_hypothesis_id,
                "source_primary_router_report_id": primary.report_id,
                "regeneration_unit_freeze_id": freeze.freeze_id,
            },
        )

    return execute_pre_n10_regeneration_from_router_v1(
        context=context,
        primary_router_report=primary,
        unit_freeze=freeze,
        backend_factory=backend_factory,
        output_root=root,
    )


__all__ = [
    "execute_pre_n10_regeneration_router_runtime_v1",
]
