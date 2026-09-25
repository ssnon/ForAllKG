from __future__ import annotations

from pathlib import Path
from typing import Callable

from pipeline_core.discovery.hypothesis_contracts import HypothesisContext
from pipeline_core.discovery.hypothesis_llm import HypothesisDraftBackend
from pipeline_core.discovery.pre_n10_primary_router_v1 import (
    PreN10PrimaryRouterReportV1,
)
from pipeline_core.discovery.pre_n10_regeneration_v1 import (
    PreN10RegenerationExecutionReportV1,
    execute_pre_n10_regeneration_v1,
)
from pipeline_core.discovery.prospective_regeneration_unit_v2 import (
    ProspectiveRegenerationUnitV2Freeze,
    ProspectiveRegenerationUnitV2Result,
)


BackendFactory = Callable[[str, Path], HypothesisDraftBackend]


def execute_pre_n10_regeneration_from_router_v1(
    *,
    context: HypothesisContext,
    primary_router_report: PreN10PrimaryRouterReportV1,
    unit_freeze: ProspectiveRegenerationUnitV2Freeze,
    backend_factory: BackendFactory,
    output_root: Path,
) -> tuple[
    PreN10RegenerationExecutionReportV1,
    dict[str, ProspectiveRegenerationUnitV2Result],
]:
    """Run the existing one-shot regeneration unit only for router fallbacks.

    The underlying v1 regeneration executor is structurally keyed on
    report_id/report_sha256 plus hypothesis_id/regeneration_fallback_required.
    This adapter makes that compatibility explicit without changing the frozen
    regeneration artifact schema consumed by semantic re-entry v2.
    """

    if primary_router_report.regeneration_fallback_required_count < 1:
        raise ValueError(
            "pre-N10 router regeneration requires at least one fallback lineage"
        )

    return execute_pre_n10_regeneration_v1(
        context=context,
        primary_report=primary_router_report,  # type: ignore[arg-type]
        unit_freeze=unit_freeze,
        backend_factory=backend_factory,
        output_root=output_root,
    )


__all__ = ["execute_pre_n10_regeneration_from_router_v1"]
