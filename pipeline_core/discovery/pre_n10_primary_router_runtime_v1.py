from __future__ import annotations

from pathlib import Path

from pipeline_core.discovery.pre_n10_primary_router_v1 import (
    PreN10PrimaryRouterReportV1,
    execute_pre_n10_primary_router_v1,
)
from pipeline_core.discovery.pre_n10_scientific_contract_v1 import (
    PreN10ScientificContractReportV1,
)
from pipeline_core.discovery.preverifier_specification_repair_executor import (
    InstructorSpecificationRepairBackend,
)
from pipeline_core.discovery.prospective_routed_primary_materializer_v2 import (
    InstructorSourceAlignmentAuditBackend,
)


def execute_pre_n10_primary_router_runtime_v1(
    *,
    portfolio_path: Path,
    query_plan_path: Path,
    contract_report_path: Path,
    output_root: Path,
    model: str,
    specification_repair_model: str | None = None,
    specification_audit_model: str | None = None,
    source_alignment_model: str | None = None,
    api_key_env: str = "OPENAI_API_KEY",
    base_url: str | None = None,
    temperature: float = 0.0,
    parse_retries: int = 1,
    timeout_seconds: float = 180.0,
) -> tuple[object, PreN10ScientificContractReportV1, PreN10PrimaryRouterReportV1]:
    """Execute the unified pre-N10 primary router with lazy concrete backends.

    This adapter changes no routing or scientific policy.  It only gives the
    already-frozen router a concrete runtime surface suitable for a CLI and a
    later campaign dispatcher.

    LLM-backed primary backends are constructed lazily:
    - READY / decomposition / mixed-route fallback never initialize them.
    - specification repair initializes only the bounded repair+audit backend.
    - source alignment initializes only the zero-delta alignment audit backend.
    """

    portfolio_file = portfolio_path.expanduser().resolve()
    query_file = query_plan_path.expanduser().resolve()
    contract_file = contract_report_path.expanduser().resolve()
    root = output_root.expanduser().resolve()

    for path, label in (
        (portfolio_file, "portfolio"),
        (query_file, "query plan"),
        (contract_file, "contract report"),
    ):
        if not path.is_file():
            raise ValueError(f"missing pre-N10 primary-router {label}: {path}")

    default_model = str(model).strip()
    if not default_model:
        raise ValueError("pre-N10 primary router runtime requires --model")

    repair_model = str(
        specification_repair_model or default_model
    ).strip()
    repair_audit_model = str(
        specification_audit_model or default_model
    ).strip()
    alignment_model = str(
        source_alignment_model or default_model
    ).strip()
    if not repair_model or not repair_audit_model or not alignment_model:
        raise ValueError("resolved primary-router model names must be non-empty")

    contract = PreN10ScientificContractReportV1.model_validate_json(
        contract_file.read_text(encoding="utf-8")
    )

    def specification_factory(hypothesis_id: str, lineage_root: Path):
        return InstructorSpecificationRepairBackend(
            repair_model=repair_model,
            audit_model=repair_audit_model,
            api_key_env=str(api_key_env),
            base_url=str(base_url or ""),
            temperature=float(temperature),
            parse_retries=int(parse_retries),
            timeout_seconds=float(timeout_seconds),
            telemetry_path=(
                lineage_root / "specification_repair.telemetry.jsonl"
            ),
            telemetry_context={
                "pipeline": "pre_n10_primary_router_runtime_v1",
                "route": "SPECIFICATION_REPAIR",
                "hypothesis_id": hypothesis_id,
                "source_contract_report_id": contract.report_id,
            },
        )

    def source_alignment_factory(
        hypothesis_id: str,
        lineage_root: Path,
    ):
        resolved_base_url = str(base_url or "").strip()
        if not resolved_base_url:
            raise ValueError(
                "source-alignment primary route requires --base-url"
            )
        return InstructorSourceAlignmentAuditBackend(
            model=alignment_model,
            api_key_env=str(api_key_env),
            base_url=resolved_base_url,
            temperature=float(temperature),
            parse_retries=int(parse_retries),
            timeout_seconds=float(timeout_seconds),
            telemetry_path=(
                lineage_root / "source_alignment.telemetry.jsonl"
            ),
            telemetry_context={
                "pipeline": "pre_n10_primary_router_runtime_v1",
                "route": "SOURCE_ALIGNMENT",
                "hypothesis_id": hypothesis_id,
                "source_contract_report_id": contract.report_id,
            },
        )

    return execute_pre_n10_primary_router_v1(
        portfolio_path=portfolio_file,
        query_plan_path=query_file,
        contract_report=contract,
        output_root=root,
        specification_repair_backend_factory=specification_factory,
        source_alignment_audit_backend_factory=source_alignment_factory,
    )


__all__ = [
    "execute_pre_n10_primary_router_runtime_v1",
]
