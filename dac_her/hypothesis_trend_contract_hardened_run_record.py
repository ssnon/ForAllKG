from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


HYPOTHESIS_TREND_HARDENED_RUNTIME_SEMANTICS_ID = (
    "hypothesis_trend_contract_hardened_runtime_v1_alpha4c5i"
)


class ContractHardenedTrendHypothesisMakerRunRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[
        "contract-hardened-trend-hypothesis-maker-run-v1"
    ] = "contract-hardened-trend-hypothesis-maker-run-v1"

    runtime_semantics_id: str
    run_id: str

    context_id: str
    context_sha256: str
    source_report_id: str
    source_report_sha256: str
    source_trend_input_id: str
    source_trend_input_sha256: str

    source_directional_exposure_id: str
    source_directional_exposure_sha256: str
    hardened_exposure_id: str
    hardened_exposure_sha256: str

    portfolio_id: str | None = None
    portfolio_sha256: str | None = None

    hardened_compiler_semantics_id: str
    hardened_validator_semantics_id: str
    renderer_semantics_id: str
    downstream_directional_compiler_semantics_id: str
    prompt_version: str
    prompt_sha256: str

    backend: str
    model: str
    generation_attempts: int
    repair_attempts: int
    final_validation_passed: bool
    failure_stage: Literal[
        "none",
        "compile",
        "validation",
        "generation",
    ] = "none"

    validation_errors: int = 0
    validation_warnings: int = 0
    compile_issue_count: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    elapsed_seconds: float | None = None
    temperature: float | None = None
    backend_mode: str | None = None
    base_url: str | None = None
    parse_retries: int | None = None
    max_repairs: int = 1
