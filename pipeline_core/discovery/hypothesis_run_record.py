from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class HypothesisMakerRunRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["hypothesis-maker-run-v1"] = "hypothesis-maker-run-v1"
    run_id: str
    context_id: str
    context_sha256: str
    source_packet_id: str
    source_packet_sha256: str
    source_report_id: str
    source_report_sha256: str
    portfolio_id: str | None = None
    portfolio_sha256: str | None = None
    prompt_version: str
    prompt_sha256: str
    backend: str
    model: str
    generation_attempts: int
    repair_attempts: int
    final_validation_passed: bool
    failure_stage: Literal["none", "compile", "validation", "generation"] = "none"
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
