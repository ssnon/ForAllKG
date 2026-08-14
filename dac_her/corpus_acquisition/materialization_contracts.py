from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


MaterializationStatus = Literal[
    "materialized",
    "unsupported",
    "failed",
    "skipped",
]


class MaterializationPolicy(StrictModel):
    schema_version: Literal[
        "document-materialization-policy-v1"
    ] = "document-materialization-policy-v1"
    policy_id: str

    pdf_materializer: Literal["marker_cli"] = "marker_cli"
    marker_command: str = "marker_single"
    marker_extra_args: list[str] = Field(default_factory=list)
    marker_timeout_seconds: int = Field(default=1200, ge=30)
    marker_environment_overrides: dict[str, str] = Field(
        default_factory=lambda: {
            "MKL_THREADING_LAYER": "GNU",
        }
    )
    marker_environment_unset: list[str] = Field(
        default_factory=lambda: [
            "MKL_SERVICE_FORCE_INTEL",
        ]
    )

    materialize_pdf: bool = True
    materialize_txt: bool = True
    materialize_csv: bool = True
    materialize_xlsx: bool = True
    materialize_docx: bool = True
    materialize_pptx: bool = True

    # Old binary Office files and arbitrary ZIP bundles are retained as source
    # artifacts but are not silently transformed in M4 v1.
    unsupported_extensions: list[str] = Field(
        default_factory=lambda: [".zip", ".xls", ".doc", ".bin"]
    )

    xlsx_max_sheets: int = Field(default=20, ge=1, le=100)
    xlsx_max_rows_per_sheet: int = Field(default=5000, ge=1)
    xlsx_max_columns: int = Field(default=100, ge=1)
    csv_max_rows: int = Field(default=10000, ge=1)
    text_max_bytes: int = Field(default=52428800, ge=1024)

    main_selection_mode: Literal["whole_document"] = "whole_document"
    si_selection_mode: Literal["referenced_blocks"] = "referenced_blocks"
    si_fallback: Literal["skip"] = "skip"
    si_reference_scope: Literal["whole_main"] = "whole_main"
    figure_processing_mode: Literal[
        "none", "caption_first", "always_vision"
    ] = "caption_first"

    @model_validator(mode="after")
    def _command_nonempty(self) -> "MaterializationPolicy":
        if not self.marker_command.strip():
            raise ValueError("marker_command must be non-empty")
        return self


class MaterializedDocument(StrictModel):
    schema_version: Literal[
        "materialized-document-v1"
    ] = "materialized-document-v1"
    materialization_id: str
    paper_id: str
    work_id: str
    document_id: str
    role: Literal["main", "supporting_information"]
    source_artifact_id: str
    source_artifact_sha256: str | None = None
    source_path: str
    source_extension: str
    status: MaterializationStatus
    materializer: str
    package_dir: str | None = None
    markdown_path: str | None = None
    metadata_path: str | None = None
    markdown_sha256: str | None = None
    markdown_char_count: int | None = None
    copied_asset_count: int = 0
    error: str | None = None
    source_scientific_content_modified: Literal[False] = False
    scientific_result_inferred: Literal[False] = False
    positive_evidence_promotion_performed: Literal[False] = False


class PaperMaterializationRecord(StrictModel):
    paper_id: str
    work_id: str
    title: str
    doi: str | None = None
    main_document_status: MaterializationStatus
    supplementary_document_count: int = 0
    supplementary_materialized_count: int = 0
    extraction_ready: bool = False
    document_ids: list[str] = Field(default_factory=list)


class CorpusMaterializationReport(StrictModel):
    schema_version: Literal[
        "corpus-materialization-report-v1"
    ] = "corpus-materialization-report-v1"
    materialization_id: str
    source_profile_id: str
    source_catalog_id: str
    source_m3_report_path: str
    source_m3_1_report_path: str | None = None
    policy_id: str
    selected_work_count: int
    main_downloaded_source_count: int = 0
    main_materialized_count: int = 0
    main_materialization_failed_count: int = 0
    supplementary_downloaded_source_count: int = 0
    supplementary_materialized_count: int = 0
    supplementary_materialization_failed_count: int = 0
    unsupported_source_count: int = 0
    extraction_ready_paper_count: int = 0
    not_extraction_ready_paper_count: int = 0
    generated_config_path: str
    extraction_plan_path: str
    output_root: str
    llm_calls_performed: Literal[False] = False
    scientific_result_inference_performed: Literal[False] = False
    positive_evidence_promotion_performed: Literal[False] = False
