from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from pipeline_core.document_config import load_paper_configs
from pipeline_core.literature.acquisition.contracts import SelectedCorpusWork
from pipeline_core.literature.acquisition.materialization_contracts import CorpusMaterializationReport, MaterializedDocument, PaperMaterializationRecord
from pipeline_core.literature.acquisition.materialization_package import (
    write_extraction_plan,
    write_generated_config,
)
from scripts.materialization_plan_runtime import EXTRACT_PAPER_COMMAND_PREFIX
from pipeline_core.literature.acquisition.materialization_state import atomic_write_json, write_jsonl
from pipeline_core.literature.acquisition.pre_extraction_gate import assess_pre_extraction_gate, build_pre_extraction_gate_report, load_pre_extraction_gate_policy
from pipeline_core.literature.acquisition.profile import load_acquisition_profile
from pipeline_core.literature.catalog_contracts import (
    CatalogWork,
    LiteratureCatalogPacket,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read_jsonl(path: Path, model) -> list[Any]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(model.model_validate_json(line))
    return rows


def _resolve_project_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _load_raw_papers_config(path: Path) -> dict[str, dict[str, Any]]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict) or loaded.get("version") != 3:
        raise ValueError("M4 input config must be a version 3 papers config")
    papers = loaded.get("papers")
    if not isinstance(papers, dict):
        raise ValueError("M4 input config is missing papers mapping")
    return {
        str(paper_id): entry
        for paper_id, entry in papers.items()
        if isinstance(entry, dict)
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generic Corpus Acquisition M4.5: verify downloaded-paper "
            "bibliographic identity and deterministic full-text bridge "
            "suitability before handing papers to strict LLM extraction."
        )
    )
    parser.add_argument("--acquisition-profile", required=True, type=Path)
    parser.add_argument("--gate-policy", required=True, type=Path)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--selected-works", required=True, type=Path)
    parser.add_argument("--m4-dir", required=True, type=Path)
    parser.add_argument("--input-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output-config", required=True, type=Path)
    parser.add_argument("--domain-profile-id", required=True)
    parser.add_argument("--data-root", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    acquisition_profile = load_acquisition_profile(args.acquisition_profile)
    gate_policy = load_pre_extraction_gate_policy(args.gate_policy)
    packet = LiteratureCatalogPacket.model_validate_json(
        args.catalog.read_text(encoding="utf-8")
    )
    selected = _read_jsonl(args.selected_works, SelectedCorpusWork)

    m4_report_path = args.m4_dir / "materialization_report.json"
    if not m4_report_path.exists():
        raise FileNotFoundError(f"Completed M4 report required: {m4_report_path}")
    m4_report = CorpusMaterializationReport.model_validate_json(
        m4_report_path.read_text(encoding="utf-8")
    )
    paper_records = _read_jsonl(
        args.m4_dir / "paper_materialization_records.jsonl",
        PaperMaterializationRecord,
    )
    materialized_documents = _read_jsonl(
        args.m4_dir / "materialized_documents.jsonl",
        MaterializedDocument,
    )
    raw_config = _load_raw_papers_config(args.input_config)

    if packet.acquisition_profile_id != acquisition_profile.profile_id:
        raise ValueError("Catalog/acquisition-profile mismatch")
    if acquisition_profile.domain_profile_id != args.domain_profile_id:
        raise ValueError("Acquisition/domain-profile mismatch")
    if m4_report.source_profile_id != acquisition_profile.profile_id:
        raise ValueError("M4/acquisition-profile mismatch")
    if m4_report.source_catalog_id != packet.catalog_id:
        raise ValueError("M4/catalog mismatch")

    work_map: dict[str, CatalogWork] = {
        row.work_id: row for row in packet.works
    }
    selected_map = {row.work_id: row for row in selected}
    if len(selected_map) != len(selected):
        raise ValueError("Duplicate selected work_id")

    main_rows = [
        row for row in materialized_documents
        if row.role == "main"
    ]
    main_document_by_paper = {row.paper_id: row for row in main_rows}
    if len(main_document_by_paper) != len(main_rows):
        raise ValueError("Duplicate materialized main document for paper_id")
    if len({row.paper_id for row in paper_records}) != len(paper_records):
        raise ValueError("Duplicate paper_id in M4 materialization records")
    ready_records = [row for row in paper_records if row.extraction_ready]

    assessments = []
    filtered_papers: dict[str, dict[str, Any]] = {}
    gate_rows = []

    for record in ready_records:
        work = work_map.get(record.work_id)
        selected_work = selected_map.get(record.work_id)
        if work is None or selected_work is None:
            raise ValueError(
                f"M4 paper references missing catalog/selection work: {record.work_id}"
            )
        if record.paper_id not in raw_config:
            raise ValueError(
                f"M4-ready paper missing from input config: {record.paper_id}"
            )

        main_document = main_document_by_paper.get(record.paper_id)
        main_markdown = ""
        markdown_path = None
        if (
            main_document is not None
            and main_document.status == "materialized"
            and main_document.markdown_path
        ):
            markdown_path = _resolve_project_path(main_document.markdown_path)
            if markdown_path.exists():
                main_markdown = markdown_path.read_text(
                    encoding="utf-8",
                    errors="replace",
                )

        assessment = assess_pre_extraction_gate(
            paper_id=record.paper_id,
            work=work,
            selected_work=selected_work,
            acquisition_profile=acquisition_profile,
            main_markdown=main_markdown,
            policy=gate_policy,
        )
        assessments.append(assessment)
        if assessment.auto_extraction_allowed:
            filtered_papers[record.paper_id] = raw_config[record.paper_id]

        gate_rows.append(
            {
                "paper_id": record.paper_id,
                "work_id": record.work_id,
                "markdown_path": str(markdown_path) if markdown_path else None,
                "identity_status": assessment.identity.status,
                "identity_method": assessment.identity.method,
                "suitability_status": assessment.suitability.status,
                "suitable_axes": assessment.suitability.suitable_axes,
                "auto_extraction_allowed": assessment.auto_extraction_allowed,
            }
        )

    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    write_generated_config(
        config_path=args.output_config,
        papers=filtered_papers,
    )

    parsed = load_paper_configs(
        args.output_config,
        project_root=PROJECT_ROOT,
    )
    if set(parsed) != set(filtered_papers):
        raise RuntimeError("Filtered config parser roundtrip mismatch")

    extraction_plan_path = output_root / "extraction_plan.jsonl"
    write_extraction_plan(
        path=extraction_plan_path,
        paper_ids=sorted(filtered_papers),
        generated_config_path=args.output_config,
        domain_profile_id=args.domain_profile_id,
        data_root=args.data_root,
        extract_command_prefix=EXTRACT_PAPER_COMMAND_PREFIX,
        project_root=PROJECT_ROOT,
    )

    write_jsonl(
        output_root / "pre_extraction_gate_assessments.jsonl",
        assessments,
    )
    write_jsonl(
        output_root / "pre_extraction_gate_summary.jsonl",
        gate_rows,
    )

    report = build_pre_extraction_gate_report(
        assessments=assessments,
        policy=gate_policy,
        acquisition_profile=acquisition_profile,
    ).model_copy(
        update={
            "source_catalog_id": packet.catalog_id,
            "source_m4_materialization_id": m4_report.materialization_id,
            "source_m4_ready_paper_count": len(ready_records),
            "source_m4_dir": str(args.m4_dir),
            "input_config_path": str(args.input_config),
            "output_config_path": str(args.output_config),
            "extraction_plan_path": str(extraction_plan_path),
        }
    )
    atomic_write_json(
        output_root / "pre_extraction_gate_report.json",
        report,
    )

    print("Generic corpus acquisition M4.5 complete")
    print("M4 extraction-ready papers:", len(ready_records))
    print("Gate evaluated papers:", len(assessments))
    print("Strict extraction-ready papers:", report.auto_extraction_ready_count)
    print("Blocked papers:", report.blocked_paper_count)
    print("Identity status counts:", report.identity_status_counts)
    print("Suitability status counts:", report.suitability_status_counts)
    print("Filtered config:", args.output_config)
    print("Strict extraction plan:", extraction_plan_path)
    print("LLM calls performed:", report.llm_calls_performed)
    print(
        "Positive-evidence promotion:",
        report.positive_evidence_promotion_performed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
