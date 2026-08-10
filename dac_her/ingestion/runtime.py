from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .article_list import parse_article_rows
from .contracts import IngestionIssue, PaperRegistryEntry
from .corpus_manifest import build_corpus_manifest
from .discovery import match_articles_to_drive
from .drive_google import GoogleWorkspaceReader
from .marker_runner import MarkerSingleRunner
from .qc import markdown_qc, qc_status
from .registry import PaperRegistry


@dataclass
class SyncConfig:
    credentials_path: Path
    drive_folder_id: str
    spreadsheet_id: str
    sheet_range: str = "Sheet1!A:H"
    data_root: Path = Path("data_dac/ingestion")
    registry_path: Path = Path("data_dac/ingestion/registry/papers.json")
    alias_map_path: Path | None = None
    corpus_id: str = "dac_drive_latest"
    dry_run: bool = False
    convert: bool = True
    force_reconvert: bool = False
    marker_command: str = "marker_single"
    marker_extra_args: tuple[str, ...] = ()
    show_progress: bool = True
    heartbeat_seconds: float = 30.0




class _ProgressReporter:
    def __init__(self, enabled: bool, data_root: Path, run_dir: Path):
        self.enabled = enabled
        self.data_root = data_root
        self.run_dir = run_dir
        self.started = time.monotonic()

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.started

    def log(self, message: str) -> None:
        if self.enabled:
            print(f"[ingestion] {message}", flush=True)

    def checkpoint(
        self,
        *,
        phase: str,
        completed: int = 0,
        total: int = 0,
        current_paper_id: str | None = None,
        statuses: list[dict] | None = None,
        detail: str | None = None,
    ) -> None:
        rows = list(statuses or [])
        status_counts = {
            status: sum(1 for row in rows if row.get("status") == status)
            for status in sorted({str(row.get("status")) for row in rows})
        }
        payload = {
            "schema_version": "graphagentsdac-ingestion-progress-v01",
            "phase": phase,
            "completed": completed,
            "total": total,
            "current_paper_id": current_paper_id,
            "status_counts": status_counts,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "detail": detail,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        (self.run_dir / "progress.json").write_text(text, encoding="utf-8")
        latest = self.data_root / "runs" / "latest_progress.json"
        latest.write_text(text, encoding="utf-8")


def _progress_summary(statuses: list[dict]) -> str:
    counts: dict[str, int] = {}
    for row in statuses:
        status = str(row.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    if not counts:
        return "no completed papers yet"
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))


def _load_aliases(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): str(v) for k, v in payload.items()}


def _document_unchanged(entry: PaperRegistryEntry | None, paper, marker_version: str) -> bool:
    if entry is None or entry.marker_version != marker_version:
        return False
    if entry.qc_status not in {"passed", "passed_with_warnings"}:
        return False
    if not entry.main_markdown or not Path(entry.main_markdown).exists():
        return False
    main_fp = paper.main_file.fingerprint() if paper.main_file else None
    old_main = entry.main_drive_file.get("fingerprint")
    if main_fp != old_main:
        return False
    new_si = [item.fingerprint() for item in paper.si_files]
    old_si = [item.get("fingerprint") for item in entry.si_drive_files]
    if new_si != old_si:
        return False
    return all(Path(item).exists() for item in entry.si_markdown)


def _drive_dict(item):
    value = item.to_dict()
    value["fingerprint"] = item.fingerprint()
    return value


def run_sync(config: SyncConfig) -> dict:
    aliases = _load_aliases(config.alias_map_path)
    registry = PaperRegistry(config.registry_path)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = config.data_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    progress = _ProgressReporter(config.show_progress, config.data_root, run_dir)
    progress.checkpoint(phase="starting")

    progress.log("reading Article_lists metadata...")
    reader = GoogleWorkspaceReader(config.credentials_path)
    values = reader.read_sheet(config.spreadsheet_id, config.sheet_range)
    articles = parse_article_rows(values)
    progress.log(f"Article_lists loaded: {len(articles)} paper rows")
    progress.checkpoint(phase="article_list_loaded", detail=f"rows={len(articles)}")

    progress.log("scanning Drive folder recursively...")
    drive_files = reader.list_recursive(config.drive_folder_id)
    progress.log(f"Drive scan complete: {len(drive_files)} files discovered")
    progress.checkpoint(phase="drive_scanned", detail=f"files={len(drive_files)}")

    progress.log("matching Sheet rows to main PDFs and SI files...")
    papers, global_issues = match_articles_to_drive(articles, drive_files, aliases)
    progress.log(
        f"matching complete: {len(papers)} papers, {len(global_issues)} global issues"
    )

    marker = MarkerSingleRunner(
        command=config.marker_command,
        paginate_output=True,
        extra_args=list(config.marker_extra_args),
    )
    if config.convert and not config.dry_run:
        progress.log(
            f"checking marker_single environment (version={marker.version})..."
        )
        marker.preflight()
        progress.log("marker_single preflight OK")

    statuses = []
    all_issues: list[IngestionIssue] = list(global_issues)
    total_papers = len(papers)
    progress.checkpoint(
        phase="processing_papers",
        completed=0,
        total=total_papers,
        statuses=statuses,
    )

    for paper_index, paper in enumerate(papers, start=1):
        prefix = f"[{paper_index:>3}/{total_papers}] {paper.paper_id}"
        progress.log(f"{prefix} | checking")
        progress.checkpoint(
            phase="processing_papers",
            completed=paper_index - 1,
            total=total_papers,
            current_paper_id=paper.paper_id,
            statuses=statuses,
        )

        entry = registry.get(paper.paper_id)
        if paper.issues:
            all_issues.extend(paper.issues)
        if not paper.ready_for_download:
            statuses.append({"paper_id": paper.paper_id, "status": "metadata_blocked"})
            progress.log(
                f"{prefix} | metadata_blocked | {_progress_summary(statuses)}"
            )
            progress.checkpoint(
                phase="processing_papers",
                completed=paper_index,
                total=total_papers,
                statuses=statuses,
            )
            continue
        assert paper.main_file is not None
        if _document_unchanged(entry, paper, marker.version) and not config.force_reconvert:
            statuses.append({"paper_id": paper.paper_id, "status": "unchanged"})
            progress.log(f"{prefix} | unchanged/skip | {_progress_summary(statuses)}")
            progress.checkpoint(
                phase="processing_papers",
                completed=paper_index,
                total=total_papers,
                statuses=statuses,
            )
            continue
        if config.dry_run:
            statuses.append({"paper_id": paper.paper_id, "status": "would_process"})
            progress.log(f"{prefix} | would_process | {_progress_summary(statuses)}")
            progress.checkpoint(
                phase="processing_papers",
                completed=paper_index,
                total=total_papers,
                statuses=statuses,
            )
            continue

        source_dir = config.data_root / "sources" / paper.paper_id
        markdown_dir = config.data_root / "markdown" / paper.paper_id
        main_pdf = source_dir / "main.pdf"
        progress.log(f"{prefix} | downloading main PDF: {paper.main_file.name}")
        reader.download_file(paper.main_file.file_id, main_pdf)
        si_paths = []
        for index, si in enumerate(paper.si_files, start=1):
            path = source_dir / f"si_{index}.pdf"
            progress.log(
                f"{prefix} | downloading SI {index}/{len(paper.si_files)}: {si.name}"
            )
            reader.download_file(si.file_id, path)
            si_paths.append(path)

        conversion_issues: list[IngestionIssue] = []
        main_md = None
        si_md: list[str] = []
        if config.convert:
            base_meta = {
                "paper_id": paper.paper_id,
                "title": paper.article.title,
                "annotator": paper.article.annotator,
                "conversion_engine": "marker_single",
                "marker_version": marker.version,
                "source_drive_file_id": paper.main_file.file_id,
                "source_drive_filename": paper.main_file.name,
            }
            marker_progress = lambda message, p=prefix: progress.log(f"{p} | {message}")
            main_result = marker.convert(
                main_pdf,
                markdown_dir / "main",
                document_id=f"{paper.paper_id}_main",
                role="main",
                metadata={
                    **base_meta,
                    "document_id": f"{paper.paper_id}_main",
                    "document_role": "main",
                    "si_count": len(paper.si_files),
                },
                force=config.force_reconvert,
                progress=marker_progress,
                heartbeat_seconds=config.heartbeat_seconds,
            )
            conversion_issues.extend(markdown_qc(main_result, paper.article.title))
            main_md = main_result.normalized_markdown
            for index, (si_file, si_pdf) in enumerate(zip(paper.si_files, si_paths), start=1):
                progress.log(
                    f"{prefix} | converting SI {index}/{len(si_paths)} with marker_single"
                )
                si_result = marker.convert(
                    si_pdf,
                    markdown_dir / f"si_{index}",
                    document_id=f"{paper.paper_id}_SI_{index}",
                    role="supporting_information",
                    metadata={
                        "paper_id": paper.paper_id,
                        "document_id": f"{paper.paper_id}_SI_{index}",
                        "document_role": "supporting_information",
                        "parent_document_id": f"{paper.paper_id}_main",
                        "si_index": index,
                        "title": paper.article.title,
                        "annotator": paper.article.annotator,
                        "conversion_engine": "marker_single",
                        "marker_version": marker.version,
                        "source_drive_file_id": si_file.file_id,
                        "source_drive_filename": si_file.name,
                    },
                    force=config.force_reconvert,
                    progress=marker_progress,
                    heartbeat_seconds=config.heartbeat_seconds,
                )
                conversion_issues.extend(markdown_qc(si_result, "", min_chars=500))
                if si_result.normalized_markdown:
                    si_md.append(si_result.normalized_markdown)
        else:
            conversion_issues.append(
                IngestionIssue(
                    code="conversion_disabled",
                    message="PDF was downloaded but Marker conversion was disabled.",
                    severity="info",
                    paper_id=paper.paper_id,
                )
            )

        paper_issues = list(paper.issues) + conversion_issues
        status = qc_status([item for item in paper_issues if item.severity != "info"])
        new_entry = PaperRegistryEntry(
            paper_id=paper.paper_id,
            title=paper.article.title,
            annotator=paper.article.annotator,
            source_file_name=paper.article.file_name,
            main_drive_file=_drive_dict(paper.main_file),
            si_drive_files=[_drive_dict(item) for item in paper.si_files],
            local_main_pdf=str(main_pdf),
            local_si_pdfs=[str(item) for item in si_paths],
            marker_version=marker.version,
            main_markdown=main_md,
            si_markdown=si_md,
            qc_status=status,
            issues=[item.to_dict() for item in paper_issues],
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        registry.put(new_entry)
        registry.save()
        all_issues.extend(conversion_issues)
        statuses.append({"paper_id": paper.paper_id, "status": status})
        progress.log(f"{prefix} | {status} | {_progress_summary(statuses)}")
        progress.checkpoint(
            phase="processing_papers",
            completed=paper_index,
            total=total_papers,
            statuses=statuses,
        )

    if not config.dry_run:
        registry.save()
    progress.log("building corpus manifest..." if not config.dry_run else "dry-run complete; manifest skipped")
    manifest_path = config.data_root / "corpora" / config.corpus_id / "manifest.json"
    manifest = (
        build_corpus_manifest(registry, manifest_path, config.corpus_id)
        if not config.dry_run
        else None
    )
    report = {
        "schema_version": "graphagentsdac-ingestion-run-v01",
        "run_id": run_id,
        "dry_run": config.dry_run,
        "drive_folder_id": config.drive_folder_id,
        "spreadsheet_id": config.spreadsheet_id,
        "sheet_range": config.sheet_range,
        "article_rows": len(articles),
        "drive_files": len(drive_files),
        "papers": len(papers),
        "status_counts": {
            status: sum(1 for row in statuses if row["status"] == status)
            for status in sorted({row["status"] for row in statuses})
        },
        "statuses": statuses,
        "issues": [item.to_dict() for item in all_issues],
        "manifest_path": str(manifest_path) if manifest else None,
        "manifest_document_count": manifest["document_count"] if manifest else None,
    }
    (run_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    latest = config.data_root / "runs" / "latest.json"
    latest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    progress.checkpoint(
        phase="complete",
        completed=total_papers,
        total=total_papers,
        statuses=statuses,
        detail=f"report={run_dir / 'report.json'}",
    )
    progress.log(
        f"complete in {progress.elapsed_seconds:.1f}s | {_progress_summary(statuses)}"
    )
    return report

