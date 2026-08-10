from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml

from dac_her.domains.extraction_registry import get_extraction_adapter
from dac_her.domains.registry import get_domain_profile
from dac_her.ingestion.marker_runner import MarkerSingleRunner
from dac_her.ingestion.naming import parse_pdf_name


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class LocalPaper:
    paper_id: str
    main_pdf: Path
    si_pdfs: tuple[Path, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_local_papers(input_dir: str | Path) -> list[LocalPaper]:
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    grouped: dict[tuple[str, int], dict[str, object]] = {}
    ignored: list[str] = []
    for path in sorted(input_dir.glob("*.pdf")):
        parsed = parse_pdf_name(path.name)
        if parsed is None:
            ignored.append(path.name)
            continue
        key = (parsed.owner, parsed.number)
        row = grouped.setdefault(key, {"main": None, "si": []})
        if parsed.role == "main":
            if row["main"] is not None:
                raise ValueError(f"Multiple main PDFs for {key}: {path.name}")
            row["main"] = path
        else:
            row["si"].append((int(parsed.si_index or 0), path))

    papers: list[LocalPaper] = []
    missing_main: list[str] = []
    for (owner, number), row in sorted(grouped.items(), key=lambda item: item[0]):
        main = row["main"]
        if main is None:
            missing_main.append(f"{owner}_{number}")
            continue
        si = tuple(path for _, path in sorted(row["si"], key=lambda pair: pair[0]))
        papers.append(LocalPaper(
            paper_id=f"{owner}_{number}",
            main_pdf=Path(main),
            si_pdfs=si,
        ))

    if missing_main:
        raise ValueError(
            "Supporting-information files without a main PDF: "
            + ", ".join(missing_main)
        )
    if not papers:
        extra = f" Ignored: {ignored}" if ignored else ""
        raise ValueError(f"No recognized local paper PDFs found.{extra}")

    seen: dict[str, LocalPaper] = {}
    duplicates: list[tuple[str, str]] = []
    for paper in papers:
        fingerprint = _sha256(paper.main_pdf)
        prior = seen.get(fingerprint)
        if prior is not None:
            duplicates.append((prior.paper_id, paper.paper_id))
        else:
            seen[fingerprint] = paper
    if duplicates:
        raise ValueError(
            "Duplicate main-PDF content detected: "
            + ", ".join(f"{a} == {b}" for a, b in duplicates)
        )
    return papers


def _repo_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _convert_document(
    *,
    marker: MarkerSingleRunner,
    input_pdf: Path,
    output_dir: Path,
    document_id: str,
    role: str,
    metadata: dict[str, object],
    force: bool,
) -> Path:
    result = marker.convert(
        input_pdf,
        output_dir,
        document_id=document_id,
        role=role,
        metadata=metadata,
        force=force,
        progress=lambda message: print(f"[marker] {message}", flush=True),
    )
    if result.return_code != 0 or not result.normalized_markdown:
        raise RuntimeError(
            f"Marker conversion failed for {input_pdf}: "
            f"{result.error or 'normalized markdown missing'}"
        )
    return Path(result.normalized_markdown)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ingest local main/SI PDFs with Marker and generate a papers YAML "
            "for one scientific domain."
        )
    )
    parser.add_argument("--domain-profile", default="sers_au_ag")
    parser.add_argument("--input-dir", default="data_sers/inbox")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--config-out", default="configs/papers_sers_au_ag.yaml")
    parser.add_argument("--marker-command", default="marker_single")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = get_domain_profile(args.domain_profile)
    adapter = get_extraction_adapter(profile.profile_id)
    data_root = Path(args.data_root or adapter.default_data_root)
    input_dir = Path(args.input_dir)
    papers = discover_local_papers(input_dir)

    print("Domain profile:", profile.profile_id)
    print("Extraction adapter:", adapter.adapter_id)
    print("Input directory:", input_dir)
    print("Data root:", data_root)
    print("Papers:", len(papers))
    for paper in papers:
        print(
            " ", paper.paper_id,
            "| main=", paper.main_pdf.name,
            "| SI=", [path.name for path in paper.si_pdfs],
        )

    if args.dry_run:
        return

    marker = MarkerSingleRunner(command=args.marker_command, paginate_output=True)
    marker.preflight()

    config: dict[str, object] = {"version": 3, "papers": {}}
    papers_config: dict[str, object] = config["papers"]  # type: ignore[assignment]

    for index, paper in enumerate(papers, start=1):
        print(f"\n[{index}/{len(papers)}] {paper.paper_id}", flush=True)
        source_dir = data_root / "ingestion" / "sources" / paper.paper_id
        markdown_root = data_root / "ingestion" / "markdown" / paper.paper_id
        source_dir.mkdir(parents=True, exist_ok=True)
        markdown_root.mkdir(parents=True, exist_ok=True)

        main_copy = source_dir / "main.pdf"
        shutil.copy2(paper.main_pdf, main_copy)
        main_md = _convert_document(
            marker=marker,
            input_pdf=main_copy,
            output_dir=markdown_root / "main",
            document_id=f"{paper.paper_id}_main",
            role="main",
            metadata={
                "paper_id": paper.paper_id,
                "domain_profile_id": profile.profile_id,
                "source_filename": paper.main_pdf.name,
                "document_role": "main",
            },
            force=args.force,
        )

        documents: list[dict[str, object]] = [{
            "document_id": "main",
            "role": "main",
            "package_dir": _repo_path(main_md.parent),
            "markdown_file": main_md.name,
            "selection": {"mode": "whole_document"},
            "figure_processing": {"mode": "caption_first", "vision_assets": []},
        }]

        for si_index, si_pdf in enumerate(paper.si_pdfs, start=1):
            si_copy = source_dir / f"si_{si_index}.pdf"
            shutil.copy2(si_pdf, si_copy)
            si_md = _convert_document(
                marker=marker,
                input_pdf=si_copy,
                output_dir=markdown_root / f"si{si_index}",
                document_id=f"{paper.paper_id}_si{si_index}",
                role="supporting_information",
                metadata={
                    "paper_id": paper.paper_id,
                    "domain_profile_id": profile.profile_id,
                    "source_filename": si_pdf.name,
                    "document_role": "supporting_information",
                },
                force=args.force,
            )
            documents.append({
                "document_id": f"si{si_index}",
                "role": "supporting_information",
                "package_dir": _repo_path(si_md.parent),
                "markdown_file": si_md.name,
                "selection": {
                    "mode": "referenced_blocks",
                    "fallback": "skip",
                    "reference_scope": "whole_main",
                },
                "figure_processing": {"mode": "caption_first", "vision_assets": []},
            })

        papers_config[paper.paper_id] = {
            "enabled": True,
            "documents": documents,
            "resolution_file": None,
        }

    config_path = Path(args.config_out)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False, width=100),
        encoding="utf-8",
    )

    manifest = {
        "schema_version": "graphagents-local-ingestion-v01",
        "domain_profile_id": profile.profile_id,
        "extraction_adapter_id": adapter.adapter_id,
        "data_root": str(data_root),
        "input_dir": str(input_dir),
        "paper_ids": [paper.paper_id for paper in papers],
        "config_path": str(config_path),
        "source_sha256": {
            paper.paper_id: _sha256(paper.main_pdf) for paper in papers
        },
    }
    manifest_path = data_root / "ingestion" / "local_ingestion_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nLocal ingestion complete")
    print("Config:", config_path)
    print("Manifest:", manifest_path)


if __name__ == "__main__":
    main()
