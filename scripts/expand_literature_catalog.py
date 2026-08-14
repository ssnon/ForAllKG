from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dac_her.corpus_acquisition.catalog_expansion import append_catalog_expansion
from dac_her.corpus_acquisition.profile import build_catalog_queries, load_acquisition_profile
from dac_her.corpus_acquisition.progress import compact_text, progress_prefix
from dac_her.corpus_acquisition.openalex_catalog_adapter import OpenAlexCatalogProvider
from dac_her.literature_catalog import (
    CrossrefCatalogProvider,
    LiteratureCatalogRetriever,
    SemanticScholarCatalogProvider,
)
from dac_her.literature_catalog_contracts import LiteratureCatalogPacket


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            if hasattr(row, "model_dump"):
                row = row.model_dump(mode="json")
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _providers(names: list[str]):
    result = []
    for name in names:
        if name == "semantic_scholar":
            result.append(SemanticScholarCatalogProvider())
        elif name == "crossref":
            result.append(CrossrefCatalogProvider())
        elif name == "openalex":
            result.append(
                OpenAlexCatalogProvider(
                    api_key=os.getenv("OPENALEX_API_KEY"),
                    mailto=os.getenv("OPENALEX_MAILTO") or os.getenv("CROSSREF_MAILTO"),
                )
            )
        else:
            raise ValueError(f"Unknown catalog provider: {name}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Append-only M1 discovery expansion. Search deeper and/or through "
            "additional metadata providers, then append only genuinely new "
            "canonical works while freezing every base CatalogWork row."
        )
    )
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--base-catalog", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expansion-id", required=True)
    parser.add_argument(
        "--providers",
        default="semantic_scholar,crossref,openalex",
        help="Comma-separated semantic_scholar,crossref,openalex.",
    )
    parser.add_argument("--results-per-query", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = load_acquisition_profile(args.profile)
    base = LiteratureCatalogPacket.model_validate_json(
        args.base_catalog.read_text(encoding="utf-8")
    )
    if base.acquisition_profile_id != profile.profile_id:
        raise ValueError("Base catalog/profile mismatch")
    if args.results_per_query < 1 or args.results_per_query > 100:
        raise ValueError("results-per-query must be between 1 and 100")

    provider_names = [
        value.strip()
        for value in args.providers.split(",")
        if value.strip()
    ]
    if not provider_names:
        raise ValueError("At least one expansion provider is required")

    queries = build_catalog_queries(profile)

    def progress(event: dict[str, Any]) -> None:
        if event.get("stage") != "m1_retrieval":
            return
        if event.get("event") == "start":
            print(
                progress_prefix("M1 expand", int(event["current"]), int(event["total"])),
                f"{str(event['provider']):<17}",
                f"axis={event['axis_id']}",
                f'query="{compact_text(str(event["query_text"]))}"',
                flush=True,
            )
        elif event.get("event") == "complete":
            print(
                progress_prefix("M1 expand", int(event["current"]), int(event["total"])),
                "ok" if event.get("success") else "FAIL",
                f"results={int(event.get('result_count') or 0):>3}",
                f"elapsed={float(event.get('elapsed_seconds') or 0.0):.2f}s",
                flush=True,
            )

    incoming = LiteratureCatalogRetriever(
        _providers(provider_names),
        results_per_query=args.results_per_query,
        progress_callback=progress,
    ).retrieve(
        profile_id=profile.profile_id,
        queries=queries,
    ).packet

    result = append_catalog_expansion(
        base=base,
        incoming=incoming,
        expansion_id=args.expansion_id,
    )

    output = args.output_dir
    _write_json(output / "catalog.json", result.packet)
    _write_json(output / "incoming_catalog.json", incoming)
    _write_jsonl(output / "queries.jsonl", result.packet.queries)
    _write_jsonl(output / "candidates.jsonl", result.packet.works)
    _write_json(output / "expansion_report.json", result.report)

    print("Append-only literature catalog expansion complete")
    print("Base works:", result.report["base_work_count"])
    print("Incoming canonical works:", result.report["incoming_work_count"])
    print("Overlaps skipped:", result.report["overlap_work_count"])
    print("New works appended:", result.report["new_work_count"])
    print("Expanded works:", result.report["expanded_work_count"])
    print("Expanded catalog:", output / "catalog.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
