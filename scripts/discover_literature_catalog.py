from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dac_her.corpus_acquisition.profile import (
    build_catalog_queries,
    load_acquisition_profile,
)
from dac_her.corpus_acquisition.progress import (
    compact_text,
    progress_prefix,
)
from pipeline_core.literature.catalog import (
    CrossrefCatalogProvider,
    LiteratureCatalogRetriever,
    SemanticScholarCatalogProvider,
)


def _write_json(
    path: Path,
    value: Any,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(
    path: Path,
    rows: list[Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            if hasattr(row, "model_dump"):
                row = row.model_dump(
                    mode="json"
                )
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "M1: discover a neutral literature catalog for a generic "
            "corpus-acquisition profile. This stage retrieves metadata/"
            "abstracts only and does not promote external literature into "
            "positive KG evidence."
        )
    )
    parser.add_argument(
        "--profile",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--providers",
        default=None,
        help=(
            "Comma-separated semantic_scholar,crossref. "
            "Defaults to the profile."
        ),
    )
    parser.add_argument(
        "--results-per-query",
        type=int,
        default=None,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = load_acquisition_profile(
        args.profile
    )
    queries = build_catalog_queries(profile)

    provider_names = (
        [
            value.strip()
            for value in args.providers.split(",")
            if value.strip()
        ]
        if args.providers
        else list(
            profile.discovery.default_providers
        )
    )
    providers = []
    for name in provider_names:
        if name == "semantic_scholar":
            providers.append(
                SemanticScholarCatalogProvider()
            )
        elif name == "crossref":
            providers.append(
                CrossrefCatalogProvider()
            )
        else:
            raise ValueError(
                f"Unknown catalog provider: {name}"
            )

    results_per_query = (
        args.results_per_query
        if args.results_per_query is not None
        else profile.discovery.results_per_query
    )
    def _progress(event: dict[str, Any]) -> None:
        if event.get("stage") != "m1_retrieval":
            return
        if event.get("event") == "start":
            print(
                progress_prefix(
                    "M1",
                    int(event["current"]),
                    int(event["total"]),
                ),
                f"{str(event['provider']):<17}",
                f"axis={event['axis_id']}",
                f'query="{compact_text(str(event["query_text"]))}"',
                flush=True,
            )
        elif event.get("event") == "complete":
            status = "ok" if event.get("success") else "FAIL"
            print(
                progress_prefix(
                    "M1",
                    int(event["current"]),
                    int(event["total"]),
                ),
                f"{status:<4}",
                f"results={int(event.get('result_count') or 0):>3}",
                f"elapsed={float(event.get('elapsed_seconds') or 0.0):.2f}s",
                flush=True,
            )

    packet = LiteratureCatalogRetriever(
        providers,
        results_per_query=results_per_query,
        progress_callback=_progress,
    ).retrieve(
        profile_id=profile.profile_id,
        queries=queries,
    ).packet

    output = args.output_dir
    _write_json(
        output / "catalog.json",
        packet,
    )
    _write_jsonl(
        output / "queries.jsonl",
        queries,
    )
    _write_jsonl(
        output / "candidates.jsonl",
        packet.works,
    )

    successful = sum(
        execution.success
        for execution in packet.executions
    )
    report = {
        "stage": "generic_corpus_acquisition_m1",
        "profile_id": profile.profile_id,
        "domain_profile_id": (
            profile.domain_profile_id
        ),
        "catalog_id": packet.catalog_id,
        "query_count": len(queries),
        "provider_count": len(providers),
        "successful_provider_query_executions": (
            successful
        ),
        "provider_query_execution_count": len(
            packet.executions
        ),
        "raw_work_count": packet.raw_work_count,
        "canonical_work_count": (
            packet.canonical_work_count
        ),
        "deduplicated_work_count": (
            packet.deduplicated_work_count
        ),
        "supplementary_records_collapsed": (
            packet.supplementary_records_collapsed
        ),
        "epistemic_usage": packet.epistemic_usage,
        "positive_evidence_promotion_performed": False,
    }
    _write_json(
        output / "discovery_report.json",
        report,
    )

    print("Generic corpus acquisition M1 complete")
    print("Profile:", profile.profile_id)
    print("Queries:", len(queries))
    print(
        "Provider-query executions:",
        successful,
        "/",
        len(packet.executions),
    )
    print(
        "Works:",
        f"raw={packet.raw_work_count}",
        f"canonical={packet.canonical_work_count}",
        f"deduplicated={packet.deduplicated_work_count}",
    )
    print("Catalog:", output / "catalog.json")
    print(
        "Epistemic usage:",
        packet.epistemic_usage,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
