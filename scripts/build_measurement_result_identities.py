from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import networkx as nx

from dac_her.domains.extraction_registry import get_extraction_adapter
from dac_her.domains.registry import get_domain_profile
from dac_her.measurement_result_identity import (
    MEASUREMENT_RESULT_IDENTITY_SEMANTICS_ID,
    audit_measurement_result_identities,
    build_measurement_result_identities,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_jsonl(
    path: Path,
    rows: Iterable[dict[str, Any]],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build non-destructive Measurement source-mention to scientific-"
            "result identity sidecars."
        )
    )
    parser.add_argument("--domain-profile", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument(
        "--mode",
        choices=("evidence", "mechanism", "exploratory"),
        default="exploratory",
    )
    parser.add_argument(
        "--measurement-result-identity-id",
        required=True,
    )
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = get_domain_profile(args.domain_profile)
    extraction_adapter = get_extraction_adapter(profile.profile_id)

    data_root = Path(
        args.data_root or extraction_adapter.default_data_root
    )
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root

    corpus_root = (
        data_root / "corpus" / args.corpus_id / args.mode
    )
    manifest_path = corpus_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Corpus manifest not found: {manifest_path}"
        )
    corpus_manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    if not isinstance(corpus_manifest, dict):
        raise ValueError("Corpus manifest must be a JSON object.")
    if str(corpus_manifest.get("domain_profile_id", "")) != profile.profile_id:
        raise ValueError("Measurement-result identity corpus/domain mismatch.")
    if int(
        corpus_manifest.get("destructive_cross_paper_merges", -1)
    ) != 0:
        raise ValueError(
            "Measurement-result identity requires a non-destructive corpus."
        )

    paper_ids = [
        str(value)
        for value in corpus_manifest.get("paper_ids", [])
        if str(value).strip()
    ]
    if not paper_ids or len(paper_ids) != len(set(paper_ids)):
        raise ValueError(
            "Corpus manifest must contain unique paper IDs."
        )

    source_graphs: dict[str, nx.Graph] = {}
    source_rows: list[dict[str, object]] = []
    identities = []
    candidates = []

    for paper_id in paper_ids:
        graph_path = (
            data_root
            / "extracted"
            / paper_id
            / f"{paper_id}.graphml"
        )
        if not graph_path.exists():
            raise FileNotFoundError(
                f"Canonical graph not found: {graph_path}"
            )
        graph = nx.read_graphml(
            graph_path,
            force_multigraph=True,
        )
        graph_domain = str(
            graph.graph.get("domain_profile_id", "")
        )
        if graph_domain and graph_domain != profile.profile_id:
            raise ValueError(
                f"Canonical graph/domain mismatch for {paper_id}."
            )

        paper_identities, paper_candidates = (
            build_measurement_result_identities(
                graph,
                paper_id,
            )
        )
        source_graphs[paper_id] = graph
        identities.extend(paper_identities)
        candidates.extend(paper_candidates)
        source_rows.append(
            {
                "paper_id": paper_id,
                "canonical_graphml": str(graph_path),
                "canonical_graph_sha256": _sha256_file(graph_path),
                "source_mention_count": sum(
                    str(attrs.get("type", "")) == "Measurement"
                    for _, attrs in graph.nodes(data=True)
                ),
                "scientific_result_count": len(
                    paper_identities
                ),
                "consolidated_exact_result_count": sum(
                    item.status == "consolidated_exact"
                    for item in paper_identities
                ),
            }
        )

    audit = audit_measurement_result_identities(
        identities=identities,
        candidates=candidates,
        source_graphs=source_graphs,
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (
            corpus_root
            / "measurement_result_identity"
            / args.measurement_result_identity_id
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    identities_path = _write_jsonl(
        output_dir / "identities.jsonl",
        (item.to_row() for item in identities),
    )
    candidates_path = _write_jsonl(
        output_dir / "same_lineage_candidates.jsonl",
        (item.to_row() for item in candidates),
    )
    audit_path = output_dir / "audit.json"
    audit_path.write_text(
        json.dumps(
            audit.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    summary = {
        "measurement_result_identity_id": (
            args.measurement_result_identity_id
        ),
        "measurement_result_identity_semantics_id": (
            MEASUREMENT_RESULT_IDENTITY_SEMANTICS_ID
        ),
        "domain_profile_id": profile.profile_id,
        "corpus_id": args.corpus_id,
        "corpus_mode": args.mode,
        "corpus_manifest": str(manifest_path),
        "paper_ids": paper_ids,
        "paper_count": len(paper_ids),
        "source_graphs": source_rows,
        **{
            key: value
            for key, value in audit.to_dict().items()
            if key not in {"semantics_id", "paper_count"}
        },
        "policy": {
            "destructive_graph_merge": False,
            "same_value_alone_never_merges": True,
            "same_origin_local_measurement_id_required": True,
            "missing_condition_is_not_conflict": True,
            "explicit_condition_conflict_blocks_consolidation": True,
            "subject_identity_must_be_compatible": True,
        },
        "outputs": {
            "identities": str(identities_path),
            "same_lineage_candidates": str(candidates_path),
            "audit": str(audit_path),
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print("Measurement-result identities built")
    print(
        "Measurement-result identity ID:",
        args.measurement_result_identity_id,
    )
    print(
        "Identity semantics:",
        MEASUREMENT_RESULT_IDENTITY_SEMANTICS_ID,
    )
    print("Papers:", len(paper_ids))
    print("Source mentions:", audit.source_mention_count)
    print("Scientific results:", audit.scientific_result_count)
    print(
        "Consolidated exact results:",
        audit.consolidated_exact_result_count,
    )
    print(
        "Consolidated source mentions:",
        audit.consolidated_source_mention_count,
    )
    print(
        "Unresolved same-lineage groups:",
        audit.unresolved_same_lineage_group_count,
    )
    print("Structural gate:", audit.structural_gate)
    print("Saved:", output_dir)
    return 0 if audit.structural_gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
