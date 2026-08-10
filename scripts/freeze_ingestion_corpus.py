from __future__ import annotations

import argparse
import json
from pathlib import Path

from dac_her.corpus_freeze import load_and_freeze


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Freeze a QC-passed ingestion manifest with exact-PDF dedupe and review diagnostics."
    )
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", default=None)
    p.add_argument("--corpus-id", default=None)
    p.add_argument("--project-root", default=".")
    p.add_argument("--exclude-warnings", action="store_true")
    p.add_argument(
        "--skip-path-verification",
        action="store_true",
        help="Do not require Markdown paths to exist; hashes become null for missing files.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.manifest)
    raw = json.loads(source.read_text(encoding="utf-8"))
    corpus_id = args.corpus_id or str(raw.get("corpus_id") or "corpus")
    output = Path(args.output) if args.output else (
        Path("data_dac") / "frozen_corpora" / corpus_id / "manifest.json"
    )
    result = load_and_freeze(
        source,
        output,
        project_root=args.project_root,
        include_warnings=not args.exclude_warnings,
        verify_paths=not args.skip_path_verification,
    )

    print("[freeze] complete")
    print(f"[freeze] source documents:       {result['source_document_count']}")
    print(f"[freeze] eligible documents:     {result['eligible_document_count']}")
    print(f"[freeze] exact duplicates removed: {result['deduplicated_document_count']}")
    print(f"[freeze] frozen documents:       {result['document_count']}")
    print(f"[freeze] title review groups:    {sum(1 for g in result['title_review_groups'] if g['review_required'])}")
    print(f"[freeze] repeated SI groups:     {len(result['si_fingerprint_review_groups'])}")
    print(f"[freeze] manifest: {output}")
    if result["exact_duplicate_groups"]:
        print("\n[freeze] exact duplicate groups")
        for group in result["exact_duplicate_groups"]:
            print(
                "  - " + group["canonical_paper_id"] + " <= "
                + ", ".join(group["duplicate_paper_ids"])
            )
    review = [g for g in result["title_review_groups"] if g["review_required"]]
    if review:
        print("\n[freeze] same-title review candidates (kept, not deleted)")
        for group in review:
            print("  - " + " | ".join(group["paper_ids"]))


if __name__ == "__main__":
    main()
