from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _paper_status_map(manifest: dict) -> dict[str, str]:
    rows = manifest.get("papers") or []
    out: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        paper_id = str(row.get("paper_id") or "").strip()
        status = str(
            row.get("extraction_quality_status") or ""
        ).strip()
        if paper_id:
            out[paper_id] = status
    return out


def _projection_complete(root: Path) -> bool:
    return all(
        (root / name).is_file()
        for name in (
            "graph.graphml",
            "node_text.jsonl",
            "edge_evidence.jsonl",
            "summary.json",
        )
    )


def _validated_existing_evidence(
    *,
    root: Path,
    paper_id: str,
    canonical: Path,
) -> bool:
    if not _projection_complete(root):
        return False

    summary = _load_json(root / "summary.json")
    if str(summary.get("paper_id") or "") != paper_id:
        return False
    if str(summary.get("mode") or "") != "evidence":
        return False

    recorded = str(summary.get("canonical_graphml") or "").strip()
    if not recorded:
        return False

    try:
        recorded_path = Path(recorded).expanduser().resolve()
    except Exception:
        return False

    return recorded_path == canonical.resolve()


def _run(command: list[str], *, cwd: Path) -> None:
    subprocess.run(
        command,
        cwd=str(cwd),
        check=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "S27l: materialize per-paper evidence projections from frozen "
            "canonical graphs recorded by the existing exploratory corpus, "
            "then build a corpus-level evidence graph. Dry-run by default."
        )
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Current discovery runtime data root.",
    )
    parser.add_argument(
        "--corpus-id",
        default="sers500_final_v2",
    )
    parser.add_argument(
        "--domain-profile",
        default="sers_au_ag",
    )
    parser.add_argument(
        "--source-mode",
        default="exploratory",
        choices=("mechanism", "exploratory"),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
    )
    parser.add_argument(
        "--rebuild-existing",
        action="store_true",
        help=(
            "Rebuild already-valid evidence bundles. Off by default; "
            "normally valid bundles are reused."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    project_root = Path(__file__).resolve().parents[2]
    data_root = Path(args.data_root).expanduser().resolve()
    corpus_root = data_root / "corpus" / args.corpus_id
    source_manifest_path = (
        corpus_root / args.source_mode / "manifest.json"
    )

    if not source_manifest_path.is_file():
        raise FileNotFoundError(source_manifest_path)

    source_manifest = _load_json(source_manifest_path)
    paper_ids = [
        str(value)
        for value in (source_manifest.get("paper_ids") or [])
        if str(value).strip()
    ]
    if not paper_ids:
        raise RuntimeError("source manifest paper_ids is empty")
    if len(set(paper_ids)) != len(paper_ids):
        raise RuntimeError("source manifest contains duplicate paper_ids")

    source_status = _paper_status_map(source_manifest)
    partial_critical = sorted(
        paper_id
        for paper_id in paper_ids
        if source_status.get(paper_id) == "partial_critical"
    )

    plan = []
    missing_canonical = []

    for paper_id in paper_ids:
        source_summary_path = (
            data_root
            / "extracted"
            / paper_id
            / "graphagents"
            / args.source_mode
            / "summary.json"
        )
        if not source_summary_path.is_file():
            raise FileNotFoundError(source_summary_path)

        source_summary = _load_json(source_summary_path)

        if str(source_summary.get("paper_id") or "") != paper_id:
            raise RuntimeError(
                f"source summary paper mismatch: {source_summary_path}"
            )
        if str(source_summary.get("mode") or "") != args.source_mode:
            raise RuntimeError(
                f"source summary mode mismatch: {source_summary_path}"
            )

        canonical_text = str(
            source_summary.get("canonical_graphml") or ""
        ).strip()
        if not canonical_text:
            raise RuntimeError(
                f"canonical_graphml missing: {source_summary_path}"
            )

        canonical = Path(canonical_text).expanduser().resolve()
        if not canonical.is_file():
            missing_canonical.append(
                {
                    "paper_id": paper_id,
                    "canonical": str(canonical),
                }
            )
            continue

        evidence_root = (
            data_root
            / "extracted"
            / paper_id
            / "graphagents"
            / "evidence"
        )

        valid_existing = _validated_existing_evidence(
            root=evidence_root,
            paper_id=paper_id,
            canonical=canonical,
        )

        plan.append(
            {
                "paper_id": paper_id,
                "canonical": canonical,
                "canonical_sha256": _sha256(canonical),
                "evidence_root": evidence_root,
                "reuse": (
                    valid_existing
                    and not args.rebuild_existing
                ),
            }
        )

    if missing_canonical:
        raise RuntimeError(
            "canonical provenance is unresolved for "
            f"{len(missing_canonical)} paper(s): "
            f"{missing_canonical[:5]!r}"
        )

    to_build = [row for row in plan if not row["reuse"]]
    to_reuse = [row for row in plan if row["reuse"]]

    print("S27L_EVIDENCE_LANE_PLAN=PASS")
    print("data_root:", data_root)
    print("corpus_id:", args.corpus_id)
    print("domain_profile:", args.domain_profile)
    print("source_mode:", args.source_mode)
    print("paper_count:", len(paper_ids))
    print("canonical_resolved_count:", len(plan))
    print("evidence_projection_build_count:", len(to_build))
    print("evidence_projection_reuse_count:", len(to_reuse))
    print("source_partial_critical_count:", len(partial_critical))
    print("apply:", args.apply)
    print("reextract_authorized=false")
    print("canonical_graph_mutation_authorized=false")
    print("exploratory_graph_mutation_authorized=false")
    print("mechanism_graph_mutation_authorized=false")
    print("production_selection_changed=false")
    print("novelty_authority_created=false")

    if not args.apply:
        print("S27L_DRY_RUN=PASS")
        print("NEXT=rerun with --apply")
        return 0

    projection_script = (
        project_root
        / "scripts"
        / "corpus"
        / "build_graphagents_projection.py"
    )
    corpus_script = (
        project_root
        / "scripts"
        / "corpus"
        / "build_corpus_graph.py"
    )

    for script in (projection_script, corpus_script):
        if not script.is_file():
            raise FileNotFoundError(script)

    build_ledger = []

    for index, row in enumerate(plan, start=1):
        paper_id = str(row["paper_id"])
        canonical = Path(row["canonical"])
        evidence_root = Path(row["evidence_root"])

        if row["reuse"]:
            status = "reused"
        else:
            command = [
                sys.executable,
                "-m",
                "scripts.corpus.build_graphagents_projection",
                "--paper-id",
                paper_id,
                "--domain-profile",
                args.domain_profile,
                "--data-root",
                str(data_root),
                "--mode",
                "evidence",
                "--canonical-graphml",
                str(canonical),
                "--output-dir",
                str(evidence_root),
            ]
            _run(command, cwd=project_root)

            if not _validated_existing_evidence(
                root=evidence_root,
                paper_id=paper_id,
                canonical=canonical,
            ):
                raise RuntimeError(
                    "new evidence bundle failed structural validation: "
                    f"{paper_id}"
                )
            status = "built"

        build_ledger.append(
            {
                "paper_id": paper_id,
                "canonical_graphml": str(canonical),
                "canonical_sha256": row["canonical_sha256"],
                "evidence_root": str(evidence_root),
                "status": status,
            }
        )
        print(
            f"S27L_EVIDENCE_PAPER {index}/{len(plan)} "
            f"{paper_id} {status}"
        )

    # Build exactly the same paper set as the source corpus.
    corpus_command = [
        sys.executable,
        "-m",
        "scripts.corpus.build_corpus_graph",
        "--corpus-id",
        args.corpus_id,
        "--domain-profile",
        args.domain_profile,
        "--data-root",
        str(data_root),
        "--paper-ids",
        *paper_ids,
        "--mode",
        "evidence",
    ]

    # Reproduce, but do not broaden beyond, an already-materialized source
    # corpus that contains PARTIAL_CRITICAL papers.
    if partial_critical:
        corpus_command.append("--allow-critical-partial")

    _run(corpus_command, cwd=project_root)

    evidence_corpus_root = (
        corpus_root / "evidence"
    )
    evidence_manifest_path = (
        evidence_corpus_root / "manifest.json"
    )
    evidence_graph_path = (
        evidence_corpus_root / "graph.graphml"
    )
    evidence_audit_path = (
        evidence_corpus_root / "audit.json"
    )

    for path in (
        evidence_manifest_path,
        evidence_graph_path,
        evidence_audit_path,
    ):
        if not path.is_file():
            raise RuntimeError(
                f"evidence corpus output missing: {path}"
            )

    evidence_manifest = _load_json(evidence_manifest_path)
    if list(evidence_manifest.get("paper_ids") or []) != paper_ids:
        raise RuntimeError(
            "evidence corpus paper set/order differs from source corpus"
        )
    if str(evidence_manifest.get("mode") or "") != "evidence":
        raise RuntimeError("evidence corpus mode mismatch")

    evidence_audit = _load_json(evidence_audit_path)
    if not bool(evidence_audit.get("passes_structural_gate")):
        raise RuntimeError(
            "evidence corpus structural audit did not pass"
        )

    lane_manifest = {
        "schema_version": "s27l-evidence-operationalization-lane-v1",
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": _sha256(source_manifest_path),
        "source_mode": args.source_mode,
        "corpus_id": args.corpus_id,
        "domain_profile": args.domain_profile,
        "paper_count": len(paper_ids),
        "paper_ids": paper_ids,
        "partial_critical_paper_ids": partial_critical,
        "evidence_projection_built_count": sum(
            row["status"] == "built"
            for row in build_ledger
        ),
        "evidence_projection_reused_count": sum(
            row["status"] == "reused"
            for row in build_ledger
        ),
        "evidence_graph": str(evidence_graph_path),
        "evidence_graph_sha256": _sha256(evidence_graph_path),
        "evidence_manifest": str(evidence_manifest_path),
        "evidence_manifest_sha256": _sha256(evidence_manifest_path),
        "evidence_audit": str(evidence_audit_path),
        "evidence_structural_gate_passed": True,
        "papers": build_ledger,
        "reextract_authorized": False,
        "canonical_graph_mutated": False,
        "exploratory_graph_mutated": False,
        "mechanism_graph_mutated": False,
        "production_selection_changed": False,
        "positive_premise_authority_created": False,
        "gap_authority_created": False,
        "novelty_authority_created": False,
        "measurement_independence_certification_authority": False,
    }

    lane_manifest_path = (
        evidence_corpus_root
        / "s27l_operationalization_lane_manifest.json"
    )
    lane_manifest_path.write_text(
        json.dumps(
            lane_manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print("S27L_EVIDENCE_LANE_MATERIALIZATION=PASS")
    print("evidence_graph:", evidence_graph_path)
    print("evidence_graph_sha256:", _sha256(evidence_graph_path))
    print("evidence_manifest:", evidence_manifest_path)
    print(
        "evidence_structural_gate_passed:",
        evidence_audit.get("passes_structural_gate"),
    )
    print("lane_manifest:", lane_manifest_path)
    print("reextract_performed=false")
    print("canonical_graph_mutated=false")
    print("exploratory_graph_mutated=false")
    print("mechanism_graph_mutated=false")
    print("production_selection_changed=false")
    print("positive_premise_authority_created=false")
    print("gap_authority_created=false")
    print("novelty_authority_created=false")
    print(
        "measurement_independence_certification_authority=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
