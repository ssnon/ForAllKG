from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import networkx as nx

from dac_her.config import get_paper_config
from dac_her.resolution_candidates import (
    generate_resolution_candidates,
    sync_decisions_jsonl,
    write_candidates_csv,
)
from dac_her.run_state import (
    paper_output_root,
    resolve_run_directory,
    write_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "papers.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate paper-level entity and measurement resolution "
            "candidates from the raw merged graph."
        )
    )
    parser.add_argument("--paper-id", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--raw-graph",
        default=None,
        help="Override raw_merged.graphml path.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Default: <run_dir>/resolution",
    )
    parser.add_argument(
        "--decisions",
        default=None,
        help=(
            "Stable decisions JSONL path. Default: configured .jsonl "
            "resolution_file, otherwise <paper_root>/resolution/decisions.jsonl"
        ),
    )
    parser.add_argument(
        "--fuzzy-threshold",
        type=float,
        default=0.72,
    )
    parser.add_argument(
        "--measurement-threshold",
        type=float,
        default=0.90,
    )
    return parser.parse_args()


def resolve_decisions_path(
    *,
    paper_resolution_file: Path | None,
    paper_root: Path,
    override: str | None,
) -> Path:
    if override:
        return Path(override).resolve()
    if (
        paper_resolution_file is not None
        and paper_resolution_file.suffix.lower() == ".jsonl"
    ):
        return paper_resolution_file.resolve()
    return (paper_root / "resolution" / "decisions.jsonl").resolve()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.fuzzy_threshold <= 1.0:
        raise ValueError("--fuzzy-threshold must be between 0 and 1.")
    if not 0.0 <= args.measurement_threshold <= 1.0:
        raise ValueError("--measurement-threshold must be between 0 and 1.")

    paper = get_paper_config(
        args.config,
        project_root=PROJECT_ROOT,
        paper_id=args.paper_id,
    )
    run_dir = resolve_run_directory(
        project_root=PROJECT_ROOT,
        paper_id=paper.paper_id,
        run_id=args.run_id,
    )

    raw_graph_path = (
        Path(args.raw_graph).resolve()
        if args.raw_graph
        else run_dir / "raw_merged.graphml"
    )
    if not raw_graph_path.exists():
        raise FileNotFoundError(
            "Raw merged graph not found. First run:\n"
            f"  python -m scripts.build_paper_graph --paper-id {paper.paper_id}\n"
            f"Expected: {raw_graph_path}"
        )

    graph = nx.read_graphml(raw_graph_path)
    candidates, summary = generate_resolution_candidates(
        graph,
        fuzzy_minimum_score=args.fuzzy_threshold,
        measurement_minimum_score=args.measurement_threshold,
    )

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else run_dir / "resolution"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates_path = write_candidates_csv(
        output_dir / "candidates.csv",
        candidates,
    )
    summary_payload = {
        **summary.to_dict(),
        "paper_id": paper.paper_id,
        "run_id": run_dir.name,
        "raw_graph": str(raw_graph_path),
        "fuzzy_threshold": args.fuzzy_threshold,
        "measurement_threshold": args.measurement_threshold,
    }
    write_json(output_dir / "candidate_summary.json", summary_payload)

    paper_root = paper_output_root(PROJECT_ROOT, paper.paper_id)
    decisions_path = resolve_decisions_path(
        paper_resolution_file=paper.resolution_file,
        paper_root=paper_root,
        override=args.decisions,
    )
    decisions_path, sync_summary = sync_decisions_jsonl(
        decisions_path,
        candidates,
    )

    template_path = output_dir / "decisions.template.jsonl"
    shutil.copyfile(decisions_path, template_path)

    stable_resolution_dir = paper_root / "resolution"
    stable_resolution_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        candidates_path,
        stable_resolution_dir / "latest_candidates.csv",
    )
    write_json(
        stable_resolution_dir / "latest_candidate_summary.json",
        summary_payload,
    )

    print("Resolution candidate generation successful")
    print("Paper:", paper.paper_id)
    print("Run ID:", run_dir.name)
    print("Raw graph:", raw_graph_path)
    print("Candidates:", len(candidates))
    print("  Exact entity:", summary.exact_entity_candidates)
    print(
        "  Fuzzy cross-component:",
        summary.fuzzy_cross_component_candidates,
    )
    print(
        "  Fuzzy intra-component:",
        summary.fuzzy_intra_component_candidates,
    )
    print(
        "  Auto-approved safe Metal/Reaction:",
        summary.auto_approved_candidates,
    )
    print(
        "  Measurement duplicate:",
        summary.measurement_duplicate_candidates,
    )
    print("Candidate CSV:", candidates_path)
    print("Decisions JSONL:", decisions_path)
    print("Decision sync:", json.dumps(sync_summary, ensure_ascii=False))
    print()
    print("Only records with both fields below are applied:")
    print('  "decision": "same_entity"')
    print('  "approved": true')


if __name__ == "__main__":
    main()
