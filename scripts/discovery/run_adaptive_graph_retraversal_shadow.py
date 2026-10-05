from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from pipeline_core.discovery.adaptive_graph_retraversal import (
    audit_context_delta,
    context_footprint,
    eligible_premise_ids,
    materialize_selected_traversal,
    merge_footprints,
    select_retraversal_paths,
    traversal_footprint,
)
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisContext,
)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write(path: Path, value: Any) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_structural_exhaustion_summary(
    *,
    request_id: str | None,
    source_context_id: str,
    output_context_id: str,
    selected_path_count: int,
    new_eligible_premise_ids: list[str],
    selected_paper_expansion_count: int,
    paper_expansion_candidate_count: int,
    selected_traversal: str,
    output_context: str,
    context_delta_audit: str,
    context_delta_audit_id: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": "adaptive-graph-retraversal-summary-v1",
        "implementation_version": "adaptive-graph-retraversal-v2b",
        "status": "NO_STRUCTURALLY_NEW_ELIGIBLE_POSITIVE_PREMISE",
        "request_id": request_id,
        "source_context_id": source_context_id,
        "output_context_id": output_context_id,
        "selected_path_count": int(selected_path_count),
        "new_eligible_premise_ids": list(new_eligible_premise_ids),
        "new_eligible_premise_count": len(new_eligible_premise_ids),
        "structurally_new_eligible_premise_ids": [],
        "structurally_new_eligible_premise_count": 0,
        "selected_paper_expansion_count": int(selected_paper_expansion_count),
        "paper_expansion_candidate_count": int(paper_expansion_candidate_count),
        "selected_traversal": selected_traversal,
        "output_context": output_context,
        "context_delta_audit": context_delta_audit,
        "context_delta_audit_id": context_delta_audit_id,
        "reason": (
            "Retraversal produced a new context but no structurally new "
            "eligible positive premise. New paths/statements alone do not "
            "create positive-premise authority."
        ),
        "external_prior_art_as_positive_premise": False,
        "external_boundary_used_for_positive_path_selection": False,
        "canonical_graph_mutated": False,
    }


def run(label: str, cmd: list[str], output_dir: Path) -> None:
    print()
    print("=" * 88)
    print(label)
    print("=" * 88)
    print("$", " ".join(cmd))
    result = subprocess.run(cmd, text=True, capture_output=True)
    safe = "_".join(label.lower().split())
    (output_dir / f"{safe}.stdout.txt").write_text(
        result.stdout or "", encoding="utf-8"
    )
    (output_dir / f"{safe}.stderr.txt").write_text(
        result.stderr or "", encoding="utf-8"
    )
    if result.stdout:
        print(result.stdout.rstrip())
    if result.stderr:
        print(result.stderr.rstrip(), file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(
            f"{label} failed with return code {result.returncode}"
        )


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--source-context", required=True, type=Path)
    p.add_argument("--prior-traversal", action="append", required=True, type=Path)
    p.add_argument("--request", required=True, type=Path)
    p.add_argument("--model", required=True)
    p.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    p.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    p.add_argument("--objective", default="explain_connection")
    p.add_argument("--candidate-top-k", type=int, default=32)
    p.add_argument("--selected-top-k", type=int, default=6)
    p.add_argument("--node-map-k", type=int, default=30)
    p.add_argument("--endpoint-pair-k", type=int, default=24)
    p.add_argument("--max-depth-increment", type=int, default=2)
    p.add_argument("--max-depth-cap", type=int, default=16)
    p.add_argument("--min-new-edge-fraction", type=float, default=0.35)
    p.add_argument("--min-new-paper-fraction", type=float, default=0.25)
    p.add_argument("--max-prior-edge-jaccard", type=float, default=0.85)
    p.add_argument("--max-selected-edge-jaccard", type=float, default=0.85)
    p.add_argument("--paper-expansion-reserve", type=int, default=2)
    p.add_argument("--allow-top-n-fallback", action="store_true")
    p.add_argument("--disable-path-lineage-propagation", action="store_true")
    p.add_argument("--output-dir", required=True, type=Path)
    return p


def _traversal_command(
    *,
    template: dict[str, Any],
    args: argparse.Namespace,
    algorithm: str,
    output: Path,
) -> list[str]:
    source = str(template.get("source_query") or "").strip()
    target = str(template.get("target_query") or "").strip()
    stop = str(template.get("semantic_stop_query") or "").strip()

    if not source or not target:
        raise RuntimeError(
            "Graph retraversal requires source_query and target_query "
            "in the prior traversal."
        )

    constraints = template.get("constraints") or {}
    if not isinstance(constraints, dict):
        constraints = {}
    old_depth = int(
        template.get("effective_max_depth")
        or constraints.get("max_depth")
        or 8
    )
    new_depth = min(
        int(args.max_depth_cap),
        max(old_depth, old_depth + int(args.max_depth_increment)),
    )

    cmd = [
        sys.executable,
        "-m",
        "scripts.discovery.run_graph_traversal",
        "--corpus-id",
        str(template.get("corpus_id") or ""),
        "--domain-profile",
        str(template.get("domain_profile_id") or ""),
        "--data-root",
        str(template.get("data_root") or ""),
        "--mode",
        str(template.get("mode") or "mechanism"),
        "--algorithm",
        algorithm,
        "--source",
        source,
        "--target",
        target,
        "--node-map-k",
        str(args.node_map_k),
        "--endpoint-pair-k",
        str(args.endpoint_pair_k),
        "--top-k",
        str(args.candidate_top_k),
        "--max-depth",
        str(new_depth),
        "--include-candidate-paths",
        "--output",
        str(output),
    ]

    if algorithm == "semantic_stop":
        if not stop:
            raise RuntimeError(
                "semantic_stop retraversal requires semantic_stop_query."
            )
        cmd += [
            "--stop",
            stop,
            "--semantic-stop-max-depth",
            str(new_depth),
        ]
    return cmd


def _select(
    *,
    broad: dict[str, Any],
    prior: dict[str, set[str]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates = broad.get("candidate_paths", [])
    if not isinstance(candidates, list):
        candidates = []
    if not candidates:
        candidates = broad.get("paths", [])
    return select_retraversal_paths(
        [row for row in candidates if isinstance(row, dict)],
        prior=prior,
        top_k=args.selected_top_k,
        min_new_edge_fraction=args.min_new_edge_fraction,
        min_new_paper_fraction=args.min_new_paper_fraction,
        max_prior_edge_jaccard=args.max_prior_edge_jaccard,
        max_selected_edge_jaccard=args.max_selected_edge_jaccard,
        paper_expansion_reserve=args.paper_expansion_reserve,
    )


def main() -> int:
    args = parser().parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    context = HypothesisContext.model_validate_json(
        args.source_context.read_text(encoding="utf-8")
    )
    request = load(args.request)
    prior_paths = [path.expanduser().resolve() for path in args.prior_traversal]
    prior_payloads = [load(path) for path in prior_paths]
    if not prior_payloads:
        raise RuntimeError("At least one prior traversal is required.")

    if str(request.get("source_context_id") or "") != str(context.context_id):
        raise RuntimeError("Retraversal request/context lineage mismatch.")

    template = prior_payloads[-1]
    prior = merge_footprints(
        [
            context_footprint(context.model_dump(mode="json")),
            *[traversal_footprint(row) for row in prior_payloads],
        ]
    )

    original_algorithm = str(template.get("algorithm") or "top_n")
    broad_path = args.output_dir / "broad.traversal.json"
    run(
        "adaptive graph retraversal broad search",
        _traversal_command(
            template=template,
            args=args,
            algorithm=original_algorithm,
            output=broad_path,
        ),
        args.output_dir,
    )
    broad = load(broad_path)
    selected, selection = _select(broad=broad, prior=prior, args=args)

    fallback_used = False
    if (
        not selected
        and original_algorithm == "semantic_stop"
        and args.allow_top_n_fallback
    ):
        fallback_used = True
        fallback_path = args.output_dir / "broad.top_n.traversal.json"
        run(
            "adaptive graph retraversal top_n fallback",
            _traversal_command(
                template=template,
                args=args,
                algorithm="top_n",
                output=fallback_path,
            ),
            args.output_dir,
        )
        broad = load(fallback_path)
        selected, selection = _select(broad=broad, prior=prior, args=args)

    selection = {
        **selection,
        "source_context_id": str(context.context_id),
        "request_id": request.get("request_id"),
        "original_algorithm": original_algorithm,
        "effective_algorithm": broad.get("algorithm"),
        "top_n_fallback_used": fallback_used,
    }
    write(args.output_dir / "retraversal.selection.json", selection)

    if not selected:
        summary = {
            "schema_version": "adaptive-graph-retraversal-summary-v1",
        "implementation_version": "adaptive-graph-retraversal-v2b",
            "status": "NO_NOVEL_GROUNDED_NEIGHBORHOOD",
            "request_id": request.get("request_id"),
            "source_context_id": str(context.context_id),
            "selected_path_count": 0,
            "external_prior_art_as_positive_premise": False,
            "canonical_graph_mutated": False,
        }
        write(args.output_dir / "retraversal.summary.json", summary)
        print("No admissible new grounded neighborhood was found.")
        return 0

    selected_payload = materialize_selected_traversal(
        broad,
        selected_paths=selected,
        selection_report=selection,
        source_context_id=str(context.context_id),
    )
    selected_path = args.output_dir / "retraversal.selected_traversal.json"
    write(selected_path, selected_payload)

    packet = args.output_dir / "retraversal.explorer.packet.json"
    run(
        "adaptive graph retraversal explorer packet",
        [
            sys.executable,
            "-m",
            "scripts.discovery.build_explorer_packet",
            "--traversal-result",
            str(selected_path),
            "--domain-profile",
            str(context.domain_profile_id),
            "--question",
            str(context.question),
            "--objective",
            str(args.objective),
            "--output",
            str(packet),
        ],
        args.output_dir,
    )

    explorer_prefix = args.output_dir / "retraversal.explorer"
    explorer_report = Path(str(explorer_prefix) + ".report.json")
    cmd = [
        sys.executable,
        "-m",
        "scripts.discovery.run_graph_explorer",
        "--packet",
        str(packet),
        "--model",
        str(args.model),
        "--api-key-env",
        str(args.api_key_env),
        "--output-prefix",
        str(explorer_prefix),
        "--save-prompt",
    ]
    if args.base_url:
        cmd += ["--base-url", str(args.base_url)]
    run("adaptive graph retraversal explorer", cmd, args.output_dir)

    new_context_path = args.output_dir / "retraversal.context.json"
    compression = args.output_dir / "retraversal.evidence_compression.json"
    family = args.output_dir / "retraversal.evidence_family_diagnostics.json"
    lineage_diag = args.output_dir / "retraversal.path_lineage_diagnostics.json"
    propagation = args.output_dir / "retraversal.path_lineage_propagation.json"

    build_context_cmd = [
        sys.executable,
        "-m",
        "scripts.discovery.build_hypothesis_context",
        "--packet",
        str(packet),
        "--report",
        str(explorer_report),
        "--output",
        str(new_context_path),
        "--compression-output",
        str(compression),
        "--family-diagnostics-output",
        str(family),
        "--path-lineage-output",
        str(lineage_diag),
    ]
    if args.disable_path_lineage_propagation:
        build_context_cmd.append("--disable-path-lineage-propagation")
    else:
        build_context_cmd += [
            "--path-lineage-propagation-output",
            str(propagation),
        ]
    run(
        "adaptive graph retraversal context build",
        build_context_cmd,
        args.output_dir,
    )

    new_context = HypothesisContext.model_validate_json(
        new_context_path.read_text(encoding="utf-8")
    )
    if new_context.domain_profile_id != context.domain_profile_id:
        raise RuntimeError("Retraversal changed domain profile.")
    if new_context.corpus_id != context.corpus_id:
        raise RuntimeError("Retraversal changed corpus.")
    if new_context.question != context.question:
        raise RuntimeError("Retraversal changed research question.")
    if new_context.context_id == context.context_id:
        raise RuntimeError("Retraversal failed to create a new context.")

    source_context_payload = context.model_dump(mode="json")
    output_context_payload = new_context.model_dump(mode="json")
    old_eligible = eligible_premise_ids(source_context_payload)
    new_eligible = eligible_premise_ids(output_context_payload)
    new_only = sorted(new_eligible - old_eligible)

    context_delta = audit_context_delta(
        source_context=source_context_payload,
        output_context=output_context_payload,
    )
    context_delta_path = (
        args.output_dir / "retraversal.context_delta.json"
    )
    write(context_delta_path, context_delta)
    structural_new = list(
        context_delta.get(
            "structurally_new_eligible_premise_ids",
            [],
        )
    )
    if not structural_new:
        summary = build_structural_exhaustion_summary(
            request_id=request.get("request_id"),
            source_context_id=str(context.context_id),
            output_context_id=str(new_context.context_id),
            selected_path_count=len(selected),
            new_eligible_premise_ids=new_only,
            selected_paper_expansion_count=selection.get(
                "selected_paper_expansion_count", 0
            ),
            paper_expansion_candidate_count=selection.get(
                "paper_expansion_candidate_count", 0
            ),
            selected_traversal=str(selected_path),
            output_context=str(new_context_path),
            context_delta_audit=str(context_delta_path),
            context_delta_audit_id=context_delta.get("audit_id"),
        )
        write(
            args.output_dir / "retraversal.summary.json",
            summary,
        )

        print()
        print("===== ADAPTIVE GRAPH RETRAVERSAL V2B =====")
        print("source context:", context.context_id)
        print("output context:", new_context.context_id)
        print("selected paths:", len(selected))
        print("new eligible premises:", len(new_only))
        print("structurally new eligible premises: 0")
        print(
            "status:",
            "NO_STRUCTURALLY_NEW_ELIGIBLE_POSITIVE_PREMISE",
        )
        print("EXTERNAL_PRIOR_ART_AS_POSITIVE_PREMISE=False")
        print("CANONICAL_GRAPH_MUTATED=False")
        return 0

    lineage = {
        "schema_version": "adaptive-context-retraversal-lineage-v1",
        "request_id": request.get("request_id"),
        "source_context_id": context.context_id,
        "source_context_sha256": context.context_sha256,
        "output_context_id": new_context.context_id,
        "output_context_sha256": new_context.context_sha256,
        "source_traversal_paths": [str(path) for path in prior_paths],
        "selected_traversal": str(selected_path),
        "selection_id": selection.get("selection_id"),
        "selected_path_ids": selection.get("selected_path_ids", []),
        "source_eligible_premise_count": len(old_eligible),
        "output_eligible_premise_count": len(new_eligible),
        "new_eligible_premise_ids": new_only,
        "new_eligible_premise_count": len(new_only),
        "structurally_new_eligible_premise_ids": structural_new,
        "structurally_new_eligible_premise_count": len(structural_new),
        "context_delta_audit": str(context_delta_path),
        "context_delta_audit_id": context_delta.get("audit_id"),
        "external_prior_art_as_positive_premise": False,
        "external_boundary_used_for_positive_path_selection": False,
        "canonical_graph_mutated": False,
    }
    write(args.output_dir / "retraversal.lineage.json", lineage)

    summary = {
        "schema_version": "adaptive-graph-retraversal-summary-v1",
        "implementation_version": "adaptive-graph-retraversal-v2b",
        "status": "COMPLETE_NEW_GROUNDED_CONTEXT",
        "request_id": request.get("request_id"),
        "source_context_id": context.context_id,
        "output_context_id": new_context.context_id,
        "selected_path_count": len(selected),
        "new_eligible_premise_count": len(new_only),
        "structurally_new_eligible_premise_count": len(structural_new),
        "selected_paper_expansion_count": selection.get(
            "selected_paper_expansion_count", 0
        ),
        "paper_expansion_candidate_count": selection.get(
            "paper_expansion_candidate_count", 0
        ),
        "selected_traversal": str(selected_path),
        "output_context": str(new_context_path),
        "lineage": str(args.output_dir / "retraversal.lineage.json"),
        "external_prior_art_as_positive_premise": False,
        "canonical_graph_mutated": False,
    }
    write(args.output_dir / "retraversal.summary.json", summary)

    print()
    print("===== ADAPTIVE GRAPH RETRAVERSAL V2B =====")
    print("source context:", context.context_id)
    print("output context:", new_context.context_id)
    print("selected paths:", len(selected))
    print("new eligible premises:", len(new_only))
    print("structurally new eligible premises:", len(structural_new))
    print(
        "selected new-paper paths:",
        selection.get("selected_paper_expansion_count", 0),
        "/ candidates:",
        selection.get("paper_expansion_candidate_count", 0),
    )
    print("EXTERNAL_PRIOR_ART_AS_POSITIVE_PREMISE=False")
    print("CANONICAL_GRAPH_MUTATED=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
