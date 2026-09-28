from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx

from pipeline_core.discovery.direct_relationpattern_task_shadow import (
    build_report,
    candidate_from_relationpattern_mapping,
    evaluate_candidates,
    relationpattern_mapping_from_graph_node,
)
from pipeline_core.discovery.node_mapping import NodeMapper, QueryConcept
from pipeline_core.discovery.question_axis_responsiveness_llm import (
    OpenRouterQuestionAxisResponsivenessBackend,
)


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _clean(value: object) -> str:
    return " ".join(str(value or "").split()).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Shadow-only direct accepted RelationPattern retrieval "
            "+ task-responsiveness evaluation."
        )
    )
    parser.add_argument("--final-traversal", required=True, type=Path)
    parser.add_argument("--question", required=True)
    parser.add_argument("--retrieval-source", required=True)
    parser.add_argument("--retrieval-target", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--retrieval-top-k", type=int, default=20)
    parser.add_argument("--debug-dir", type=Path, default=None)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.retrieval_top_k < 1:
        raise ValueError("--retrieval-top-k must be >= 1")

    traversal = _json(args.final_traversal)

    data_root = Path(str(traversal["data_root"]))
    if not data_root.is_absolute():
        data_root = Path.cwd() / data_root

    corpus_id = str(traversal["corpus_id"])
    mode = str(traversal["mode"])

    corpus_root = data_root / "corpus" / corpus_id / mode
    graph_path = corpus_root / "graph.graphml"
    index_dir = corpus_root / "navigation" / "node_index"

    if not graph_path.is_file():
        raise FileNotFoundError(graph_path)

    mapper = NodeMapper.from_directory(index_dir)
    graph = nx.read_graphml(graph_path, force_multigraph=True)

    source = _clean(args.retrieval_source)
    target = _clean(args.retrieval_target)
    question = _clean(args.question)

    if not source or not target:
        raise ValueError("requested task endpoints must be non-empty")
    if not question:
        raise ValueError("question must be non-empty")

    joint_query = _clean(source + " " + target)

    matches = mapper.map(
        QueryConcept(
            text=joint_query,
            top_k=args.retrieval_top_k,
            allow_candidates=False,
            allow_alignment_hubs=False,
        )
    )

    candidates = []
    validation_rejections = []

    for rank, match in enumerate(matches, start=1):
        node_id = str(match.node_id)
        if node_id not in graph:
            continue

        attrs = dict(graph.nodes[node_id])

        if str(attrs.get("retention_lane", "")).strip() != "accepted_pattern":
            continue
        if str(attrs.get("concept_type", "")).strip() != "RelationPattern":
            continue

        row = relationpattern_mapping_from_graph_node(
            node_id=node_id,
            attrs=attrs,
        )

        try:
            candidate = candidate_from_relationpattern_mapping(
                row=row,
                joint_query=joint_query,
                retrieval_rank=rank,
                semantic_similarity=match.semantic_similarity,
            )
        except Exception as exc:
            validation_rejections.append(
                {
                    "node_id": node_id,
                    "retrieval_rank": rank,
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            continue

        candidates.append(candidate)

    if args.debug_dir is not None:
        args.debug_dir.mkdir(parents=True, exist_ok=True)

    backend = OpenRouterQuestionAxisResponsivenessBackend(
        model=args.model,
        provider=args.provider,
        temperature=0.0,
        reasoning_effort="medium",
        telemetry_context={
            "stage": "direct_relationpattern_task_shadow",
            "shadow_only": True,
            "production_selection_changed": False,
        },
    )

    assessments = evaluate_candidates(
        question=question,
        candidates=candidates,
        backend=backend,
        debug_dir=(
            str(args.debug_dir)
            if args.debug_dir is not None
            else None
        ),
    )

    report = build_report(
        question=question,
        retrieval_source=source,
        retrieval_target=target,
        joint_query=joint_query,
        retrieval_top_k=args.retrieval_top_k,
        retrieved_node_count=len(matches),
        validation_rejection_count=len(validation_rejections),
        candidates=candidates,
        assessments=assessments,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        report.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    print("=== DIRECT RELATIONPATTERN TASK SHADOW ===")
    print("joint query:", joint_query)
    print("retrieved nodes:", len(matches))
    print("accepted RelationPatterns:", len(candidates))
    print("validation rejections:", len(validation_rejections))
    print("stable DIRECT:", report.stable_direct_count)
    print("stable SUBORDINATE:", report.stable_subordinate_count)
    print("stable TASK_REPLACING:", report.stable_task_replacing_count)
    print("unresolved:", report.unresolved_count)

    by_id = {row.candidate_id: row for row in assessments}

    print("\n=== CANDIDATES ===")
    for candidate in candidates:
        assessment = by_id[candidate.candidate_id]
        component = candidate.relation_component

        print(
            f"\nrank={candidate.retrieval_rank} "
            f"sim={candidate.semantic_similarity:.4f}"
        )
        print(
            " ",
            component.subject,
            "--" + component.relation + "-->",
            component.object,
        )
        print(
            " task:",
            assessment.task_class,
            "| stable=",
            assessment.decision_stable,
            "| status=",
            assessment.stable_status,
            "| role=",
            assessment.stable_role,
        )

    print("\nresponsive candidate ids:")
    for candidate_id in report.responsive_candidate_ids:
        print(" ", candidate_id)

    print("\nshadow_only:", report.shadow_only)
    print(
        "production_selection_changed:",
        report.production_selection_changed,
    )
    print(
        "novelty_authority_created:",
        report.novelty_authority_created,
    )
    print(
        "positive_premise_authority_created:",
        report.positive_premise_authority_created,
    )
    print("report:", args.output)

    if validation_rejections:
        rejects_path = Path(
            str(args.output)
            + ".validation_rejections.json"
        )
        rejects_path.write_text(
            json.dumps(
                validation_rejections,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print("validation rejections:", rejects_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
