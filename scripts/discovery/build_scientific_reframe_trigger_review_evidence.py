from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.corpus.semantic_ir.grounded_scope import (
    resolve_grounded_semantic_task_scope,
)
from pipeline_core.discovery.hypothesis_contracts import HypothesisContext
from pipeline_core.discovery.reframing.evidence_tension import (
    extract_evidence_level_tensions,
)
from pipeline_core.discovery.reframing.reframe_evidence import (
    build_scientific_reframe_evidence_packet,
)
from pipeline_core.discovery.reframing.review_evidence import (
    build_trigger_review_evidence_pack,
)
from pipeline_core.discovery.reframing.trigger_calibration import (
    TriggerReviewTemplate,
)


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a zero-LLM, evidence-complete review pack for manual calibration "
            "of scientific reframing trigger allocation."
        )
    )
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--benchmark-root", required=True, type=Path)
    parser.add_argument(
        "--root",
        action="append",
        required=True,
        help="Extraction root. Repeat for merged/legacy corpus roots.",
    )
    parser.add_argument(
        "--scope-mode",
        choices=("premise_and_gap", "premise_only"),
        default="premise_and_gap",
    )
    parser.add_argument(
        "--duplicate-paper-policy",
        choices=("error", "latest_attempt"),
        default="error",
    )
    parser.add_argument(
        "--cross-root-duplicate-policy",
        choices=("error", "prefer_last_root"),
        default="error",
    )
    parser.add_argument("--max-condition-examples", type=int, default=30)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    template = TriggerReviewTemplate.model_validate(_load(args.template))
    benchmark_root = args.benchmark_root.resolve()
    if not benchmark_root.is_dir():
        raise FileNotFoundError(f"benchmark root not found: {benchmark_root}")

    task_keys = sorted({row.task_key for row in template.rows})
    task_evidence = {}
    for task_key in task_keys:
        canonical_dir = benchmark_root / task_key / "canonical"
        packet_path = canonical_dir / "explorer.packet.json"
        report_path = canonical_dir / "explorer.report.json"
        context_path = canonical_dir / "hypothesis.context.json"
        missing = [
            path.name
            for path in (packet_path, report_path, context_path)
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"{task_key}: missing canonical artifact(s): " + ", ".join(missing)
            )

        grounded, bundles = resolve_grounded_semantic_task_scope(
            packet_path=packet_path,
            context_path=context_path,
            corpus_roots=list(args.root),
            scope_mode=args.scope_mode,
            duplicate_paper_policy=args.duplicate_paper_policy,
            cross_root_duplicate_policy=args.cross_root_duplicate_policy,
        )
        context = HypothesisContext.model_validate(_load(context_path))
        evidence = build_scientific_reframe_evidence_packet(
            context=context,
            grounded=grounded,
            bundles=bundles,
            max_condition_examples=args.max_condition_examples,
        )
        tensions = extract_evidence_level_tensions(
            explorer_report=_load(report_path),
            evidence=evidence,
            explorer_packet=_load(packet_path),
        )
        task_evidence[task_key] = (evidence, tensions)
        print(
            f"{task_key}: premises={len(evidence.premise_statements)}; "
            f"gaps={len(evidence.gap_statements)}; "
            f"tensions={len(tensions.witnesses)}; "
            f"chunks={evidence.grounded_source_chunk_count}"
        )

    pack = build_trigger_review_evidence_pack(
        template=template,
        benchmark_root=str(benchmark_root),
        extraction_roots=list(args.root),
        task_evidence=task_evidence,
    )
    output = args.output or (
        benchmark_root / "scientific_reframing_trigger_review_evidence.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(pack.model_dump_json(indent=2) + "\n", encoding="utf-8")

    print()
    print("Scientific reframing trigger review evidence pack built")
    print("LLM calls: 0")
    print("Tasks:", pack.task_count)
    print("Operator review rows:", pack.operator_review_row_count)
    print("Labels prefilled: false")
    print("Current trigger decisions are displayed but are not ground truth.")
    print("Output:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
