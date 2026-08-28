from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.question_axis_responsiveness_llm import (
    OpenRouterQuestionAxisResponsivenessBackend,
)
from pipeline_core.discovery.question_task_conflict_responsiveness import (
    conflict_responsiveness_artifact,
    evaluate_conflict_candidates_shadow,
)


def load_json(
    path: Path,
) -> dict:
    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError(
            f"JSON root must be object: {path}"
        )

    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Shadow-only two-pass Question↔candidate responsiveness "
            "review for candidate units participating in observed "
            "DiscoveryBundle semantic conflicts."
        )
    )

    parser.add_argument(
        "--raw-conflicts",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--traversal",
        type=Path,
        action="append",
        required=True,
    )

    parser.add_argument(
        "--group",
        required=True,
    )

    parser.add_argument(
        "--question",
        required=True,
    )

    parser.add_argument(
        "--model",
        required=True,
    )

    parser.add_argument(
        "--provider",
        default=None,
    )

    parser.add_argument(
        "--reasoning-effort",
        default="medium",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--telemetry-path",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    raw = load_json(
        args.raw_conflicts
    )

    traversals = [
        load_json(path)
        for path in args.traversal
    ]

    backend = (
        OpenRouterQuestionAxisResponsivenessBackend(
            model=args.model,
            provider=args.provider,
            temperature=args.temperature,
            reasoning_effort=(
                args.reasoning_effort
            ),
            telemetry_path=(
                args.telemetry_path
            ),
            telemetry_context={
                "stage":
                    "question_task_conflict_responsiveness_shadow",

                "group":
                    args.group,
            },
        )
    )

    debug_prefix = None

    if args.debug_dir is not None:
        args.debug_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        debug_prefix = str(
            args.debug_dir
            / "candidate"
        )

    results = (
        evaluate_conflict_candidates_shadow(
            group=args.group,
            question=args.question,
            raw_conflict_payload=raw,
            traversal_payloads=traversals,
            backend=backend,
            debug_path_prefix=(
                debug_prefix
            ),
        )
    )

    payload = (
        conflict_responsiveness_artifact(
            group=args.group,
            question=args.question,
            results=results,
            raw_conflict_source=str(
                args.raw_conflicts
            ),
            traversal_sources=[
                str(path)
                for path in args.traversal
            ],
        )
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "Conflict candidates reviewed:",
        payload[
            "candidate_count"
        ],
    )

    print(
        "LLM calls:",
        2
        * payload[
            "candidate_count"
        ],
    )

    print(
        "Saved:",
        args.output,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
