from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.question_task_preservation_shadow_artifact import (
    build_task_preservation_shadow_artifact,
)


def _load_json(
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
            f"JSON root must be an object: {path}"
        )

    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a shadow-only pair-level question-task-preservation "
            "report from DiscoveryBundle semantic conflicts and a "
            "candidate responsiveness audit."
        )
    )

    parser.add_argument(
        "--raw-conflicts",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--responsiveness-audit",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--group",
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    payload = (
        build_task_preservation_shadow_artifact(
            raw_conflict_payload=(
                _load_json(
                    args.raw_conflicts
                )
            ),
            responsiveness_payload=(
                _load_json(
                    args.responsiveness_audit
                )
            ),
            group=args.group,
            raw_conflict_source=str(
                args.raw_conflicts
            ),
            responsiveness_source=str(
                args.responsiveness_audit
            ),
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
        "Raw observations:",
        payload[
            "raw_observation_count"
        ],
    )

    print(
        "Unique pairs:",
        payload[
            "unique_pair_count"
        ],
    )

    print(
        "Assessed pairs:",
        payload[
            "assessed_pair_count"
        ],
    )

    print(
        "Replacement pairs:",
        payload[
            "replacement_pair_count"
        ],
    )

    print(
        "Missing-assessment pairs:",
        payload[
            "missing_assessment_pair_count"
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
