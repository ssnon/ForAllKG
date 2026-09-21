from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.reframing.response_semantics_probe import (
    ScientificResponseSemanticsProbeReport,
)
from pipeline_core.discovery.reframing.response_semantics_validation import (
    build_response_semantics_candidate_freeze,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the candidate v2 response-semantics profile after adaptation "
            "and before untouched validation. No trigger decisions are changed."
        )
    )
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    probe_path = args.probe.resolve()
    probe = ScientificResponseSemanticsProbeReport.model_validate_json(
        probe_path.read_text(encoding="utf-8")
    )
    freeze = build_response_semantics_candidate_freeze(
        probe=probe,
        source_probe_path=probe_path,
        repository_root=args.repository_root,
    )
    output = args.output or probe_path.parent / "scientific_reframing_response_semantics_v2_freeze.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(freeze.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Scientific reframing response-semantics candidate freeze complete")
    print("LLM calls: 0")
    print("Candidate profile frozen: true")
    print("Adaptation tasks:", len(freeze.adaptation_task_ids))
    print("Adaptation domains:", ",".join(freeze.adaptation_domain_labels))
    print("Profile IDs:", ",".join(freeze.semantics_profile_ids))
    print("Candidate semantics fingerprint:", freeze.candidate_semantics_fingerprint)
    print("Adaptation data excluded from untouched validation: true")
    print("Production trigger semantics modified: false")
    print("Output:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
