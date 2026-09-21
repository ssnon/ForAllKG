from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.reframing.response_semantics_probe import (
    load_response_semantics_probe,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare frozen v1 response recognition with a shadow generic-core + "
            "domain-adapter response-semantics proposal. No trigger decision changes."
        )
    )
    parser.add_argument("--diagnostics", required=True)
    parser.add_argument("--output", default=None)
    return parser


def main() -> int:
    args = _parser().parse_args()
    diagnostics_path = Path(args.diagnostics).resolve()
    report = load_response_semantics_probe(diagnostics_path)
    output = (
        Path(args.output).resolve()
        if args.output
        else diagnostics_path.parent / "scientific_reframing_response_semantics_probe.json"
    )
    output.write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Scientific reframing response-semantics adaptation probe complete")
    print("LLM calls: 0")
    print(f"Domain: {report.validation_domain_label}")
    print(f"Profile: {report.semantics_profile_id}")
    print(f"Premises: {report.premise_count}")
    print(
        "Response-bearing premises/statements: "
        f"frozen_v1={report.v1_response_match_count}; "
        f"candidate_v2={report.candidate_response_match_count}"
    )
    if report.candidate_law_signal_counts:
        print("Candidate response-law signals:")
        for kind, count in report.candidate_law_signal_counts.items():
            print(f"  {kind}: {count}")
    else:
        print("Candidate response-law signals: none")
    print(
        "Candidate LATENT response-support floor: "
        f"premises={report.candidate_latent_response_support_premise_count}; "
        f"papers={report.candidate_latent_support_paper_count}; "
        f"floor_met={str(report.candidate_latent_current_floor_met).lower()}"
    )
    print("This task is adaptation data after this design use; it is not untouched validation for v2.")
    print("Frozen trigger decisions were not changed.")
    print(f"Output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
