
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisPortfolio,
)
from pipeline_core.discovery.sers_effective_gen1_certification_closeout import (
    build_certification_closeout,
    render_markdown,
)


def load(path: Path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--effective-portfolio", required=True, type=Path)
    p.add_argument("--n9-intake", required=True, type=Path)
    p.add_argument("--n9-full", required=True, type=Path)
    p.add_argument("--candidate-gate", required=True, type=Path)
    p.add_argument("--production-gate", required=True, type=Path)
    p.add_argument("--certification-report", required=True, type=Path)
    p.add_argument("--certified-portfolio", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--gen0-projection", required=True, type=Path)
    p.add_argument("--gen0-baseline-comparison", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    args = p.parse_args()

    effective = HypothesisPortfolio.model_validate_json(
        args.effective_portfolio.read_text(encoding="utf-8")
    )
    certified = HypothesisPortfolio.model_validate_json(
        args.certified_portfolio.read_text(encoding="utf-8")
    )

    report = build_certification_closeout(
        effective_portfolio=effective,
        n9_intake=load(args.n9_intake),
        n9_full=load(args.n9_full),
        candidate_gate=load(args.candidate_gate),
        production_gate=load(args.production_gate),
        certification=load(args.certification_report),
        certified_portfolio=certified,
        manifest=load(args.manifest),
        gen0_projection=load(args.gen0_projection),
        gen0_baseline_comparison=load(
            args.gen0_baseline_comparison
        ),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "sers_research_candidate.freeze.json"
    md_path = args.output_dir / "SERS_RESEARCH_CANDIDATE_FINAL_REPORT.md"

    json_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(
        render_markdown(report),
        encoding="utf-8",
    )

    print("===== SERS EFFECTIVE GEN1 CERTIFICATION CLOSEOUT =====")
    print("N9 intake:", report["n9_intake"]["claim_state_counts"])
    print("N9 full:", report["n9_full"]["claim_state_counts"])
    print("N10:", report["n10_certification"]["counts"])
    print("N10 reference:", report["n10_reference_state_counts"])
    print("Certified IDs:", report["certified_hypothesis_ids"])
    for row in report["final_candidates"]:
        print(
            row["hypothesis_id"],
            "|", row["selection_class"],
            "| positive=", row[
                "positive_nonobviousness_authority"
            ],
            "|", row["certification_status"],
            "| candidate=", row["research_candidate_state"],
            "| N10-ref=", row["n10_reference_state"],
        )
    print("CERTIFICATION_ONLY=True")
    print("PRODUCTION_SELECTION_CHANGED=False")
    print("CANONICAL_GRAPH_MUTATED=False")
    print("Output:", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
