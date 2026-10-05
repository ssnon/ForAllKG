from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from pipeline_core.discovery.hypothesis_contracts import HypothesisContext, HypothesisPortfolio
from pipeline_core.discovery.prospective_identification_materialization_shadow import compact_shadow_record, run_prospective_identification_shadow


def write(path: Path, payload) -> None:
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--context", required=True, type=Path)
    p.add_argument("--portfolio", required=True, type=Path)
    p.add_argument("--source-stage", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    p.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    p.add_argument("--parse-retries", type=int, default=3)
    p.add_argument("--output-dir", required=True, type=Path)
    args = p.parse_args()

    context = HypothesisContext.model_validate_json(args.context.read_text(encoding="utf-8"))
    portfolio = HypothesisPortfolio.model_validate_json(args.portfolio.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, candidate in enumerate(portfolio.hypotheses, start=1):
        prefix = args.output_dir / f"{i:02d}_{str(candidate.hypothesis_id).split(':')[-1]}"
        artifact = run_prospective_identification_shadow(
            context=context,
            candidate=candidate,
            source_stage=args.source_stage,
            model=args.model,
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            parse_retries=args.parse_retries,
            output_prefix=prefix,
        )
        rows.append({"hypothesis_id": str(candidate.hypothesis_id), **compact_shadow_record(artifact)})
    summary = {
        "schema_version": "prospective-identification-materialization-shadow-summary-v1",
        "source_stage": args.source_stage,
        "candidate_count": len(rows),
        "rows": rows,
        "production_selection_changed": False,
        "shadow_has_generation_authority": False,
        "shadow_has_selection_authority": False,
    }
    write(args.output_dir / "summary.json", summary)
    print("Prospective identification materialization shadow complete")
    print("candidates:", len(rows))
    for row in rows:
        print(row["hypothesis_id"], "|", row["status"], "| prospective=", row["prospective_identifiability"], "| would_abstain=", row["would_abstain_if_authoritative"])
    print("PRODUCTION_SELECTION_CHANGED=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
