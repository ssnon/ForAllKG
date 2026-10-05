from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from pipeline_core.discovery.adaptive_yield_evaluation import AdaptiveYieldCaseAudit
from pipeline_core.discovery.hypothesis_contracts import HypothesisPortfolio


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def write(path: Path, value: Any) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def one_card_portfolio(source_path: Path, hypothesis_id: str) -> HypothesisPortfolio:
    source = HypothesisPortfolio.model_validate_json(
        source_path.read_text(encoding="utf-8")
    )
    rows = [
        x for x in source.hypotheses
        if str(x.hypothesis_id) == str(hypothesis_id)
    ]
    if len(rows) != 1:
        raise RuntimeError(f"hypothesis {hypothesis_id} not unique in {source_path}")
    payload = source.model_dump(mode="json")
    payload["portfolio_id"] = "independent_stress:" + str(rows[0].hypothesis_id).split(":")[-1]
    payload["hypotheses"] = [rows[0].model_dump(mode="json")]
    payload["abstention_reason"] = None
    return HypothesisPortfolio.model_validate(payload)


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Independently re-search frozen adaptive-yield candidates using "
            "fresh query decomposition and no prior-art memory."
        )
    )
    p.add_argument("--case-audit", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--model", required=True)
    p.add_argument("--domain-profile", default=None)
    p.add_argument("--provider-plan", type=Path, default=None)
    p.add_argument("--providers", default="openalex,crossref")
    p.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    p.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    p.add_argument("--results-per-query", type=int, default=24)
    p.add_argument("--max-ranked-works", type=int, default=20)
    p.add_argument("--parse-retries", type=int, default=3)
    args = p.parse_args()

    audit = AdaptiveYieldCaseAudit.model_validate_json(
        args.case_audit.read_text(encoding="utf-8")
    )
    domain = args.domain_profile or audit.domain_profile_id
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    unique = {}
    for row in audit.candidates:
        unique.setdefault(row.scientific_fingerprint, row)

    rows = []
    for index, (fingerprint, row) in enumerate(sorted(unique.items()), start=1):
        candidate_dir = out / f"candidate_{index:02d}"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        source_path = Path(row.portfolio_path).expanduser().resolve()
        portfolio = one_card_portfolio(source_path, row.hypothesis_id)
        portfolio_path = candidate_dir / "portfolio.json"
        write(portfolio_path, portfolio)
        prefix = candidate_dir / "independent_external"

        cmd = [
            sys.executable,
            "-m",
            "scripts.discovery.run_external_novelty",
            "--portfolio", str(portfolio_path),
            "--domain-profile", domain,
            "--model", args.model,
            "--api-key-env", args.api_key_env,
            "--results-per-query", str(args.results_per_query),
            "--max-ranked-works", str(args.max_ranked_works),
            "--parse-retries", str(args.parse_retries),
            "--pre-review-metadata-resolution",
            "--pre-review-coverage-shadow",
            "--downstream-gate-shadow",
            "--independent-evidence-review",
            "--source-bound-topology-shadow",
            "--output-prefix", str(prefix),
            "--save-prompts",
        ]
        if args.provider_plan is not None:
            cmd += ["--provider-plan", str(args.provider_plan)]
        else:
            cmd += ["--providers", args.providers]
        if args.base_url:
            cmd += ["--base-url", args.base_url]

        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise RuntimeError(
                f"independent novelty candidate {index} failed: return code {result.returncode}"
            )

        report_path = Path(str(prefix) + ".report.json")
        report = load(report_path)
        cards = report.get("cards", []) or []
        if len(cards) != 1:
            raise RuntimeError(f"expected one external novelty card, got {len(cards)}")
        status = str(cards[0].get("status") or "UNKNOWN")
        rows.append(
            {
                "scientific_fingerprint": fingerprint,
                "source_record_id": row.record_id,
                "source_hypothesis_id": row.hypothesis_id,
                "status": status,
                "report": str(report_path),
                "fresh_query_decomposition": True,
                "prior_art_memory_used": False,
                "independent_evidence_review": True,
            }
        )

    counts = Counter(x["status"] for x in rows)
    payload = {
        "schema_version": "adaptive-independent-novelty-stress-v1",
        "case_id": audit.case_id,
        "candidate_count": len(rows),
        "status_counts": dict(counts),
        "candidates": rows,
        "fresh_query_decomposition": True,
        "prior_art_memory_used": False,
        "candidate_freeze_precedes_search": True,
        "external_prior_art_as_positive_premise": False,
        "novelty_authority_created": False,
        "production_selection_changed": False,
    }
    write(out / "independent_novelty.summary.json", payload)
    print("Adaptive independent novelty stress complete")
    print("candidates:", len(rows))
    print("statuses:", dict(counts))
    print("prior-art memory used: false")
    print("output:", out / "independent_novelty.summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
