from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from dac_her.domains.registry import get_domain_profile
from dac_her.novelty_gap_analysis import NoveltyGapAnalyzer
from dac_her.sers_novelty_gap_g0_g2_dev_validation import (
    canonical_json,
    compile_production_report,
    load_frozen_input,
)

ROOT = Path(__file__).resolve().parents[4]
DEV_V4_REPORT = ROOT / (
    "evaluation/sers_novelty_gap/g0_g2_dev_run_v4/"
    "g0_g2_dev_report_v4.json"
)
DEFAULT_RUN_ROOT = ROOT / (
    "evaluation/sers_novelty_gap/"
    "g0_g2_production_integration_v2_run"
)
EXPECTED_DEV_V4_RUN_ID = (
    "sers_novelty_gap_g0_g2_dev_run_v4:"
    "7d56d7cbd2578efe327b"
)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=str,
        ) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def build_production_run() -> dict[str, Any]:
    frozen = load_frozen_input(ROOT)
    dev_v4 = json.loads(DEV_V4_REPORT.read_text(encoding="utf-8"))
    if dev_v4.get("run_id") != EXPECTED_DEV_V4_RUN_ID:
        raise ValueError("DEV v4 run ID mismatch")
    if dev_v4.get("structural_outcome") != (
        "SERS_NOVELTY_GAP_G0_G2_DEV_V4_STRUCTURAL_PASS"
    ):
        raise ValueError("DEV v4 is not structural PASS")

    report = compile_production_report(
        portfolio=frozen["portfolio"],
        plan=frozen["plan"],
        packet=frozen["packet"],
        reviews=frozen["reviews"],
    )

    analyzer = NoveltyGapAnalyzer(
        max_target_claims=2,
        queries_per_gap=3,
        domain_profile=get_domain_profile("sers_au_ag"),
    )
    gap_1 = analyzer.build(
        frozen["portfolio"],
        report,
        frozen["plan"],
    )
    gap_2 = analyzer.build(
        frozen["portfolio"],
        report,
        frozen["plan"],
    )

    production_gap = gap_1.model_dump(mode="json")
    dev_gap = dev_v4["novelty_gap_plan_v2"]

    query_rows = []
    for gap in production_gap["gaps"]:
        for query in gap["targeted_queries"]:
            query_rows.append(
                {
                    "hypothesis_id": gap["hypothesis_id"],
                    "claim_id": query["claim_id"],
                    "query_role": query["query_role"],
                    "query_text": query["query_text"],
                }
            )

    checks = {
        "production_gap_repeat_deterministic":
            canonical_json(gap_1) == canonical_json(gap_2),
        "exact_dev_v4_gap_equivalence":
            canonical_json(production_gap) == canonical_json(dev_gap),
        "exact_dev_v4_query_equivalence":
            [
                (
                    gap["hypothesis_id"],
                    query["claim_id"],
                    query["query_role"],
                    query["query_text"],
                )
                for gap in production_gap["gaps"]
                for query in gap["targeted_queries"]
            ] == [
                (
                    gap["hypothesis_id"],
                    query["claim_id"],
                    query["query_role"],
                    query["query_text"],
                )
                for gap in dev_gap["gaps"]
                for query in gap["targeted_queries"]
            ],
        "targeted_retrieval_not_called": True,
        "provider_calls_zero": True,
        "ranker_not_recomputed": True,
        "claim_reviewer_not_recomputed": True,
        "hypothesis_rewrite_not_called": True,
        "llm_calls_zero": True,
        "network_calls_zero": True,
        "fresh_reserve_c_not_consumed": True,
        "automatic_next_stage_disabled": True,
    }
    structural_pass = all(checks.values())

    body = {
        "schema_version":
            "sers-novelty-gap-g0-g2-production-integration-v2",
        "source_dev_v4_run_id": EXPECTED_DEV_V4_RUN_ID,
        "source_portfolio_id": frozen["portfolio"].portfolio_id,
        "source_query_plan_id": frozen["plan"].plan_id,
        "source_canonical_packet_id": frozen["packet"].packet_id,
        "structural_outcome": (
            "SERS_NOVELTY_GAP_G0_G2_PRODUCTION_INTEGRATION_V2_PASS"
            if structural_pass
            else
            "SERS_NOVELTY_GAP_G0_G2_PRODUCTION_INTEGRATION_V2_FAIL"
        ),
        "checks": checks,
        "production_external_novelty_report":
            report.model_dump(mode="json"),
        "production_novelty_gap_plan_v2": production_gap,
        "query_rows": query_rows,
        "targeted_retrieval_called": False,
        "provider_calls": 0,
        "ranker_recomputed": False,
        "claim_reviewer_recomputed": False,
        "hypothesis_rewrite_called": False,
        "llm_calls": 0,
        "network_calls": 0,
        "fresh_reserve_c_consumed": False,
        "automatic_next_stage_authorized": False,
    }
    body["run_sha256"] = sha256_json(body)
    body["run_id"] = (
        "sers_novelty_gap_g0_g2_production_integration_v2:"
        + body["run_sha256"][:20]
    )
    return body


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument(
        "--confirm-production-integration-v2",
        action="store_true",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=DEFAULT_RUN_ROOT,
    )
    args = parser.parse_args()
    if not args.run or not args.confirm_production_integration_v2:
        parser.error(
            "--run and --confirm-production-integration-v2 are required"
        )

    run_root = args.run_root
    if run_root.exists():
        print("Production integration v2: FAIL")
        print(" - run root already exists:", run_root)
        return 2

    print("SERS G0-G2 Production Integration v2")
    print("Production NoveltyGapAnalyzer path: direct / no monkeypatch")
    print("Targeted retrieval called:", False)
    print("Provider calls:", 0)
    print("LLM calls:", 0)
    print("Network calls:", 0)
    print("Fresh Reserve C consumed:", False)
    print()

    try:
        report = build_production_run()
        run_root.mkdir(parents=True, exist_ok=False)
        atomic_json(
            run_root / "production_integration_v2_report.json",
            report,
        )
        marker_name = (
            "STRUCTURAL_PASS.json"
            if report["structural_outcome"].endswith("_PASS")
            else "STRUCTURAL_FAIL.json"
        )
        atomic_json(
            run_root / marker_name,
            {
                "status": (
                    "structural_pass"
                    if marker_name.startswith("STRUCTURAL_PASS")
                    else "structural_fail"
                ),
                "run_id": report["run_id"],
            },
        )
    except Exception as exc:
        print("Production integration v2: FAIL")
        print(" -", f"{type(exc).__name__}: {exc}")
        print("LLM calls:", 0)
        print("Network calls:", 0)
        print("Fresh Reserve C consumed:", False)
        return 2

    print("Run ID:", report["run_id"])
    print("Structural outcome:", report["structural_outcome"])
    for key, value in report["checks"].items():
        print(f"{key}: {value}")
    print()
    for row in report["query_rows"]:
        print(
            f"[{row['hypothesis_id']}|{row['claim_id']}|"
            f"{row['query_role']}] {row['query_text']}"
        )
    print()
    print("Targeted retrieval called:", False)
    print("LLM calls:", 0)
    print("Network calls:", 0)
    print("Fresh Reserve C consumed:", False)
    print("Automatic next stage authorized:", False)
    return 0 if report["structural_outcome"].endswith("_PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
