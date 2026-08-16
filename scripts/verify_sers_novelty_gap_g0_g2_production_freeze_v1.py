from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FREEZE_ROOT = ROOT / (
    "evaluation/sers_novelty_gap/g0_g2_production_freeze_v1"
)

EXPECTED_DEV_V4_RUN_ID = (
    "sers_novelty_gap_g0_g2_dev_run_v4:"
    "7d56d7cbd2578efe327b"
)
EXPECTED_PROD_V2_RUN_ID = (
    "sers_novelty_gap_g0_g2_production_integration_v2:"
    "83b9f9724d3182fedc5c"
)
EXPECTED_PROD_OUTCOME = (
    "SERS_NOVELTY_GAP_G0_G2_PRODUCTION_INTEGRATION_V2_PASS"
)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        canonical_json(value).encode("utf-8")
    ).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def manifest_body(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in manifest.items()
        if key not in {"freeze_id", "manifest_sha256"}
    }


def verify_freeze(
    root: Path,
    freeze_root: Path,
) -> tuple[bool, list[str], dict[str, Any]]:
    issues: list[str] = []

    manifest_path = freeze_root / "freeze_manifest.json"
    ready_path = freeze_root / "FREEZE_READY.json"
    if not manifest_path.is_file():
        return False, ["freeze_manifest.json missing"], {}
    if not ready_path.is_file():
        return False, ["FREEZE_READY.json missing"], {}

    manifest = load_json(manifest_path)
    ready = load_json(ready_path)

    body_sha = sha256_json(manifest_body(manifest))
    if manifest.get("manifest_sha256") != body_sha:
        issues.append("manifest_sha256 mismatch")
    expected_freeze_id = (
        "sers_novelty_gap_g0_g2_production_freeze_v1:"
        + body_sha[:20]
    )
    if manifest.get("freeze_id") != expected_freeze_id:
        issues.append("freeze_id mismatch")
    if ready.get("freeze_id") != manifest.get("freeze_id"):
        issues.append("FREEZE_READY freeze_id mismatch")
    if ready.get("manifest_sha256") != body_sha:
        issues.append("FREEZE_READY manifest_sha256 mismatch")
    if ready.get("status") != "FREEZE_READY":
        issues.append("FREEZE_READY status mismatch")

    for rel, meta in manifest.get("critical_files", {}).items():
        path = root / rel
        if not path.is_file():
            issues.append(f"critical file missing: {rel}")
            continue
        data = path.read_bytes()
        if sha256_bytes(data) != meta.get("sha256"):
            issues.append(f"critical file SHA256 mismatch: {rel}")
        if len(data) != meta.get("size_bytes"):
            issues.append(f"critical file size mismatch: {rel}")

    for name, meta in manifest.get("evidence_files", {}).items():
        path = freeze_root / name
        if not path.is_file():
            issues.append(f"freeze evidence missing: {name}")
            continue
        data = path.read_bytes()
        if sha256_bytes(data) != meta.get("sha256"):
            issues.append(f"freeze evidence SHA256 mismatch: {name}")
        if len(data) != meta.get("size_bytes"):
            issues.append(f"freeze evidence size mismatch: {name}")

    dev_path = freeze_root / "source_dev_v4_report.json"
    prod_path = freeze_root / "source_production_integration_v2_report.json"
    if dev_path.is_file() and prod_path.is_file():
        dev = load_json(dev_path)
        prod = load_json(prod_path)

        if dev.get("run_id") != EXPECTED_DEV_V4_RUN_ID:
            issues.append("DEV v4 run ID mismatch")
        if dev.get("structural_outcome") != (
            "SERS_NOVELTY_GAP_G0_G2_DEV_V4_STRUCTURAL_PASS"
        ):
            issues.append("DEV v4 structural outcome is not PASS")

        if prod.get("run_id") != EXPECTED_PROD_V2_RUN_ID:
            issues.append("production integration v2 run ID mismatch")
        if prod.get("structural_outcome") != EXPECTED_PROD_OUTCOME:
            issues.append("production integration v2 outcome is not PASS")

        checks = prod.get("checks", {})
        required_true = (
            "production_gap_repeat_deterministic",
            "exact_dev_v4_gap_equivalence",
            "exact_dev_v4_query_equivalence",
            "targeted_retrieval_not_called",
            "provider_calls_zero",
            "ranker_not_recomputed",
            "claim_reviewer_not_recomputed",
            "hypothesis_rewrite_not_called",
            "llm_calls_zero",
            "network_calls_zero",
            "fresh_reserve_c_not_consumed",
            "automatic_next_stage_disabled",
        )
        for key in required_true:
            if checks.get(key) is not True:
                issues.append(f"production integration check false: {key}")

        if dev.get("novelty_gap_plan_v2") != prod.get(
            "production_novelty_gap_plan_v2"
        ):
            issues.append("DEV v4 / production gap plan mismatch")

        expected = {
            "direction_aware_trend_hypothesis:ad13dac8334238124899": (
                "LITERATURE_SUPPORTED_EXTENSION",
                "targeted_search_then_refine",
                3,
            ),
            "direction_aware_trend_hypothesis:8507f8cadfc46d8d80de": (
                "NEW_COMBINATION_OF_KNOWN_EFFECTS",
                "keep",
                0,
            ),
            "direction_aware_trend_hypothesis:1cf889e57332402d88c9": (
                "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP",
                "targeted_search_only",
                3,
            ),
        }
        gaps = prod.get(
            "production_novelty_gap_plan_v2",
            {},
        ).get("gaps", [])
        observed_ids = {gap.get("hypothesis_id") for gap in gaps}
        if observed_ids != set(expected):
            issues.append("production hypothesis set mismatch")
        for gap in gaps:
            hypothesis_id = gap.get("hypothesis_id")
            if hypothesis_id not in expected:
                continue
            status, action, query_count = expected[hypothesis_id]
            if gap.get("source_external_status") != status:
                issues.append(
                    f"status mismatch for {hypothesis_id}"
                )
            if gap.get("action") != action:
                issues.append(
                    f"action mismatch for {hypothesis_id}"
                )
            if len(gap.get("targeted_queries", [])) != query_count:
                issues.append(
                    f"query count mismatch for {hypothesis_id}"
                )

        if prod.get("targeted_retrieval_called") is not False:
            issues.append("targeted retrieval unexpectedly called")
        if prod.get("provider_calls") != 0:
            issues.append("provider calls are nonzero")
        if prod.get("llm_calls") != 0:
            issues.append("LLM calls are nonzero")
        if prod.get("network_calls") != 0:
            issues.append("network calls are nonzero")
        if prod.get("fresh_reserve_c_consumed") is not False:
            issues.append("Fresh Reserve C marked consumed")
        if prod.get("automatic_next_stage_authorized") is not False:
            issues.append("automatic next stage unexpectedly authorized")

    production_source = (
        root / "dac_her/novelty_gap_analysis.py"
    )
    if production_source.is_file():
        text = production_source.read_text(encoding="utf-8")
        if "sers_novelty_gap_query_compaction_v4" in text:
            issues.append("production imports DEV v4 compactor")
        if "_QUERY_TOKEN_RE" not in text:
            issues.append("production contiguous compactor marker missing")
        if "def _query_terms(text: str, *, max_chars: int = 270)" not in text:
            issues.append("production _query_terms signature mismatch")

    meta = {
        "freeze_id": manifest.get("freeze_id"),
        "manifest_sha256": manifest.get("manifest_sha256"),
        "source_dev_v4_run_id": manifest.get("source_dev_v4_run_id"),
        "source_production_integration_v2_run_id":
            manifest.get("source_production_integration_v2_run_id"),
    }
    return not issues, issues, meta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--freeze-root",
        type=Path,
        default=DEFAULT_FREEZE_ROOT,
    )
    args = parser.parse_args()

    ok, issues, meta = verify_freeze(ROOT, args.freeze_root)
    if not ok:
        print("G0-G2 production freeze v1 verification: FAIL")
        for issue in issues:
            print(" -", issue)
        print("Network calls during verification:", 0)
        print("LLM calls during verification:", 0)
        print("Fresh Reserve C consumed:", False)
        return 2

    print("G0-G2 production freeze v1 verification: PASS")
    print("Freeze ID:", meta["freeze_id"])
    print("Manifest SHA256:", meta["manifest_sha256"])
    print("DEV v4:", meta["source_dev_v4_run_id"])
    print(
        "Production integration v2:",
        meta["source_production_integration_v2_run_id"],
    )
    print("Exact DEV-v4 / production plan equality: True")
    print("Targeted retrieval called:", False)
    print("Network calls during verification:", 0)
    print("LLM calls during verification:", 0)
    print("Fresh Reserve C consumed:", False)
    print("Automatic next stage authorized:", False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
