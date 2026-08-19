from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

from dac_her.external_novelty_contracts import (
    LiteratureQueryPlan,
    PriorArtPacket,
)
from dac_her.novelty_refinement_contracts import NoveltyGapPlan

ROOT = Path(__file__).resolve().parents[4]
BUNDLE_ROOT = ROOT / (
    "evaluation/sers_novelty_gap/t1_frozen_input_bundle_v1"
)
MANIFEST = BUNDLE_ROOT / "bundle_manifest.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_json(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


def main() -> int:
    if not MANIFEST.is_file():
        print("T1 frozen input bundle verification: FAIL")
        print(" - bundle manifest missing")
        return 2

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    body = dict(manifest)
    observed_id = body.pop("bundle_id", None)
    observed_sha = body.pop("manifest_sha256", None)
    expected_sha = _sha256_json(body)
    expected_id = (
        "sers_targeted_retrieval_t1_input_bundle_v1:"
        + expected_sha[:20]
    )

    issues: list[str] = []
    if observed_sha != expected_sha:
        issues.append("manifest SHA mismatch")
    if observed_id != expected_id:
        issues.append("bundle ID mismatch")

    for rel, expected in manifest.get("files", {}).items():
        path = BUNDLE_ROOT / rel
        if not path.is_file():
            issues.append(f"missing bundle file: {rel}")
            continue
        if _sha256(path) != expected:
            issues.append(f"bundle file SHA mismatch: {rel}")

    try:
        plan = LiteratureQueryPlan.model_validate_json(
            (BUNDLE_ROOT / "base_query_plan.json")
            .read_text(encoding="utf-8")
        )
        packet = PriorArtPacket.model_validate_json(
            (BUNDLE_ROOT / "base_prior_art_packet.json")
            .read_text(encoding="utf-8")
        )
        gap = NoveltyGapPlan.model_validate_json(
            (BUNDLE_ROOT / "novelty_gap_plan.json")
            .read_text(encoding="utf-8")
        )
        if packet.source_query_plan_id != plan.plan_id:
            issues.append("base packet/query plan ID mismatch")
        if packet.source_portfolio_id != plan.source_portfolio_id:
            issues.append("base packet/query plan portfolio mismatch")
        if gap.source_portfolio_id != plan.source_portfolio_id:
            issues.append("gap plan/base plan portfolio mismatch")
        if manifest.get("base_query_plan_id") != plan.plan_id:
            issues.append("manifest base query plan ID mismatch")
        if manifest.get("base_prior_art_packet_id") != packet.packet_id:
            issues.append("manifest base packet ID mismatch")
        if manifest.get("novelty_gap_plan_id") != gap.plan_id:
            issues.append("manifest gap plan ID mismatch")
    except Exception as exc:
        issues.append(
            "bundle model validation failed: "
            + type(exc).__name__
            + ": "
            + str(exc)
        )

    t0 = subprocess.run(
        [
            sys.executable,
            "-m",
            "campaigns.sers_novelty_gap.history.cli.verify_sers_targeted_retrieval_t0_freeze_v2",
        ],
        cwd=ROOT,
        text=True,
    )
    if t0.returncode != 0:
        issues.append("T0 freeze v2 verifier failed")

    if issues:
        print("T1 frozen input bundle verification: FAIL")
        for issue in issues:
            print(" -", issue)
        print("Network calls during verification:", 0)
        print("LLM calls during verification:", 0)
        print("Fresh Reserve C consumed:", False)
        return 2

    print("T1 frozen input bundle verification: PASS")
    print("Bundle ID:", observed_id)
    print("Manifest SHA256:", observed_sha)
    print("Base query plan:", manifest["base_query_plan_id"])
    print("Base prior-art packet:", manifest["base_prior_art_packet_id"])
    print("Novelty gap plan:", manifest["novelty_gap_plan_id"])
    print("T0 freeze:", manifest["t0_freeze_id"])
    print("Parent G0-G2 freeze:", manifest["parent_g0_g2_freeze_id"])
    print("Network calls during verification:", 0)
    print("LLM calls during verification:", 0)
    print("Fresh Reserve C consumed:", False)
    print("Automatic next stage authorized:", False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
