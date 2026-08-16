from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from dac_her.external_novelty_contracts import (
    LiteratureQueryPlan,
    PriorArtPacket,
)
from dac_her.literature_provider_plan import (
    LiteratureProviderPlan,
    require_standard_or_full_auto_plan,
    resolve_literature_provider_plan,
)
from dac_her.novelty_refinement_contracts import NoveltyGapPlan

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = ROOT / (
    "evaluation/sers_novelty_gap/t1_frozen_input_bundle_v1"
)
DEFAULT_SPEC_ROOT = ROOT / (
    "evaluation/sers_novelty_gap/"
    "t1_live_targeted_retrieval_spec_v1"
)


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


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _run_verify(module: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", module],
        cwd=ROOT,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Verifier failed: {module}")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
    )
    parser.add_argument(
        "--spec-root",
        type=Path,
        default=DEFAULT_SPEC_ROOT,
    )
    parser.add_argument(
        "--results-per-query",
        type=int,
        default=12,
    )
    args = parser.parse_args()
    if not args.run:
        parser.error("--run is required")
    if args.spec_root.exists():
        print("T1 prepare: FAIL")
        print(" - spec root already exists:", args.spec_root)
        return 2

    # T0 verifier itself performs isolated parent G0-G2 verification.
    _run_verify(
        "scripts.verify_sers_targeted_retrieval_t0_freeze_v2"
    )
    _run_verify(
        "scripts.verify_sers_targeted_retrieval_t1_input_bundle_v1"
    )

    input_root = args.input_root.resolve()
    manifest = json.loads(
        (input_root / "bundle_manifest.json")
        .read_text(encoding="utf-8")
    )
    base_plan = LiteratureQueryPlan.model_validate_json(
        (input_root / "base_query_plan.json")
        .read_text(encoding="utf-8")
    )
    base_packet = PriorArtPacket.model_validate_json(
        (input_root / "base_prior_art_packet.json")
        .read_text(encoding="utf-8")
    )
    gap_plan = NoveltyGapPlan.model_validate_json(
        (input_root / "novelty_gap_plan.json")
        .read_text(encoding="utf-8")
    )

    if base_packet.source_query_plan_id != base_plan.plan_id:
        raise RuntimeError(
            "Frozen base packet does not reference frozen base query plan."
        )
    if base_packet.source_portfolio_id != base_plan.source_portfolio_id:
        raise RuntimeError(
            "Frozen base packet/query plan portfolio mismatch."
        )
    if (
        gap_plan.source_portfolio_id
        != base_plan.source_portfolio_id
    ):
        raise RuntimeError(
            "Frozen novelty gap plan/base plan portfolio mismatch."
        )
    if manifest.get("base_query_plan_id") != base_plan.plan_id:
        raise RuntimeError("Input bundle base query plan ID mismatch.")
    if manifest.get("base_prior_art_packet_id") != base_packet.packet_id:
        raise RuntimeError("Input bundle base packet ID mismatch.")
    if manifest.get("novelty_gap_plan_id") != gap_plan.plan_id:
        raise RuntimeError("Input bundle novelty gap plan ID mismatch.")

    provider_plan = resolve_literature_provider_plan()
    require_standard_or_full_auto_plan(provider_plan)
    if provider_plan.requested_mode != "auto":
        raise RuntimeError("T1 v1 requires auto provider mode.")

    targeted_gaps = [
        gap for gap in gap_plan.gaps
        if gap.targeted_queries
    ]
    skipped = [
        gap for gap in gap_plan.gaps
        if not gap.targeted_queries
    ]
    if any(gap.action != "keep" for gap in skipped):
        raise RuntimeError(
            "A non-keep gap has zero targeted queries; fail closed."
        )
    target_query_count = sum(
        len(gap.targeted_queries)
        for gap in targeted_gaps
    )
    if target_query_count <= 0:
        raise RuntimeError("Frozen gap plan has no targeted queries.")

    # Frozen G0-G2 contract currently contains exactly H1=3, H2=0, H3=3.
    expected_shape = {
        "direction_aware_trend_hypothesis:ad13dac8334238124899":
            ("targeted_search_then_refine", 3),
        "direction_aware_trend_hypothesis:8507f8cadfc46d8d80de":
            ("keep", 0),
        "direction_aware_trend_hypothesis:1cf889e57332402d88c9":
            ("targeted_search_only", 3),
    }
    observed_shape = {
        gap.hypothesis_id:
            (gap.action, len(gap.targeted_queries))
        for gap in gap_plan.gaps
    }
    if observed_shape != expected_shape:
        raise RuntimeError(
            "Frozen G0-G2 targeted-query shape drift: "
            + repr(observed_shape)
        )

    args.spec_root.mkdir(parents=True, exist_ok=False)
    copies = {
        "base_query_plan.json": base_plan,
        "base_prior_art_packet.json": base_packet,
        "novelty_gap_plan.json": gap_plan,
        "provider_plan.json": provider_plan,
    }
    for name, model in copies.items():
        _atomic_json(
            args.spec_root / name,
            model.model_dump(mode="json"),
        )
    _atomic_json(
        args.spec_root / "input_bundle_manifest.json",
        manifest,
    )

    body = {
        "schema_version":
            "sers-targeted-retrieval-t1-live-spec-v1",
        "parent_t0_freeze_verifier":
            "scripts.verify_sers_targeted_retrieval_t0_freeze_v2",
        "parent_g0_g2_verification":
            "inherited_via_t0_freeze_v2_isolated_parent_verification",
        "input_bundle_verifier":
            "scripts.verify_sers_targeted_retrieval_t1_input_bundle_v1",
        "input_bundle_id": manifest["bundle_id"],
        "t0_freeze_id": manifest["t0_freeze_id"],
        "parent_g0_g2_freeze_id":
            manifest["parent_g0_g2_freeze_id"],
        "frozen_copy_sha256": {
            name: _sha256(args.spec_root / name)
            for name in [
                "base_query_plan.json",
                "base_prior_art_packet.json",
                "novelty_gap_plan.json",
                "provider_plan.json",
                "input_bundle_manifest.json",
            ]
        },
        "base_query_plan_id": base_plan.plan_id,
        "base_prior_art_packet_id": base_packet.packet_id,
        "novelty_gap_plan_id": gap_plan.plan_id,
        "provider_plan_id": provider_plan.plan_id,
        "provider_mode": provider_plan.mode,
        "providers": list(provider_plan.active_providers),
        "results_per_query": args.results_per_query,
        "gap_count": len(gap_plan.gaps),
        "targeted_gap_count": len(targeted_gaps),
        "skipped_keep_gap_count": len(skipped),
        "targeted_query_count": target_query_count,
        "gaps": [
            {
                "gap_id": gap.gap_id,
                "hypothesis_id": gap.hypothesis_id,
                "action": gap.action,
                "targeted_query_count":
                    len(gap.targeted_queries),
                "target_claim_ids":
                    list(gap.target_claim_ids),
            }
            for gap in gap_plan.gaps
        ],
        "network_authorized_only_for_t1_run": True,
        "provider_set_frozen_for_run": True,
        "runtime_provider_fallback_authorized": False,
        "query_rewrite_authorized": False,
        "ranker_authorized": False,
        "claim_reviewer_authorized": False,
        "novelty_reassessment_authorized": False,
        "llm_authorized": False,
        "hypothesis_rewrite_authorized": False,
        "fresh_reserve_c_authorized": False,
        "automatic_next_stage_authorized": False,
    }
    digest = _sha256_json(body)
    spec = {
        **body,
        "spec_sha256": digest,
        "spec_id":
            "sers_targeted_retrieval_t1_live_spec:"
            + digest[:20],
    }
    _atomic_json(args.spec_root / "t1_spec.json", spec)
    _atomic_json(
        args.spec_root / "PREPARE_PASS.json",
        {
            "spec_id": spec["spec_id"],
            "network_calls": 0,
            "llm_calls": 0,
            "fresh_reserve_c_consumed": False,
        },
    )

    print("SERS T1 live targeted-retrieval prepare: PASS")
    print("Spec ID:", spec["spec_id"])
    print("Input bundle:", spec["input_bundle_id"])
    print("T0 freeze:", spec["t0_freeze_id"])
    print("Base query plan:", base_plan.plan_id)
    print("Base prior-art packet:", base_packet.packet_id)
    print("Gap plan:", gap_plan.plan_id)
    print("Provider mode:", provider_plan.mode)
    print("Providers:", provider_plan.active_providers)
    print("Targeted gaps:", len(targeted_gaps))
    print("Skipped keep gaps:", len(skipped))
    print("Targeted queries:", target_query_count)
    print("Results per query:", args.results_per_query)
    print("Network calls during prepare:", 0)
    print("LLM calls:", 0)
    print("Fresh Reserve C consumed:", False)
    print("Automatic next stage authorized:", False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
