from __future__ import annotations

import argparse
import json
from pathlib import Path

from dac_her.external_novelty_contracts import (
    LiteratureQueryPlan,
    PriorArtPacket,
)
from dac_her.literature_provider_plan import LiteratureProviderPlan
from dac_her.novelty_refinement_contracts import NoveltyGapPlan
from dac_her.sers_targeted_retrieval_t1_live_guard import (
    validate_t1_pre_network_guard,
)

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SPEC_ROOT = ROOT / (
    "evaluation/sers_novelty_gap/"
    "t1_live_targeted_retrieval_spec_v1"
)
DEFAULT_RUN_ROOT = ROOT / (
    "evaluation/sers_novelty_gap/"
    "t1_live_targeted_retrieval_run_v1"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec-root",
        type=Path,
        default=DEFAULT_SPEC_ROOT,
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=DEFAULT_RUN_ROOT,
    )
    args = parser.parse_args()

    if args.run_root.exists():
        print("T1 guarded live preflight: FAIL")
        print(" - live run root already exists:", args.run_root)
        return 2

    spec = json.loads(
        (args.spec_root / "t1_spec.json").read_text(encoding="utf-8")
    )
    base_plan = LiteratureQueryPlan.model_validate_json(
        (args.spec_root / "base_query_plan.json").read_text(encoding="utf-8")
    )
    base_packet = PriorArtPacket.model_validate_json(
        (args.spec_root / "base_prior_art_packet.json")
        .read_text(encoding="utf-8")
    )
    gap_plan = NoveltyGapPlan.model_validate_json(
        (args.spec_root / "novelty_gap_plan.json").read_text(encoding="utf-8")
    )
    provider_plan = LiteratureProviderPlan.model_validate_json(
        (args.spec_root / "provider_plan.json").read_text(encoding="utf-8")
    )

    try:
        guard = validate_t1_pre_network_guard(
            root=ROOT,
            spec_root=args.spec_root,
            spec=spec,
            base_plan=base_plan,
            base_packet=base_packet,
            gap_plan=gap_plan,
            provider_plan=provider_plan,
        )
    except Exception as exc:
        print("T1 guarded live preflight: FAIL")
        print("Exception type:", type(exc).__name__)
        print("Reason:", str(exc))
        print("Network calls:", 0)
        print("LLM calls:", 0)
        print("Fresh Reserve C consumed:", False)
        return 2

    print("T1 guarded live preflight: PASS")
    print("Source git HEAD:", guard["source_git_head"])
    print("Source branch:", guard["source_git_branch"])
    print("Spec ID:", guard["spec_id"])
    print("Spec SHA256:", guard["spec_sha256"])
    print("Provider plan:", guard["provider_plan_id"])
    print("Provider mode:", guard["provider_mode"])
    print("Providers:", guard["providers"])
    print("Targeted queries:", guard["targeted_query_count"])
    print("Results per query:", guard["results_per_query"])
    print("T0 freeze verified:", guard["t0_freeze_verified"])
    print("T1 input bundle verified:", guard["t1_input_bundle_verified"])
    print("Working tree clean:", guard["working_tree_clean"])
    print("Runtime files tracked:", guard["runtime_files_tracked"])
    print("Live run root exists:", False)
    print("Network calls:", 0)
    print("LLM calls:", 0)
    print("Fresh Reserve C consumed:", False)
    print("Automatic next stage authorized:", False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
