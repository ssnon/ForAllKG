from __future__ import annotations

import argparse
from pathlib import Path

from dac_her.discovery_contracts import DiscoveryBundle
from dac_her.dual_hypothesis_context import DualHypothesisContext
from dac_her.hypothesis_contracts import HypothesisContext


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Combine a grounded HypothesisContext with a non-evidentiary DiscoveryBundle."
    )
    parser.add_argument("--context", required=True, type=Path)
    parser.add_argument("--discovery-bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    context = HypothesisContext.model_validate_json(args.context.read_text(encoding="utf-8"))
    bundle = DiscoveryBundle.model_validate_json(args.discovery_bundle.read_text(encoding="utf-8"))
    dual = DualHypothesisContext.build(context, bundle)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(dual.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print("DualHypothesisContext built")
    print("Dual context ID:", dual.dual_context_id)
    print("Dual context SHA256:", dual.dual_context_sha256)
    print("Grounded context:", context.context_id)
    print("Discovery bundle:", bundle.bundle_id)
    print("Discovery inspirations:", len(bundle.inspirations))
    print("Saved:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
