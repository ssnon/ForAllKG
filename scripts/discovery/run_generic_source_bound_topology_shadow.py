
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.source_bound_topology_shadow import (
    build_topology_report,
)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--query-plan", required=True, type=Path)
    p.add_argument("--saturation-report", required=True, type=Path)
    p.add_argument("--external-report", required=True, type=Path)
    p.add_argument("--source-binding", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()

    topology = build_topology_report(
        query_plan=load(args.query_plan),
        saturation_report=load(args.saturation_report),
        external_report=load(args.external_report),
        source_binding_bundle=load(args.source_binding),
    )
    payload = {
        "schema_version": "generic-source-bound-topology-shadow-v1",
        "shadow_only": True,
        "topology_residual": topology,
        "novelty_authority_created": False,
        "n9_authority_created": False,
        "n10_authority_created": False,
        "production_selection_changed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("Generic source-bound topology shadow complete")
    print("Topology states:", topology.get("topology_state_counts", {}))
    print("Closure states:", topology.get("component_closure_counts", {}))
    print("Residual states:", topology.get("residual_state_counts", {}))
    print("SHADOW_ONLY=True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
