from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.direct_higher_order_synthesis_context import (
    build_direct_higher_order_synthesis_contexts,
    render_direct_higher_order_shadow_prompt,
)
from pipeline_core.discovery.direct_higher_order_topology import (
    DirectHigherOrderTopologyCandidate,
)


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prompt-dir", type=Path, default=None)
    args = parser.parse_args()

    payload = _json(args.topology_report)

    topologies = tuple(
        DirectHigherOrderTopologyCandidate.model_validate(row)
        for row in payload.get("topologies", [])
    )

    contexts = build_direct_higher_order_synthesis_contexts(
        topologies=topologies
    )

    rendered = [
        render_direct_higher_order_shadow_prompt(context)
        for context in contexts
    ]

    prompt_paths = []

    if args.prompt_dir is not None:
        args.prompt_dir.mkdir(parents=True, exist_ok=True)
        for index, text in enumerate(rendered, start=1):
            path = args.prompt_dir / f"{index:02d}.prompt.txt"
            path.write_text(text + "\n", encoding="utf-8")
            prompt_paths.append(str(path))

    out = {
        "schema_version":
            "direct-higher-order-synthesis-context-report-v1",
        "topology_report": str(args.topology_report),
        "topology_count": len(topologies),
        "context_count": len(contexts),
        "contexts": [
            row.model_dump(mode="json")
            for row in contexts
        ],
        "rendered_prompts": rendered,
        "prompt_paths": prompt_paths,
        "authority": {
            "shadow_only": True,
            "prompt_rendering_authorized": True,
            "llm_call_authorized": False,
            "interaction_claim_authorized": False,
            "endpoint_equivalence_assertion_authorized": False,
            "scientific_identity_assertion_authorized": False,
            "novelty_authority_created": False,
            "positive_premise_authority_created": False,
            "production_selection_authority": False,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=== S203 DIRECT HIGHER-ORDER SYNTHESIS CONTEXT ===")
    print("topologies:", len(topologies))
    print("contexts:", len(contexts))

    for index, context in enumerate(contexts, start=1):
        opportunity = context.structural_opportunity
        print(f"\ncontext {index}: {context.context_id}")
        print(" source-role:", opportunity.source_role_text)
        print(" target-role:", opportunity.target_role_text)
        print(" modifier C:", opportunity.modifier_text)
        print(" anchor role:", opportunity.modifier_anchor_role)
        print(" llm call authorized:", context.guard.llm_call_authorized)

    print("\nPrompt rendering only; no LLM call or authority change.")
    print("artifact:", args.output)
    print("S203_CONTEXT_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
