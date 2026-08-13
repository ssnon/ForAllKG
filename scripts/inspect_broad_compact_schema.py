from __future__ import annotations

import json

from dac_her.broad_compact_schema import BroadMechanismGraphDraft
from dac_her.draft_schema import KnowledgeGraphDraft
from dac_her.llm_telemetry import estimate_tokens


def _schema_stats(model: type[KnowledgeGraphDraft]) -> dict[str, object]:
    payload = model.model_json_schema()
    defs = payload.get("$defs")
    if not isinstance(defs, dict):
        defs = {}

    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    tokens, estimator = estimate_tokens(serialized)
    return {
        "model": model.__name__,
        "characters": len(serialized),
        "estimated_tokens": tokens,
        "estimator": estimator,
        "contains_measurement_node_definition": "MeasurementNode" in defs,
        "contains_measurement_group_definition": "MeasurementGroupDraft" in defs,
    }


def main() -> None:
    full = _schema_stats(KnowledgeGraphDraft)
    compact = _schema_stats(BroadMechanismGraphDraft)
    full_tokens = int(full["estimated_tokens"])
    compact_tokens = int(compact["estimated_tokens"])
    saved = full_tokens - compact_tokens
    payload = {
        "full": full,
        "compact": compact,
        "estimated_tokens_saved_per_generation": saved,
        "estimated_reduction_fraction": (
            saved / full_tokens if full_tokens else None
        ),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
