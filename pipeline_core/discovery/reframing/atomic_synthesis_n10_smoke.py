from __future__ import annotations

from collections import Counter
from typing import Any, Mapping


def summarize_n9_states(payload: Mapping[str, Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    rows = payload.get("hypotheses")
    if not isinstance(rows, list):
        raise ValueError("N9 intake hypotheses must be a list")

    for hypothesis in rows:
        if not isinstance(hypothesis, Mapping):
            raise ValueError("N9 intake hypothesis row must be an object")
        claims = hypothesis.get("claims")
        if not isinstance(claims, list):
            raise ValueError("N9 intake claims must be a list")
        for decision in claims:
            if not isinstance(decision, Mapping):
                raise ValueError("N9 intake claim decision must be an object")
            state = str(decision.get("shadow_state") or "").strip()
            if not state:
                raise ValueError("N9 intake claim decision lacks shadow_state")
            counts[state] += 1

    return dict(sorted(counts.items()))


def ready_for_closure_count(payload: Mapping[str, Any]) -> int:
    return summarize_n9_states(payload).get("READY_FOR_CLOSURE", 0)


__all__ = [
    "ready_for_closure_count",
    "summarize_n9_states",
]
