"""Exact textual provenance only; never repairs or adjudicates a claim."""
from __future__ import annotations

import hashlib


def trace_specification_sources(hypothesis, raw_fields, accepted_fields):
    """Keep omission, source matching and sanitizer acceptance independent.

    A text match proves where a string occurs, not branch ownership, scientific
    support, or novelty. Empty drafts remain empty even if sources exist.
    Offsets are Python Unicode character offsets with an exclusive end.
    """
    sources = {
        "required_bridge": [("inferential_bridge", hypothesis.inferential_bridge)],
        "predicted_observation": [],
        "falsification_condition": [],
    }
    sources["required_bridge"].extend(
        (f"assumptions[{i}]", value)
        for i, value in enumerate(hypothesis.assumptions)
    )
    for i, item in enumerate(hypothesis.predicted_observations):
        for name in ("observable", "expected_direction", "rationale"):
            sources["predicted_observation"].append(
                (f"predicted_observations[{i}].{name}", getattr(item, name, ""))
            )
    for i, item in enumerate(hypothesis.falsification_criteria):
        for name in ("observable", "falsifying_outcome"):
            sources["falsification_condition"].append(
                (f"falsification_criteria[{i}].{name}", getattr(item, name, ""))
            )

    def locate(text, pool):
        matches = []
        if not text.strip():
            return matches
        for path, value in pool:
            value = str(value or "")
            start = value.find(text)
            while start >= 0:
                matches.append({
                    "source_path": path,
                    "source_sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
                    "start": start,
                    "end": start + len(text),
                    "quote": text,
                })
                start = value.find(text, start + 1)
        return matches

    result = {}
    for field, pool in sources.items():
        raw = str(raw_fields.get(field) or "")
        accepted = str(accepted_fields.get(field) or "")
        matches = locate(raw, pool)
        result[field] = {
            "draft_state": "PRESENT" if raw.strip() else "EMPTY",
            "sanitizer_state": (
                "RETAINED" if accepted.strip()
                else "REJECTED" if raw.strip() else "EMPTY"
            ),
            "raw_source_match_state": (
                "NOT_APPLICABLE_EMPTY" if not raw.strip()
                else "EXACT_UNIQUE" if len(matches) == 1
                else "EXACT_MULTIPLE" if matches
                else "NO_EXACT_MATCH"
            ),
            "raw_exact_matches": matches,
            "accepted_exact_matches": locate(accepted, pool),
            "nonempty_source_paths": [
                path for path, value in pool if str(value or "").strip()
            ],
        }
    return {
        "schema_version": "novelty-specification-source-trace-v1",
        "diagnostic_only": True,
        "hypothesis_id": hypothesis.hypothesis_id,
        "match_policy": "EXACT_CASE_SENSITIVE_UNICODE_SUBSTRING",
        "offset_unit": "UNICODE_CODE_POINT",
        "branch_attribution_assessed": False,
        "scientific_sufficiency_assessed": False,
        "fields": result,
    }
