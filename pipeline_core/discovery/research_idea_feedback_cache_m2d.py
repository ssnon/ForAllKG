"""Opt-in, fail-closed reuse of SIS scientific *feedback*, never evidence authority.

Pair cache is content-addressed by the exact two idea payloads + evaluator identity.
Full-portfolio prior-art cache requires exact portfolio bytes and limited freshness.
Neither cache creates scientific truth, novelty certification, or grounding.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


PAIR_CACHE_SCHEMA = "sis-m2d-semantic-pair-cache-v1"
PRIOR_CACHE_SCHEMA = "sis-m2d-exact-portfolio-prior-art-cache-v1"
PAIR_EVALUATION_POLICY = "SIS-M2D-PARTIAL-PAIR-REVIEW-V1"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _digest(value: Any) -> str:
    return _sha(_canonical(value).encode("utf-8"))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"cache entry must be JSON object: {path}")
    return value


def _atomic_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".m2d-", suffix=".json", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _pair_key(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((a, b)))


def _check_pair_row(row: Mapping[str, Any], expected: tuple[str, str]) -> None:
    if _pair_key(str(row.get("idea_id_a", "")), str(row.get("idea_id_b", ""))) != expected:
        raise ValueError(f"program-family assessment pair ID mismatch: {expected}")
    if row.get("relation") not in ("SAME_PROGRAM", "ADJACENT_PROGRAM", "DISTINCT_PROGRAM"):
        raise ValueError(f"program-family assessment relation invalid: {expected}")
    if not str(row.get("rationale") or "").strip():
        raise ValueError(f"program-family assessment missing rationale: {expected}")


def incremental_program_pairs(
    *,
    idea_payloads: Sequence[Mapping[str, Any]],
    source_contexts: Mapping[str, str],
    model: str,
    base_url: str | None,
    system_prompt: str,
    cache_dir: Path,
    infer_missing: Callable[[list[tuple[str, str]]], Sequence[Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Infer all missing pairs in one batch; cached edges are pair-local judgments.

    `idea_payloads` must match program_family_prompt_payload()['ideas'] exactly.
    `infer_missing` must respond for *only* its specified pair keys.
    """
    by_id = {str(x["idea_id"]): dict(x) for x in idea_payloads}
    if len(by_id) != len(idea_payloads):
        raise ValueError("duplicate idea IDs in pair-cache input")
    all_pairs = list(combinations(sorted(by_id), 2))
    cached: dict[tuple[str, str], dict[str, Any]] = {}
    files: dict[tuple[str, str], tuple[Path, str]] = {}
    for a, b in all_pairs:
        fp = _digest({
            "policy": PAIR_EVALUATION_POLICY,
            "model": model,
            "base_url": base_url or "",
            "system_prompt": system_prompt,
            "context_a": source_contexts[a],
            "context_b": source_contexts[b],
            "idea_a": by_id[a],
            "idea_b": by_id[b],
        })
        path = cache_dir / "pairs" / f"{fp}.json"
        files[(a, b)] = (path, fp)
        if not path.is_file():
            continue
        try:
            doc = _read_json(path)
            if (doc.get("schema_version") != PAIR_CACHE_SCHEMA
                or doc.get("fingerprint") != fp
                or doc.get("row_sha256") != _digest(doc.get("row"))
                or not isinstance(doc.get("row"), dict)):
                continue
            _check_pair_row(doc["row"], (a, b))
            cached[(a, b)] = doc["row"]
        except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError):
            # A damaged or untrusted cache entry never produces a scientific decision.
            continue

    missing = [pair for pair in all_pairs if pair not in cached]
    new_rows: dict[tuple[str, str], dict[str, Any]] = {}
    if missing:
        response = infer_missing(missing)
        for raw in response:
            row = dict(raw)
            key = _pair_key(str(row.get("idea_id_a", "")), str(row.get("idea_id_b", "")))
            if key not in missing or key in new_rows:
                raise ValueError(f"unexpected or duplicate incremental program pair: {key}")
            _check_pair_row(row, key)
            new_rows[key] = row
        if set(new_rows) != set(missing):
            raise ValueError(f"missing incremental program pairs: {sorted(set(missing) - set(new_rows))}")
        # Only write after the entire new batch is validated: no partial coverage.
        for key, row in new_rows.items():
            path, fp = files[key]
            _atomic_json(path, {
                "schema_version": PAIR_CACHE_SCHEMA,
                "fingerprint": fp,
                "row_sha256": _digest(row),
                "row": row,
            })
    merged = [cached.get(key) or new_rows[key] for key in all_pairs]
    return merged, {"pairs_total": len(all_pairs), "pairs_cache_hit": len(cached),
                    "pairs_inferred": len(missing), "model_calls": int(bool(missing))}


def prior_art_cache_key(
    *,
    portfolio_bytes: bytes,
    context_id: str,
    context_sha256: str,
    model: str,
    base_url: str | None,
    providers: str,
    results_per_query: int,
    provider_plan_bytes: bytes | None,
    runner_bytes: bytes,
) -> str:
    """Exact-input cache. Never key on ResearchIdea identity alone."""
    return _digest({
        "schema_version": PRIOR_CACHE_SCHEMA,
        "portfolio_sha256": _sha(portfolio_bytes),
        "context_id": context_id,
        "context_sha256": context_sha256,
        "model": model,
        "base_url": base_url or "",
        "providers": providers,
        "results_per_query": results_per_query,
        "provider_plan_sha256": _sha(provider_plan_bytes) if provider_plan_bytes is not None else None,
        "runner_sha256": _sha(runner_bytes),
        "flags": ["pre-review-coverage-shadow", "downstream-gate-shadow", "save-prompts"],
    })


def load_prior_art_cache(
    *, cache_dir: Path, fingerprint: str, max_age_hours: float,
    source_portfolio_id: str, now: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Require exactly matching portfolio ID and a positive, bounded freshness TTL."""
    if not 0 < max_age_hours <= 24:
        raise ValueError("prior-art TTL must be greater than zero and at most 24 hours")
    path = cache_dir / "prior_art" / f"{fingerprint}.json"
    if not path.is_file():
        return None
    try:
        record = _read_json(path)
        if record.get("schema_version") != PRIOR_CACHE_SCHEMA or record.get("fingerprint") != fingerprint:
            return None
        cached_at = float(record["cached_at_unix"])
        current = time.time() if now is None else now
        if cached_at > current + 60 or current - cached_at > max_age_hours * 3600:
            return None
        plan, report = record["plan"], record["report"]
        if not isinstance(plan, dict) or not isinstance(report, dict):
            return None
        if record.get("data_sha256") != _digest({"plan": plan, "report": report}):
            return None
        if str(report.get("source_portfolio_id") or "") != source_portfolio_id:
            return None
        return plan, report
    except (ValueError, TypeError, OSError, KeyError, json.JSONDecodeError):
        return None


def save_prior_art_cache(
    *, cache_dir: Path, fingerprint: str,
    plan: Mapping[str, Any], report: Mapping[str, Any],
    now: float | None = None,
) -> None:
    _atomic_json(cache_dir / "prior_art" / f"{fingerprint}.json", {
        "schema_version": PRIOR_CACHE_SCHEMA,
        "fingerprint": fingerprint,
        "cached_at_unix": time.time() if now is None else now,
        "data_sha256": _digest({"plan": dict(plan), "report": dict(report)}),
        "plan": dict(plan),
        "report": dict(report),
    })
