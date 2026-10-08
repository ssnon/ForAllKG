from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline_core.discovery.research_idea_feedback_cache_m2d import (
    incremental_program_pairs,
    load_prior_art_cache,
    prior_art_cache_key,
    save_prior_art_cache,
)


def _ideas(n=4):
    return [
        {"idea_id": f"research_idea:{i}", "canonical_intent": f"mechanism {i}"}
        for i in range(n)
    ]


def _infer(pairs):
    return [
        {"idea_id_a": a, "idea_id_b": b, "relation": "DISTINCT_PROGRAM",
         "rationale": "Distinct mechanistic explanations."}
        for a, b in pairs
    ]


def _run_pair(tmp_path, ideas=None, model="test-model", infer=_infer):
    ideas = ideas or _ideas()
    return incremental_program_pairs(
        idea_payloads=ideas,
        source_contexts={row["idea_id"]: "context123" for row in ideas},
        model=model,
        base_url="https://example.invalid/v1",
        system_prompt="fixed program policy",
        cache_dir=tmp_path,
        infer_missing=infer,
    )


def _prior_key(*, portfolio=b'{"portfolio_id":"portfolio:1"}', model="m", runner=b"version-1"):
    return prior_art_cache_key(
        portfolio_bytes=portfolio, context_id="context:1",
        context_sha256="a" * 64, model=model, base_url="https://example.invalid/v1",
        providers="auto", results_per_query=6, provider_plan_bytes=None,
        runner_bytes=runner,
    )


def test_initial_pair_batch_is_complete_and_caches(tmp_path):
    invoked = []
    def infer(pairs):
        invoked.append(list(pairs))
        return _infer(pairs)
    rows, stats = _run_pair(tmp_path, infer=infer)
    assert len(rows) == 6
    assert stats == {"pairs_total": 6, "pairs_cache_hit": 0, "pairs_inferred": 6, "model_calls": 1}
    assert len(invoked) == 1
    assert len(list((tmp_path / "pairs").glob("*.json"))) == 6


def test_unchanged_pair_batch_is_fully_reused_without_call(tmp_path):
    _run_pair(tmp_path)
    def forbidden(_):
        raise AssertionError("cached batch should not call LLM")
    rows, stats = _run_pair(tmp_path, infer=forbidden)
    assert len(rows) == 6 and stats["pairs_cache_hit"] == 6
    assert stats["model_calls"] == 0


def test_next_generation_unchanged_pairs_reused_and_new_only_inferred(tmp_path):
    _run_pair(tmp_path)
    newer = _ideas()[1:] + [{"idea_id": "research_idea:new", "canonical_intent": "new mediator"}]
    requested = []
    def infer(pairs):
        requested.extend(pairs)
        return _infer(pairs)
    rows, stats = _run_pair(tmp_path, ideas=newer, infer=infer)
    assert len(rows) == 6
    assert stats["pairs_cache_hit"] == 3
    assert stats["pairs_inferred"] == 3
    assert len(requested) == 3


def test_changed_science_invalidates_pair_cache(tmp_path):
    _run_pair(tmp_path)
    changed = _ideas()
    changed[0]["canonical_intent"] = "completely different mediator"
    rows, stats = _run_pair(tmp_path, ideas=changed)
    assert len(rows) == 6
    assert stats["pairs_cache_hit"] == 3
    assert stats["pairs_inferred"] == 3


def test_evaluator_change_invalidates_all_pair_cache(tmp_path):
    _run_pair(tmp_path)
    _, stats = _run_pair(tmp_path, model="different-evaluator")
    assert stats["pairs_inferred"] == 6


def test_invalid_and_missing_pair_batches_do_not_write_partial_cache(tmp_path):
    def partial(pairs):
        return _infer(pairs[:2])
    with pytest.raises(ValueError, match="missing incremental"):
        _run_pair(tmp_path, infer=partial)
    assert not list((tmp_path / "pairs").glob("*.json"))


def test_tampered_pair_cache_entry_is_not_trusted(tmp_path):
    _run_pair(tmp_path)
    file = next((tmp_path / "pairs").glob("*.json"))
    x = json.loads(file.read_text())
    x["row"]["relation"] = "SAME_PROGRAM"
    file.write_text(json.dumps(x))
    _, stats = _run_pair(tmp_path)
    assert stats["pairs_cache_hit"] == 5
    assert stats["pairs_inferred"] == 1


def test_exact_prior_art_cache_and_freshness(tmp_path):
    key = _prior_key()
    plan = {"id": "plan"}
    report = {"source_portfolio_id": "portfolio:1", "result": "observational"}
    save_prior_art_cache(cache_dir=tmp_path, fingerprint=key, plan=plan, report=report, now=10000)
    assert load_prior_art_cache(cache_dir=tmp_path, fingerprint=key, max_age_hours=1,
                                source_portfolio_id="portfolio:1", now=10010) == (plan, report)
    assert load_prior_art_cache(cache_dir=tmp_path, fingerprint=key, max_age_hours=1,
                                source_portfolio_id="portfolio:1", now=14001) is None
    assert load_prior_art_cache(cache_dir=tmp_path, fingerprint=key, max_age_hours=1,
                                source_portfolio_id="portfolio:other", now=10010) is None


def test_prior_art_cache_key_includes_content_and_runner_policy():
    assert _prior_key() != _prior_key(model="changed")
    assert _prior_key() != _prior_key(runner=b"version-2")
    assert _prior_key() != _prior_key(portfolio=b'{"portfolio_id":"different"}')


def test_tampered_prior_art_cache_falls_back_to_retrieval(tmp_path):
    key = _prior_key()
    save_prior_art_cache(cache_dir=tmp_path, fingerprint=key,
                         plan={"id": "p"}, report={"source_portfolio_id": "portfolio:1"}, now=10000)
    file = tmp_path / "prior_art" / (key + ".json")
    row = json.loads(file.read_text())
    row["report"]["source_portfolio_id"] = "portfolio:fake"
    file.write_text(json.dumps(row))
    assert load_prior_art_cache(cache_dir=tmp_path, fingerprint=key, max_age_hours=1,
                                source_portfolio_id="portfolio:1", now=10020) is None


def test_prior_art_cache_rejects_unbounded_or_negative_ttl(tmp_path):
    for ttl in (0, -1, 25):
        with pytest.raises(ValueError, match="at most 24"):
            load_prior_art_cache(cache_dir=tmp_path, fingerprint=_prior_key(),
                                 max_age_hours=ttl, source_portfolio_id="portfolio:1")
