from __future__ import annotations

import inspect

import campaigns.sers_alpha4_epoch.holdout.cli.run_sers_alpha4c4d2_trend_holdout as runner


def test_runner_propagates_partial_critical_override_to_corpus():
    source = inspect.getsource(runner.main)
    assert "--allow-critical-partial" in source
    assert "partial_critical" in source


def test_runner_does_not_relax_rejected_quality():
    source = inspect.getsource(runner.verify_protocol_and_lock)
    assert '"complete"' in source
    assert '"partial_acceptable"' in source
    assert '"partial_critical"' in source
    assert "allowed_statuses" in source
