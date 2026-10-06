import json
from types import SimpleNamespace as NS

from pipeline_core.discovery.research_idea_feedback_resolution import (
    resolve_feedback_artifacts,
)


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _materialization():
    return NS(
        report_id="mat:1",
        output_portfolio_id="portfolio:1",
        source_context_id="ctx:1",
        records=[
            NS(status="MATERIALIZED", hypothesis_id="h1"),
            NS(status="MATERIALIZED", hypothesis_id="h2"),
            NS(status="ABSTAINED", hypothesis_id=None),
        ],
    )


def test_resolver_uses_exact_portfolio_context_and_hypothesis_lineage(tmp_path):
    _write(
        tmp_path / "run" / "epistemic_state.json",
        {
            "schema_version": "scientific-portfolio-residual-epistemic-state-v1",
            "report_id": "residual:1",
            "source_portfolio_id": "portfolio:1",
            "hypotheses": [{"hypothesis_id": "h1"}],
        },
    )
    _write(
        tmp_path / "other" / "epistemic_state.json",
        {
            "schema_version": "scientific-portfolio-residual-epistemic-state-v1",
            "report_id": "residual:wrong",
            "source_portfolio_id": "portfolio:other",
            "hypotheses": [],
        },
    )
    _write(
        tmp_path / "p" / "h1.prospective.result.json",
        {
            "schema_version": "prospective-identification-materialization-shadow-v1",
            "source_context_id": "ctx:1",
            "source_hypothesis_id": "h1",
            "prospective_identifiability": "PROSPECTIVELY_IDENTIFIABLE",
        },
    )
    _write(
        tmp_path / "p" / "wrong_context.prospective.result.json",
        {
            "schema_version": "prospective-identification-materialization-shadow-v1",
            "source_context_id": "ctx:other",
            "source_hypothesis_id": "h2",
            "prospective_identifiability": "CURRENTLY_IDENTIFIED",
        },
    )

    resolved = resolve_feedback_artifacts(
        materialization=_materialization(),
        search_roots=[tmp_path],
    )
    assert resolved.residual_state["report_id"] == "residual:1"
    assert set(resolved.prospective_by_hypothesis) == {"h1"}
    assert resolved.report.prospective_unmatched_hypothesis_ids == ["h2"]
    assert resolved.report.residual_ambiguous is False


def test_resolver_fails_closed_on_conflicting_exact_lineage_residuals(tmp_path):
    for idx, state in enumerate(["A", "B"], start=1):
        _write(
            tmp_path / f"r{idx}" / "epistemic_state.json",
            {
                "schema_version": "scientific-portfolio-residual-epistemic-state-v1",
                "report_id": f"residual:{idx}",
                "source_portfolio_id": "portfolio:1",
                "state_counts": {state: 1},
                "hypotheses": [],
            },
        )
    resolved = resolve_feedback_artifacts(
        materialization=_materialization(),
        search_roots=[tmp_path],
    )
    assert resolved.residual_state is None
    assert resolved.report.residual_ambiguous is True
    assert resolved.report.ambiguous_feedback_attached is False
