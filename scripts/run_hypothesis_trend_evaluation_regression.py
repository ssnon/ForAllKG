from __future__ import annotations

import argparse
from pathlib import Path

from dac_her.hypothesis_trend_directional_contracts import (
    DirectionAwareTrendHypothesisPortfolio,
    DirectionAwareTrendHypothesisPortfolioDraft,
)
from dac_her.hypothesis_trend_directional_run_record import (
    DirectionAwareTrendHypothesisMakerRunRecord,
)
from dac_her.hypothesis_trend_evaluation import (
    detect_claim_scope_issues,
    evaluate_run,
    load_protocol,
    verify_protocol_integrity,
)
from dac_her.hypothesis_trend_input import TrendAwareHypothesisInput


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--seen-input", required=True, type=Path)
    parser.add_argument("--seen-run", required=True, type=Path)
    parser.add_argument("--seen-final-draft", required=True, type=Path)
    parser.add_argument("--seen-portfolio", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _load(path: Path, model):
    return model.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def main() -> int:
    args = parse_args()
    protocol = load_protocol(args.protocol)
    drift = verify_protocol_integrity(
        protocol,
        root=Path.cwd(),
    )
    if drift:
        raise RuntimeError(
            "Frozen protocol implementation drift:\n"
            + "\n".join(drift)
        )

    source = _load(args.seen_input, TrendAwareHypothesisInput)
    run = _load(
        args.seen_run,
        DirectionAwareTrendHypothesisMakerRunRecord,
    )
    draft = _load(
        args.seen_final_draft,
        DirectionAwareTrendHypothesisPortfolioDraft,
    )
    portfolio = _load(
        args.seen_portfolio,
        DirectionAwareTrendHypothesisPortfolio,
    )

    report = evaluate_run(
        root=Path.cwd(),
        protocol=protocol,
        source=source,
        final_draft=draft,
        run_record=run,
        portfolio=portfolio,
        evaluation_mode="seen_regression",
    )
    if not report.accepted:
        raise RuntimeError(
            "Reviewed 5d.1 seen smoke failed frozen 5e evaluation: "
            + ", ".join(
                row.code
                for row in report.issues
                if row.severity == "fatal"
            )
        )

    assert "TREND_UNIVERSAL_ESCALATION" in (
        detect_claim_scope_issues(
            "This relation always holds in all contexts.",
            cross_paper_synthesis=False,
        )
    )
    assert "CROSS_PAPER_OVERCLAIM" in (
        detect_claim_scope_issues(
            "The relation is replicated across independent studies.",
            cross_paper_synthesis=False,
        )
    )
    assert "TREND_CAUSAL_ESCALATION" in (
        detect_claim_scope_issues(
            "The trend demonstrates a causal relationship.",
            cross_paper_synthesis=False,
        )
    )
    clean = detect_claim_scope_issues(
        (
            "This is a provisional contextual association rather than "
            "a causal or universal claim, and cross-paper replication "
            "is not established."
        ),
        cross_paper_synthesis=False,
    )
    if clean:
        raise RuntimeError(
            f"Conservative seen wording was falsely rejected: {clean}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        report.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    print("alpha4c.5e frozen evaluation regression")
    print("Reviewed 5d.1 seen smoke: PASS")
    print("Fatal issues:", report.fatal_issue_count)
    print("Hypotheses:", report.hypothesis_count)
    print("Count thresholds used: False")
    print("Universal overclaim guard: ENFORCED")
    print("Cross-paper overclaim guard: ENFORCED")
    print("Causal-evidence escalation guard: ENFORCED")
    print("Conservative limitation wording: ALLOWED")
    print("Reserve consumed: False")
    print("LLM calls: 0")
    print("Output:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
