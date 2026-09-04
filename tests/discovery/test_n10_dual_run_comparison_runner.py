import json
import sys

from pipeline_core.discovery.external_novelty_contracts import (
    HypothesisNoveltyClaims,
    LiteratureQueryPlan,
    NoveltyClaim,
)
from scripts.discovery import (
    build_nonobviousness_dual_run_comparison as runner,
)


def _plan():
    claim = NoveltyClaim(
        claim_id="claim:novel",
        hypothesis_id="hypothesis:1",
        claim_rank=1,
        kind="mediator",
        importance="core",
        novelty_selection_role="NOVELTY_BEARING",
        text="Synthetic novelty claim.",
        rationale="Synthetic runner test.",
    )

    return LiteratureQueryPlan(
        plan_id="plan:1",
        plan_sha256="plan-sha",
        source_portfolio_id="portfolio:1",
        claims=[
            HypothesisNoveltyClaims(
                hypothesis_id="hypothesis:1",
                title="Synthetic",
                claims=[claim],
            )
        ],
    )


def _intake():
    return {
        "schema_version":
            "nonobviousness-shadow-v1",
        "shadow_only": True,
        "source_portfolio_id":
            "portfolio:1",
        "source_query_plan_id":
            "plan:1",
        "source_external_report_id":
            "report:1",
        "hypotheses": [
            {
                "hypothesis_id":
                    "hypothesis:1",
                "claims": [
                    {
                        "claim": {
                            "claim_id":
                                "claim:novel",
                            "importance":
                                "core",
                        },
                        "shadow_state":
                            "READY_FOR_CLOSURE",
                        "specification": {},
                    }
                ],
            }
        ],
    }


def _full():
    return {
        "schema_version":
            "nonobviousness-full-shadow-v1",
        "shadow_only": True,
        "source_portfolio_id":
            "portfolio:1",
        "claims": [
            {
                "claim_id":
                    "claim:novel",
                "final_verdict":
                    "POTENTIALLY_NON_OBVIOUS",
                "final_reason_codes": [
                    "synthetic_positive",
                ],
            }
        ],
    }


def test_runner_writes_comparison_only_artifact(
    tmp_path,
    monkeypatch,
):
    plan_path = tmp_path / "plan.json"
    intake_path = tmp_path / "intake.json"
    full_path = tmp_path / "full.json"
    output_path = tmp_path / "comparison.json"

    plan_path.write_text(
        _plan().model_dump_json(
            indent=2
        )
        + "\n",
        encoding="utf-8",
    )

    intake_path.write_text(
        json.dumps(
            _intake(),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    full_path.write_text(
        json.dumps(
            _full(),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_nonobviousness_dual_run_comparison",
            "--query-plan",
            str(plan_path),
            "--intake-shadow",
            str(intake_path),
            "--full-shadow",
            str(full_path),
            "--output",
            str(output_path),
        ],
    )

    assert runner.main() == 0

    artifact = json.loads(
        output_path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        artifact["schema_version"]
        == "nonobviousness-dual-run-comparison-v1"
    )

    assert artifact["comparison_only"] is True
    assert artifact["production_authority"] is False

    assert (
        artifact[
            "candidate_has_production_authority"
        ]
        is False
    )

    assert artifact["authority_policy"] == "v1_only"

    row = artifact["comparisons"][0]

    assert (
        row[
            "observed_production_fallback_allowed"
        ]
        is True
    )

    assert (
        row["v1"]["fallback_allowed"]
        is True
    )


def test_runner_has_no_production_output_argument(
    monkeypatch,
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_nonobviousness_dual_run_comparison",
            "--help",
        ],
    )

    import pytest

    with pytest.raises(SystemExit) as exc:
        runner.parse_args()

    assert exc.value.code == 0
