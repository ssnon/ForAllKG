from __future__ import annotations

import json
import subprocess
import sys

from scripts.discovery import (
    build_nonobviousness_production_gate_v2 as production_cli,
)


def _candidate():
    return {
        "schema_version":
            "scientific-novelty-fallback-"
            "gate-v2-candidate",

        "candidate_only":
            True,

        "production_authority":
            False,

        "alpha6_original_fallback_authority":
            False,

        "authority_policy":
            "none_candidate_only",

        "source_portfolio_id":
            "portfolio:1",

        "source_query_plan_id":
            "plan:1",

        "gate_count":
            1,

        "candidate_fallback_allowed_count":
            0,

        "candidate_fallback_blocked_count":
            1,

        "selection_counts": {
            "ELIGIBLE": 0,
            "CONDITIONAL": 1,
            "INELIGIBLE": 0,
        },

        "gates": [
            {
                "hypothesis_id":
                    "hypothesis:1",

                "selection_class":
                    "CONDITIONAL",

                "candidate_fallback_allowed":
                    False,

                "candidate_positive_nonobviousness_authority":
                    False,

                "production_authority":
                    False,

                "action":
                    "RESOLVE_NOVELTY_BEARING_EVIDENCE",

                "base_aggregation_action":
                    "RESOLVE_NOVELTY_BEARING_EVIDENCE",

                "blocking_claim_ids":
                    [],

                "unresolved_claim_ids":
                    ["claim:1"],

                "unresolved_selection_role_claim_ids":
                    [],

                "structurally_unresolved_claim_ids":
                    [],

                "resolution_requirements":
                    [],

                "reason_codes":
                    [
                        "role_aware_v2_conditional_fail_closed"
                    ],
            }
        ],
    }


def test_candidate_cli_module_help():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.discovery."
            "build_nonobviousness_production_gate_v2_candidate",
            "--help",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--query-plan" in result.stdout
    assert "--intake-shadow" in result.stdout
    assert "--full-shadow" in result.stdout
    assert "--output" in result.stdout


def test_production_cli_module_help():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.discovery."
            "build_nonobviousness_production_gate_v2",
            "--help",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--candidate-gate" in result.stdout
    assert "--output" in result.stdout


def test_production_cli_preserves_conditional_negative(
    monkeypatch,
    tmp_path,
):
    source = (
        tmp_path
        / "candidate.json"
    )

    output = (
        tmp_path
        / "production.json"
    )

    source.write_text(
        json.dumps(
            _candidate()
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_nonobviousness_production_gate_v2",
            "--candidate-gate",
            str(source),
            "--output",
            str(output),
        ],
    )

    production_cli.main()

    artifact = json.loads(
        output.read_text(
            encoding="utf-8"
        )
    )

    assert (
        artifact["schema_version"]
        == "scientific-novelty-fallback-gate-v2"
    )

    assert (
        artifact["production_authority"]
        is True
    )

    assert (
        artifact["gates"][0][
            "selection_class"
        ]
        == "CONDITIONAL"
    )

    assert (
        artifact["gates"][0][
            "fallback_allowed"
        ]
        is False
    )
