
from __future__ import annotations

import json
from pathlib import Path

from pipeline_core.discovery.development_human_review import (
    ARMS,
    build_review_bundle,
    render_blind_html,
    aggregate_review_responses,
)


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _hypothesis(hid: str) -> dict:
    return {
        "hypothesis_id": hid,
        "domain_profile_id": "demo",
        "title": f"title {hid}",
        "hypothesis_statement": f"statement {hid}",
        "hypothesis_type": "context_dependency",
        "premise_statement_ids": ["s1"],
        "inferential_bridge": "bridge",
        "predicted_observations": [
            {
                "observable": "y",
                "expected_direction": "increase",
                "rationale": "r",
            }
        ],
        "falsification_criteria": [
            {
                "observable": "y",
                "falsifying_outcome": "no change",
            }
        ],
        "assumptions": [],
        "source_paper_ids": ["p1"],
        "cross_paper_synthesis": False,
        "candidate_dependency": "none",
    }


def _make_case(root: Path, case_id: str, counts: dict[str, int]) -> None:
    run_dir = root / "source_runs" / case_id.lower()
    _write(
        run_dir / "hypothesis.context.json",
        {
            "question": "Q?",
            "domain_profile_id": "demo",
            "evidence_statements": [
                {
                    "statement_id": "s1",
                    "text": "grounded premise",
                    "epistemic_role": "reported",
                    "claim_kind": "result",
                    "paper_ids": ["p1"],
                    "eligible_as_premise": True,
                }
            ],
        },
    )
    status_path = root / "execution_status.json"
    status = _read(status_path) if status_path.exists() else {"cases": []}
    status["cases"].append({"case_id": case_id, "run_dir": str(run_dir)})
    _write(status_path, status)

    for arm in ARMS:
        name = (
            "portfolio.json"
            if arm in {"LEGACY", "PORTFOLIO_SELECTED"}
            else "materialized.portfolio.json"
        )
        _write(
            root / case_id / arm / name,
            {
                "hypotheses": [
                    _hypothesis(f"{arm}:{i}")
                    for i in range(counts[arm])
                ]
            },
        )


def test_balanced_sampling_is_deterministic_and_blind(tmp_path: Path) -> None:
    counts = {arm: 3 for arm in ARMS}
    _make_case(tmp_path, "CASE_A", counts)
    a = build_review_bundle(tmp_path, per_arm_per_case=2, seed="fixed")
    b = build_review_bundle(tmp_path, per_arm_per_case=2, seed="fixed")
    assert a["key"] == b["key"]
    assert a["manifest"]["sample_count"] == 8
    assert len({row["blind_id"] for row in a["key"]["rows"]}) == 8

    for item in a["blind"]["items"]:
        assert "arm" not in item
        assert "hypothesis_id" not in item
        assert "candidate_id" not in item
        assert "premises" not in item
        assert "verification" not in item

    html_text = render_blind_html(a["blind"])
    assert "production_selection_authority" not in html_text


def test_zero_arm_case_is_excluded_in_paired_mode(tmp_path: Path) -> None:
    counts = {arm: 2 for arm in ARMS}
    counts["EVOLUTION_BALANCED"] = 0
    _make_case(tmp_path, "CASE_B", counts)
    bundle = build_review_bundle(tmp_path, per_arm_per_case=2, seed="fixed")
    assert bundle["manifest"]["sample_count"] == 0
    assert bundle["manifest"]["excluded_cases"][0]["case_id"] == "CASE_B"


def test_aggregation_unblinds_only_with_private_key(tmp_path: Path) -> None:
    counts = {arm: 2 for arm in ARMS}
    _make_case(tmp_path, "CASE_C", counts)
    bundle = build_review_bundle(tmp_path, per_arm_per_case=1, seed="fixed")
    key_path = tmp_path / "key.json"
    _write(key_path, bundle["key"])

    ratings = {
        row["blind_id"]: {
            "ratings": {
                "scientific_interest": 4,
                "conceptual_distinctiveness": 5,
                "mechanistic_plausibility": 4,
                "falsifiability": 4,
                "actionability": 3,
                "information_gain": 5,
                "pursue_likelihood": 4,
            },
            "strengths": "",
            "concerns": "",
        }
        for row in bundle["key"]["rows"]
    }
    response = tmp_path / "r.json"
    _write(
        response,
        {
            "phase": "BLIND",
            "bundle_id": bundle["manifest"]["bundle_id"],
            "reviewer_code": "r1",
            "ratings": ratings,
        },
    )
    result = aggregate_review_responses(
        key_path=key_path,
        response_paths=[response],
    )
    assert result["rating_row_count"] == 4
    for arm in ARMS:
        assert (
            result["arm_summary"][arm]["conceptual_distinctiveness"]["mean"]
            == 5.0
        )
        assert result["arm_summary"][arm]["pursue_likelihood"]["n"] == 1


def test_aggregation_prefers_reveal_over_duplicate_blind_file(tmp_path: Path) -> None:
    counts = {arm: 1 for arm in ARMS}
    _make_case(tmp_path, "CASE_D", counts)
    bundle = build_review_bundle(tmp_path, per_arm_per_case=1, seed="fixed")
    key_path = tmp_path / "key.json"
    _write(key_path, bundle["key"])
    ratings = {
        row["blind_id"]: {
            "ratings": {
                "scientific_interest": 4,
                "conceptual_distinctiveness": 4,
                "mechanistic_plausibility": 4,
                "falsifiability": 4,
                "actionability": 4,
                "information_gain": 4,
                "pursue_likelihood": 4,
            }
        }
        for row in bundle["key"]["rows"]
    }
    blind = {
        "phase": "BLIND",
        "bundle_id": bundle["manifest"]["bundle_id"],
        "reviewer_code": "r1",
        "ratings": ratings,
    }
    reveal = {
        "phase": "REVEAL",
        "bundle_id": bundle["manifest"]["bundle_id"],
        "reviewer_code": "r1",
        "blind_response": blind,
        "post_reveal": {
            row["blind_id"]: {"trust_delta": 1}
            for row in bundle["key"]["rows"]
        },
    }
    blind_path = tmp_path / "blind.json"
    reveal_path = tmp_path / "reveal.json"
    _write(blind_path, blind)
    _write(reveal_path, reveal)
    result = aggregate_review_responses(
        key_path=key_path,
        response_paths=[blind_path, reveal_path],
    )
    assert result["rating_row_count"] == 4
    assert result["unique_reviewer_item_count"] == 4
    for arm in ARMS:
        assert result["arm_summary"][arm]["scientific_interest"]["n"] == 1
        assert result["post_reveal_trust_delta_by_arm"][arm]["mean"] == 1.0
