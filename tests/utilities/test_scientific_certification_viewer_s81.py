from __future__ import annotations

import json
from pathlib import Path

from scripts.utilities.scientific_certification_viewer_runtime import (
    build_scientific_certification_viewer,
    load_scientific_certification_payload,
)


def _dump(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value),
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    hid = "hypothesis:scientific"

    _dump(
        run / "hypothesis.context.json",
        {
            "context_id": "ctx",
            "question": "How does architecture affect SERS reliability?",
            "domain_profile_id": "sers_au_ag",
        },
    )
    _dump(
        run / "scientific_atomic_merged_candidates.portfolio.json",
        {
            "portfolio_id": "portfolio:candidates",
            "domain_profile_id": "sers_au_ag",
            "source_context_id": "ctx",
            "hypotheses": [
                {
                    "hypothesis_id": hid,
                    "title": "Accessible-hotspot stability",
                    "hypothesis_statement": (
                        "Accessible-hotspot stability controls transfer."
                    ),
                    "hypothesis_type": "cross_evidence_synthesis",
                    "premise_statement_ids": ["p1"],
                    "gap_statement_ids": ["g1"],
                    "inferential_bridge": "A grounded bridge.",
                    "assumptions": ["Comparable batches."],
                    "source_paper_ids": ["paper:1"],
                    "predicted_observations": [
                        {
                            "observation_id": "pred:1",
                            "observable": "transfer error",
                            "expected_direction": "decrease",
                            "rationale": "Stability improves transfer.",
                        }
                    ],
                    "falsification_criteria": [
                        {
                            "criterion_id": "false:1",
                            "observable": "transfer error",
                            "falsifying_outcome": "No relation is observed.",
                        }
                    ],
                    "evidence_profile": {"premise_count": 1},
                }
            ],
        },
    )
    _dump(
        run / "scientific_atomic_n10_certification.report.json",
        {
            "candidate_count": 1,
            "certified_count": 0,
            "unresolved_count": 1,
            "rejected_count": 0,
            "decisions": [
                {
                    "hypothesis_id": hid,
                    "selection_class": "CONDITIONAL",
                    "positive_nonobviousness_authority": False,
                    "fallback_allowed": False,
                    "certification_status": "NOVELTY_UNRESOLVED",
                    "novelty_certified": False,
                    "candidate_retained": True,
                    "action": "RESOLVE_NOVELTY_BEARING_EVIDENCE",
                    "unresolved_dimensions": ["EVIDENCE_CLOSURE"],
                    "reason_codes": [
                        "role_aware_v2_conditional_fail_closed"
                    ],
                }
            ],
        },
    )
    _dump(
        run / "scientific_atomic_merged_certified.portfolio.json",
        {
            "portfolio_id": "portfolio:certified",
            "domain_profile_id": "sers_au_ag",
            "hypotheses": [],
            "abstention_reason": "none certified",
        },
    )
    _dump(
        run / "scientific_atomic_merged_final_semantic.review.json",
        {
            "overall_summary": "Accepted with bounded warnings.",
            "dimensions": [
                {
                    "dimension": "premise_fidelity",
                    "verdict": "pass",
                    "rationale": "Grounded.",
                    "hypothesis_ids": [hid],
                },
                {
                    "dimension": "directional_specificity",
                    "verdict": "warning",
                    "rationale": "Direction remains broad.",
                    "hypothesis_ids": [hid],
                },
            ],
        },
    )
    _dump(
        run / "scientific_atomic_cross_lane_v2.report.json",
        {
            "hypotheses": [
                {
                    "hypothesis_id": hid,
                    "atomic_specifications": [
                        {
                            "claim_id": "claim:1",
                            "novelty_selection_role": "NOVELTY_BEARING",
                            "text": "Atomic claim.",
                            "prior_art_identity_terms": [
                                "architecture-conditioned hotspot accessibility"
                            ],
                            "relation_endpoint_anchors": [
                                "hotspot stability",
                                "transfer error",
                            ],
                            "relation_nucleus_terms": [
                                "hotspot stability",
                                "transfer error",
                            ],
                            "required_bridge": "Atomic bridge.",
                            "predicted_observation": "Lower error.",
                            "falsification_condition": "No lower error.",
                        }
                    ],
                }
            ],
        },
    )
    return run


def test_payload_preserves_candidate_and_unresolved_certification(
    tmp_path: Path,
) -> None:
    run = _fixture(tmp_path)
    payload = load_scientific_certification_payload(
        run_dir=run,
    )
    assert payload["candidate_count"] == 1
    assert payload["certified_count"] == 0
    assert payload["unresolved_count"] == 1
    row = payload["hypotheses"][0]
    assert row["semantic"]["status"] == "pass_with_warnings"
    assert (
        row["certification"]["certification_status"]
        == "NOVELTY_UNRESOLVED"
    )
    assert len(row["atomic"]["atomic_specifications"]) == 1


def test_builds_self_contained_certification_html(
    tmp_path: Path,
) -> None:
    run = _fixture(tmp_path)
    output = run / "demo" / "scientific_certification.html"
    result = build_scientific_certification_viewer(
        run_dir=run,
        output=output,
    )
    rendered = result.read_text(encoding="utf-8")
    assert "Accessible-hotspot stability" in rendered
    assert "NOVELTY_UNRESOLVED" in rendered
    assert "Candidate retention and novelty certification" in rendered
    assert "Atomic specification" in rendered
    assert "<script src=" not in rendered
    assert "<link rel=" not in rendered
