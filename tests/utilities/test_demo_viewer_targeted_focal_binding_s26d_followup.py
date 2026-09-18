from __future__ import annotations

import json
from pathlib import Path

from scripts.utilities.demo_viewer_runtime import (
    load_core_demo_payload,
)


def _dump(path: Path, value: dict) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        json.dumps(value),
        encoding="utf-8",
    )


def test_kept_original_uses_focal_card_from_multi_card_targeted_report(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"

    original_a = "hypothesis:aaaaaaaaaaaaaaaaaaaa"
    original_b = "hypothesis:bbbbbbbbbbbbbbbbbbbb"
    original_c = "hypothesis:cccccccccccccccccccc"
    final_b = "hypothesis:dddddddddddddddddddd"

    _dump(
        run / "hypothesis.context.json",
        {
            "context_id": "context:test",
            "task_id": "task:test",
            "question": "Q",
            "corpus_id": "corpus:test",
            "domain_profile_id": "sers_au_ag",
            "evidence_statements": [],
        },
    )

    _dump(
        run / "novelty_refinement_a6.portfolio.json",
        {
            "portfolio_id": "portfolio:final",
            "domain_profile_id": "sers_au_ag",
            "hypotheses": [
                {
                    "hypothesis_id": final_b,
                    "title": "Final B",
                    "hypothesis_statement": "Final B statement",
                    "hypothesis_type": "context_dependency",
                    "premise_statement_ids": [],
                    "predicted_observations": [],
                    "falsification_criteria": [],
                }
            ],
        },
    )

    _dump(
        run / "external_novelty_a52.report.json",
        {
            "cards": [
                {
                    "hypothesis_id": original_b,
                    "status": "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP",
                    "interpretation": "initial-B",
                    "claim_reviews": [],
                }
            ]
        },
    )

    _dump(
        run / "novelty_refinement_a6.report.json",
        {
            "attempts": [
                {
                    "original_hypothesis_id": original_b,
                    "candidate_hypothesis_id": original_b,
                    "final_hypothesis_id": final_b,
                    "decision": "kept_original",
                    "targeted_external_status": "NEW_COMBINATION_OF_KNOWN_EFFECTS",
                    "final_external_status": "NEW_COMBINATION_OF_KNOWN_EFFECTS",
                }
            ]
        },
    )

    detail = (
        run
        / "novelty_refinement_a6.external"
    )

    # This report contains the whole portfolio, but its filename declares B
    # as the focal targeted reassessment.
    _dump(
        detail
        / "targeted_02_bbbbbbbbbbbbbbbbbbbb.report.json",
        {
            "cards": [
                {
                    "hypothesis_id": original_a,
                    "status": "LITERATURE_SUPPORTED_EXTENSION",
                    "interpretation": "nonfocal-A",
                    "claim_reviews": [],
                },
                {
                    "hypothesis_id": original_b,
                    "status": "NEW_COMBINATION_OF_KNOWN_EFFECTS",
                    "interpretation": "focal-B",
                    "claim_reviews": [
                        {
                            "claim_id": "B",
                            "status": "COMPONENTS_ONLY",
                        }
                    ],
                },
                {
                    "hypothesis_id": original_c,
                    "status": "NEW_COMBINATION_OF_KNOWN_EFFECTS",
                    "interpretation": "nonfocal-C",
                    "claim_reviews": [],
                },
            ]
        },
    )

    # A second reassessment also contains B, but C is focal here. Flattening
    # all report cards would incorrectly overwrite B with this stale/nonfocal
    # assessment.
    _dump(
        detail
        / "targeted_03_cccccccccccccccccccc.report.json",
        {
            "cards": [
                {
                    "hypothesis_id": original_b,
                    "status": "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP",
                    "interpretation": "nonfocal-B-wrong",
                    "claim_reviews": [],
                },
                {
                    "hypothesis_id": original_c,
                    "status": "NEW_COMBINATION_OF_KNOWN_EFFECTS",
                    "interpretation": "focal-C",
                    "claim_reviews": [],
                },
            ]
        },
    )

    _dump(
        run / "e2e_runner.manifest.json",
        {
            "domain_profile_id": "sers_au_ag",
        },
    )

    row = load_core_demo_payload(
        run
    )["hypotheses"][0]

    assert (
        row["novelty"]["hypothesis_id"]
        == original_b
    )
    assert (
        row["novelty"]["status"]
        == "NEW_COMBINATION_OF_KNOWN_EFFECTS"
    )
    assert (
        row["novelty"]["interpretation"]
        == "focal-B"
    )
    assert (
        row["novelty_binding"]["source_kind"]
        == "targeted"
    )
    assert (
        row["novelty_binding"]["source_hypothesis_id"]
        == original_b
    )
    assert (
        row["novelty_binding"]["source_card_found"]
        is True
    )
    assert (
        row["novelty_binding"]["status_consistent"]
        is True
    )


def test_ambiguous_or_missing_focal_card_fails_closed(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run2"
    original = "hypothesis:bbbbbbbbbbbbbbbbbbbb"
    final_id = "hypothesis:dddddddddddddddddddd"

    _dump(
        run / "hypothesis.context.json",
        {
            "context_id": "context:test",
            "task_id": "task:test",
            "question": "Q",
            "corpus_id": "corpus:test",
            "domain_profile_id": "sers_au_ag",
            "evidence_statements": [],
        },
    )
    _dump(
        run / "novelty_refinement_a6.portfolio.json",
        {
            "portfolio_id": "portfolio:final",
            "domain_profile_id": "sers_au_ag",
            "hypotheses": [
                {
                    "hypothesis_id": final_id,
                    "title": "Final",
                    "hypothesis_statement": "Final",
                    "hypothesis_type": "context_dependency",
                    "premise_statement_ids": [],
                    "predicted_observations": [],
                    "falsification_criteria": [],
                }
            ],
        },
    )
    _dump(
        run / "external_novelty_a52.report.json",
        {
            "cards": [
                {
                    "hypothesis_id": original,
                    "status": "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP",
                    "interpretation": "initial",
                    "claim_reviews": [],
                }
            ]
        },
    )
    _dump(
        run / "novelty_refinement_a6.report.json",
        {
            "attempts": [
                {
                    "original_hypothesis_id": original,
                    "candidate_hypothesis_id": original,
                    "final_hypothesis_id": final_id,
                    "decision": "kept_original",
                    "final_external_status": "NEW_COMBINATION_OF_KNOWN_EFFECTS",
                }
            ]
        },
    )
    _dump(
        run
        / "novelty_refinement_a6.external"
        / "targeted_02_bbbbbbbbbbbbbbbbbbbb.report.json",
        {
            "cards": [
                {
                    "hypothesis_id": "hypothesis:aaaaaaaaaaaaaaaaaaaa",
                    "status": "NEW_COMBINATION_OF_KNOWN_EFFECTS",
                    "interpretation": "wrong-focal",
                    "claim_reviews": [],
                }
            ]
        },
    )
    _dump(
        run / "e2e_runner.manifest.json",
        {
            "domain_profile_id": "sers_au_ag",
        },
    )

    row = load_core_demo_payload(
        run
    )["hypotheses"][0]

    assert row["novelty"] == {}
    assert (
        row["novelty_binding"]["source_card_found"]
        is False
    )
