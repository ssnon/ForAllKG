from __future__ import annotations

import json
from pathlib import Path

from scripts.utilities.demo_viewer_runtime import (
    load_core_demo_payload,
)


def _dump(
    path: Path,
    value: dict,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        json.dumps(value),
        encoding="utf-8",
    )


def _base_run(
    tmp_path: Path,
    *,
    decision: str,
    original_id: str,
    candidate_id: str,
    final_id: str,
    initial_status: str,
    final_status: str,
) -> Path:
    run = tmp_path / decision

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
        run
        / "novelty_refinement_a6.portfolio.json",
        {
            "portfolio_id": "portfolio:final",
            "domain_profile_id": "sers_au_ag",
            "hypotheses": [
                {
                    "hypothesis_id": final_id,
                    "title": "Final",
                    "hypothesis_statement": "Final H",
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
                    "hypothesis_id": original_id,
                    "status": initial_status,
                    "claim_reviews": [
                        {
                            "claim_id": "initial",
                            "status": "INITIAL",
                        }
                    ],
                    "interpretation": "initial",
                }
            ]
        },
    )

    _dump(
        run / "novelty_refinement_a6.report.json",
        {
            "attempts": [
                {
                    "original_hypothesis_id": original_id,
                    "candidate_hypothesis_id": candidate_id,
                    "final_hypothesis_id": final_id,
                    "decision": decision,
                    "final_external_status": final_status,
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

    return run


def test_accepted_refinement_binds_fresh_final_card(
    tmp_path: Path,
) -> None:
    run = _base_run(
        tmp_path,
        decision="accepted_refinement",
        original_id="hypothesis:original",
        candidate_id="hypothesis:candidate",
        final_id="hypothesis:final",
        initial_status="LITERATURE_SUPPORTED_EXTENSION",
        final_status="KNOWN_COMPONENTS_WITH_RELATIONAL_GAP",
    )

    _dump(
        run
        / "novelty_refinement_a6.external"
        / "final_01_candidate.report.json",
        {
            "cards": [
                {
                    "hypothesis_id": "hypothesis:candidate",
                    "status": "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP",
                    "claim_reviews": [
                        {
                            "claim_id": "fresh",
                            "status": "COMPONENTS_ONLY",
                        }
                    ],
                    "interpretation": "fresh-final",
                }
            ]
        },
    )

    row = load_core_demo_payload(
        run
    )["hypotheses"][0]

    assert (
        row["hypothesis"]["hypothesis_id"]
        == "hypothesis:final"
    )
    assert (
        row["novelty"]["hypothesis_id"]
        == "hypothesis:candidate"
    )
    assert (
        row["novelty"]["status"]
        == "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP"
    )
    assert (
        row["novelty"]["interpretation"]
        == "fresh-final"
    )
    assert (
        row["novelty_binding"]["source_kind"]
        == "fresh_final"
    )
    assert (
        row["novelty_binding"]["status_consistent"]
        is True
    )


def test_kept_original_prefers_targeted_card(
    tmp_path: Path,
) -> None:
    run = _base_run(
        tmp_path,
        decision="kept_original",
        original_id="hypothesis:original",
        candidate_id="hypothesis:original",
        final_id="hypothesis:final",
        initial_status="KNOWN_COMPONENTS_WITH_RELATIONAL_GAP",
        final_status="NEW_COMBINATION_OF_KNOWN_EFFECTS",
    )

    _dump(
        run
        / "novelty_refinement_a6.external"
        / "targeted_01_original.report.json",
        {
            "cards": [
                {
                    "hypothesis_id": "hypothesis:original",
                    "status": "NEW_COMBINATION_OF_KNOWN_EFFECTS",
                    "claim_reviews": [
                        {
                            "claim_id": "targeted",
                            "status": "NO_DIRECT_MATCH_FOUND",
                        }
                    ],
                    "interpretation": "targeted-final",
                }
            ]
        },
    )

    row = load_core_demo_payload(
        run
    )["hypotheses"][0]

    assert (
        row["novelty"]["status"]
        == "NEW_COMBINATION_OF_KNOWN_EFFECTS"
    )
    assert (
        row["novelty"]["interpretation"]
        == "targeted-final"
    )
    assert (
        row["novelty_binding"]["source_kind"]
        == "targeted"
    )


def test_missing_fresh_card_fails_closed_instead_of_showing_initial(
    tmp_path: Path,
) -> None:
    run = _base_run(
        tmp_path,
        decision="accepted_refinement",
        original_id="hypothesis:original",
        candidate_id="hypothesis:candidate",
        final_id="hypothesis:final",
        initial_status="LITERATURE_SUPPORTED_EXTENSION",
        final_status="KNOWN_COMPONENTS_WITH_RELATIONAL_GAP",
    )

    row = load_core_demo_payload(
        run
    )["hypotheses"][0]

    assert row["novelty"] == {}
    assert (
        row["hypothesis"]["novelty_status"]
        == "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP"
    )
    assert (
        row["novelty_binding"]["source_card_found"]
        is False
    )
