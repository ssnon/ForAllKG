import json
from pathlib import Path

from scripts.utilities.demo_viewer_runtime import (
    _exploratory_novelty_by_source_hypothesis,
    render_core_demo_html,
)


def test_loads_exploratory_novelty_by_source_hypothesis(tmp_path: Path):
    root = (
        tmp_path
        / "canonical_e2e_v1"
        / "06_exploratory_novelty"
    )
    lineage_dir = root / "lineage" / "source"
    lineage_dir.mkdir(parents=True)

    external_path = lineage_dir / "external_novelty.report.json"
    external_path.write_text(
        json.dumps(
            {
                "cards": [
                    {
                        "hypothesis_id": "hypothesis:regen",
                        "title": "Regenerated orientation hypothesis",
                        "status": "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP",
                        "interpretation": "Components are known; relation remains unresolved.",
                        "reason_codes": [],
                        "coverage": {
                            "query_count": 4,
                            "successful_query_count": 4,
                            "unique_work_count": 78,
                            "abstract_work_count": 40,
                        },
                        "strongest_prior_art_work_ids": ["work:1"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report_path = root / "exploratory_novelty.report.json"
    report_path.write_text(
        json.dumps(
            {
                "external_status_counts": {
                    "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP": 1
                },
                "lineages": [
                    {
                        "source_hypothesis_id": "hypothesis:source",
                        "regenerated_hypothesis_id": "hypothesis:regen",
                        "selected_claim_ids": ["claim:1", "claim:2"],
                        "claim_statuses": {
                            "claim:1": "COMPONENTS_ONLY",
                            "claim:2": "COMPONENTS_ONLY",
                        },
                        "external_status": "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP",
                        "external_report_path": str(external_path),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    observed_path, by_source, report = (
        _exploratory_novelty_by_source_hypothesis(tmp_path)
    )

    assert observed_path == report_path
    assert report["external_status_counts"] == {
        "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP": 1
    }
    row = by_source["hypothesis:source"]
    assert row["regenerated_hypothesis_id"] == "hypothesis:regen"
    assert row["regenerated_title"] == "Regenerated orientation hypothesis"
    assert row["external_status"] == "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP"
    assert row["claim_statuses"] == {
        "claim:1": "COMPONENTS_ONLY",
        "claim:2": "COMPONENTS_ONLY",
    }
    assert row["coverage"]["unique_work_count"] == 78
    assert row["coverage"]["abstract_work_count"] == 40
    assert row["novelty_certification_authority"] is False
    assert row["strict_certification_contract_unchanged"] is True


def test_core_html_labels_exploratory_result_as_non_certifying():
    payload = {
        "question": "Raman demo",
        "domain_profile_id": "sers_au_ag",
        "corpus_id": "sers",
        "portfolio_id": "portfolio:test",
        "paper_ids": [],
        "hypotheses": [
            {
                "hypothesis": {
                    "hypothesis_id": "hypothesis:source",
                    "title": "Orientation hypothesis",
                    "statement": "Orientation changes relative Raman intensity.",
                    "semantic_gate_status": "pass",
                    "novelty_status": "not_assessed",
                },
                "semantic": [],
                "novelty": {},
                "refinement": {},
                "exploratory_novelty": {
                    "regenerated_hypothesis_id": "hypothesis:regen",
                    "regenerated_title": "Regenerated orientation hypothesis",
                    "source_hypothesis_id": "hypothesis:source",
                    "external_status": "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP",
                    "claim_statuses": {
                        "claim:1": "COMPONENTS_ONLY"
                    },
                    "coverage": {
                        "query_count": 4,
                        "successful_query_count": 4,
                        "unique_work_count": 78,
                        "abstract_work_count": 40,
                    },
                    "interpretation": "Components are known.",
                },
            }
        ],
    }

    html = render_core_demo_html(payload, title="Raman demo")

    assert "Exploratory prior-art search" in html
    assert "Exploratory only — not novelty certification." in html
    assert "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP" in html
    assert "COMPONENTS_ONLY" in html
    assert "Regenerated orientation hypothesis" in html
    assert "Exploratory search target" in html
    assert "Source hypothesis" in html
