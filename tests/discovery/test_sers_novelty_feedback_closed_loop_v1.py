
from types import SimpleNamespace

from pipeline_core.discovery.sers_novelty_feedback_closed_loop import (
    _external_boundaries,
    audit_gen1,
)


def test_external_boundaries_separate_known_and_unresolved():
    direct = SimpleNamespace(
        claim_id="c1",
        status="DIRECT_PRIOR_ART",
        claim_text="A changes B",
        importance="core",
        matches=[SimpleNamespace(title="Known work")],
    )
    open_claim = SimpleNamespace(
        claim_id="c2",
        status="COMPONENTS_ONLY",
        claim_text="B jointly changes C and D",
        importance="core",
        matches=[],
    )
    card = SimpleNamespace(claim_reviews=[direct, open_claim])
    known, unresolved, targets = _external_boundaries(card)
    assert targets == ["c1"]
    assert "Known work" in known[0]
    assert "COMPONENTS_ONLY" in unresolved[0]


def test_gen1_audit_flags_known_axis_repeat():
    generation = {
        "report_id": "g",
        "records": [
            {
                "source_hypothesis_id": "h0",
                "route": "FRESH_CONTEXT_REAXIS",
                "decision": "ACCEPTED_GENERATION_SHADOW",
                "generated_hypothesis_id": "h1",
            }
        ],
    }
    external = {
        "cards": [
            {
                "hypothesis_id": "h1",
                "status": "PLAUSIBLY_NOVEL",
            }
        ]
    }
    aggregation = {
        "composites": [
            {
                "hypothesis_id": "h1",
                "aggregation_disposition": "NO_RESIDUAL",
            }
        ]
    }
    report = audit_gen1(
        generation_report=generation,
        external_report=external,
        aggregation=aggregation,
        cohort_audit={"pass": True},
    )
    assert report["known_axis_repeat_count"] == 1
    assert report["rows"][0]["outcome"] == "KNOWN_AXIS_REPEAT"
    assert report["novelty_authority_created"] is False
