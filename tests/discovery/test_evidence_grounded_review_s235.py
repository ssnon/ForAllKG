from pipeline_core.discovery.external_novelty_contracts import PriorArtMatchDraft


def test_prior_art_match_draft_accepts_evidence_spans():
    row = PriorArtMatchDraft(
        work_id="w1",
        relationship="PARTIAL_PRIOR_ART",
        confidence=0.9,
        rationale="r",
        evidence_spans=["A changes B."],
    )
    assert row.evidence_spans == ["A changes B."]


def test_prior_art_match_draft_remains_backward_compatible():
    row = PriorArtMatchDraft(
        work_id="w1",
        relationship="COMPONENT_ONLY",
        confidence=0.8,
        rationale="r",
    )
    assert row.evidence_spans == []
