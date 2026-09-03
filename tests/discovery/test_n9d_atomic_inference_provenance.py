import json

from pipeline_core.discovery.external_novelty_contracts import (
    ClaimPriorArtCandidateSet,
    HypothesisNoveltyClaims,
    LiteratureQuery,
    NoveltyClaim,
    NoveltyClaimInferenceProvenance,
    PriorArtPacket,
    PriorArtWork,
    QueryExecution,
    RankedPriorArtWork,
)
from pipeline_core.discovery.novelty_closure_execution import (
    ClosureLiteratureQueryPlan,
    ExecutableClosureTarget,
)
from pipeline_core.discovery.novelty_closure_review import (
    ClosureEvidenceMatchDraft,
    ClosureSlotReviewDraft,
    compile_closure_slot_review,
)
from pipeline_core.discovery.novelty_inference_provenance import (
    attach_atomic_inference_provenance,
)

FINAL_HYPOTHESIS_ID = "hypothesis:final"
SOURCE_HYPOTHESIS_ID = "hypothesis:source"
AXIS_ID = "axis:q4"


def _fallback() -> NoveltyClaimInferenceProvenance:
    return NoveltyClaimInferenceProvenance(
        final_hypothesis_id=FINAL_HYPOTHESIS_ID,
        source_review_hypothesis_id=SOURCE_HYPOTHESIS_ID,
        axis_id=AXIS_ID,
        review_status="pass",
        assertion_ids=["central", "pred_h", "pred_her"],
        source_classes=["S_BOUNDED_SYNTHESIS"],
        grounded_statement_ids=["stmt:g1", "stmt:g2"],
        axis_basis=[
            "source candidate: adsorption free energy of oxygenated intermediates "
            "| VARIES_WITH | integrated crystal orbital Hamilton population (ICOHP)"
        ],
    )


def _claim(
    *,
    claim_id: str,
    text: str,
    prediction: str = "",
    identity: str = "relative oxygenated intermediate stabilization",
    relation: str = "hydrogen adsorption free energy",
) -> NoveltyClaim:
    return NoveltyClaim(
        claim_id=claim_id,
        hypothesis_id=FINAL_HYPOTHESIS_ID,
        claim_rank=1,
        kind="moderator_interaction",
        importance="core",
        text=text,
        rationale="test",
        prior_art_identity_terms=[identity] if identity else [],
        relation_nucleus_terms=[relation] if relation else [],
        predicted_observation=prediction,
    )


def _write_inference(tmp_path, assertions):
    path = tmp_path / "axis.inference.json"
    payload = {
        "schema_version": "discovery-axis-inference-artifact-v2",
        "portfolio_id": "portfolio:test",
        "records": [
            {
                "final_hypothesis_id": FINAL_HYPOTHESIS_ID,
                "axis_id": AXIS_ID,
                "source_review_hypothesis_id": SOURCE_HYPOTHESIS_ID,
                "status": "pass",
                "review": {
                    "hypothesis_id": SOURCE_HYPOTHESIS_ID,
                    "axis_id": AXIS_ID,
                    "status": "pass",
                    "assertions": assertions,
                },
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _attach(tmp_path, claims, assertions):
    return attach_atomic_inference_provenance(
        decompositions=[
            HypothesisNoveltyClaims(
                hypothesis_id=FINAL_HYPOTHESIS_ID,
                title="Q4",
                claims=claims,
            )
        ],
        inference_audit_path=_write_inference(tmp_path, assertions),
        fallback_by_hypothesis={FINAL_HYPOTHESIS_ID: _fallback()},
    )[0]


def test_q4_surface_reexpression_binds_atomically(tmp_path):
    assertion = {
        "assertion_id": "central",
        "assertion_text": (
            "Relative free-energy stabilization of oxygenated intermediates may "
            "condition the relationship between M-H iCOHP and hydrogen adsorption "
            "free energy."
        ),
        "source_class": "S_BOUNDED_SYNTHESIS",
        "grounded_statement_ids": ["stmt:g1"],
        "axis_basis": ["adsorption free energy of oxygenated intermediates"],
    }
    group = _attach(
        tmp_path,
        [_claim(claim_id="claim:h", text="conditional H adsorption")],
        [assertion],
    )
    provenance = group.claims[0].inference_provenance
    assert provenance is not None
    assert provenance.binding_scope == "ATOMIC_CLAIM_ASSERTION_BINDING"
    assert provenance.assertion_ids == ["central"]
    assert "atomic_binding_identity_and_relation_lexical" in (
        provenance.binding_reason_codes
    )


def test_exact_prediction_is_preferred_over_broader_identity_matches(tmp_path):
    prediction = (
        "Among comparable M-H iCOHP sites, variation in the free-energy "
        "stabilization of oxygenated intermediates is associated with a "
        "difference in hydrogen adsorption free energy."
    )
    assertions = [
        {
            "assertion_id": "central",
            "assertion_text": (
                "Relative free-energy stabilization of oxygenated intermediates may "
                "condition M-H iCOHP and hydrogen adsorption free energy."
            ),
            "source_class": "S_BOUNDED_SYNTHESIS",
            "grounded_statement_ids": ["stmt:g1"],
            "axis_basis": ["axis basis"],
        },
        {
            "assertion_id": "pred_h",
            "assertion_text": prediction,
            "source_class": "S_BOUNDED_SYNTHESIS",
            "grounded_statement_ids": ["stmt:g2"],
            "axis_basis": ["axis basis h"],
        },
    ]
    group = _attach(
        tmp_path,
        [_claim(claim_id="claim:h", text="conditional H adsorption", prediction=prediction)],
        assertions,
    )
    provenance = group.claims[0].inference_provenance
    assert provenance is not None
    assert provenance.assertion_ids == ["pred_h"]
    assert provenance.grounded_statement_ids == ["stmt:g2"]
    assert provenance.binding_reason_codes == ["atomic_binding_exact_prediction_text"]


def test_atomic_binding_does_not_cross_prediction_branches(tmp_path):
    h_prediction = (
        "Oxygenated-intermediate free-energy stabilization differs with hydrogen "
        "adsorption free energy."
    )
    her_prediction = (
        "Oxygenated-intermediate free-energy stabilization changes the association "
        "between M-H iCOHP and HER activity."
    )
    assertions = [
        {
            "assertion_id": "pred_h",
            "assertion_text": h_prediction,
            "source_class": "S_BOUNDED_SYNTHESIS",
            "grounded_statement_ids": ["stmt:h"],
            "axis_basis": ["axis h"],
        },
        {
            "assertion_id": "pred_her",
            "assertion_text": her_prediction,
            "source_class": "S_BOUNDED_SYNTHESIS",
            "grounded_statement_ids": ["stmt:her"],
            "axis_basis": ["axis her"],
        },
    ]
    group = _attach(
        tmp_path,
        [
            _claim(claim_id="claim:h", text="H branch", prediction=h_prediction),
            _claim(
                claim_id="claim:her",
                text="HER branch",
                prediction=her_prediction,
                relation="HER activity",
            ),
        ],
        assertions,
    )
    by_id = {claim.claim_id: claim.inference_provenance for claim in group.claims}
    assert by_id["claim:h"].assertion_ids == ["pred_h"]
    assert by_id["claim:her"].assertion_ids == ["pred_her"]


def test_semantic_neighbor_does_not_create_atomic_binding(tmp_path):
    assertions = [
        {
            "assertion_id": "neighbor",
            "assertion_text": (
                "Lattice oxygen reactivity correlates with COHP and hydrogen "
                "adsorption energy."
            ),
            "source_class": "S_BOUNDED_SYNTHESIS",
            "grounded_statement_ids": ["stmt:g1"],
            "axis_basis": ["lattice oxygen reactivity"],
        }
    ]
    group = _attach(
        tmp_path,
        [_claim(claim_id="claim:h", text="conditional H adsorption")],
        assertions,
    )
    provenance = group.claims[0].inference_provenance
    assert provenance is not None
    assert provenance.binding_scope == "HYPOTHESIS_REVIEW_CONTEXT"
    assert provenance.assertion_ids == ["central", "pred_h", "pred_her"]
    assert "atomic_claim_assertion_binding_unresolved" in provenance.binding_reason_codes


def test_missing_atomic_identity_fails_closed_to_hypothesis_context(tmp_path):
    assertions = [
        {
            "assertion_id": "central",
            "assertion_text": "Some bounded synthesis.",
            "source_class": "S_BOUNDED_SYNTHESIS",
            "grounded_statement_ids": [],
            "axis_basis": ["axis"],
        }
    ]
    group = _attach(
        tmp_path,
        [_claim(claim_id="claim:no-id", text="claim", identity="")],
        assertions,
    )
    provenance = group.claims[0].inference_provenance
    assert provenance is not None
    assert provenance.binding_scope == "HYPOTHESIS_REVIEW_CONTEXT"
    assert "atomic_binding_missing_identity_terms" in provenance.binding_reason_codes


def _closure_review(*, slot: str, abstract: str, anchors):
    target = ExecutableClosureTarget(
        target_id="closure_target:1",
        slot=slot,
        source_claim_id="claim:1",
        target_basis="test",
        search_terms=("test",),
        search_query="test",
        source_text="test",
        identity_anchor_terms=tuple(anchors),
    )
    query = LiteratureQuery(
        query_id="query:1",
        hypothesis_id="hypothesis:1",
        claim_id=target.target_id,
        query_kind="claim_primary",
        query_text="test",
    )
    plan = ClosureLiteratureQueryPlan(
        plan_id="plan:1",
        plan_sha256="sha",
        source_portfolio_id="portfolio:1",
        source_hypothesis_id="hypothesis:1",
        source_claim_id="claim:1",
        queries=[query],
        targets=[target],
    )
    packet = PriorArtPacket(
        packet_id="packet:1",
        packet_sha256="sha",
        source_portfolio_id="portfolio:1",
        source_query_plan_id=plan.plan_id,
        searched_at_utc="2026-09-03T00:00:00Z",
        works=[PriorArtWork(work_id="work:1", title="Test work", abstract=abstract)],
        executions=[
            QueryExecution(
                query_id=query.query_id,
                provider="test",
                success=True,
                result_count=1,
            )
        ],
    )
    candidates = ClaimPriorArtCandidateSet(
        hypothesis_id="hypothesis:1",
        claim_id=target.target_id,
        ranked_works=[
            RankedPriorArtWork(
                work_id="work:1",
                relevance_score=0.9,
                semantic_similarity=0.9,
                lexical_coverage=0.9,
                abstract_available=True,
            )
        ],
    )
    draft = ClosureSlotReviewDraft(
        matches=[
            ClosureEvidenceMatchDraft(
                work_id="work:1",
                relationship="ESTABLISHES_SLOT",
                confidence=0.9,
                rationale="reviewer positive",
            )
        ],
        interpretation="test",
    )
    return compile_closure_slot_review(
        target=target,
        draft=draft,
        candidates=candidates,
        packet=packet,
        plan=plan,
    )


def test_positive_nonbase_match_requires_same_identity_contract():
    review = _closure_review(
        slot="DISTINGUISHING_FACTOR_EFFECT",
        abstract=(
            "Lattice oxygen reactivity correlates with COHP and hydrogen adsorption energy."
        ),
        anchors=["oxygenated intermediate stabilization"],
    )
    assert review.matches[0].relationship == "COMPONENT_ONLY"
    assert "positive_slot_match_downgraded_identity_mismatch" in review.reason_codes


def test_positive_identity_match_remains_established():
    review = _closure_review(
        slot="DISTINGUISHING_FACTOR_EFFECT",
        abstract=(
            "The stabilization energies of oxygenated intermediates were compared across catalysts."
        ),
        anchors=["relative oxygenated intermediate stabilization"],
    )
    assert review.matches[0].relationship == "ESTABLISHES_SLOT"
    assert review.evidence_state == "ESTABLISHED"


def test_base_positive_match_is_not_identity_guarded():
    review = _closure_review(
        slot="BASE_RELATION",
        abstract="M-H iCOHP correlates with hydrogen adsorption free energy.",
        anchors=[],
    )
    assert review.matches[0].relationship == "ESTABLISHES_SLOT"
    assert review.evidence_state == "ESTABLISHED"
