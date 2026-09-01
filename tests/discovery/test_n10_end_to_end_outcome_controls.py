from pipeline_core.discovery.external_novelty_contracts import (
    PriorArtPacket,
    PriorArtWork,
)
from pipeline_core.discovery.novelty_adjudication import (
    EstablishedPriorArtRelation,
    NonObviousnessAdjudicationDraft,
    NonObviousnessAdjudicationVector,
    NonObviousnessEvidencePacket,
    assess_adjudication_readiness,
    compile_nonobviousness_adjudication,
)
from pipeline_core.discovery.novelty_adjudication_llm import (
    review_and_compile_nonobviousness_adjudication,
)
from pipeline_core.discovery.novelty_closure_compiler import (
    compile_nonobviousness_evidence_closure,
)
from pipeline_core.discovery.novelty_closure_relationships import (
    ClosureRelationshipAssessmentDraft,
    compile_closure_relationship_assessment,
)
from pipeline_core.discovery.novelty_nonobviousness import (
    ResidualClaimStructure,
    assess_structural_nonobviousness,
)


BRIDGE = (
    "Laser power drives a transition at Pc that changes how "
    "spacing maps to measured SERS enhancement."
)

PREDICTION = (
    "Below and above Pc, the spacing-to-SERS response "
    "occupies two distinguishable regimes."
)

FALSIFIER = (
    "The spacing-to-SERS response varies smoothly with power "
    "and shows no reproducible regime boundary."
)


def _review(
    slot,
    state,
    *,
    ids=(),
    negative=True,
):
    return {
        "slot": slot,
        "evidence_state": state,
        "positive_work_ids": list(ids),
        "negative_coverage_sufficient": negative,
    }


def _lower_order_reviews(full_state="NOT_FOUND"):
    rows = [
        _review(
            "BASE_RELATION",
            "ESTABLISHED",
            ids=("work:base",),
        ),
        _review(
            "DISTINGUISHING_FACTOR_EFFECT",
            "ESTABLISHED",
            ids=("work:factor",),
        ),
        _review(
            "BRIDGE_RELATION",
            "ESTABLISHED",
            ids=("work:bridge",),
        ),
    ]

    if full_state == "ESTABLISHED":
        rows.append(
            _review(
                "FULL_RELATION",
                "ESTABLISHED",
                ids=("work:full",),
            )
        )
    elif full_state == "UNASSESSED":
        rows.append(
            _review(
                "FULL_RELATION",
                "UNASSESSED",
                negative=False,
            )
        )
    else:
        rows.append(
            _review(
                "FULL_RELATION",
                "NOT_FOUND",
                negative=True,
            )
        )

    return rows


def _relationship(
    reviews,
    bridge_kind,
):
    return compile_closure_relationship_assessment(
        reviews=reviews,
        draft=ClosureRelationshipAssessmentDraft(
            bridge_kind=bridge_kind,
            scope_compatibility="COMPATIBLE",
            bridge_basis_work_ids=[
                "work:bridge",
            ],
            scope_basis_work_ids=[
                "work:base",
                "work:factor",
                "work:bridge",
            ],
            interpretation="Synthetic deterministic control.",
        ),
    )


def _closure(
    reviews,
    relationship,
):
    return compile_nonobviousness_evidence_closure(
        reviews=reviews,
        bridge_kind=relationship.bridge_kind,
        scope_compatible=relationship.scope_compatible,
    ).closure


def _generic_vector():
    return NonObviousnessAdjudicationVector(
        inferential_distance="LOCAL_REPHRASE",
        mechanistic_necessity="NO_NEW_MECHANISM",
        regime_specificity="NONE",
        counterintuitiveness="EXPECTED",
        testable_distinctiveness="GENERIC",
        required_bridge="",
        predicted_observation="",
        falsification_condition="",
    )


def _threshold_vector():
    return NonObviousnessAdjudicationVector(
        inferential_distance="NEW_REGIME_STRUCTURE",
        mechanistic_necessity="NEW_BRIDGE_REQUIRED",
        regime_specificity="THRESHOLD",
        counterintuitiveness="NONTRIVIAL",
        testable_distinctiveness="QUANTITATIVE",
        required_bridge=BRIDGE,
        predicted_observation=PREDICTION,
        falsification_condition=FALSIFIER,
    )


def _neutral_draft():
    return NonObviousnessAdjudicationDraft(
        proposed_verdict="INSUFFICIENT_FOR_JUDGMENT",
        direct_reconstruction_from_known_relations=False,
        additional_scientific_assumptions=(),
        prediction_distinguishes_from_routine_baseline=False,
        falsifier_is_specific=False,
        concise_basis="Synthetic control.",
    )


def _packet(
    *,
    structural_status,
    vector,
    full_known=False,
):
    return NonObviousnessEvidencePacket(
        claim_id="claim:test",
        claim_text=(
            "A critical laser power Pc separates two distinct "
            "spacing-to-SERS regimes."
        ),
        structural_status=structural_status,
        vector=vector,
        established_relations=(
            EstablishedPriorArtRelation(
                relation_statement=(
                    "Interparticle spacing affects SERS enhancement."
                ),
                relationship_status="ESTABLISHED",
                work_ids=("work:base",),
                scope_note="compatible",
            ),
            EstablishedPriorArtRelation(
                relation_statement=(
                    "Laser power affects a relevant local state."
                ),
                relationship_status="ESTABLISHED",
                work_ids=("work:factor",),
                scope_note="compatible",
            ),
            EstablishedPriorArtRelation(
                relation_statement=(
                    "Laser power connects to a state relevant to "
                    "spacing-sensitive SERS."
                ),
                relationship_status="ESTABLISHED",
                work_ids=("work:bridge",),
                scope_note="compatible",
            ),
        ),
        direct_full_claim_prior_art=full_known,
        evidence_closure_sufficient=(
            structural_status != "INSUFFICIENT_CLOSURE"
        ),
    )


def _prior():
    return PriorArtPacket(
        packet_id="packet:p",
        packet_sha256="sha",
        source_portfolio_id="portfolio:p",
        source_query_plan_id="plan:p",
        searched_at_utc="2026-09-01T00:00:00Z",
        works=[
            PriorArtWork(
                work_id="work:base",
                title="Base",
                abstract="Spacing affects SERS enhancement.",
            ),
            PriorArtWork(
                work_id="work:factor",
                title="Factor",
                abstract="Laser power affects a relevant local state.",
            ),
            PriorArtWork(
                work_id="work:bridge",
                title="Bridge",
                abstract=(
                    "Laser power connects to a state relevant to "
                    "spacing-sensitive SERS."
                ),
            ),
        ],
    )


class _PotentialBackend:
    def adjudicate(
        self,
        *,
        packet,
        prior_art,
    ):
        return NonObviousnessAdjudicationDraft(
            proposed_verdict="POTENTIALLY_NON_OBVIOUS",
            direct_reconstruction_from_known_relations=False,
            additional_scientific_assumptions=(BRIDGE,),
            prediction_distinguishes_from_routine_baseline=True,
            falsifier_is_specific=True,
            concise_basis=(
                "The established lower-order relations do not "
                "supply the explicit Pc regime transition."
            ),
        )


def test_direct_full_prior_art_short_circuits_to_routine():
    reviews = _lower_order_reviews(
        full_state="ESTABLISHED"
    )

    relationship = _relationship(
        reviews,
        "MEDIATION_CHAIN",
    )

    closure = _closure(
        reviews,
        relationship,
    )

    structural = assess_structural_nonobviousness(
        closure,
        ResidualClaimStructure(
            claim_kind="distinctive_prediction",
            introduces_threshold=True,
            introduces_regime_change=True,
        ),
    )

    assert structural.status == "DIRECTLY_KNOWN"

    readiness = assess_adjudication_readiness(
        structural_status=structural.status,
        vector=_threshold_vector(),
    )

    final = compile_nonobviousness_adjudication(
        readiness=readiness,
        packet=_packet(
            structural_status=structural.status,
            vector=_threshold_vector(),
            full_known=True,
        ),
        draft=_neutral_draft(),
    )

    assert final.verdict == "ROUTINE_FROM_PRIOR_ART"


def test_interaction_compatible_lower_order_closure_is_routine():
    reviews = _lower_order_reviews()

    relationship = _relationship(
        reviews,
        "INTERACTION_COMPATIBLE",
    )

    assert relationship.scope_compatible is True

    closure = _closure(
        reviews,
        relationship,
    )

    structural = assess_structural_nonobviousness(
        closure,
        ResidualClaimStructure(
            claim_kind="moderator_interaction",
        ),
    )

    assert structural.status == "ROUTINE_COMPOSITION"

    readiness = assess_adjudication_readiness(
        structural_status=structural.status,
        vector=_generic_vector(),
    )

    final = compile_nonobviousness_adjudication(
        readiness=readiness,
        packet=_packet(
            structural_status=structural.status,
            vector=_generic_vector(),
        ),
        draft=_neutral_draft(),
    )

    assert final.verdict == "ROUTINE_FROM_PRIOR_ART"


def test_unassessed_full_relation_stops_as_insufficient():
    reviews = _lower_order_reviews(
        full_state="UNASSESSED"
    )

    relationship = _relationship(
        reviews,
        "MEDIATION_CHAIN",
    )

    closure = _closure(
        reviews,
        relationship,
    )

    structural = assess_structural_nonobviousness(
        closure,
        ResidualClaimStructure(
            claim_kind="distinctive_prediction",
            introduces_threshold=True,
            introduces_regime_change=True,
        ),
    )

    assert structural.status == "INSUFFICIENT_CLOSURE"

    readiness = assess_adjudication_readiness(
        structural_status=structural.status,
        vector=_threshold_vector(),
    )

    final = compile_nonobviousness_adjudication(
        readiness=readiness,
        packet=_packet(
            structural_status=structural.status,
            vector=_threshold_vector(),
        ),
        draft=_neutral_draft(),
    )

    assert final.verdict == "INSUFFICIENT_FOR_JUDGMENT"


def test_explicit_threshold_can_reach_potentially_nonobvious():
    reviews = _lower_order_reviews()

    relationship = _relationship(
        reviews,
        "MEDIATION_CHAIN",
    )

    assert relationship.scope_compatible is True
    assert relationship.bridge_kind == "MEDIATION_CHAIN"

    closure = _closure(
        reviews,
        relationship,
    )

    structural = assess_structural_nonobviousness(
        closure,
        ResidualClaimStructure(
            claim_kind="distinctive_prediction",
            introduces_threshold=True,
            introduces_regime_change=True,
        ),
    )

    assert structural.status == "REGIME_OR_THRESHOLD_LEAP"

    readiness = assess_adjudication_readiness(
        structural_status=structural.status,
        vector=_threshold_vector(),
    )

    assert (
        readiness.readiness
        == "READY_FOR_NONOBVIOUSNESS_REVIEW"
    )

    result = review_and_compile_nonobviousness_adjudication(
        backend=_PotentialBackend(),
        readiness=readiness,
        packet=_packet(
            structural_status=structural.status,
            vector=_threshold_vector(),
        ),
        prior_art=_prior(),
    )

    assert result.review_performed is True
    assert (
        result.compiled.verdict
        == "POTENTIALLY_NON_OBVIOUS"
    )
    assert (
        result.compiled.required_additional_assumptions
        == (BRIDGE,)
    )
