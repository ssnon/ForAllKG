from pipeline_core.discovery.external_novelty_contracts import (
    PriorArtPacket,
    PriorArtWork,
)
from pipeline_core.discovery.novelty_adjudication import (
    EstablishedPriorArtRelation,
    NonObviousnessAdjudicationDraft,
    NonObviousnessAdjudicationVector,
    NonObviousnessEvidencePacket,
    NonObviousnessReviewGate,
)
from pipeline_core.discovery.novelty_adjudication_llm import (
    build_nonobviousness_adjudication_user_prompt,
    review_and_compile_nonobviousness_adjudication,
    sanitize_adjudication_draft,
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


def _packet():
    return NonObviousnessEvidencePacket(
        claim_id="claim:threshold",
        claim_text=(
            "A critical laser power Pc separates two distinct "
            "spacing-to-SERS regimes."
        ),
        structural_status="REGIME_OR_THRESHOLD_LEAP",
        vector=NonObviousnessAdjudicationVector(
            inferential_distance="NEW_REGIME_STRUCTURE",
            mechanistic_necessity="NEW_BRIDGE_REQUIRED",
            regime_specificity="THRESHOLD",
            counterintuitiveness="NONTRIVIAL",
            testable_distinctiveness="QUANTITATIVE",
            required_bridge=BRIDGE,
            predicted_observation=PREDICTION,
            falsification_condition=FALSIFIER,
        ),
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
                    "Laser power connects to the local state "
                    "relevant to spacing-sensitive SERS."
                ),
                relationship_status="ESTABLISHED",
                work_ids=("work:bridge",),
                scope_note="compatible",
            ),
        ),
        direct_full_claim_prior_art=False,
        evidence_closure_sufficient=True,
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
                abstract=(
                    "Spacing affects SERS enhancement."
                ),
            ),
            PriorArtWork(
                work_id="work:factor",
                title="Factor",
                abstract=(
                    "Laser power changes a local state."
                ),
            ),
            PriorArtWork(
                work_id="work:bridge",
                title="Bridge",
                abstract=(
                    "Laser power changes a state relevant to "
                    "spacing-sensitive response."
                ),
            ),
            PriorArtWork(
                work_id="work:noise",
                title="Noise",
                abstract=(
                    "This must not be shown to the adjudicator."
                ),
            ),
        ],
    )


def _ready():
    return NonObviousnessReviewGate(
        readiness="READY_FOR_NONOBVIOUSNESS_REVIEW",
        reason_codes=("ready",),
        interpretation="ready",
    )


def test_adjudication_prompt_exposes_established_positive_works_only():
    prompt = (
        build_nonobviousness_adjudication_user_prompt(
            packet=_packet(),
            prior_art=_prior(),
        )
    )

    assert "work:base" in prompt
    assert "work:factor" in prompt
    assert "work:bridge" in prompt

    assert "work:noise" not in prompt
    assert (
        "This must not be shown"
        not in prompt
    )

    assert "search_query:" not in prompt
    assert "SPECIFICATION_SOURCE_TEXTS" in prompt


def test_nonextractive_additional_assumption_is_dropped():
    raw = NonObviousnessAdjudicationDraft(
        proposed_verdict="POTENTIALLY_NON_OBVIOUS",
        direct_reconstruction_from_known_relations=False,
        additional_scientific_assumptions=(
            "Laser heating irreversibly restructures the junction.",
        ),
        prediction_distinguishes_from_routine_baseline=True,
        falsifier_is_specific=True,
        concise_basis="Requires a new assumption.",
    )

    result = sanitize_adjudication_draft(
        packet=_packet(),
        draft=raw,
    )

    assert (
        result.draft.additional_scientific_assumptions
        == ()
    )

    assert (
        "non_extractive_additional_assumption_dropped"
        in result.reason_codes
    )


def test_exact_required_bridge_survives_sanitizer():
    raw = NonObviousnessAdjudicationDraft(
        proposed_verdict="POTENTIALLY_NON_OBVIOUS",
        direct_reconstruction_from_known_relations=False,
        additional_scientific_assumptions=(
            BRIDGE,
        ),
        prediction_distinguishes_from_routine_baseline=True,
        falsifier_is_specific=True,
        concise_basis=(
            "The threshold bridge is not reconstructed "
            "by lower-order relations."
        ),
    )

    result = sanitize_adjudication_draft(
        packet=_packet(),
        draft=raw,
    )

    assert (
        result.draft.additional_scientific_assumptions
        == (BRIDGE,)
    )

    assert result.reason_codes == ()


class _PotentialBackend:
    def __init__(self):
        self.calls = 0

    def adjudicate(
        self,
        *,
        packet,
        prior_art,
    ):
        self.calls += 1

        return NonObviousnessAdjudicationDraft(
            proposed_verdict="POTENTIALLY_NON_OBVIOUS",
            direct_reconstruction_from_known_relations=False,
            additional_scientific_assumptions=(
                BRIDGE,
            ),
            prediction_distinguishes_from_routine_baseline=True,
            falsifier_is_specific=True,
            concise_basis=(
                "Established relations do not contain "
                "the explicit Pc regime transition."
            ),
        )


def test_valid_ready_candidate_can_compile_potentially_nonobvious():
    backend = _PotentialBackend()

    result = (
        review_and_compile_nonobviousness_adjudication(
            backend=backend,
            readiness=_ready(),
            packet=_packet(),
            prior_art=_prior(),
        )
    )

    assert backend.calls == 1
    assert result.review_performed is True
    assert (
        result.compiled.verdict
        == "POTENTIALLY_NON_OBVIOUS"
    )


class _InventingBackend:
    def adjudicate(
        self,
        *,
        packet,
        prior_art,
    ):
        return NonObviousnessAdjudicationDraft(
            proposed_verdict="POTENTIALLY_NON_OBVIOUS",
            direct_reconstruction_from_known_relations=False,
            additional_scientific_assumptions=(
                "Laser heating irreversibly restructures the junction.",
            ),
            prediction_distinguishes_from_routine_baseline=True,
            falsifier_is_specific=True,
            concise_basis="Invented mechanism.",
        )


def test_invented_assumption_cannot_compile_potentially_nonobvious():
    result = (
        review_and_compile_nonobviousness_adjudication(
            backend=_InventingBackend(),
            readiness=_ready(),
            packet=_packet(),
            prior_art=_prior(),
        )
    )

    assert (
        result.compiled.verdict
        == "INSUFFICIENT_FOR_JUDGMENT"
    )

    assert (
        "non_extractive_additional_assumption_dropped"
        in result.sanitizer_reason_codes
    )


class _RoutineBackend:
    def adjudicate(
        self,
        *,
        packet,
        prior_art,
    ):
        return NonObviousnessAdjudicationDraft(
            proposed_verdict="ROUTINE_FROM_PRIOR_ART",
            direct_reconstruction_from_known_relations=True,
            additional_scientific_assumptions=(),
            prediction_distinguishes_from_routine_baseline=False,
            falsifier_is_specific=True,
            concise_basis=(
                "The established relations directly reconstruct "
                "the residual claim."
            ),
        )


def test_ready_candidate_can_compile_routine_from_positive_relations():
    result = (
        review_and_compile_nonobviousness_adjudication(
            backend=_RoutineBackend(),
            readiness=_ready(),
            packet=_packet(),
            prior_art=_prior(),
        )
    )

    assert (
        result.compiled.verdict
        == "ROUTINE_FROM_PRIOR_ART"
    )
