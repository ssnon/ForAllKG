from pipeline_core.discovery.external_novelty_contracts import (
    PriorArtPacket,
    PriorArtWork,
)
from pipeline_core.discovery.novelty_closure_relationships import (
    ClosureRelationshipAssessmentDraft,
)
from pipeline_core.discovery.novelty_closure_relationships_llm import (
    build_closure_relationship_user_prompt,
    relationship_review_needed,
    review_and_compile_closure_relationships,
)


def _review(
    slot,
    *,
    state="ESTABLISHED",
    ids=(),
):
    return {
        "slot": slot,
        "evidence_state": state,
        "positive_work_ids": list(ids),
    }


def _reviews():
    return [
        _review(
            "BASE_RELATION",
            ids=("work:base",),
        ),
        _review(
            "DISTINGUISHING_FACTOR_EFFECT",
            ids=("work:factor",),
        ),
        _review(
            "BRIDGE_RELATION",
            ids=("work:bridge",),
        ),
        _review(
            "FULL_RELATION",
            state="NOT_FOUND",
        ),
    ]


def _packet():
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
                    "Interparticle spacing changes the measured "
                    "SERS enhancement."
                ),
            ),
            PriorArtWork(
                work_id="work:factor",
                title="Factor",
                abstract=(
                    "Laser power changes the local plasmonic state."
                ),
            ),
            PriorArtWork(
                work_id="work:bridge",
                title="Bridge",
                abstract=(
                    "Laser power changes a local state that controls "
                    "the spacing-dependent SERS response."
                ),
            ),
            PriorArtWork(
                work_id="work:noise",
                title="Noise",
                abstract=(
                    "This retrieved paper must never enter the "
                    "cross-slot relationship prompt."
                ),
            ),
        ],
    )


def _targets():
    return {
        "BASE_RELATION": {
            "source_text":
                "Interparticle spacing affects SERS enhancement.",
        },
        "DISTINGUISHING_FACTOR_EFFECT": {
            "source_text":
                "Laser power has a lower-order effect.",
        },
        "BRIDGE_RELATION": {
            "source_text": (
                "Laser power changes the state connecting spacing "
                "to SERS enhancement."
            ),
        },
    }


def test_relationship_review_requires_complete_positive_lower_order_closure():
    assert (
        relationship_review_needed(
            _reviews()
        )
        is True
    )

    rows = _reviews()
    rows[1] = _review(
        "DISTINGUISHING_FACTOR_EFFECT",
        state="UNASSESSED",
    )

    assert (
        relationship_review_needed(
            rows
        )
        is False
    )


def test_full_relation_established_skips_cross_slot_review():
    rows = _reviews()
    rows[3] = _review(
        "FULL_RELATION",
        ids=("work:full",),
    )

    assert (
        relationship_review_needed(
            rows
        )
        is False
    )


def test_relationship_prompt_exposes_positive_established_abstracts_only():
    prompt = (
        build_closure_relationship_user_prompt(
            reviews=_reviews(),
            packet=_packet(),
            targets_by_slot=_targets(),
        )
    )

    assert "work:base" in prompt
    assert "work:factor" in prompt
    assert "work:bridge" in prompt

    assert "work:noise" not in prompt
    assert (
        "This retrieved paper must never enter"
        not in prompt
    )

    assert "search_query:" not in prompt
    assert "ALLOWED_POSITIVE_WORK_IDS" in prompt


class _FakeBackend:
    def __init__(self):
        self.calls = 0

    def review_relationships(
        self,
        *,
        reviews,
        packet,
        targets_by_slot,
    ):
        self.calls += 1

        return ClosureRelationshipAssessmentDraft(
            bridge_kind="MEDIATION_CHAIN",
            scope_compatibility="COMPATIBLE",
            bridge_basis_work_ids=[
                "work:bridge",
            ],
            scope_basis_work_ids=[
                "work:base",
                "work:factor",
                "work:bridge",
            ],
            interpretation=(
                "The positive evidence supports a mediation-like "
                "bridge in compatible scope."
            ),
        )


def test_relationship_review_compiles_positive_evidence_contract():
    backend = _FakeBackend()

    result = (
        review_and_compile_closure_relationships(
            backend=backend,
            reviews=_reviews(),
            packet=_packet(),
            targets_by_slot=_targets(),
        )
    )

    assert backend.calls == 1
    assert result.review_performed is True

    assert (
        result.compiled.bridge_kind
        == "MEDIATION_CHAIN"
    )

    assert (
        result.compiled.scope_compatible
        is True
    )


def test_incomplete_lower_order_closure_skips_backend_and_fails_closed():
    backend = _FakeBackend()

    rows = _reviews()
    rows[2] = _review(
        "BRIDGE_RELATION",
        state="UNASSESSED",
    )

    result = (
        review_and_compile_closure_relationships(
            backend=backend,
            reviews=rows,
            packet=_packet(),
            targets_by_slot=_targets(),
        )
    )

    assert backend.calls == 0
    assert result.review_performed is False
    assert result.compiled.bridge_kind == "NONE"
    assert result.compiled.scope_compatible is False
