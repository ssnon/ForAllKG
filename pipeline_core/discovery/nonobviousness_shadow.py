from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, is_dataclass, replace

from pipeline_core.discovery.external_novelty_contracts import (
    ExternalNoveltyReport,
    LiteratureQueryPlan,
)
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisCard,
    HypothesisPortfolio,
)
from pipeline_core.discovery.novelty_claim_decomposition import (
    recover_required_bridge_from_hypothesis,
)
from pipeline_core.discovery.novelty_residue import (
    NoveltyResidueClaim,
    assess_residual_specification,
    extract_novelty_residue,
)


def _json_safe(
    value: object,
) -> object:
    """Recursively convert artifact values to JSON-native objects.

    N10 residue dataclasses may contain nested Pydantic scientific-
    structure contracts. dataclasses.asdict() alone does not convert
    those nested BaseModel instances.
    """

    if hasattr(value, "model_dump"):
        return _json_safe(
            value.model_dump(
                mode="json"
            )
        )

    if is_dataclass(value):
        return _json_safe(
            asdict(value)
        )

    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(
        value,
        (list, tuple, set),
    ):
        return [
            _json_safe(item)
            for item in value
        ]

    return value



_CANONICAL_BRIDGE_PROVENANCE = (
    "CANONICAL_HYPOTHESIS_INFERENTIAL_BRIDGE"
)


def recover_uniquely_attributed_required_bridge(
    *,
    hypothesis: HypothesisCard,
    claim: NoveltyResidueClaim,
    sibling_claims: Sequence[NoveltyResidueClaim],
) -> str:
    """Recover a canonical bridge only under unique atomic attribution.

    The existing exact-source/branch-identity recovery remains the
    underlying admissibility predicate.  This guard adds no semantic
    matching, synonyms, or scientific inference.

    A canonical HypothesisCard inferential bridge may fill an empty
    atomic required_bridge only when that same canonical bridge is
    admissible for exactly one atomic claim of the hypothesis.

    All hypothesis-local atomic claims count toward uniqueness,
    including claims that already possess a query-plan bridge and
    claims whose selection role is not novelty-bearing.
    """

    if hypothesis.hypothesis_id != claim.hypothesis_id:
        return ""

    local_claims = [
        sibling
        for sibling in sibling_claims
        if (
            sibling.hypothesis_id
            == hypothesis.hypothesis_id
        )
    ]

    claim_ids = [
        sibling.claim_id
        for sibling in local_claims
    ]

    # Fail closed on incomplete or malformed sibling context.
    if claim.claim_id not in claim_ids:
        return ""

    if len(claim_ids) != len(set(claim_ids)):
        return ""

    candidate = recover_required_bridge_from_hypothesis(
        hypothesis,
        claim.prior_art_identity_terms,
    )

    if not candidate:
        return ""

    matching_claim_ids = []

    for sibling in local_claims:
        sibling_candidate = (
            recover_required_bridge_from_hypothesis(
                hypothesis,
                sibling.prior_art_identity_terms,
            )
        )

        if sibling_candidate == candidate:
            matching_claim_ids.append(
                sibling.claim_id
            )

    if matching_claim_ids != [
        claim.claim_id
    ]:
        return ""

    return candidate


def reconcile_intake_required_bridge(
    claim: NoveltyResidueClaim,
    *,
    intake_claim: dict[str, object],
    specification_provenance: dict[str, object] | None,
    hypothesis: HypothesisCard | None,
    sibling_claims: Sequence[NoveltyResidueClaim] | None = None,
) -> NoveltyResidueClaim:
    """Reconcile only a provenance-validated required_bridge.

    The intake artifact may not alter claim identity, prior-art state,
    branch identity, prediction, falsifier, or scientific structure.

    If the query-plan residue lacks required_bridge, the only permitted
    recovery is the exact canonical hypothesis inferential bridge after
    the existing branch-specific extractive sanitizer accepts it.
    """

    expected = _json_safe(claim)

    if not isinstance(expected, dict):
        raise TypeError(
            "NoveltyResidueClaim did not serialize to a dictionary"
        )

    incoming = dict(intake_claim)

    # Backward-compatible metadata reconciliation.
    #
    # Older N9 intake artifacts predate these fields. Their absence
    # is not scientific drift because both values are reconstructed
    # independently from the authoritative query-plan residue.
    for metadata_field in (
        "importance",
        "specification_sanitization_reason_codes",
    ):
        if (
            metadata_field in expected
            and metadata_field not in incoming
        ):
            incoming[metadata_field] = (
                expected[metadata_field]
            )

    expected_bridge = str(
        expected.pop("required_bridge", "")
        or ""
    ).strip()

    incoming_bridge = str(
        incoming.pop("required_bridge", "")
        or ""
    ).strip()

    if incoming != expected:
        raise ValueError(
            "N10 intake claim drift outside required_bridge"
        )

    # Existing query-plan bridge remains authoritative.
    if expected_bridge:
        if incoming_bridge != expected_bridge:
            raise ValueError(
                "N10 intake attempted to replace an existing "
                "query-plan required_bridge"
            )

        return claim

    # Nothing recovered: preserve the original fail-closed claim.
    if not incoming_bridge:
        return claim

    provenance = str(
        (
            specification_provenance
            or {}
        ).get(
            "required_bridge",
            "",
        )
        or ""
    )

    if provenance != _CANONICAL_BRIDGE_PROVENANCE:
        raise ValueError(
            "Recovered N10 required_bridge lacks canonical "
            "hypothesis provenance"
        )

    if hypothesis is None:
        raise ValueError(
            "Canonical hypothesis is required to verify "
            "recovered N10 required_bridge"
        )

    if hypothesis.hypothesis_id != claim.hypothesis_id:
        raise ValueError(
            "Recovered N10 required_bridge hypothesis mismatch"
        )

    recovered = recover_required_bridge_from_hypothesis(
        hypothesis,
        claim.prior_art_identity_terms,
    )

    if not recovered:
        raise ValueError(
            "Canonical hypothesis bridge does not satisfy "
            "branch-specific extractive requirements"
        )

    if recovered != incoming_bridge:
        raise ValueError(
            "Recovered N10 required_bridge does not exactly "
            "match independently verified canonical bridge"
        )

    if sibling_claims is None:
        raise ValueError(
            "Canonical hypothesis bridge recovery requires "
            "hypothesis-local sibling claims for unique attribution"
        )

    uniquely_recovered = (
        recover_uniquely_attributed_required_bridge(
            hypothesis=hypothesis,
            claim=claim,
            sibling_claims=sibling_claims,
        )
    )

    if uniquely_recovered != recovered:
        raise ValueError(
            "Canonical hypothesis bridge is not uniquely "
            "attributable to one atomic claim"
        )

    return replace(
        claim,
        required_bridge=recovered,
    )



def compile_shadow_claim(
    claim: NoveltyResidueClaim,
) -> dict[str, object]:
    """Compile one claim into the N9 shadow-intake disposition.

    This stage deliberately stops before evidence closure.
    It must not infer scientific non-obviousness from residual status.
    """

    specification = assess_residual_specification(
        claim
    )

    if claim.disposition == "SATURATED":
        shadow_state = "SATURATED_PRIOR_ART"
        next_action = "NONE"

    elif claim.disposition == "UNRESOLVED_PARTIAL":
        shadow_state = "UNRESOLVED_PARTIAL"
        next_action = "RESOLVE_PARTIAL_PRIOR_ART"

    elif claim.disposition == "RESIDUAL":
        if (
            specification.status
            == "READY_FOR_CLOSURE"
        ):
            shadow_state = "READY_FOR_CLOSURE"
            next_action = "TARGETED_CLOSURE_REQUIRED"
        else:
            shadow_state = "NEEDS_REFINEMENT"
            next_action = (
                "REFINE_HYPOTHESIS_SPECIFICATION"
            )

    else:
        shadow_state = "UNRESOLVED"
        next_action = "REVIEW_EVIDENCE_STATE"

    return {
        "claim": _json_safe(claim),
        "specification": _json_safe(specification),
        "shadow_state": shadow_state,
        "next_action": next_action,

        # Explicitly prevent later consumers from mistaking
        # intake for completed N9 adjudication.
        "closure_status": (
            "PENDING_TARGETED_CLOSURE"
            if shadow_state == "READY_FOR_CLOSURE"
            else "NOT_RUN"
        ),
        "structural_status": "NOT_RUN",
        "adjudication_status": "NOT_RUN",
    }


def build_nonobviousness_shadow(
    *,
    plan: LiteratureQueryPlan,
    report: ExternalNoveltyReport,
    source_portfolio: HypothesisPortfolio | None = None,
) -> dict[str, object]:
    """Build shadow-only N9 residue/specification artifact."""

    if (
        plan.source_portfolio_id
        != report.source_portfolio_id
    ):
        raise ValueError(
            "N9 shadow provenance mismatch: "
            "query plan and external report refer "
            "to different source portfolios."
        )

    if (
        source_portfolio is not None
        and source_portfolio.portfolio_id
        != report.source_portfolio_id
    ):
        raise ValueError(
            "N9 shadow provenance mismatch: "
            "source portfolio and external report refer "
            "to different portfolios."
        )

    source_cards = {
        card.hypothesis_id: card
        for card in (
            source_portfolio.hypotheses
            if source_portfolio is not None
            else []
        )
    }

    residues = extract_novelty_residue(
        plan,
        report,
    )

    cards = {
        card.hypothesis_id: card
        for card in report.cards
    }

    hypothesis_rows: list[dict[str, object]] = []
    states: Counter[str] = Counter()

    for residue in residues:
        card = cards.get(residue.hypothesis_id)

        decisions: list[dict[str, object]] = []

        for claim in residue.claims:
            compiled_claim = claim

            bridge_source = (
                "QUERY_PLAN"
                if str(claim.required_bridge or "").strip()
                else "UNRESOLVED"
            )

            if (
                not str(claim.required_bridge or "").strip()
                and source_portfolio is not None
            ):
                hypothesis = source_cards.get(
                    claim.hypothesis_id
                )

                if hypothesis is not None:
                    recovered_bridge = (
                        recover_uniquely_attributed_required_bridge(
                            hypothesis=hypothesis,
                            claim=claim,
                            sibling_claims=residue.claims,
                        )
                    )

                    if recovered_bridge:
                        compiled_claim = replace(
                            claim,
                            required_bridge=recovered_bridge,
                        )
                        bridge_source = (
                            "CANONICAL_HYPOTHESIS_"
                            "INFERENTIAL_BRIDGE"
                        )

            decision = compile_shadow_claim(
                compiled_claim
            )

            decision["specification_provenance"] = {
                "required_bridge": bridge_source,
            }

            decisions.append(decision)

        for decision in decisions:
            states[str(decision["shadow_state"])] += 1

        hypothesis_rows.append(
            {
                "hypothesis_id": residue.hypothesis_id,
                "title": (
                    card.title
                    if card is not None
                    else ""
                ),
                "external_status": (
                    residue.external_status
                ),
                "claims": decisions,
                "ready_for_closure_claim_ids": [
                    str(
                        row["claim"]["claim_id"]
                    )
                    for row in decisions
                    if (
                        row["shadow_state"]
                        == "READY_FOR_CLOSURE"
                    )
                ],
                "needs_refinement_claim_ids": [
                    str(
                        row["claim"]["claim_id"]
                    )
                    for row in decisions
                    if (
                        row["shadow_state"]
                        == "NEEDS_REFINEMENT"
                    )
                ],
            }
        )

    return {
        "schema_version": (
            "nonobviousness-shadow-v1"
        ),
        "shadow_only": True,
        "scientific_selection_changed": False,

        "source_portfolio_id": (
            report.source_portfolio_id
        ),
        "source_query_plan_id": plan.plan_id,
        "source_query_plan_sha256": (
            plan.plan_sha256
        ),
        "source_external_report_id": (
            report.report_id
        ),
        "source_external_report_sha256": (
            report.report_sha256
        ),
        "source_prior_art_packet_id": (
            report.source_prior_art_packet_id
        ),

        "hypothesis_count": len(hypothesis_rows),
        "claim_count": sum(states.values()),
        "shadow_state_counts": dict(
            sorted(states.items())
        ),
        "hypotheses": hypothesis_rows,

        "epistemic_policy": {
            "residual_is_not_novelty": True,
            "missing_prior_art_is_not_positive_evidence": True,
            "under_specified_residue_requires_refinement": True,
            "ready_for_closure_is_not_nonobviousness": True,
        },
    }
