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
from pipeline_core.discovery.typed_required_bridge_binding import (
    TYPED_REQUIRED_BRIDGE_PROVENANCE,
    validate_typed_required_bridge_binding,
)
from pipeline_core.discovery.novelty_claim_decomposition import (
    _branch_identity_signature,
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


def select_uniquely_discriminated_atomic_claim_id(
    *,
    candidate_bridge: str,
    sibling_claims: Sequence[NoveltyResidueClaim],
) -> str | None:
    """Return one atomically discriminated sibling claim ID, or None.

    This is a conservative lexical helper over already-existing
    atomic decomposition metadata.  It does not perform recovery,
    alter a claim, or provide scientific/production authority.

    The helper intentionally does NOT reinterpret
    prior_art_identity_terms.  Those terms identify a prior-art
    family and may legitimately be shared by scientifically
    distinct sibling claims.

    Only relation_nucleus_terms and distinguishing_terms may act
    as secondary atomic discriminators.  A discriminator counts
    only when:

    1. its frozen surface signature is non-empty;
    2. that exact signature belongs to exactly one sibling claim;
    3. every retained signature token occurs in candidate_bridge.

    Exactly one sibling must receive at least one such hit.
    Zero or multiple selected siblings fail closed and return None.

    No synonyms, stemming, abbreviation expansion, embeddings, or
    scientific inference are performed.
    """

    candidate = str(
        candidate_bridge or ""
    ).strip()

    claims = list(
        sibling_claims
    )

    if not candidate or not claims:
        return None

    hypothesis_ids = {
        str(
            sibling.hypothesis_id
        )
        for sibling in claims
    }

    if len(hypothesis_ids) != 1:
        return None

    claim_ids = [
        str(
            sibling.claim_id
        )
        for sibling in claims
    ]

    if (
        any(
            not claim_id
            for claim_id in claim_ids
        )
        or len(claim_ids)
        != len(set(claim_ids))
    ):
        return None

    candidate_signature = set(
        _branch_identity_signature(
            candidate
        )
    )

    if not candidate_signature:
        return None

    selected_claim_ids: set[str] = set()

    for field_name in (
        "relation_nucleus_terms",
        "distinguishing_terms",
    ):
        owners: dict[
            tuple[str, ...],
            set[str],
        ] = {}

        for sibling in claims:
            sibling_id = str(
                sibling.claim_id
            )

            for value in getattr(
                sibling,
                field_name,
            ):
                signature = (
                    _branch_identity_signature(
                        str(value)
                    )
                )

                if not signature:
                    continue

                owners.setdefault(
                    signature,
                    set(),
                ).add(
                    sibling_id
                )

        for signature, owner_ids in owners.items():
            if len(owner_ids) != 1:
                continue

            if not set(
                signature
            ).issubset(
                candidate_signature
            ):
                continue

            selected_claim_ids.update(
                owner_ids
            )

    if len(selected_claim_ids) != 1:
        return None

    return next(
        iter(
            selected_claim_ids
        )
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

    # Preserve the historical fast path exactly: an identity-admissible
    # canonical bridge attributed to only this atomic claim is recovered
    # without consulting any secondary discriminator.
    if matching_claim_ids == [
        claim.claim_id
    ]:
        return candidate

    # The current claim must itself remain inside the identity-admissible
    # ambiguity set.  Secondary discrimination never creates admissibility
    # for a claim that failed the existing canonical branch-identity check.
    if (
        claim.claim_id not in matching_claim_ids
        or len(matching_claim_ids) < 2
    ):
        return ""

    ambiguous_claims = [
        sibling
        for sibling in local_claims
        if sibling.claim_id in matching_claim_ids
    ]

    selected_claim_id = (
        select_uniquely_discriminated_atomic_claim_id(
            candidate_bridge=candidate,
            sibling_claims=ambiguous_claims,
        )
    )

    if selected_claim_id != claim.claim_id:
        return ""

    return candidate


def reconcile_intake_required_bridge(
    claim: NoveltyResidueClaim,
    *,
    intake_claim: dict[str, object],
    specification_provenance: dict[str, object] | None,
    hypothesis: HypothesisCard | None,
    sibling_claims: Sequence[NoveltyResidueClaim] | None = None,
    required_bridge_binding: dict[str, object] | None = None,
    required_bridge_binding_contract: dict[str, object] | None = None,
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

    # Existing query-plan bridge remains authoritative.
    if expected_bridge:
        if incoming_bridge != expected_bridge:
            raise ValueError(
                "N10 intake attempted to replace an existing "
                "query-plan required_bridge"
            )
        if provenance == TYPED_REQUIRED_BRIDGE_PROVENANCE:
            raise ValueError(
                "typed source binding cannot replace existing query-plan bridge"
            )
        return claim

    # Typed source binding is an alternate representation of the bridge.
    # Raw intake text stays empty; only a validated existing source quote
    # is transiently materialized for legacy N9 closure/vector consumers.
    if provenance == TYPED_REQUIRED_BRIDGE_PROVENANCE:
        if incoming_bridge:
            raise ValueError(
                "typed source binding must not write free-text required_bridge"
            )
        if hypothesis is None:
            raise ValueError(
                "typed source binding requires canonical hypothesis"
            )
        if hypothesis.hypothesis_id != claim.hypothesis_id:
            raise ValueError(
                "typed source binding hypothesis mismatch"
            )
        if not isinstance(required_bridge_binding, dict):
            raise ValueError(
                "typed source binding payload missing"
            )
        if not isinstance(required_bridge_binding_contract, dict):
            raise ValueError(
                "typed source binding contract missing"
            )
        recovered = validate_typed_required_bridge_binding(
            hypothesis=hypothesis,
            claim=claim,
            binding=required_bridge_binding,
            contract=required_bridge_binding_contract,
        )
        return replace(
            claim,
            required_bridge=recovered,
        )

    # Nothing recovered: preserve the original fail-closed claim.
    if not incoming_bridge:
        return claim

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
    required_bridge_bindings: dict[str, dict[str, object]] | None = None,
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

            typed_binding = None
            typed_contract = None

            if (
                not str(compiled_claim.required_bridge or "").strip()
                and required_bridge_bindings is not None
            ):
                typed_row = required_bridge_bindings.get(
                    claim.claim_id
                )
                if typed_row is not None:
                    if not isinstance(typed_row, dict):
                        raise ValueError(
                            "typed bridge binding row must be an object"
                        )
                    typed_binding = typed_row.get("binding")
                    typed_contract = typed_row.get("contract")
                    hypothesis = source_cards.get(
                        claim.hypothesis_id
                    )
                    if hypothesis is None:
                        raise ValueError(
                            "typed bridge binding cannot resolve canonical hypothesis"
                        )
                    if not isinstance(typed_binding, dict):
                        raise ValueError(
                            "typed bridge binding payload must be an object"
                        )
                    if not isinstance(typed_contract, dict):
                        raise ValueError(
                            "typed bridge contract must be an object"
                        )
                    typed_quote = validate_typed_required_bridge_binding(
                        hypothesis=hypothesis,
                        claim=claim,
                        binding=typed_binding,
                        contract=typed_contract,
                    )
                    compiled_claim = replace(
                        claim,
                        required_bridge=typed_quote,
                    )
                    bridge_source = TYPED_REQUIRED_BRIDGE_PROVENANCE

            decision = compile_shadow_claim(
                compiled_claim
            )

            if typed_binding is not None:
                # Persist original raw atomic claim; structured binding remains
                # separate metadata and does not become generated bridge prose.
                decision_claim = decision.get("claim")
                if not isinstance(decision_claim, dict):
                    raise TypeError("compiled N9 claim is not a dictionary")
                decision_claim["required_bridge"] = claim.required_bridge
                decision["required_bridge_binding"] = typed_binding
                decision["required_bridge_binding_contract"] = typed_contract

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
