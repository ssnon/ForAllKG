from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from pipeline_core.discovery.external_novelty_contracts import (
    ExternalNoveltyCard,
    PriorArtPacket,
)
from pipeline_core.discovery.scientific_distinctiveness_contracts import (
    ScientificDistinctivenessReview,
)


SEMANTIC_DISTINCTIVENESS_PROMPT_VERSION = (
    "semantic-distinctiveness-critic-prompt-v2"
)


def _canonical_json(
    value: object,
) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256(
    text: str,
) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def _excerpt(
    text: str | None,
    *,
    limit: int = 1200,
) -> str | None:
    value = " ".join(
        str(
            text or ""
        ).split()
    )

    if not value:
        return None

    if len(value) <= limit:
        return value

    return (
        value[
            :limit
        ].rstrip()
        + " ..."
    )


def _ordered_unique(
    values,
) -> list[str]:
    result = []
    seen = set()

    for value in values:
        value = str(
            value
        )

        if (
            not value
            or value in seen
        ):
            continue

        seen.add(
            value
        )

        result.append(
            value
        )

    return result


@dataclass(frozen=True)
class SemanticDistinctivenessPrompt:
    hypothesis_id: str

    prompt_version: str
    system_prompt: str
    user_prompt: str
    prompt_sha256: str

    allowed_claim_ids: tuple[str, ...]
    allowed_work_ids: tuple[str, ...]

    @classmethod
    def create(
        cls,
        *,
        hypothesis_id: str,
        system_prompt: str,
        user_prompt: str,
        allowed_claim_ids: list[str],
        allowed_work_ids: list[str],
    ) -> "SemanticDistinctivenessPrompt":

        body = {
            "prompt_version":
                SEMANTIC_DISTINCTIVENESS_PROMPT_VERSION,

            "hypothesis_id":
                hypothesis_id,

            "system_prompt":
                system_prompt,

            "user_prompt":
                user_prompt,

            "allowed_claim_ids":
                allowed_claim_ids,

            "allowed_work_ids":
                allowed_work_ids,
        }

        return cls(
            hypothesis_id=
                hypothesis_id,

            prompt_version=(
                SEMANTIC_DISTINCTIVENESS_PROMPT_VERSION
            ),

            system_prompt=
                system_prompt,

            user_prompt=
                user_prompt,

            prompt_sha256=_sha256(
                _canonical_json(
                    body
                )
            ),

            allowed_claim_ids=tuple(
                allowed_claim_ids
            ),

            allowed_work_ids=tuple(
                allowed_work_ids
            ),
        )


SYSTEM_PROMPT = """
You are the semantic scientific-distinctiveness critic for an
evidence-grounded hypothesis discovery system.

Your task is narrow.

You are NOT deciding whether the hypothesis is:
- true;
- experimentally feasible;
- important;
- publishable;
- acceptable;
- literature-wide novel.

You are assessing whether the hypothesis is scientifically
non-obvious in STRUCTURE relative to the supplied frozen reviewed
prior-art evidence.

Use only:
1. the supplied hypothesis/claim text;
2. the supplied deterministic external-novelty evidence structure;
3. the supplied reviewed prior-art matches and excerpts;
4. ordinary logical/mechanistic reasoning needed to compare those
   supplied statements.

Do NOT:
- invent unprovided papers;
- assert external scientific facts not present in the input;
- infer literature-wide absence;
- use missing matches as strong novelty evidence when recorded search
  coverage is insufficient;
- alter external novelty status;
- recommend acceptance/rejection;
- rewrite the hypothesis.

DIMENSIONS
==========

1. conceptual_prior_art_density

HIGH:
    The supplied reviewed evidence densely covers the variables,
    mechanisms, contexts, and neighboring relations needed to construct
    the hypothesis.

LOW:
    The supplied evidence leaves substantial conceptual structure
    unrepresented.

This is density within the supplied frozen evidence only.

2. straightforward_reconstruction

This dimension asks how directly the proposed scientific structure can
be reconstructed from the supplied reviewed evidence.

HIGH:
    The hypothesis is directly represented OR can be reconstructed with
    little additional relational structure.

    Examples:
    - a reviewed work already states essentially the same relation;
    - A affects Y and B affects Y, followed only by A+B affects Y;
    - a chain of already represented relations is restated without a
      genuinely new conditional dependency.

MODERATE:
    The hypothesis adds an explicit moderator, mediator, conditional
    dependency, or interaction, but the supplied evidence already
    reconstructs most of that relation or comes very close to stating it.

LOW:
    Reconstructing the hypothesis requires introducing a genuinely new
    higher-order dependency not nearly supplied by the reviewed evidence.

Important:
    Direct prior-art representation counts as HIGH reconstruction even
    when the claim is not literally a conjunction.

3. mechanism_switch

HIGH:
    The hypothesis explicitly proposes a switch, competition, change in
    governing pathway, or condition-dependent mechanism.

LOW:
    It merely reuses the same mechanism in another material/context or
    adds variables without a mechanism transition.

Do not infer a mechanism switch merely from the claim kind label.

4. ranking_or_regime_change

HIGH:
    The hypothesis predicts a qualitative change such as:
    - optimum movement;
    - rank/order reversal;
    - sign reversal;
    - threshold;
    - transition between regimes;
    - a condition-dependent strongest/weakest configuration;
    - qualitatively different resonant/off-resonant behavior.

LOW:
    It predicts only that magnitude varies or that an effect exists.

5. counterfactual_distinctiveness

HIGH:
    The hypothesis explicitly controls or holds fixed a named major
    confounder, comparator, or alternative explanation and asks whether
    a residual relation persists, changes, or remains detectable.

MODERATE:
    The hypothesis uses an otherwise-comparable, one-variable-at-a-time,
    matched, or approximately controlled comparison that isolates one
    variable, but does not explicitly identify and control a major
    alternative explanation.

LOW:
    The hypothesis merely compares groups, architectures, observables,
    conditions, or response magnitudes without an isolating control.

Important:
    A comparison by itself is not counterfactual isolation.

6. evidence_role_complementarity

Assess complementarity of EVIDENCE ROLES, not merely whether papers come
from different topics, materials, laboratories, or publication families.

HIGH:
    At least two substantively different evidence roles are necessary to
    reconstruct the hypothesis, and no single reviewed work or strongly
    overlapping evidence role nearly reconstructs the proposed relation.

    Example role separation:
    - one evidence role establishes A -> mediator;
    - another establishes mediator -> Y;
    - the hypothesis proposes a new A-conditioned mediator -> Y relation.

MODERATE:
    Multiple evidence roles contribute, but they overlap substantially,
    or one reviewed work already covers a large fraction of the proposed
    relational structure.

LOW:
    One evidence role or one reviewed work nearly reconstructs the claim,
    or the additional works are primarily redundant support for the same
    relation.

Important:
    Multiple papers do not by themselves imply HIGH complementarity.

OVERALL-TIER POLICY
===================

Do NOT choose an overall scientific-distinctiveness tier.

The deterministic compiler derives overall_tier from:
- the six semantic dimension assessments;
- the frozen deterministic evidence pattern.

Your task is only to assess the six dimensions and confidence.

REFERENCE DISCIPLINE
====================

Every dimension may reference only:
- claim IDs from allowed_claim_ids;
- work IDs from allowed_work_ids.

Reference only IDs actually needed for the rationale.

If evidence is insufficient, use INDETERMINATE rather than inventing
support.

Return only the requested structured semantic-distinctiveness draft.
""".strip()


class SemanticDistinctivenessPromptAssembler:
    def __init__(
        self,
        *,
        max_matches_per_claim: int = 8,
        abstract_excerpt_chars: int = 1200,
    ) -> None:
        if max_matches_per_claim < 1:
            raise ValueError(
                "max_matches_per_claim must be positive"
            )

        if abstract_excerpt_chars < 100:
            raise ValueError(
                "abstract_excerpt_chars too small"
            )

        self.max_matches_per_claim = int(
            max_matches_per_claim
        )

        self.abstract_excerpt_chars = int(
            abstract_excerpt_chars
        )

    def build(
        self,
        review: ScientificDistinctivenessReview,
        card: ExternalNoveltyCard,
        packet: PriorArtPacket,
    ) -> SemanticDistinctivenessPrompt:

        if (
            review.hypothesis_id
            != card.hypothesis_id
        ):
            raise ValueError(
                "semantic prompt hypothesis/card mismatch"
            )

        card_reviews = {
            row.claim_id:
                row
            for row in card.claim_reviews
        }

        if (
            len(
                card_reviews
            )
            != len(
                card.claim_reviews
            )
        ):
            raise ValueError(
                "duplicate claim review ID"
            )

        signal_by_id = {
            row.claim_id:
                row
            for row in review.claim_signals
        }

        if (
            len(
                signal_by_id
            )
            != len(
                review.claim_signals
            )
        ):
            raise ValueError(
                "duplicate claim signal ID"
            )

        allowed_claim_ids = list(
            review.source_claim_ids
        )

        if set(
            allowed_claim_ids
        ) != set(
            card_reviews
        ):
            raise ValueError(
                "semantic prompt card/source claim set mismatch"
            )

        if set(
            allowed_claim_ids
        ) != set(
            signal_by_id
        ):
            raise ValueError(
                "semantic prompt signal/source claim set mismatch"
            )

        works = {
            row.work_id:
                row
            for row in packet.works
        }

        if (
            len(
                works
            )
            != len(
                packet.works
            )
        ):
            raise ValueError(
                "duplicate prior-art work ID"
            )

        allowed_work_ids = []
        claim_payloads = []

        for claim_id in allowed_claim_ids:
            external_review = (
                card_reviews[
                    claim_id
                ]
            )

            signal = (
                signal_by_id[
                    claim_id
                ]
            )

            if (
                external_review.claim_text
                != signal.claim_text
            ):
                raise ValueError(
                    "semantic prompt claim text drift: "
                    f"{claim_id}"
                )

            if (
                external_review.importance
                != signal.importance
            ):
                raise ValueError(
                    "semantic prompt claim importance drift: "
                    f"{claim_id}"
                )

            if (
                external_review.status
                != signal.prior_art_status
            ):
                raise ValueError(
                    "semantic prompt claim status drift: "
                    f"{claim_id}"
                )

            match_payloads = []

            for match in (
                external_review.matches[
                    :self.max_matches_per_claim
                ]
            ):
                work = works.get(
                    match.work_id
                )

                if work is None:
                    raise ValueError(
                        "semantic prompt references unknown work: "
                        f"{match.work_id}"
                    )

                allowed_work_ids.append(
                    match.work_id
                )

                match_payloads.append(
                    {
                        "work_id":
                            match.work_id,

                        "relationship":
                            match.relationship,

                        "confidence":
                            match.confidence,

                        "review_rationale":
                            match.rationale,

                        "relevance_score":
                            match.relevance_score,

                        "semantic_similarity":
                            match.semantic_similarity,

                        "lexical_coverage":
                            match.lexical_coverage,

                        "scope_compatible_for_conflict":
                            match.scope_compatible_for_conflict,

                        "scope_reason_codes":
                            list(
                                match.scope_reason_codes
                            ),

                        "title":
                            work.title,

                        "year":
                            work.year,

                        "abstract_excerpt":
                            _excerpt(
                                work.abstract,
                                limit=(
                                    self
                                    .abstract_excerpt_chars
                                ),
                            ),
                    }
                )

            claim_payloads.append(
                {
                    "claim_id":
                        claim_id,

                    "claim_kind":
                        signal.claim_kind,

                    "importance":
                        signal.importance,

                    "claim_text":
                        signal.claim_text,

                    "prior_art_status":
                        signal.prior_art_status,

                    "search_coverage": {
                        "query_count":
                            signal.query_count,

                        "successful_query_count":
                            (
                                signal
                                .successful_query_count
                            ),

                        "unique_work_count":
                            signal.unique_work_count,

                        "abstract_work_count":
                            signal.abstract_work_count,

                        "reviewed_work_count":
                            signal.reviewed_work_count,
                    },

                    "relationship_counts":
                        dict(
                            signal.relationship_counts
                        ),

                    "reason_codes":
                        list(
                            signal.reason_codes
                        ),

                    "reviewed_matches":
                        match_payloads,
                }
            )

        allowed_work_ids = _ordered_unique(
            allowed_work_ids
        )

        payload = {
            "hypothesis": {
                "hypothesis_id":
                    review.hypothesis_id,

                "title":
                    review.title,
            },

            "deterministic_distinctiveness_structure": {
                "external_novelty_status":
                    (
                        review
                        .external_novelty_status
                    ),

                "evidence_pattern":
                    review.evidence_pattern,

                "core_claim_count":
                    review.core_claim_count,

                "direct_prior_art_core_claim_count":
                    (
                        review
                        .direct_prior_art_core_claim_count
                    ),

                "relation_backed_core_claim_count":
                    (
                        review
                        .relation_backed_core_claim_count
                    ),

                "component_supported_core_claim_count":
                    (
                        review
                        .component_supported_core_claim_count
                    ),

                "no_direct_match_core_claim_count":
                    (
                        review
                        .no_direct_match_core_claim_count
                    ),

                "lower_order_supported_core_claim_count":
                    (
                        review
                        .lower_order_supported_core_claim_count
                    ),

                "higher_order_relational_gap_claim_count":
                    (
                        review
                        .higher_order_relational_gap_claim_count
                    ),

                "search_coverage_sufficient":
                    (
                        review
                        .search_coverage_sufficient
                    ),

                "source_aggregate_warnings":
                    list(
                        review
                        .source_aggregate_warnings
                    ),

                "deterministic_interpretation":
                    review.interpretation,
            },

            "claims":
                claim_payloads,

            "allowed_claim_ids":
                allowed_claim_ids,

            "allowed_work_ids":
                allowed_work_ids,
        }

        user_prompt = (
            "SEMANTIC DISTINCTIVENESS REVIEW INPUT\n"
            "=====================================\n"
            + json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n\nOUTPUT REQUIREMENTS\n"
            "===================\n"
            "- Assess all six semantic dimensions.\n"
            "- Use only the supplied frozen reviewed evidence.\n"
            "- Do not invent literature or scientific facts.\n"
            "- Do not treat missing matches as strong novelty evidence "
            "when search_coverage_sufficient is false.\n"
            "- HIGH straightforward_reconstruction means LESS "
            "distinctiveness, unlike the other positive structural "
            "dimensions.\n"
            "- HIGH conceptual_prior_art_density also generally weighs "
            "against distinctiveness.\n"
            "- Return claim_ids/work_ids only from the allowed lists.\n"
            "- HIGH overall_tier is diagnostic only, never acceptance.\n"
            "- LOW overall_tier is diagnostic only, never rejection.\n"
        )

        return (
            SemanticDistinctivenessPrompt
            .create(
                hypothesis_id=(
                    review
                    .hypothesis_id
                ),
                system_prompt=
                    SYSTEM_PROMPT,
                user_prompt=
                    user_prompt,
                allowed_claim_ids=
                    allowed_claim_ids,
                allowed_work_ids=
                    allowed_work_ids,
            )
        )


    def build_reference_validation_repair(
        self,
        *,
        original_prompt: SemanticDistinctivenessPrompt,
        previous_draft: object,
        issues: list[str],
    ) -> SemanticDistinctivenessPrompt:
        """One bounded repair for invalid claim/work references.

        This is not a second scientific search or a policy relaxation.
        The model receives the same frozen evidence surface and must
        replace hallucinated/out-of-scope references with valid supplied
        references or with empty reference lists.

        If an assessment depended on unsupported external information,
        it must be reconsidered using only the original supplied evidence.
        """

        if not issues:
            raise ValueError(
                "reference repair requires at least one issue"
            )

        if not hasattr(
            previous_draft,
            "model_dump_json",
        ):
            raise TypeError(
                "previous semantic draft is not serializable"
            )

        issue_lines = [
            f"- {str(issue)}"
            for issue in issues
        ]

        repair_request = (
            "\n\nREFERENCE-CONTRACT VALIDATION REPAIR\n"
            "====================================\n"
            "The previous semantic-distinctiveness draft failed the "
            "deterministic reference allowlist.\n"
            "\n"
            "Return ONE COMPLETE REPLACEMENT draft.\n"
            "\n"
            "Repair rules:\n"
            "- Use exactly the same frozen evidence already supplied.\n"
            "- Do NOT retrieve, invent, recall, or introduce any paper.\n"
            "- Do NOT invent, transform, shorten, extend, or regenerate "
            "a claim_id or work_id.\n"
            "- claim_ids may contain ONLY IDs from allowed_claim_ids.\n"
            "- work_ids may contain ONLY IDs from allowed_work_ids.\n"
            "- Empty claim_ids/work_ids lists are valid when no specific "
            "reference is required.\n"
            "- If a dimension rationale or level depended on an invalid "
            "reference or unsupported outside knowledge, REASSESS that "
            "dimension using only the supplied evidence.\n"
            "- Use INDETERMINATE if the supplied evidence is insufficient.\n"
            "- Do not preserve a scientific judgment merely to keep the "
            "previous draft unchanged. Evidence-bounded correction takes "
            "priority.\n"
            "- Do not alter unrelated dimensions unless needed to keep "
            "overall_tier, confidence, and rationale internally consistent "
            "with the corrected evidence-bounded assessments.\n"
            "- HIGH remains diagnostic only; LOW remains diagnostic only.\n"
            "\n"
            "VALIDATION ISSUES\n"
            "-----------------\n"
            + "\n".join(
                issue_lines
            )
            + "\n\nPREVIOUS DRAFT\n"
            "--------------\n"
            + previous_draft.model_dump_json(
                indent=2
            )
        )

        return (
            SemanticDistinctivenessPrompt
            .create(
                hypothesis_id=(
                    original_prompt
                    .hypothesis_id
                ),
                system_prompt=(
                    original_prompt
                    .system_prompt
                ),
                user_prompt=(
                    original_prompt
                    .user_prompt
                    + repair_request
                ),
                allowed_claim_ids=list(
                    original_prompt
                    .allowed_claim_ids
                ),
                allowed_work_ids=list(
                    original_prompt
                    .allowed_work_ids
                ),
            )
        )
