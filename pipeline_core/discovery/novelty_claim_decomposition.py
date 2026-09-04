from __future__ import annotations

import hashlib
import json
import re
from typing import Protocol

from pipeline_core.discovery.external_novelty_contracts import (
    HypothesisNoveltyClaims,
    LiteratureQuery,
    LiteratureQueryPlan,
    NoveltyClaim,
    NoveltyClaimDecompositionDraft,
)
from pipeline_core.discovery.hypothesis_contracts import HypothesisCard, HypothesisPortfolio
from pipeline_core.discovery.novelty_structure_validation import (
    compile_claim_scientific_structure,
)


def _canonical_json(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _stable_id(prefix: str, *parts: object, length: int = 20) -> str:
    raw = "|".join(str(x) for x in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:length]}"


def _clean_query(text: str, *, limit: int = 280) -> str:
    value = str(text or "")
    value = value.replace("ΔG_H*", "hydrogen adsorption free energy")
    value = value.replace("ΔG_H", "hydrogen adsorption free energy")
    value = value.replace("ΔG", "free energy")
    value = re.sub(r"[‐‑‒–—−-]+", " ", value)
    value = re.sub(r"[^\w\s+*/().,]", " ", value, flags=re.UNICODE)
    value = " ".join(value.split())
    return value[:limit].strip()


def _clean_diagnostic_terms(
    values: list[str],
) -> list[str]:
    """Normalize structured diagnostic terms without semantic expansion."""

    rows: list[str] = []

    for value in values:
        cleaned = _clean_query(
            value,
            limit=120,
        )

        if (
            cleaned
            and cleaned.lower()
            not in {
                row.lower()
                for row in rows
            }
        ):
            rows.append(cleaned)

    return rows


def _compile_higher_order_relation_basis(
    *,
    kind: str,
    values: list[str],
    source_texts: list[str],
) -> tuple[list[str], list[str]]:
    """Validate explicit higher-order relation provenance.

    This function never constructs a composite scientific relation
    from lower-order component claims.

    In particular:

        A -> B
        B -> C

    does not authorize:

        A -> B -> C

    A higher-order basis is accepted only when the supplied hypothesis
    material itself already contains the returned span under
    conservative surface normalization.
    """

    cleaned_values: list[str] = []

    for value in values:
        cleaned = " ".join(
            str(value or "").split()
        )

        if (
            cleaned
            and cleaned not in cleaned_values
        ):
            cleaned_values.append(cleaned)

    reason_codes: list[str] = []

    if str(kind) != "composite":
        if cleaned_values:
            reason_codes.append(
                "higher_order_basis_rejected_on_non_composite_claim"
            )

        return (
            [],
            list(
                dict.fromkeys(
                    reason_codes
                )
            ),
        )

    normalized_sources = [
        _clean_query(
            source,
            limit=12000,
        ).lower()
        for source in source_texts
        if str(source or "").strip()
    ]

    accepted: list[str] = []

    for value in cleaned_values:
        normalized_value = _clean_query(
            value,
            limit=6000,
        ).lower()

        supported = bool(
            normalized_value
            and any(
                normalized_value in source
                for source in normalized_sources
            )
        )

        if not supported:
            reason_codes.append(
                "unsupported_higher_order_relation_basis"
            )
            continue

        accepted.append(value)

    if not accepted:
        reason_codes.append(
            "composite_missing_valid_higher_order_relation_basis"
        )

    return (
        accepted,
        list(
            dict.fromkeys(
                reason_codes
            )
        ),
    )


def _compile_higher_order_component_claim_ids(
    *,
    kind: str,
    local_id: str,
    component_local_ids: list[str],
    claim_id_by_local_id: dict[str, str],
) -> list[str]:
    """Resolve explicit decomposition topology fail closed.

    This function does NOT infer component relationships from claim
    text, shared variables, lexical overlap, or scientific semantics.

    Only explicit local-ID references returned by the decomposition
    are accepted.
    """

    values = [
        str(value or "").strip()
        for value in component_local_ids
    ]

    values = [
        value
        for value in values
        if value
    ]

    if len(values) != len(set(values)):
        raise ValueError(
            "duplicate higher-order component local_id"
        )

    if str(kind) != "composite":
        if values:
            raise ValueError(
                "non-composite claim cannot declare "
                "higher-order components"
            )

        return []

    if str(local_id) in values:
        raise ValueError(
            "composite claim cannot reference itself "
            "as a component"
        )

    unknown = [
        value
        for value in values
        if value not in claim_id_by_local_id
    ]

    if unknown:
        raise ValueError(
            "unknown higher-order component local_id: "
            + ", ".join(unknown)
        )

    return [
        claim_id_by_local_id[value]
        for value in values
    ]


def _clean_branch_specific_specification(
    text: str,
    identity_terms: list[str],
) -> str:
    """Preserve a specification only when it names this atomic branch.

    This is deliberately conservative. An umbrella hypothesis-level
    statement such as "laser excitation conditions ..." must not be
    silently instantiated as a "laser power" or "excitation wavelength"
    bridge unless that branch identity is explicitly represented.

    Empty identity terms retain the cleaned text for backward
    compatibility; claims with a usable branch identity are guarded.
    """

    cleaned = " ".join(
        str(text or "").split()
    )

    if not cleaned:
        return ""

    identities = _clean_diagnostic_terms(
        identity_terms
    )

    if not identities:
        return cleaned

    if _normalized_identity_present(
        cleaned,
        identities,
    ):
        return cleaned

    return ""


def _clean_branch_specific_bridge(
    text: str,
    identity_terms: list[str],
    source_texts: list[str],
) -> str:
    """Preserve only an extractively supported branch-specific bridge.

    A bridge is stronger than a branch-specific prediction/falsifier:
    it asserts the scientific proposition connecting the atomic factor
    to the residual relation.

    Therefore it must satisfy BOTH:
      1. the atomic branch identity is explicitly named; and
      2. the proposed bridge is an extractive span of the original
         hypothesis inferential bridge or assumptions.

    Paraphrasing an umbrella bridge into a new branch-specific
    proposition is intentionally rejected.
    """

    cleaned = " ".join(
        str(text or "").split()
    )

    if not cleaned:
        return ""

    identities = _clean_diagnostic_terms(
        identity_terms
    )

    if not identities:
        return ""

    normalized_bridge = _clean_query(
        cleaned,
        limit=4000,
    ).lower()

    if not _normalized_identity_present(
        cleaned,
        identities,
    ):
        return ""

    for source in source_texts:
        normalized_source = _clean_query(
            source,
            limit=8000,
        ).lower()

        if (
            normalized_bridge
            and normalized_bridge
            in normalized_source
        ):
            return cleaned

    return ""


_BRANCH_IDENTITY_NONDISCRIMINATING_TOKENS = frozenset(
    {
        # Grammatical / comparison qualifiers that do not identify
        # the scientific branch itself.
        "identity",
        "relative",
        "fixed",
        "same",
        "different",
        "distinct",
        "matched",
    }
)


def _branch_identity_signature(
    text: str,
) -> tuple[str, ...]:
    """Return a conservative lexical signature for branch identity.

    This performs only surface normalization. It never adds synonyms,
    stems terms, expands abbreviations, invokes an embedding model, or
    infers scientific equivalence.

    Example:
        "metal pair identity" -> ("metal", "pair")
        "relative oxygenated intermediate stabilization"
            -> ("oxygenated", "intermediate", "stabilization")

    All retained tokens must still occur in the candidate
    specification for branch attribution to succeed.
    """

    normalized = _clean_query(
        text,
        limit=1000,
    ).lower()

    tokens = [
        token
        for token in re.findall(
            r"\w+",
            normalized,
            flags=re.UNICODE,
        )
        if (
            token
            and token
            not in _BRANCH_IDENTITY_NONDISCRIMINATING_TOKENS
        )
    ]

    return tuple(
        dict.fromkeys(tokens)
    )


def _identity_token_surface_variants(
    token: str,
) -> frozenset[str]:
    """Return conservative surface variants for one identity token.

    Only simple English plural morphology is added. This function does
    NOT perform stemming, synonym expansion, abbreviation expansion,
    embedding similarity, or semantic inference.

    Examples:
        environment -> {environment, environments}
        pair -> {pair, pairs}
        intermediate -> {intermediate, intermediates}
        activity -> {activity, activities}

    The original token is always retained.
    """

    value = str(
        token or ""
    ).strip().lower()

    if not value:
        return frozenset()

    variants = {
        value,
    }

    # Avoid inventing morphology for very short tokens and tokens that
    # are not simple alphabetic lexical items.
    if (
        len(value) < 4
        or not value.isalpha()
    ):
        return frozenset(
            variants
        )

    if (
        value.endswith("y")
        and len(value) >= 2
        and value[-2]
        not in "aeiou"
    ):
        variants.add(
            value[:-1] + "ies"
        )

    elif value.endswith(
        (
            "ch",
            "sh",
            "x",
            "z",
        )
    ):
        variants.add(
            value + "es"
        )

    else:
        variants.add(
            value + "s"
        )

    return frozenset(
        variants
    )


def _normalized_identity_present(
    text: str,
    identity_terms: list[str],
) -> bool:
    """Check branch identity by conservative lexical containment.

    Each identity term is an alternative branch-identity expression.
    For one identity term to match, every informative token retained
    from that identity must occur in the specification text.

    This is deliberately weaker than exact phrase matching but much
    stronger than semantic similarity.
    """

    normalized_text = _clean_query(
        text,
        limit=8000,
    ).lower()

    text_tokens = set(
        re.findall(
            r"\w+",
            normalized_text,
            flags=re.UNICODE,
        )
    )

    for identity in _clean_diagnostic_terms(
        identity_terms
    ):
        signature = (
            _branch_identity_signature(
                identity
            )
        )

        # Fail closed if normalization removes the whole identity.
        if not signature:
            continue

        if all(
            any(
                variant in text_tokens
                for variant
                in _identity_token_surface_variants(
                    token
                )
            )
            for token in signature
        ):
            return True

    return False


def _extractively_present(
    text: str,
    source_texts: list[str],
) -> bool:
    normalized = _clean_query(
        text,
        limit=8000,
    ).lower()

    if not normalized:
        return False

    for source in source_texts:
        normalized_source = _clean_query(
            source,
            limit=12000,
        ).lower()

        if normalized in normalized_source:
            return True

    return False


def _diagnose_specification_sanitization(
    *,
    raw_required_bridge: str,
    required_bridge_source: str,
    sanitized_required_bridge: str,
    raw_predicted_observation: str,
    sanitized_predicted_observation: str,
    raw_falsification_condition: str,
    sanitized_falsification_condition: str,
    identity_terms: list[str],
    bridge_source_texts: list[str],
) -> list[str]:
    """Explain specification loss without changing acceptance policy."""

    codes: list[str] = []

    raw_bridge = " ".join(
        str(raw_required_bridge or "").split()
    )

    raw_prediction = " ".join(
        str(raw_predicted_observation or "").split()
    )

    raw_falsifier = " ".join(
        str(raw_falsification_condition or "").split()
    )

    identities = _clean_diagnostic_terms(
        identity_terms
    )

    # --------------------------------------------------------------
    # required_bridge
    # --------------------------------------------------------------

    if not raw_bridge:
        codes.append(
            "required_bridge_source_empty"
        )

    else:
        codes.append(
            "required_bridge_source_"
            + required_bridge_source
        )

        if not sanitized_required_bridge:
            if not identities:
                codes.append(
                    "required_bridge_rejected_"
                    "missing_branch_identity_terms"
                )

            elif not _normalized_identity_present(
                raw_bridge,
                identities,
            ):
                codes.append(
                    "required_bridge_rejected_"
                    "branch_identity"
                )

            elif not _extractively_present(
                raw_bridge,
                bridge_source_texts,
            ):
                codes.append(
                    "required_bridge_rejected_"
                    "nonextractive"
                )

            else:
                codes.append(
                    "required_bridge_rejected_"
                    "unspecified"
                )

    # --------------------------------------------------------------
    # predicted_observation
    # --------------------------------------------------------------

    if not raw_prediction:
        codes.append(
            "predicted_observation_draft_empty"
        )

    elif not sanitized_predicted_observation:
        if not identities:
            codes.append(
                "predicted_observation_rejected_"
                "missing_branch_identity_terms"
            )
        else:
            codes.append(
                "predicted_observation_rejected_"
                "branch_identity"
            )

    # --------------------------------------------------------------
    # falsification_condition
    # --------------------------------------------------------------

    if not raw_falsifier:
        codes.append(
            "falsification_condition_draft_empty"
        )

    elif not sanitized_falsification_condition:
        if not identities:
            codes.append(
                "falsification_condition_rejected_"
                "missing_branch_identity_terms"
            )
        else:
            codes.append(
                "falsification_condition_rejected_"
                "branch_identity"
            )

    return list(
        dict.fromkeys(codes)
    )


def recover_required_bridge_from_hypothesis(
    hypothesis: HypothesisCard,
    identity_terms: list[str] | tuple[str, ...],
) -> str:
    """Recover only a branch-specific, exact-source hypothesis bridge.

    This never invents or paraphrases scientific content. The canonical
    hypothesis inferential bridge must itself satisfy the existing
    branch-identity and extractive-support sanitizer.
    """

    return _clean_branch_specific_bridge(
        hypothesis.inferential_bridge,
        list(identity_terms),
        [
            hypothesis.inferential_bridge,
            *hypothesis.assumptions,
        ],
    )



def _assemble_diagnostic_relation_query(
    structural_terms: list[str],
    relation_terms: list[str],
    *,
    fallback: str,
) -> tuple[str, list[str], list[str]]:
    """Build a relation-first query without deleting or inventing terms."""

    structural = _clean_diagnostic_terms(
        structural_terms
    )

    relation = _clean_diagnostic_terms(
        relation_terms
    )

    candidate = _clean_query(
        " ".join(
            [
                *structural,
                *relation,
            ]
        )
    )

    # Fail safe for incomplete structured output.
    if len(candidate.split()) < 3:
        candidate = _clean_query(
            fallback
        )

    return (
        candidate,
        structural,
        relation,
    )


class NoveltyClaimBackend(Protocol):
    def decompose(
        self,
        hypothesis: HypothesisCard,
        *,
        max_claims: int,
    ) -> NoveltyClaimDecompositionDraft: ...


class NoveltyClaimDecomposer:
    def __init__(
        self,
        backend: NoveltyClaimBackend,
        *,
        max_claims_per_hypothesis: int = 4,
        max_queries_per_claim: int = 2,
    ) -> None:
        self.backend = backend
        self.max_claims = int(max_claims_per_hypothesis)
        self.max_queries = int(max_queries_per_claim)

        # Diagnostic-only observability channel.
        #
        # Raw decomposition values stored here must never be
        # promoted into NoveltyClaim, LiteratureQueryPlan,
        # retrieval vocabulary, evidence closure, or
        # novelty/non-obviousness authority.
        self.specification_sanitization_records: list[
            dict[str, object]
        ] = []

        if self.max_claims < 1:
            raise ValueError("max_claims_per_hypothesis must be >= 1")
        if self.max_queries < 1:
            raise ValueError("max_queries_per_claim must be >= 1")

    def decompose(self, hypothesis: HypothesisCard) -> HypothesisNoveltyClaims:
        draft = self.backend.decompose(hypothesis, max_claims=self.max_claims)

        draft_rows = list(
            draft.claims[: self.max_claims]
        )

        claim_id_by_local_id: dict[str, str] = {}

        for rank, draft_row in enumerate(
            draft_rows,
            start=1,
        ):
            if (
                draft_row.local_id
                in claim_id_by_local_id
            ):
                raise ValueError(
                    "duplicate bounded decomposition local_id: "
                    + draft_row.local_id
                )

            claim_id_by_local_id[
                draft_row.local_id
            ] = _stable_id(
                "external_novelty_claim",
                hypothesis.hypothesis_id,
                rank,
                draft_row.kind,
                draft_row.text,
            )

        rows: list[NoveltyClaim] = []

        for rank, row in enumerate(
            draft_rows,
            start=1,
        ):
            concepts = []
            for value in row.search_concepts:
                cleaned = _clean_query(value, limit=120)
                if cleaned and cleaned not in concepts:
                    concepts.append(cleaned)
            queries = []
            for value in row.search_queries:
                cleaned = _clean_query(value)
                if cleaned and cleaned not in queries:
                    queries.append(cleaned)
                if len(queries) >= self.max_queries:
                    break
            if not queries:
                fallback = _clean_query(row.text)
                if fallback:
                    queries.append(fallback)
            if len(queries) < self.max_queries and concepts:
                concept_query = _clean_query(" ".join(concepts))
                if concept_query and concept_query not in queries:
                    queries.append(concept_query)

            diagnostic_kind = row.diagnostic_query_kind

            diagnostic_source_query = _clean_query(
                row.diagnostic_search_query or ""
            )

            (
                diagnostic_execution_query,
                diagnostic_structural_terms,
                diagnostic_relation_terms,
            ) = _assemble_diagnostic_relation_query(
                row.diagnostic_structural_terms,
                row.diagnostic_relation_terms,
                fallback=diagnostic_source_query,
            )

            prior_art_identity_terms = (
                _clean_diagnostic_terms(
                    row.prior_art_identity_terms
                )
            )

            relation_nucleus_terms = (
                _clean_diagnostic_terms(
                    row.relation_nucleus_terms
                )
            )

            higher_order_source_texts = [
                hypothesis.hypothesis_statement,
                hypothesis.inferential_bridge,
                *hypothesis.assumptions,
                *[
                    item.observable
                    for item
                    in hypothesis.predicted_observations
                ],
                *[
                    item.rationale
                    for item
                    in hypothesis.predicted_observations
                ],
                *[
                    item.observable
                    for item
                    in hypothesis.falsification_criteria
                ],
                *[
                    item.falsifying_outcome
                    for item
                    in hypothesis.falsification_criteria
                ],
            ]

            (
                higher_order_relation_basis,
                higher_order_relation_reason_codes,
            ) = _compile_higher_order_relation_basis(
                kind=row.kind,
                values=row.higher_order_relation_basis,
                source_texts=higher_order_source_texts,
            )

            higher_order_component_claim_ids = (
                _compile_higher_order_component_claim_ids(
                    kind=row.kind,
                    local_id=row.local_id,
                    component_local_ids=(
                        row.higher_order_component_local_ids
                    ),
                    claim_id_by_local_id=(
                        claim_id_by_local_id
                    ),
                )
            )

            scientific_structure, structure_reason_codes = (
                compile_claim_scientific_structure(
                    row.scientific_structure,
                    identity_terms=prior_art_identity_terms,
                    source_texts=[
                        hypothesis.hypothesis_statement,
                        hypothesis.inferential_bridge,
                        *hypothesis.assumptions,
                        *[
                            item.observable
                            for item
                            in hypothesis.predicted_observations
                        ],
                        *[
                            item.rationale
                            for item
                            in hypothesis.predicted_observations
                        ],
                        *[
                            item.observable
                            for item
                            in hypothesis.falsification_criteria
                        ],
                        *[
                            item.falsifying_outcome
                            for item
                            in hypothesis.falsification_criteria
                        ],
                    ],
                )
            )

            # Preserve an explicit empty bridge from the atomic
            # decomposition. A hypothesis-level inferential bridge is
            # not a safe fallback for an atomic claim because it may
            # contain sibling branches or additional relation nuclei.
            raw_required_bridge = str(
                row.required_bridge or ""
            )

            required_bridge_source = (
                "draft"
                if raw_required_bridge.strip()
                else "empty"
            )

            bridge_source_texts = [
                hypothesis.inferential_bridge,
                *hypothesis.assumptions,
            ]

            sanitized_required_bridge = (
                _clean_branch_specific_bridge(
                    raw_required_bridge,
                    prior_art_identity_terms,
                    bridge_source_texts,
                )
            )

            sanitized_predicted_observation = (
                _clean_branch_specific_specification(
                    row.predicted_observation,
                    prior_art_identity_terms,
                )
            )

            sanitized_falsification_condition = (
                _clean_branch_specific_specification(
                    row.falsification_condition,
                    prior_art_identity_terms,
                )
            )

            specification_sanitization_reason_codes = (
                _diagnose_specification_sanitization(
                    raw_required_bridge=(
                        raw_required_bridge
                    ),
                    required_bridge_source=(
                        required_bridge_source
                    ),
                    sanitized_required_bridge=(
                        sanitized_required_bridge
                    ),
                    raw_predicted_observation=(
                        row.predicted_observation
                    ),
                    sanitized_predicted_observation=(
                        sanitized_predicted_observation
                    ),
                    raw_falsification_condition=(
                        row.falsification_condition
                    ),
                    sanitized_falsification_condition=(
                        sanitized_falsification_condition
                    ),
                    identity_terms=(
                        prior_art_identity_terms
                    ),
                    bridge_source_texts=(
                        bridge_source_texts
                    ),
                )
            )

            claim_id = claim_id_by_local_id[
                row.local_id
            ]

            # Preserve pre-sanitization specification values
            # outside the canonical NoveltyClaim contract.
            #
            # This is diagnostic provenance only. Rejected text
            # must not become evidence, query vocabulary, or
            # novelty/non-obviousness authority.
            self.specification_sanitization_records.append(
                {
                    "schema_version": (
                        "novelty-claim-specification-"
                        "sanitization-v1"
                    ),
                    "diagnostic_only": True,
                    "hypothesis_id": (
                        hypothesis.hypothesis_id
                    ),
                    "claim_id": claim_id,
                    "claim_rank": rank,
                    "claim_local_id": row.local_id,
                    "prior_art_identity_terms": list(
                        prior_art_identity_terms
                    ),
                    "required_bridge_source": (
                        required_bridge_source
                    ),
                    "raw_required_bridge": (
                        raw_required_bridge
                    ),
                    "sanitized_required_bridge": (
                        sanitized_required_bridge
                    ),
                    "raw_predicted_observation": str(
                        row.predicted_observation or ""
                    ),
                    "sanitized_predicted_observation": (
                        sanitized_predicted_observation
                    ),
                    "raw_falsification_condition": str(
                        row.falsification_condition or ""
                    ),
                    "sanitized_falsification_condition": (
                        sanitized_falsification_condition
                    ),
                    "reason_codes": list(
                        specification_sanitization_reason_codes
                    ),
                }
            )

            rows.append(
                NoveltyClaim(
                    claim_id=claim_id,
                    hypothesis_id=hypothesis.hypothesis_id,
                    claim_rank=rank,
                    kind=row.kind,
                    importance=row.importance,
                    novelty_selection_role=(
                        row.novelty_selection_role
                    ),
                    text=row.text,
                    rationale=row.rationale,
                    search_concepts=concepts,
                    search_queries=queries[: self.max_queries],
                    distinguishing_terms=_clean_diagnostic_terms(
                        row.distinguishing_terms
                    ),
                    prior_art_identity_terms=(
                        prior_art_identity_terms
                    ),
                    relation_nucleus_terms=(
                        relation_nucleus_terms
                    ),
                    higher_order_relation_basis=(
                        higher_order_relation_basis
                    ),
                    higher_order_component_claim_ids=(
                        higher_order_component_claim_ids
                    ),
                    required_bridge=(
                        sanitized_required_bridge
                    ),
                    predicted_observation=(
                        sanitized_predicted_observation
                    ),
                    falsification_condition=(
                        sanitized_falsification_condition
                    ),
                    scientific_structure=scientific_structure,
                    diagnostic_query_kind=diagnostic_kind,
                    diagnostic_search_query=(
                        diagnostic_source_query or None
                    ),
                    diagnostic_execution_query=(
                        diagnostic_execution_query or None
                    ),
                    diagnostic_structural_terms=(
                        diagnostic_structural_terms
                    ),
                    diagnostic_relation_terms=(
                        diagnostic_relation_terms
                    ),
                    scientific_structure_reason_codes=list(
                        structure_reason_codes
                    ),
                    higher_order_relation_reason_codes=(
                        higher_order_relation_reason_codes
                    ),
                    specification_sanitization_reason_codes=(
                        specification_sanitization_reason_codes
                    ),
                )
            )
        return HypothesisNoveltyClaims(
            hypothesis_id=hypothesis.hypothesis_id,
            title=hypothesis.title,
            claims=rows,
            decomposition_notes=draft.decomposition_notes,
        )


class LiteratureQueryPlanner:
    def __init__(self, *, include_hypothesis_composite: bool = True) -> None:
        self.include_hypothesis_composite = bool(include_hypothesis_composite)

    def build(
        self,
        portfolio: HypothesisPortfolio,
        decompositions: list[HypothesisNoveltyClaims],
    ) -> LiteratureQueryPlan:
        by_hypothesis = {row.hypothesis_id: row for row in decompositions}
        queries: list[LiteratureQuery] = []
        seen: set[tuple[str, str | None, str]] = set()

        for hypothesis in portfolio.hypotheses:
            row = by_hypothesis.get(hypothesis.hypothesis_id)
            if row is None:
                raise ValueError(
                    f"missing novelty-claim decomposition for {hypothesis.hypothesis_id}"
                )
            for claim in row.claims:
                for index, query_text in enumerate(claim.search_queries):
                    cleaned = _clean_query(query_text)
                    if not cleaned:
                        continue

                    if index == 0:
                        kind = "claim_primary"
                    else:
                        kind = "claim_variant"
                    key = (hypothesis.hypothesis_id, claim.claim_id, cleaned.lower())
                    if key in seen:
                        continue
                    seen.add(key)
                    queries.append(
                        LiteratureQuery(
                            query_id=_stable_id(
                                "literature_query",
                                hypothesis.hypothesis_id,
                                claim.claim_id,
                                kind,
                                cleaned,
                            ),
                            hypothesis_id=hypothesis.hypothesis_id,
                            claim_id=claim.claim_id,
                            query_kind=kind,
                            query_text=cleaned,
                        )
                    )
            if self.include_hypothesis_composite:
                composite = _clean_query(
                    " ".join(
                        [
                            hypothesis.title,
                            hypothesis.hypothesis_statement,
                        ]
                    )
                )
                if composite:
                    key = (hypothesis.hypothesis_id, None, composite.lower())
                    if key not in seen:
                        seen.add(key)
                        queries.append(
                            LiteratureQuery(
                                query_id=_stable_id(
                                    "literature_query",
                                    hypothesis.hypothesis_id,
                                    "composite",
                                    composite,
                                ),
                                hypothesis_id=hypothesis.hypothesis_id,
                                claim_id=None,
                                query_kind="hypothesis_composite",
                                query_text=composite,
                            )
                        )

        payload = {
            "schema_version": "literature-query-plan-v1",
            "source_portfolio_id": portfolio.portfolio_id,
            "queries": [row.model_dump(mode="json") for row in queries],
            "claims": [row.model_dump(mode="json") for row in decompositions],
            "policy_version": "external-novelty-query-policy-v1",
        }
        plan_id = _stable_id(
            "literature_query_plan",
            portfolio.portfolio_id,
            *[row.query_id for row in queries],
        )
        body = {**payload, "plan_id": plan_id}
        return LiteratureQueryPlan(**body, plan_sha256=_sha256_json(body))
