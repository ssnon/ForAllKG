from __future__ import annotations

import re
import unicodedata
from typing import Any

from pipeline_core.discovery.external_novelty_contracts import (
    NoveltyClaimDraft,
)
from pipeline_core.discovery.hypothesis_contracts import HypothesisCard


_ORDERED_MARKER_RE = re.compile(
    r"\b(?:"
    r"higher|lower|greater|smaller|stronger|weaker|"
    r"increase(?:d|s|ing)?|decrease(?:d|s|ing)?|"
    r"maximum|min(?:imum)?|peak|"
    r"non[\s-]?monotonic|reversal|revers(?:e|es|ed|ing)|"
    r"more|less"
    r")\b",
    flags=re.I,
)

_CONDITION_MARKER_RE = re.compile(
    r"\b(?:"
    r"under|when|whenever|if|provided|given|"
    r"at\s+comparable|with\s+comparable|for\s+comparable|"
    r"holding|held|within"
    r")\b",
    flags=re.I,
)

_ANAPHORIC_RELATION_RE = re.compile(
    r"\b(?:this|that|these|those|such)\s+"
    r"(?:[a-z0-9+/_-]+\s+){0,3}"
    r"(?:relationship|relation|association|interaction|effect|modulation|pathway)\b",
    flags=re.I,
)


def has_anaphoric_relation_reference(text: object) -> bool:
    """Return True only for demonstrative relation references.

    This is intentionally narrower than a generic pronoun check. It rejects
    forms such as "this relationship" or "that modulation", while preserving
    self-contained wording such as "the relationship between A and B" and the
    complementizer in "I propose that A modulates B".
    """

    return bool(
        _ANAPHORIC_RELATION_RE.search(
            _surface(text)
        )
    )


def _surface(text: object) -> str:
    value = unicodedata.normalize("NFKC", str(text or ""))
    value = value.replace("–", "-").replace("—", "-").replace("−", "-")
    return " ".join(value.split())


def _norm(text: object) -> str:
    return _surface(text).casefold()


def _contains(text: object, needle: object) -> bool:
    n = _norm(needle)
    return bool(n) and n in _norm(text)


def _source_rows(hypothesis: HypothesisCard) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = [
        ("hypothesis_statement", hypothesis.hypothesis_statement),
        ("inferential_bridge", hypothesis.inferential_bridge),
    ]

    rows.extend(
        (f"assumptions[{i}]", value)
        for i, value in enumerate(hypothesis.assumptions)
    )

    for i, item in enumerate(hypothesis.predicted_observations):
        rows.append(
            (f"predicted_observations[{i}].observable", item.observable)
        )
        rows.append(
            (f"predicted_observations[{i}].rationale", item.rationale)
        )

    for i, item in enumerate(hypothesis.falsification_criteria):
        rows.append(
            (f"falsification_criteria[{i}].observable", item.observable)
        )
        rows.append(
            (
                f"falsification_criteria[{i}].falsifying_outcome",
                item.falsifying_outcome,
            )
        )

    return rows


def _exact_source_paths(
    value: str,
    rows: list[tuple[str, str]],
) -> list[str]:
    cleaned = _surface(value)
    if not cleaned:
        return []

    return [
        path
        for path, source in rows
        if _contains(source, cleaned)
    ]


def _enclosing_sentence(source: str, basis: str) -> str:
    """Return the normalized source sentence that contains an exact basis span."""

    source_surface = _surface(source)
    basis_surface = _surface(basis)
    if not source_surface or not basis_surface:
        return ""

    start = source_surface.casefold().find(basis_surface.casefold())
    if start < 0:
        return ""

    end = start + len(basis_surface)

    # A colon/semicolon can separate proposition-level clauses inside one
    # sentence. Treat them as context boundaries so a basis for the first
    # proposition does not inherit qualifiers from a later proposition.
    boundary_marks = (".", "?", "!", ";", ":")

    left_candidates = [
        source_surface.rfind(mark, 0, start)
        for mark in boundary_marks
    ]
    left = max(left_candidates)

    # If the exact basis already ends at a boundary, do not absorb the next
    # clause/sentence merely because the search begins after that boundary.
    stripped_basis = basis_surface.rstrip()
    if stripped_basis and stripped_basis[-1] in boundary_marks:
        right = end
    else:
        right_candidates = [
            pos
            for mark in boundary_marks
            for pos in [source_surface.find(mark, end)]
            if pos >= 0
        ]
        right = (
            min(right_candidates) + 1
            if right_candidates
            else len(source_surface)
        )

    return _surface(source_surface[left + 1 : right])


def _basis_source_contexts(
    basis: str,
    rows: list[tuple[str, str]],
) -> list[dict[str, str]]:
    cleaned = _surface(basis)
    if not cleaned:
        return []

    contexts: list[dict[str, str]] = []
    for path, source in rows:
        if not _contains(source, cleaned):
            continue
        sentence = _enclosing_sentence(source, cleaned)
        contexts.append(
            {
                "source_path": path,
                "sentence": sentence,
            }
        )
    return contexts


def _markers(pattern: re.Pattern[str], text: object) -> list[str]:
    value = _surface(text)
    return sorted(
        {
            _surface(match.group(0))
            for match in pattern.finditer(value)
            if _surface(match.group(0))
        },
        key=lambda marker: (marker.casefold(), marker),
    )


_ORDERED_LANGUAGE_REVIEW_REASON_CODES = {
    "atomic_claim_ordered_language_not_in_source_basis",
    "atomic_claim_ordered_language_not_in_source_context",
}


def compile_ordered_marker_source_channel_shadow(
    hypothesis: HypothesisCard,
    claim: NoveltyClaimDraft,
    *,
    semantic_fidelity_shadow: dict[str, Any],
) -> dict[str, Any]:
    """Partition ordered markers by their generated/source channels.

    This is diagnostic-only. It does not remove or rewrite the current
    semantic-fidelity reason codes, does not infer synonymy, and grants no
    production, repair, or novelty authority.

    Channel contract:
      * markers present in claim.text require exact support in the bound
        proposition basis or its enclosing source context;
      * markers present only in predicted_observation require exact support
        in the prediction record selected by prediction_observation_id;
      * prediction-source wording must never launder unsupported claim.text
        wording.
    """

    basis = _surface(
        semantic_fidelity_shadow.get("proposition_basis")
    )
    contexts = list(
        semantic_fidelity_shadow.get(
            "proposition_basis_source_contexts"
        )
        or []
    )
    context_text = " ".join(
        _surface(dict(row).get("sentence"))
        for row in contexts
        if isinstance(row, dict)
        and _surface(dict(row).get("sentence"))
    )

    claim_text_markers = _markers(
        _ORDERED_MARKER_RE,
        claim.text,
    )
    predicted_observation_markers = _markers(
        _ORDERED_MARKER_RE,
        claim.predicted_observation,
    )
    generated_markers = sorted(
        set(claim_text_markers)
        | set(predicted_observation_markers),
        key=str.casefold,
    )

    prediction_id = str(
        semantic_fidelity_shadow.get(
            "prediction_observation_id"
        )
        or ""
    ).strip()
    selected_prediction = next(
        (
            row
            for row in hypothesis.predicted_observations
            if row.observation_id == prediction_id
        ),
        None,
    )
    selected_prediction_source_text = (
        " ".join(
            [
                str(selected_prediction.observable or ""),
                str(selected_prediction.rationale or ""),
                str(selected_prediction.expected_direction or ""),
            ]
        )
        if selected_prediction is not None
        else ""
    )
    selected_prediction_source_markers = _markers(
        _ORDERED_MARKER_RE,
        selected_prediction_source_text,
    )

    marker_provenance: list[dict[str, Any]] = []
    for marker in generated_markers:
        in_claim_text = marker in claim_text_markers
        in_predicted_observation = (
            marker in predicted_observation_markers
        )
        basis_or_context_supported = bool(
            _contains(basis, marker)
            or _contains(context_text, marker)
        )
        selected_prediction_supported = bool(
            selected_prediction_source_text
            and _contains(
                selected_prediction_source_text,
                marker,
            )
        )

        if in_claim_text:
            required_channel = (
                "CLAIM_TEXT_BASIS_OR_CONTEXT"
            )
            channel_supported = (
                basis_or_context_supported
            )
        elif in_predicted_observation:
            required_channel = (
                "PREDICTED_OBSERVATION_SELECTED_PREDICTION"
            )
            channel_supported = (
                selected_prediction_supported
            )
        else:
            required_channel = "UNKNOWN_GENERATED_FIELD"
            channel_supported = False

        marker_provenance.append(
            {
                "marker": marker,
                "in_claim_text": in_claim_text,
                "in_predicted_observation": (
                    in_predicted_observation
                ),
                "basis_or_context_exact_supported": (
                    basis_or_context_supported
                ),
                "selected_prediction_exact_supported": (
                    selected_prediction_supported
                ),
                "required_channel": required_channel,
                "channel_supported": channel_supported,
            }
        )

    unsupported_markers = [
        row["marker"]
        for row in marker_provenance
        if not row["channel_supported"]
    ]

    current_reason_codes = list(
        semantic_fidelity_shadow.get("reason_codes")
        or []
    )
    current_ordered_reason_codes = [
        code
        for code in current_reason_codes
        if code in _ORDERED_LANGUAGE_REVIEW_REASON_CODES
    ]
    other_current_reason_codes = [
        code
        for code in current_reason_codes
        if code not in _ORDERED_LANGUAGE_REVIEW_REASON_CODES
    ]

    candidate_clearable = bool(
        current_ordered_reason_codes
        and not unsupported_markers
        and not other_current_reason_codes
        and set(current_ordered_reason_codes)
        == _ORDERED_LANGUAGE_REVIEW_REASON_CODES
    )

    return {
        "schema_version": (
            "novelty-claim-ordered-marker-"
            "source-channel-shadow-v1"
        ),
        "diagnostic_only": True,
        "production_authority": False,
        "production_recompile_enabled": False,
        "recompile_performed": False,
        "reason_codes_modified": False,
        "semantic_synonymy_allowed": False,
        "scientific_truth_assessed": False,
        "novelty_assessed": False,
        "claim_local_id": claim.local_id,
        "prediction_observation_id": (
            prediction_id or None
        ),
        "selected_prediction_source_found": (
            selected_prediction is not None
        ),
        "claim_text_ordered_markers": (
            claim_text_markers
        ),
        "predicted_observation_ordered_markers": (
            predicted_observation_markers
        ),
        "generated_ordered_markers": (
            generated_markers
        ),
        "source_basis_ordered_markers": _markers(
            _ORDERED_MARKER_RE,
            basis,
        ),
        "source_context_ordered_markers": _markers(
            _ORDERED_MARKER_RE,
            context_text,
        ),
        "selected_prediction_source_ordered_markers": (
            selected_prediction_source_markers
        ),
        "marker_provenance": marker_provenance,
        "unsupported_markers_by_required_channel": (
            unsupported_markers
        ),
        "current_ordered_language_reason_codes": (
            current_ordered_reason_codes
        ),
        "other_current_reason_codes": (
            other_current_reason_codes
        ),
        "channel_aware_exact_status": (
            "NO_UNSUPPORTED_ORDERED_MARKERS"
            if not unsupported_markers
            else "REVIEW_REQUIRED"
        ),
        "would_clear_current_ordered_reason_pair_if_authoritative": (
            candidate_clearable
        ),
    }


_EPISTEMIC_MODAL_TOKENS = (
    "may",
    "might",
    "can",
    "could",
)

_EPISTEMIC_MODAL_PREDICATE_FORMS = {
    "affect": ("affect", "affects", "affected", "affecting"),
    "alter": ("alter", "alters", "altered", "altering"),
    "change": ("change", "changes", "changed", "changing"),
    "decrease": ("decrease", "decreases", "decreased", "decreasing"),
    "improve": ("improve", "improves", "improved", "improving"),
    "increase": ("increase", "increases", "increased", "increasing"),
    "influence": ("influence", "influences", "influenced", "influencing"),
    "mediate": ("mediate", "mediates", "mediated", "mediating"),
    "moderate": ("moderate", "moderates", "moderated", "moderating"),
    "modulate": ("modulate", "modulates", "modulated", "modulating"),
    "predict": ("predict", "predicts", "predicted", "predicting"),
    "promote": ("promote", "promotes", "promoted", "promoting"),
    "reduce": ("reduce", "reduces", "reduced", "reducing"),
    "regulate": ("regulate", "regulates", "regulated", "regulating"),
    "support": ("support", "supports", "supported", "supporting"),
}


def _epistemic_modal_mentions(text: object) -> list[dict[str, str]]:
    """Extract only immediate modal+predicate lexical pairs."""

    surface = _surface(text)
    mentions: list[dict[str, str]] = []

    for lemma, forms in _EPISTEMIC_MODAL_PREDICATE_FORMS.items():
        form_pattern = "|".join(
            re.escape(form)
            for form in sorted(forms, key=len, reverse=True)
        )
        pattern = re.compile(
            rf"\b(?P<modal>{'|'.join(_EPISTEMIC_MODAL_TOKENS)})"
            rf"\s+(?P<predicate>{form_pattern})\b",
            flags=re.I,
        )
        for match in pattern.finditer(surface):
            mentions.append(
                {
                    "modal": match.group("modal").casefold(),
                    "predicate_lemma": lemma,
                    "predicate_surface": match.group("predicate").casefold(),
                    "surface": match.group(0),
                }
            )

    return sorted(
        mentions,
        key=lambda row: (
            row["predicate_lemma"],
            row["modal"],
            row["surface"].casefold(),
        ),
    )


def _unqualified_predicate_mentions(
    text: object,
    predicate_lemma: str,
) -> list[dict[str, str]]:
    """Find exact lexical predicate forms not immediately modal/negation qualified."""

    forms = _EPISTEMIC_MODAL_PREDICATE_FORMS.get(
        predicate_lemma,
        (),
    )
    if not forms:
        return []

    surface = _surface(text)
    pattern = re.compile(
        r"\b(?:"
        + "|".join(
            re.escape(form)
            for form in sorted(forms, key=len, reverse=True)
        )
        + r")\b",
        flags=re.I,
    )
    modal_or_negation = re.compile(
        r"\b(?:may|might|can|could|not|never)\s+$",
        flags=re.I,
    )

    mentions: list[dict[str, str]] = []
    for match in pattern.finditer(surface):
        prefix = surface[max(0, match.start() - 24):match.start()]
        if modal_or_negation.search(prefix):
            continue
        mentions.append(
            {
                "predicate_lemma": predicate_lemma,
                "predicate_surface": match.group(0).casefold(),
            }
        )
    return mentions


def compile_epistemic_modality_fidelity_shadow(
    claim: NoveltyClaimDraft,
    *,
    semantic_fidelity_shadow: dict[str, Any],
    sanitized_required_bridge: str,
) -> dict[str, Any]:
    """Diagnose obvious source-modal -> unqualified atomic-claim strengthening.

    S15-X is shadow-only. Only the bound proposition_basis can authorize a
    candidate. Enclosing source context is recorded but cannot supply a modal
    that the bound atomic proposition omitted.

    The detector requires the same lexical predicate lemma on both sides.
    It does not equate predicates, modal words, scientific entities, or
    scientific propositions.
    """

    basis = _surface(
        semantic_fidelity_shadow.get("proposition_basis")
    )
    contexts = list(
        semantic_fidelity_shadow.get(
            "proposition_basis_source_contexts"
        )
        or []
    )
    context_text = " ".join(
        _surface(dict(row).get("sentence"))
        for row in contexts
        if isinstance(row, dict)
        and _surface(dict(row).get("sentence"))
    )
    bridge = _surface(sanitized_required_bridge)
    claim_text = _surface(claim.text)

    basis_modal_mentions = _epistemic_modal_mentions(basis)
    context_modal_mentions = _epistemic_modal_mentions(
        context_text
    )
    bridge_modal_mentions = _epistemic_modal_mentions(bridge)
    claim_modal_mentions = _epistemic_modal_mentions(
        claim_text
    )

    source_predicate_lemmas = sorted(
        {
            row["predicate_lemma"]
            for row in basis_modal_mentions
        }
    )

    modal_drop_candidates: list[dict[str, Any]] = []
    for lemma in source_predicate_lemmas:
        source_mentions = [
            row
            for row in basis_modal_mentions
            if row["predicate_lemma"] == lemma
        ]
        bridge_mentions = [
            row
            for row in bridge_modal_mentions
            if row["predicate_lemma"] == lemma
        ]
        claim_modal_same_lemma = [
            row
            for row in claim_modal_mentions
            if row["predicate_lemma"] == lemma
        ]
        claim_unqualified = _unqualified_predicate_mentions(
            claim_text,
            lemma,
        )

        if not claim_unqualified:
            continue

        modal_drop_candidates.append(
            {
                "predicate_lemma": lemma,
                "source_basis_modal_mentions": source_mentions,
                "required_bridge_modal_mentions": bridge_mentions,
                "claim_modal_mentions": claim_modal_same_lemma,
                "claim_unqualified_mentions": claim_unqualified,
                "same_lexical_predicate_required": True,
                "bridge_modal_preservation_observed": bool(
                    bridge_mentions
                ),
            }
        )

    current_reason_codes = list(
        semantic_fidelity_shadow.get("reason_codes")
        or []
    )
    current_modal_reason_codes = [
        code
        for code in current_reason_codes
        if "modal" in str(code).casefold()
        or "epistemic" in str(code).casefold()
    ]

    return {
        "schema_version": (
            "novelty-claim-epistemic-modality-"
            "fidelity-shadow-v1"
        ),
        "diagnostic_only": True,
        "production_authority": False,
        "reason_codes_modified": False,
        "taxonomy_modified": False,
        "claim_acceptance_authority_changed": False,
        "novelty_authority_changed": False,
        "semantic_synonymy_allowed": False,
        "modal_token_equivalence_assumed": False,
        "scientific_equivalence_assessed": False,
        "scientific_truth_assessed": False,
        "novelty_assessed": False,
        "claim_local_id": claim.local_id,
        "proposition_basis": basis,
        "source_basis_modal_mentions": basis_modal_mentions,
        "source_context_modal_mentions": context_modal_mentions,
        "required_bridge_modal_mentions": bridge_modal_mentions,
        "claim_modal_mentions": claim_modal_mentions,
        "modal_drop_candidates": modal_drop_candidates,
        "candidate_reason_code": (
            "atomic_claim_epistemic_modality_"
            "may_be_strengthened"
            if modal_drop_candidates
            else None
        ),
        "current_modal_reason_codes": current_modal_reason_codes,
        "shadow_status": (
            "MODAL_STRENGTHENING_CANDIDATE"
            if modal_drop_candidates
            else "NO_MODAL_STRENGTHENING_CANDIDATE"
        ),
    }


def plan_modal_preserving_claim_text_repair_shadow(
    claim: NoveltyClaimDraft,
    *,
    epistemic_modality_fidelity_shadow: dict[str, Any],
) -> dict[str, Any]:
    """Plan one exact source-modal restoration without granting authority.

    S16-B consumes only the already-bound proposition_basis diagnostics emitted
    by the S15-X modality shadow. It never uses required_bridge as repair
    authority and never infers modal equivalence or predicate synonymy.
    """

    shadow = dict(epistemic_modality_fidelity_shadow)
    common = {
        "schema_version": (
            "novelty-claim-modal-preserving-repair-plan-shadow-v1"
        ),
        "diagnostic_only": True,
        "production_authority": False,
        "canonical_mutation_performed": False,
        "source_authority": "bound_proposition_basis",
        "required_bridge_is_authority": False,
        "semantic_synonymy_allowed": False,
        "modal_token_equivalence_assumed": False,
    }

    if shadow.get("shadow_status") != "MODAL_STRENGTHENING_CANDIDATE":
        return {
            **common,
            "status": "NO_REPAIR_NOT_MODAL_STRENGTHENING_CANDIDATE",
        }

    candidates = list(shadow.get("modal_drop_candidates") or [])
    if len(candidates) != 1:
        return {
            **common,
            "status": "NO_REPAIR_MODAL_DROP_CANDIDATE_CARDINALITY",
            "candidate_count": len(candidates),
        }

    candidate = candidates[0]
    source_mentions = list(
        candidate.get("source_basis_modal_mentions")
        or []
    )
    unqualified_mentions = list(
        candidate.get("claim_unqualified_mentions")
        or []
    )

    if len(source_mentions) != 1:
        return {
            **common,
            "status": "NO_REPAIR_SOURCE_MODAL_CARDINALITY",
            "source_modal_count": len(source_mentions),
        }

    if len(unqualified_mentions) != 1:
        return {
            **common,
            "status": "NO_REPAIR_CLAIM_PREDICATE_CARDINALITY",
            "claim_unqualified_count": len(unqualified_mentions),
        }

    source = source_mentions[0]
    unqualified = unqualified_mentions[0]
    predicate_lemma = _surface(
        candidate.get("predicate_lemma")
    )
    if not predicate_lemma:
        return {
            **common,
            "status": "NO_REPAIR_PREDICATE_LEMMA_EMPTY",
        }

    if (
        _surface(source.get("predicate_lemma"))
        != predicate_lemma
        or _surface(unqualified.get("predicate_lemma"))
        != predicate_lemma
    ):
        return {
            **common,
            "status": "NO_REPAIR_PREDICATE_LEMMA_MISMATCH",
        }

    basis = _surface(shadow.get("proposition_basis"))
    source_modal = _surface(source.get("modal")).casefold()
    source_predicate_surface = _surface(
        source.get("predicate_surface")
    ).casefold()
    source_surface = _surface(source.get("surface"))

    if (
        not source_modal
        or not source_predicate_surface
        or not source_surface
        or source_surface.casefold()
        != f"{source_modal} {source_predicate_surface}"
        or source_surface.casefold()
        not in basis.casefold()
    ):
        return {
            **common,
            "status": "NO_REPAIR_SOURCE_MODAL_NOT_EXACT_IN_BOUND_BASIS",
        }

    target_surface = _surface(
        unqualified.get("predicate_surface")
    )
    if not target_surface:
        return {
            **common,
            "status": "NO_REPAIR_CLAIM_TARGET_EMPTY",
        }

    # Exact replacement offsets must be computed against the raw
    # claim text. _surface() is for matching/diagnostics and normalizes
    # Unicode dashes, so using it here would make preview offsets refer to a
    # normalized copy rather than the canonical text.
    claim_text = str(claim.text or "")
    target_pattern = re.compile(
        rf"\b{re.escape(target_surface)}\b",
        flags=re.I,
    )
    target_matches = list(
        target_pattern.finditer(claim_text)
    )
    if len(target_matches) != 1:
        return {
            **common,
            "status": "NO_REPAIR_CLAIM_TARGET_NOT_EXACT_UNIQUE",
            "target_surface": target_surface,
            "target_match_count": len(target_matches),
        }

    match = target_matches[0]
    # Preserve the exact modal+predicate surface captured from the bound basis.
    # The lower-cased fields above are diagnostic keys, not rewrite text.
    replacement_surface = source_surface

    return {
        **common,
        "status": "MODAL_PRESERVING_TEXT_REPAIR_CANDIDATE_SHADOW",
        "predicate_lemma": predicate_lemma,
        "source_modal": source_modal,
        "source_predicate_surface": source_predicate_surface,
        "claim_target_surface": target_surface,
        "replacement_surface": replacement_surface,
        "replacement_start": match.start(),
        "replacement_end": match.end(),
    }


def preview_modal_preserving_claim_text_repair_shadow(
    claim: NoveltyClaimDraft,
    *,
    semantic_fidelity_shadow: dict[str, Any],
    sanitized_required_bridge: str,
    repair_plan_shadow: dict[str, Any],
) -> dict[str, Any]:
    """Preview the bounded modal restoration and re-run the modality shadow."""

    plan = dict(repair_plan_shadow)
    common = {
        "schema_version": (
            "novelty-claim-modal-preserving-repair-preview-shadow-v1"
        ),
        "diagnostic_only": True,
        "production_authority": False,
        "canonical_mutation_performed": False,
    }

    if plan.get("status") != (
        "MODAL_PRESERVING_TEXT_REPAIR_CANDIDATE_SHADOW"
    ):
        return {
            **common,
            "status": "NO_PREVIEW_PLAN_NOT_READY",
            "plan_status": plan.get("status"),
        }

    # Rewrite from the raw claim text so every non-target code point,
    # including Unicode punctuation, remains byte-for-byte unchanged.
    claim_text = str(claim.text or "")
    start = plan.get("replacement_start")
    end = plan.get("replacement_end")
    target_surface = _surface(
        plan.get("claim_target_surface")
    )
    replacement_surface = _surface(
        plan.get("replacement_surface")
    )

    if (
        not isinstance(start, int)
        or isinstance(start, bool)
        or not isinstance(end, int)
        or isinstance(end, bool)
        or start < 0
        or end <= start
        or end > len(claim_text)
        or claim_text[start:end].casefold()
        != target_surface.casefold()
        or not replacement_surface
    ):
        return {
            **common,
            "status": "NO_PREVIEW_STALE_OR_INVALID_PLAN",
        }

    preview_text = (
        claim_text[:start]
        + replacement_surface
        + claim_text[end:]
    )
    preview_claim = claim.model_copy(
        update={"text": preview_text}
    )
    post_preview_modality_shadow = (
        compile_epistemic_modality_fidelity_shadow(
            preview_claim,
            semantic_fidelity_shadow=(
                semantic_fidelity_shadow
            ),
            sanitized_required_bridge=(
                sanitized_required_bridge
            ),
        )
    )

    return {
        **common,
        "source_authority": "bound_proposition_basis",
        "required_bridge_is_authority": False,
        "semantic_synonymy_allowed": False,
        "modal_token_equivalence_assumed": False,
        "status": "PREVIEW_READY_SHADOW",
        "before_text": claim_text,
        "preview_text": preview_text,
        "changed_field_names": ["text"],
        "replacement_surface": replacement_surface,
        "post_preview_modality_shadow": (
            post_preview_modality_shadow
        ),
        "modal_strengthening_candidate_cleared": (
            post_preview_modality_shadow.get(
                "shadow_status"
            )
            == "NO_MODAL_STRENGTHENING_CANDIDATE"
        ),
    }


def assess_atomic_semantic_fidelity(
    hypothesis: HypothesisCard,
    claim: NoveltyClaimDraft,
) -> dict[str, Any]:
    """Compile diagnostic-only semantic-fidelity source binding.

    The assessment deliberately performs only surface/extractive checks.
    It does not infer synonymy, scientific equivalence, causal validity,
    truth, novelty, or non-obviousness.

    No result from this function has production authority in S14-A.
    """

    binding = claim.semantic_fidelity_binding
    source_rows = _source_rows(hypothesis)
    basis = _surface(binding.proposition_basis)
    reason_codes: list[str] = []

    basis_paths = _exact_source_paths(basis, source_rows)
    basis_source_contexts = _basis_source_contexts(basis, source_rows)
    basis_context_text = " ".join(
        row["sentence"]
        for row in basis_source_contexts
        if row["sentence"]
    )

    if not basis:
        reason_codes.append("atomic_claim_source_basis_empty")
    elif not basis_paths:
        reason_codes.append("atomic_claim_source_basis_not_exact")

    anchors = [
        _surface(value)
        for value in binding.relation_endpoint_anchors
        if _surface(value)
    ]

    if not anchors:
        reason_codes.append("atomic_claim_relation_endpoint_anchors_empty")

    missing_anchor_claim = [
        value
        for value in anchors
        if not _contains(claim.text, value)
    ]
    if missing_anchor_claim:
        reason_codes.append("atomic_claim_relation_endpoint_missing_from_claim")

    missing_anchor_basis = [
        value
        for value in anchors
        if not _contains(basis, value)
    ]
    if missing_anchor_basis:
        reason_codes.append("atomic_claim_relation_endpoint_missing_from_basis")

    scope_qualifiers = [
        _surface(value)
        for value in binding.scope_qualifier_spans
        if _surface(value)
    ]
    missing_scope_claim = [
        value
        for value in scope_qualifiers
        if not _contains(claim.text, value)
    ]
    missing_scope_basis = [
        value
        for value in scope_qualifiers
        if not _contains(basis, value)
    ]
    if missing_scope_claim:
        reason_codes.append("atomic_claim_scope_qualifier_not_preserved")
    if missing_scope_basis:
        reason_codes.append("atomic_claim_scope_qualifier_not_source_backed")

    directional_qualifiers = [
        _surface(value)
        for value in binding.directional_qualifier_spans
        if _surface(value)
    ]
    missing_direction_claim = [
        value
        for value in directional_qualifiers
        if not _contains(claim.text, value)
        and not _contains(claim.predicted_observation, value)
    ]
    missing_direction_basis = [
        value
        for value in directional_qualifiers
        if not _contains(basis, value)
    ]
    if missing_direction_claim:
        reason_codes.append("atomic_claim_direction_qualifier_not_preserved")
    if missing_direction_basis:
        reason_codes.append("atomic_claim_direction_qualifier_not_source_backed")

    claim_ordered_markers = _markers(
        _ORDERED_MARKER_RE,
        " ".join(
            [
                str(claim.text or ""),
                str(claim.predicted_observation or ""),
            ]
        ),
    )
    basis_ordered_markers = _markers(_ORDERED_MARKER_RE, basis)
    source_context_ordered_markers = _markers(
        _ORDERED_MARKER_RE,
        basis_context_text,
    )

    ordered_not_in_basis = [
        value
        for value in claim_ordered_markers
        if not _contains(basis, value)
    ]
    ordered_not_in_claim = [
        value
        for value in basis_ordered_markers
        if not _contains(claim.text, value)
        and not _contains(claim.predicted_observation, value)
    ]
    context_ordered_not_in_claim = [
        value
        for value in source_context_ordered_markers
        if not _contains(claim.text, value)
        and not _contains(claim.predicted_observation, value)
    ]
    claim_ordered_not_in_context = [
        value
        for value in claim_ordered_markers
        if not _contains(basis_context_text, value)
    ]

    if ordered_not_in_basis:
        reason_codes.append("atomic_claim_ordered_language_not_in_source_basis")
    if ordered_not_in_claim:
        reason_codes.append("atomic_claim_comparative_scope_may_be_broadened")
    if claim_ordered_not_in_context:
        reason_codes.append(
            "atomic_claim_ordered_language_not_in_source_context"
        )
    if context_ordered_not_in_claim:
        reason_codes.append(
            "atomic_claim_context_comparative_scope_may_be_broadened"
        )

    basis_condition_markers = _markers(_CONDITION_MARKER_RE, basis)
    source_context_condition_markers = _markers(
        _CONDITION_MARKER_RE,
        basis_context_text,
    )
    claim_condition_markers = _markers(
        _CONDITION_MARKER_RE,
        " ".join(
            [
                str(claim.text or ""),
                str(claim.predicted_observation or ""),
            ]
        ),
    )

    if basis_condition_markers and not claim_condition_markers:
        reason_codes.append("atomic_claim_condition_scope_may_be_dropped")
    if source_context_condition_markers and not claim_condition_markers:
        reason_codes.append(
            "atomic_claim_context_condition_scope_may_be_dropped"
        )

    prediction_by_id = {
        row.observation_id: row
        for row in hypothesis.predicted_observations
    }
    falsifier_by_id = {
        row.criterion_id: row
        for row in hypothesis.falsification_criteria
    }

    prediction_id = binding.prediction_observation_id
    if str(claim.predicted_observation or "").strip():
        if not prediction_id:
            reason_codes.append("atomic_prediction_source_id_missing")
        elif prediction_id not in prediction_by_id:
            reason_codes.append("atomic_prediction_source_id_unknown")
    elif prediction_id:
        reason_codes.append("atomic_prediction_source_id_without_prediction")

    falsifier_id = binding.falsification_criterion_id
    if str(claim.falsification_condition or "").strip():
        if not falsifier_id:
            reason_codes.append("atomic_falsifier_source_id_missing")
        elif falsifier_id not in falsifier_by_id:
            reason_codes.append("atomic_falsifier_source_id_unknown")
    elif falsifier_id:
        reason_codes.append("atomic_falsifier_source_id_without_falsifier")

    raw_bridge = _surface(claim.required_bridge)
    bridge_missing_endpoints = [
        value
        for value in anchors
        if raw_bridge and not _contains(raw_bridge, value)
    ]

    if raw_bridge and has_anaphoric_relation_reference(raw_bridge):
        reason_codes.append("required_bridge_anaphoric_relation_reference")

    if raw_bridge and bridge_missing_endpoints:
        reason_codes.append("required_bridge_relation_endpoint_alignment_unresolved")

    bridge_missing_scope = [
        value
        for value in scope_qualifiers
        if raw_bridge and not _contains(raw_bridge, value)
    ]
    if raw_bridge and bridge_missing_scope:
        reason_codes.append("required_bridge_scope_qualifier_missing")

    bridge_condition_markers = _markers(_CONDITION_MARKER_RE, raw_bridge)
    bridge_ordered_markers = _markers(_ORDERED_MARKER_RE, raw_bridge)

    if (
        raw_bridge
        and source_context_condition_markers
        and not bridge_condition_markers
    ):
        reason_codes.append(
            "required_bridge_condition_scope_may_be_dropped"
        )

    context_ordered_not_in_bridge = [
        value
        for value in source_context_ordered_markers
        if raw_bridge and not _contains(raw_bridge, value)
    ]
    if raw_bridge and context_ordered_not_in_bridge:
        reason_codes.append(
            "required_bridge_comparative_scope_may_be_broadened"
        )

    # Stable, non-duplicated diagnostic codes.
    reason_codes = list(dict.fromkeys(reason_codes))

    selected_prediction = (
        prediction_by_id.get(prediction_id)
        if prediction_id
        else None
    )
    selected_falsifier = (
        falsifier_by_id.get(falsifier_id)
        if falsifier_id
        else None
    )

    return {
        "schema_version": "novelty-claim-semantic-fidelity-shadow-v1",
        "diagnostic_only": True,
        "production_authority": False,
        "scientific_truth_assessed": False,
        "scientific_equivalence_assessed": False,
        "novelty_assessed": False,
        "claim_local_id": claim.local_id,
        "proposition_basis": basis,
        "proposition_basis_source_paths": basis_paths,
        "proposition_basis_source_contexts": basis_source_contexts,
        "relation_endpoint_anchors": anchors,
        "scope_qualifier_spans": scope_qualifiers,
        "directional_qualifier_spans": directional_qualifiers,
        "prediction_observation_id": prediction_id,
        "prediction_source_expected_direction": (
            selected_prediction.expected_direction
            if selected_prediction is not None
            else None
        ),
        "falsification_criterion_id": falsifier_id,
        "claim_ordered_markers": claim_ordered_markers,
        "source_basis_ordered_markers": basis_ordered_markers,
        "source_context_ordered_markers": source_context_ordered_markers,
        "source_basis_condition_markers": basis_condition_markers,
        "source_context_condition_markers": source_context_condition_markers,
        "claim_condition_markers": claim_condition_markers,
        "bridge_condition_markers": bridge_condition_markers,
        "bridge_ordered_markers": bridge_ordered_markers,
        "bridge_missing_relation_endpoint_anchors": bridge_missing_endpoints,
        "bridge_missing_scope_qualifiers": bridge_missing_scope,
        "reason_codes": reason_codes,
    }

_BINDING_CONTRACT_REASON_CODES = {
    "atomic_claim_source_basis_empty",
    "atomic_claim_source_basis_not_exact",
    "atomic_claim_relation_endpoint_anchors_empty",
    "atomic_claim_relation_endpoint_missing_from_claim",
    "atomic_claim_relation_endpoint_missing_from_basis",
    "atomic_claim_scope_qualifier_not_preserved",
    "atomic_claim_scope_qualifier_not_source_backed",
    "atomic_claim_direction_qualifier_not_preserved",
    "atomic_claim_direction_qualifier_not_source_backed",
    "atomic_prediction_source_id_missing",
    "atomic_prediction_source_id_unknown",
    "atomic_prediction_source_id_without_prediction",
    "atomic_falsifier_source_id_missing",
    "atomic_falsifier_source_id_unknown",
    "atomic_falsifier_source_id_without_falsifier",
}

_CLAIM_FIDELITY_REVIEW_REASON_CODES = {
    "atomic_claim_ordered_language_not_in_source_basis",
    "atomic_claim_comparative_scope_may_be_broadened",
    "atomic_claim_ordered_language_not_in_source_context",
    "atomic_claim_context_comparative_scope_may_be_broadened",
    "atomic_claim_condition_scope_may_be_dropped",
    "atomic_claim_context_condition_scope_may_be_dropped",
}


def compile_atomic_semantic_fidelity_taxonomy_shadow(
    claim: NoveltyClaimDraft,
    *,
    semantic_fidelity_shadow: dict[str, Any],
    sanitized_required_bridge: str,
    sanitized_predicted_observation: str,
    sanitized_falsification_condition: str,
) -> dict[str, Any]:
    """Partition semantic-fidelity diagnostics without granting authority.

    S14-D deliberately separates:
      1. whether the model-authored binding satisfies its literal contract;
      2. whether a valid binding still raises claim-fidelity review signals;
      3. whether a novelty-bearing atomic claim lost required specification;
      4. whether the sanitized bridge remains non-self-contained.

    None of these statuses is an acceptance/rejection gate in S14-D.
    """

    reason_codes = list(
        semantic_fidelity_shadow.get("reason_codes") or []
    )

    binding_findings = [
        code
        for code in reason_codes
        if code in _BINDING_CONTRACT_REASON_CODES
    ]
    binding_status = (
        "INVALID"
        if binding_findings
        else "VALID"
    )

    claim_fidelity_findings = [
        code
        for code in reason_codes
        if code in _CLAIM_FIDELITY_REVIEW_REASON_CODES
    ]
    if binding_status != "VALID":
        claim_fidelity_status = "NOT_ASSESSABLE_FROM_BINDING"
    elif claim_fidelity_findings:
        claim_fidelity_status = "REVIEW_REQUIRED"
    else:
        claim_fidelity_status = "NO_FLAG"

    completeness_findings: list[str] = []
    role = str(claim.novelty_selection_role or "")
    if role == "NOVELTY_BEARING":
        if not _surface(sanitized_required_bridge):
            completeness_findings.append(
                "NOVELTY_BEARING_REQUIRED_BRIDGE_EMPTY"
            )
        if not _surface(sanitized_predicted_observation):
            completeness_findings.append(
                "NOVELTY_BEARING_PREDICTION_EMPTY"
            )
        if not _surface(sanitized_falsification_condition):
            completeness_findings.append(
                "NOVELTY_BEARING_FALSIFIER_EMPTY"
            )

    completeness_status = (
        "REVIEW_REQUIRED"
        if completeness_findings
        else "NO_FLAG"
    )

    sanitized_bridge = _surface(
        sanitized_required_bridge
    )
    self_containment_findings: list[str] = []
    if (
        sanitized_bridge
        and has_anaphoric_relation_reference(
            sanitized_bridge
        )
    ):
        self_containment_findings.append(
            "SANITIZED_REQUIRED_BRIDGE_RETAINS_ANAPHORIC_REFERENCE"
        )

    self_containment_status = (
        "REVIEW_REQUIRED"
        if self_containment_findings
        else "NO_FLAG"
    )

    if (
        completeness_status == "REVIEW_REQUIRED"
        or self_containment_status == "REVIEW_REQUIRED"
    ):
        overall_review_status = "REVIEW_REQUIRED"
    elif binding_status != "VALID":
        overall_review_status = "NOT_ASSESSABLE_FROM_BINDING"
    elif claim_fidelity_status == "REVIEW_REQUIRED":
        overall_review_status = "REVIEW_REQUIRED"
    else:
        overall_review_status = "PASS_SHADOW"

    return {
        "schema_version": (
            "novelty-claim-semantic-fidelity-taxonomy-shadow-v1"
        ),
        "diagnostic_only": True,
        "production_authority": False,
        "semantic_fidelity_enforcement_enabled": False,
        "flat_reason_codes_are_authority": False,
        "binding_contract": {
            "status": binding_status,
            "findings": binding_findings,
        },
        "claim_fidelity": {
            "status": claim_fidelity_status,
            "findings": claim_fidelity_findings,
        },
        "specification_completeness": {
            "status": completeness_status,
            "findings": completeness_findings,
        },
        "specification_self_containment": {
            "status": self_containment_status,
            "findings": self_containment_findings,
        },
        "overall_review_status": overall_review_status,
    }
