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
from pipeline_core.discovery.novelty_specification_source_trace import (
    trace_specification_sources,
)
from pipeline_core.discovery.novelty_atomic_semantic_fidelity import (
    assess_atomic_semantic_fidelity,
    compile_atomic_semantic_fidelity_taxonomy_shadow,
    compile_epistemic_modality_fidelity_shadow,
    compile_ordered_marker_source_channel_shadow,
    has_anaphoric_relation_reference,
    plan_modal_preserving_claim_text_repair_shadow,
    preview_modal_preserving_claim_text_repair_shadow,
)
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



def _exact_source_recompile_proposition_units(
    text: str,
) -> list[str]:
    """Split source text only at strong proposition boundaries.

    Commas are intentionally preserved so leading conditions remain attached
    to the scientific proposition they delimit.
    """

    cleaned = " ".join(str(text or "").split()).strip()
    if not cleaned:
        return []

    return [
        part.strip()
        for part in re.split(
            r"(?<=[.!?;:])\s+",
            cleaned,
        )
        if part.strip()
    ]


def _literal_facet_present(
    text: str,
    facet: str,
) -> bool:
    normalized_facet = _clean_query(
        facet,
        limit=4000,
    ).casefold()
    normalized_text = _clean_query(
        text,
        limit=8000,
    ).casefold()
    return bool(
        normalized_facet
        and normalized_facet in normalized_text
    )


_SCOPE_WRAPPER_REBIND_PAIRS = {
    ("at", "under the condition of"),
    ("under the condition of", "at"),
}


def _normalize_scope_wrapper_core(text: str) -> str:
    return _clean_query(
        str(text or "").strip(" \t\r\n,;:.!?"),
        limit=4000,
    ).casefold()


def _split_scope_wrapper(
    text: str,
) -> tuple[str | None, str | None]:
    cleaned = " ".join(str(text or "").split()).strip()
    cleaned = cleaned.strip(" \t\r\n,;:.!?")

    patterns = (
        (
            "under the condition of",
            r"^under\s+the\s+condition\s+of\s+(.+)$",
        ),
        ("at", r"^at\s+(.+)$"),
    )
    for wrapper, pattern in patterns:
        match = re.match(
            pattern,
            cleaned,
            flags=re.IGNORECASE,
        )
        if match:
            core = match.group(1).strip(" \t\r\n,;:.!?")
            return wrapper, core

    return None, None


def _leading_scope_wrapper_before_comma(
    proposition: str,
) -> str:
    cleaned = " ".join(str(proposition or "").split()).strip()
    if "," not in cleaned:
        return ""

    prefix = cleaned.split(",", 1)[0].strip()
    wrapper, core = _split_scope_wrapper(prefix)
    if not wrapper or not core:
        return ""
    return prefix


def _scope_wrapper_exact_core_match(
    binding_scope: str,
    source_scope: str,
) -> dict[str, object]:
    binding_wrapper, binding_core = _split_scope_wrapper(binding_scope)
    source_wrapper, source_core = _split_scope_wrapper(source_scope)

    normalized_binding_core = _normalize_scope_wrapper_core(
        binding_core or ""
    )
    normalized_source_core = _normalize_scope_wrapper_core(
        source_core or ""
    )

    matched = bool(
        binding_wrapper
        and source_wrapper
        and (binding_wrapper, source_wrapper)
        in _SCOPE_WRAPPER_REBIND_PAIRS
        and normalized_binding_core
        and normalized_binding_core == normalized_source_core
    )

    return {
        "matched": matched,
        "binding_wrapper": binding_wrapper,
        "source_wrapper": source_wrapper,
        "binding_core": binding_core,
        "source_core": source_core,
        "normalized_binding_core": normalized_binding_core,
        "normalized_source_core": normalized_source_core,
    }


def _plan_scope_wrapper_exact_source_atomic_recompile_shadow(
    hypothesis: HypothesisCard,
    claim: object,
    *,
    sanitized_required_bridge: str,
    exact_source_plan: dict[str, object],
) -> dict[str, object]:
    """Plan a narrow grammatical-scope rebind as diagnostic fallback only.

    This planner is considered only when the existing exact-source planner
    reports NO_EXACT_SOURCE_RECOMPILE_CANDIDATE. It does not treat scope
    phrases as semantic synonyms. Exactly one model-declared scope qualifier
    must differ from the source only by an approved grammatical wrapper, and
    the scientific scope core must match exactly after lexical normalization.
    Endpoints and directions remain literal-match requirements, the source
    proposition must independently pass the required-bridge sanitizer, and
    multiple candidates are never selected.
    """

    binding = claim.semantic_fidelity_binding
    anchors = [
        " ".join(str(value or "").split())
        for value in binding.relation_endpoint_anchors
        if " ".join(str(value or "").split())
    ]
    scope_qualifiers = [
        " ".join(str(value or "").split())
        for value in binding.scope_qualifier_spans
        if " ".join(str(value or "").split())
    ]
    directional_qualifiers = [
        " ".join(str(value or "").split())
        for value in binding.directional_qualifier_spans
        if " ".join(str(value or "").split())
    ]

    candidates: list[dict[str, object]] = []
    seen: set[str] = set()

    exact_classification = str(
        exact_source_plan.get("classification") or ""
    )
    eligible_fallback = bool(
        not sanitized_required_bridge
        and exact_classification
        == "NO_EXACT_SOURCE_RECOMPILE_CANDIDATE"
        and anchors
        and len(scope_qualifiers) == 1
    )

    if eligible_fallback:
        sources: list[tuple[str, str]] = [
            ("inferential_bridge", hypothesis.inferential_bridge),
        ]
        sources.extend(
            (f"assumptions[{index}]", value)
            for index, value in enumerate(hypothesis.assumptions)
        )

        binding_scope = scope_qualifiers[0]

        for parent_path, source_text in sources:
            for unit_index, unit in enumerate(
                _exact_source_recompile_proposition_units(source_text)
            ):
                anchor_hits = {
                    value: _literal_facet_present(unit, value)
                    for value in anchors
                }
                direction_hits = {
                    value: _literal_facet_present(unit, value)
                    for value in directional_qualifiers
                }

                if not all(anchor_hits.values()):
                    continue
                if not all(direction_hits.values()):
                    continue

                sanitized_candidate = _clean_branch_specific_bridge(
                    unit,
                    list(claim.prior_art_identity_terms),
                    [unit],
                )
                if not sanitized_candidate:
                    continue

                source_scope = _leading_scope_wrapper_before_comma(unit)
                if not source_scope:
                    continue

                scope_match = _scope_wrapper_exact_core_match(
                    binding_scope,
                    source_scope,
                )
                if not scope_match["matched"]:
                    continue

                key = _clean_query(
                    sanitized_candidate,
                    limit=8000,
                ).casefold()
                if key in seen:
                    continue
                seen.add(key)

                candidates.append(
                    {
                        "source_path": (
                            f"{parent_path}.unit[{unit_index}]"
                        ),
                        "parent_path": parent_path,
                        "exact_source_text": unit,
                        "binding_scope": binding_scope,
                        "source_scope": source_scope,
                        "scope_match": scope_match,
                        "anchor_hits": anchor_hits,
                        "direction_hits": direction_hits,
                        "sanitized_candidate": sanitized_candidate,
                    }
                )

    if exact_classification != "NO_EXACT_SOURCE_RECOMPILE_CANDIDATE":
        classification = "NOT_ELIGIBLE_EXISTING_EXACT_SOURCE_PATH"
    elif sanitized_required_bridge:
        classification = "NOT_ELIGIBLE_REQUIRED_BRIDGE_PRESENT"
    elif not anchors:
        classification = "BINDING_UNUSABLE_NO_RELATION_ENDPOINTS"
    elif len(scope_qualifiers) != 1:
        classification = "BINDING_UNUSABLE_SCOPE_CARDINALITY"
    elif len(candidates) == 1:
        classification = (
            "SCOPE_WRAPPER_EXACT_SOURCE_RECOMPILE_CANDIDATE"
        )
    elif len(candidates) > 1:
        classification = (
            "AMBIGUOUS_SCOPE_WRAPPER_EXACT_SOURCE_RECOMPILE_CANDIDATES"
        )
    else:
        classification = (
            "NO_SCOPE_WRAPPER_EXACT_SOURCE_RECOMPILE_CANDIDATE"
        )

    return {
        "schema_version": (
            "novelty-claim-scope-wrapper-exact-source-"
            "recompile-shadow-v1"
        ),
        "diagnostic_only": True,
        "production_authority": False,
        "production_recompile_enabled": False,
        "recompile_performed": False,
        "semantic_scope_synonymy_allowed": False,
        "approved_wrapper_pairs": [
            list(pair)
            for pair in sorted(_SCOPE_WRAPPER_REBIND_PAIRS)
        ],
        "claim_local_id": claim.local_id,
        "fallback_from_exact_source_classification": exact_classification,
        "classification": classification,
        "relation_endpoint_anchors": anchors,
        "scope_qualifier_spans": scope_qualifiers,
        "directional_qualifier_spans": directional_qualifiers,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }



def _preview_scope_wrapper_exact_source_atomic_recompile_shadow(
    hypothesis: HypothesisCard,
    claim: object,
    *,
    recompile_plan: dict[str, object],
) -> dict[str, object]:
    """Compile a unique scope-wrapper candidate into a diagnostic preview.

    The preview applies only the S14-T bounded rebind: claim.text and
    required_bridge become the exact source proposition, proposition_basis is
    rebound to that same proposition, and the single scope qualifier is rebound
    to the exact source scope wording. Endpoint/direction/source-ID bindings and
    every other draft field must remain unchanged. No production authority is
    granted and the supplied claim is never mutated.
    """

    base = {
        "schema_version": (
            "novelty-claim-scope-wrapper-exact-source-"
            "recompile-preview-shadow-v1"
        ),
        "diagnostic_only": True,
        "production_authority": False,
        "production_recompile_enabled": False,
        "recompile_performed": False,
        "semantic_scope_synonymy_allowed": False,
        "claim_local_id": claim.local_id,
        "planner_classification": recompile_plan.get("classification"),
        "preview_status": "NOT_ELIGIBLE",
        "ready_for_bounded_recompile": False,
        "candidate_source_path": None,
        "exact_source_text": "",
        "binding_scope": "",
        "source_scope": "",
        "preview_claim": None,
        "sanitized_required_bridge": "",
        "sanitized_predicted_observation": "",
        "sanitized_falsification_condition": "",
        "semantic_fidelity_shadow": None,
        "semantic_fidelity_taxonomy_shadow": None,
        "mutation_audit": {
            "allowed_changed_fields": [
                "required_bridge",
                "semantic_fidelity_binding",
                "text",
            ],
            "allowed_binding_changed_fields": [
                "proposition_basis",
                "scope_qualifier_spans",
            ],
            "changed_fields": [],
            "unexpected_changed_fields": [],
            "changed_binding_fields": [],
            "unexpected_binding_changed_fields": [],
            "all_other_fields_preserved": True,
        },
        "preview_reason_codes": [],
    }

    if (
        recompile_plan.get("classification")
        != "SCOPE_WRAPPER_EXACT_SOURCE_RECOMPILE_CANDIDATE"
        or int(recompile_plan.get("candidate_count") or 0) != 1
    ):
        return base

    candidates = list(recompile_plan.get("candidates") or [])
    if len(candidates) != 1:
        base["preview_status"] = "REVIEW_REQUIRED_PREVIEW"
        base["preview_reason_codes"] = [
            "scope_wrapper_recompile_preview_candidate_cardinality_unexpected"
        ]
        return base

    candidate = candidates[0]
    exact_source = " ".join(
        str(candidate.get("exact_source_text") or "").split()
    ).strip()
    source_path = str(candidate.get("source_path") or "").strip()
    binding_scope = " ".join(
        str(candidate.get("binding_scope") or "").split()
    ).strip()
    source_scope = " ".join(
        str(candidate.get("source_scope") or "").split()
    ).strip()
    scope_match = dict(candidate.get("scope_match") or {})

    base["candidate_source_path"] = source_path or None
    base["exact_source_text"] = exact_source
    base["binding_scope"] = binding_scope
    base["source_scope"] = source_scope

    if (
        not exact_source
        or not source_path
        or not binding_scope
        or not source_scope
        or scope_match.get("matched") is not True
    ):
        base["preview_status"] = "REVIEW_REQUIRED_PREVIEW"
        base["preview_reason_codes"] = [
            "scope_wrapper_recompile_preview_candidate_source_invalid"
        ]
        return base

    before_payload = claim.model_dump(mode="json")
    after_payload = json.loads(json.dumps(before_payload))
    after_payload["text"] = exact_source
    after_payload["required_bridge"] = exact_source

    binding_payload = dict(
        after_payload.get("semantic_fidelity_binding") or {}
    )
    binding_payload["proposition_basis"] = exact_source
    binding_payload["scope_qualifier_spans"] = [source_scope]
    after_payload["semantic_fidelity_binding"] = binding_payload

    preview_claim = type(claim).model_validate(after_payload)
    preview_payload = preview_claim.model_dump(mode="json")

    allowed_changed_fields = {
        "text",
        "required_bridge",
        "semantic_fidelity_binding",
    }
    changed_fields = sorted(
        key
        for key in set(before_payload) | set(preview_payload)
        if before_payload.get(key) != preview_payload.get(key)
    )
    unexpected_changed_fields = [
        key
        for key in changed_fields
        if key not in allowed_changed_fields
    ]

    before_binding = dict(
        before_payload.get("semantic_fidelity_binding") or {}
    )
    preview_binding = dict(
        preview_payload.get("semantic_fidelity_binding") or {}
    )
    changed_binding_fields = sorted(
        key
        for key in set(before_binding) | set(preview_binding)
        if before_binding.get(key) != preview_binding.get(key)
    )
    allowed_binding_changed_fields = {
        "proposition_basis",
        "scope_qualifier_spans",
    }
    unexpected_binding_changed_fields = [
        key
        for key in changed_binding_fields
        if key not in allowed_binding_changed_fields
    ]

    bridge_source_texts = [
        hypothesis.inferential_bridge,
        *hypothesis.assumptions,
    ]
    sanitized_required_bridge = _clean_branch_specific_bridge(
        preview_claim.required_bridge,
        list(preview_claim.prior_art_identity_terms),
        bridge_source_texts,
    )
    sanitized_predicted_observation = (
        _clean_branch_specific_specification(
            preview_claim.predicted_observation,
            list(preview_claim.prior_art_identity_terms),
        )
    )
    sanitized_falsification_condition = (
        _clean_branch_specific_specification(
            preview_claim.falsification_condition,
            list(preview_claim.prior_art_identity_terms),
        )
    )

    semantic_fidelity_shadow = assess_atomic_semantic_fidelity(
        hypothesis,
        preview_claim,
    )
    taxonomy = compile_atomic_semantic_fidelity_taxonomy_shadow(
        preview_claim,
        semantic_fidelity_shadow=semantic_fidelity_shadow,
        sanitized_required_bridge=sanitized_required_bridge,
        sanitized_predicted_observation=sanitized_predicted_observation,
        sanitized_falsification_condition=sanitized_falsification_condition,
    )

    preview_reason_codes: list[str] = []
    if sanitized_required_bridge != exact_source:
        preview_reason_codes.append(
            "scope_wrapper_recompile_preview_required_bridge_not_preserved"
        )
    if unexpected_changed_fields:
        preview_reason_codes.append(
            "scope_wrapper_recompile_preview_unexpected_field_mutation"
        )
    if unexpected_binding_changed_fields:
        preview_reason_codes.append(
            "scope_wrapper_recompile_preview_binding_mutation_out_of_bounds"
        )
    if list(preview_binding.get("scope_qualifier_spans") or []) != [source_scope]:
        preview_reason_codes.append(
            "scope_wrapper_recompile_preview_scope_not_rebound_exactly"
        )
    if " ".join(
        str(preview_binding.get("proposition_basis") or "").split()
    ).strip() != exact_source:
        preview_reason_codes.append(
            "scope_wrapper_recompile_preview_basis_not_exact_source"
        )
    if semantic_fidelity_shadow.get("reason_codes"):
        preview_reason_codes.append(
            "scope_wrapper_recompile_preview_semantic_fidelity_review"
        )
    if taxonomy["binding_contract"]["status"] != "VALID":
        preview_reason_codes.append(
            "scope_wrapper_recompile_preview_binding_contract_not_valid"
        )
    if taxonomy["claim_fidelity"]["status"] != "NO_FLAG":
        preview_reason_codes.append(
            "scope_wrapper_recompile_preview_claim_fidelity_not_clean"
        )
    if taxonomy["specification_completeness"]["status"] != "NO_FLAG":
        preview_reason_codes.append(
            "scope_wrapper_recompile_preview_specification_incomplete"
        )
    if taxonomy["specification_self_containment"]["status"] != "NO_FLAG":
        preview_reason_codes.append(
            "scope_wrapper_recompile_preview_self_containment_not_clean"
        )
    if taxonomy["overall_review_status"] != "PASS_SHADOW":
        preview_reason_codes.append(
            "scope_wrapper_recompile_preview_overall_not_pass_shadow"
        )

    ready = not preview_reason_codes

    base.update(
        {
            "preview_status": (
                "PASS_SHADOW_PREVIEW"
                if ready
                else "REVIEW_REQUIRED_PREVIEW"
            ),
            "ready_for_bounded_recompile": ready,
            "preview_claim": preview_payload,
            "sanitized_required_bridge": sanitized_required_bridge,
            "sanitized_predicted_observation": (
                sanitized_predicted_observation
            ),
            "sanitized_falsification_condition": (
                sanitized_falsification_condition
            ),
            "semantic_fidelity_shadow": semantic_fidelity_shadow,
            "semantic_fidelity_taxonomy_shadow": taxonomy,
            "mutation_audit": {
                "allowed_changed_fields": sorted(allowed_changed_fields),
                "allowed_binding_changed_fields": sorted(
                    allowed_binding_changed_fields
                ),
                "changed_fields": changed_fields,
                "unexpected_changed_fields": unexpected_changed_fields,
                "changed_binding_fields": changed_binding_fields,
                "unexpected_binding_changed_fields": (
                    unexpected_binding_changed_fields
                ),
                "all_other_fields_preserved": bool(
                    not unexpected_changed_fields
                    and not unexpected_binding_changed_fields
                ),
            },
            "preview_reason_codes": preview_reason_codes,
        }
    )
    return base


def _authorize_scope_wrapper_exact_source_atomic_recompile_shadow(
    claim: object,
    *,
    recompile_plan: dict[str, object],
    recompile_preview: dict[str, object],
) -> dict[str, object]:
    """Re-check the wrapper-rebind mutation boundary without granting authority."""

    reason_codes: list[str] = []
    candidates = list(recompile_plan.get("candidates") or [])

    if (
        recompile_plan.get("classification")
        != "SCOPE_WRAPPER_EXACT_SOURCE_RECOMPILE_CANDIDATE"
        or int(recompile_plan.get("candidate_count") or 0) != 1
        or len(candidates) != 1
    ):
        reason_codes.append(
            "scope_wrapper_recompile_authority_plan_not_unique_candidate"
        )

    candidate = candidates[0] if len(candidates) == 1 else {}
    candidate_source_path = str(
        candidate.get("source_path") or ""
    ).strip()
    candidate_exact_source = " ".join(
        str(candidate.get("exact_source_text") or "").split()
    ).strip()
    candidate_sanitized = " ".join(
        str(candidate.get("sanitized_candidate") or "").split()
    ).strip()
    binding_scope = " ".join(
        str(candidate.get("binding_scope") or "").split()
    ).strip()
    source_scope = " ".join(
        str(candidate.get("source_scope") or "").split()
    ).strip()
    scope_match = dict(candidate.get("scope_match") or {})

    if (
        not candidate_source_path
        or not (
            candidate_source_path.startswith("inferential_bridge.unit[")
            or candidate_source_path.startswith("assumptions[")
        )
        or not candidate_exact_source
        or candidate_sanitized != candidate_exact_source
        or not binding_scope
        or not source_scope
        or scope_match.get("matched") is not True
        or scope_match.get("normalized_binding_core")
        != scope_match.get("normalized_source_core")
    ):
        reason_codes.append(
            "scope_wrapper_recompile_authority_candidate_source_invalid"
        )

    for facet_key in ("anchor_hits", "direction_hits"):
        hits = dict(candidate.get(facet_key) or {})
        if hits and not all(bool(value) for value in hits.values()):
            reason_codes.append(
                "scope_wrapper_recompile_authority_candidate_facet_guard_failed"
            )
            break

    current_binding = claim.semantic_fidelity_binding
    current_scopes = [
        " ".join(str(value or "").split())
        for value in current_binding.scope_qualifier_spans
        if " ".join(str(value or "").split())
    ]
    if current_scopes != [binding_scope]:
        reason_codes.append(
            "scope_wrapper_recompile_authority_binding_scope_identity_mismatch"
        )

    if (
        recompile_preview.get("preview_status")
        != "PASS_SHADOW_PREVIEW"
        or recompile_preview.get("ready_for_bounded_recompile") is not True
    ):
        reason_codes.append(
            "scope_wrapper_recompile_authority_preview_not_ready"
        )

    if (
        recompile_preview.get("diagnostic_only") is not True
        or recompile_preview.get("production_authority") is not False
        or recompile_preview.get("production_recompile_enabled") is not False
        or recompile_preview.get("recompile_performed") is not False
        or recompile_preview.get("semantic_scope_synonymy_allowed") is not False
    ):
        reason_codes.append(
            "scope_wrapper_recompile_authority_preview_metadata_guard_failed"
        )

    if (
        str(recompile_preview.get("candidate_source_path") or "").strip()
        != candidate_source_path
        or " ".join(
            str(recompile_preview.get("exact_source_text") or "").split()
        ).strip()
        != candidate_exact_source
        or " ".join(
            str(recompile_preview.get("binding_scope") or "").split()
        ).strip()
        != binding_scope
        or " ".join(
            str(recompile_preview.get("source_scope") or "").split()
        ).strip()
        != source_scope
    ):
        reason_codes.append(
            "scope_wrapper_recompile_authority_source_identity_mismatch"
        )

    if list(recompile_preview.get("preview_reason_codes") or []):
        reason_codes.append(
            "scope_wrapper_recompile_authority_preview_reason_codes_present"
        )

    before_payload = claim.model_dump(mode="json")
    preview_payload_raw = recompile_preview.get("preview_claim")
    preview_payload = (
        dict(preview_payload_raw)
        if isinstance(preview_payload_raw, dict)
        else None
    )

    changed_fields: list[str] = []
    unexpected_changed_fields: list[str] = []
    nested_binding_changed_fields: list[str] = []
    unexpected_binding_changed_fields: list[str] = []

    if preview_payload is None:
        reason_codes.append(
            "scope_wrapper_recompile_authority_preview_claim_missing"
        )
    else:
        try:
            validated_preview = type(claim).model_validate(preview_payload)
            preview_payload = validated_preview.model_dump(mode="json")
        except Exception:
            reason_codes.append(
                "scope_wrapper_recompile_authority_preview_claim_invalid"
            )
            preview_payload = None

    if preview_payload is not None:
        allowed_changed_fields = {
            "text",
            "required_bridge",
            "semantic_fidelity_binding",
        }
        changed_fields = sorted(
            key
            for key in set(before_payload) | set(preview_payload)
            if before_payload.get(key) != preview_payload.get(key)
        )
        unexpected_changed_fields = [
            key
            for key in changed_fields
            if key not in allowed_changed_fields
        ]
        if unexpected_changed_fields:
            reason_codes.append(
                "scope_wrapper_recompile_authority_unexpected_field_mutation"
            )

        before_binding = dict(
            before_payload.get("semantic_fidelity_binding") or {}
        )
        preview_binding = dict(
            preview_payload.get("semantic_fidelity_binding") or {}
        )
        nested_binding_changed_fields = sorted(
            key
            for key in set(before_binding) | set(preview_binding)
            if before_binding.get(key) != preview_binding.get(key)
        )
        allowed_binding_changed_fields = {
            "proposition_basis",
            "scope_qualifier_spans",
        }
        unexpected_binding_changed_fields = [
            key
            for key in nested_binding_changed_fields
            if key not in allowed_binding_changed_fields
        ]
        if unexpected_binding_changed_fields:
            reason_codes.append(
                "scope_wrapper_recompile_authority_binding_mutation_out_of_bounds"
            )

        if (
            " ".join(str(preview_payload.get("text") or "").split()).strip()
            != candidate_exact_source
            or " ".join(
                str(preview_payload.get("required_bridge") or "").split()
            ).strip()
            != candidate_exact_source
            or " ".join(
                str(preview_binding.get("proposition_basis") or "").split()
            ).strip()
            != candidate_exact_source
            or list(preview_binding.get("scope_qualifier_spans") or [])
            != [source_scope]
        ):
            reason_codes.append(
                "scope_wrapper_recompile_authority_preview_payload_not_exact_rebind"
            )

        if " ".join(
            str(
                recompile_preview.get("sanitized_required_bridge") or ""
            ).split()
        ).strip() != candidate_exact_source:
            reason_codes.append(
                "scope_wrapper_recompile_authority_required_bridge_not_preserved"
            )

        if " ".join(
            str(
                recompile_preview.get("sanitized_predicted_observation") or ""
            ).split()
        ).strip() != " ".join(
            str(preview_payload.get("predicted_observation") or "").split()
        ).strip():
            reason_codes.append(
                "scope_wrapper_recompile_authority_prediction_not_preserved"
            )

        if " ".join(
            str(
                recompile_preview.get("sanitized_falsification_condition") or ""
            ).split()
        ).strip() != " ".join(
            str(preview_payload.get("falsification_condition") or "").split()
        ).strip():
            reason_codes.append(
                "scope_wrapper_recompile_authority_falsifier_not_preserved"
            )

    audit = dict(recompile_preview.get("mutation_audit") or {})
    if (
        list(audit.get("unexpected_changed_fields") or [])
        or list(audit.get("unexpected_binding_changed_fields") or [])
        or audit.get("all_other_fields_preserved") is not True
        or sorted(audit.get("changed_fields") or []) != changed_fields
        or sorted(audit.get("changed_binding_fields") or [])
        != nested_binding_changed_fields
    ):
        reason_codes.append(
            "scope_wrapper_recompile_authority_mutation_audit_mismatch"
        )

    semantic_shadow = recompile_preview.get("semantic_fidelity_shadow")
    if (
        not isinstance(semantic_shadow, dict)
        or list(semantic_shadow.get("reason_codes") or [])
    ):
        reason_codes.append(
            "scope_wrapper_recompile_authority_semantic_fidelity_not_clean"
        )

    taxonomy = recompile_preview.get(
        "semantic_fidelity_taxonomy_shadow"
    )
    if not isinstance(taxonomy, dict):
        reason_codes.append(
            "scope_wrapper_recompile_authority_taxonomy_missing"
        )
    else:
        taxonomy_clean = (
            taxonomy.get("binding_contract", {}).get("status") == "VALID"
            and taxonomy.get("claim_fidelity", {}).get("status") == "NO_FLAG"
            and taxonomy.get("specification_completeness", {}).get("status")
            == "NO_FLAG"
            and taxonomy.get("specification_self_containment", {}).get("status")
            == "NO_FLAG"
            and taxonomy.get("overall_review_status") == "PASS_SHADOW"
        )
        if not taxonomy_clean:
            reason_codes.append(
                "scope_wrapper_recompile_authority_taxonomy_not_clean"
            )

    unique_reason_codes = list(dict.fromkeys(reason_codes))
    contract_satisfied = not unique_reason_codes

    return {
        "schema_version": (
            "novelty-claim-scope-wrapper-exact-source-"
            "recompile-authority-shadow-v1"
        ),
        "diagnostic_only": True,
        "production_authority": False,
        "production_recompile_enabled": False,
        "recompile_performed": False,
        "semantic_scope_synonymy_allowed": False,
        "claim_local_id": claim.local_id,
        "authority_status": (
            "AUTHORIZED_SHADOW"
            if contract_satisfied
            else "DENIED_SHADOW"
        ),
        "bounded_recompile_contract_satisfied": contract_satisfied,
        "candidate_source_path": candidate_source_path or None,
        "exact_source_text": candidate_exact_source,
        "binding_scope": binding_scope,
        "source_scope": source_scope,
        "changed_fields": changed_fields,
        "unexpected_changed_fields": unexpected_changed_fields,
        "nested_binding_changed_fields": nested_binding_changed_fields,
        "unexpected_binding_changed_fields": unexpected_binding_changed_fields,
        "authority_reason_codes": unique_reason_codes,
    }


def _plan_existing_bridge_scope_alignment_shadow(
    claim: object,
    *,
    sanitized_required_bridge: str,
    exact_source_plan: dict[str, object],
    semantic_fidelity_shadow: dict[str, object],
    specification_source_trace: dict[str, object],
) -> dict[str, object]:
    """Plan text-only alignment for an already-retained exact bridge."""

    bridge = " ".join(str(sanitized_required_bridge or "").split()).strip()
    exact_classification = str(exact_source_plan.get("classification") or "")
    reasons = list(semantic_fidelity_shadow.get("reason_codes") or [])
    binding = claim.semantic_fidelity_binding
    binding_scopes = [
        " ".join(str(value or "").split())
        for value in binding.scope_qualifier_spans
        if " ".join(str(value or "").split())
    ]
    claim_scope = _leading_scope_wrapper_before_comma(str(claim.text or ""))
    source_scope = _leading_scope_wrapper_before_comma(bridge)
    binding_scope = binding_scopes[0] if len(binding_scopes) == 1 else ""
    binding_wrapper, binding_core = _split_scope_wrapper(binding_scope)
    source_wrapper, source_core = _split_scope_wrapper(source_scope)
    normalized_binding_core = _normalize_scope_wrapper_core(binding_core or "")
    normalized_source_core = _normalize_scope_wrapper_core(source_core or "")
    binding_source_scope_match = bool(
        binding_wrapper
        and source_wrapper
        and binding_wrapper == source_wrapper
        and normalized_binding_core == normalized_source_core
    )
    scope_match = _scope_wrapper_exact_core_match(claim_scope, source_scope)

    fields = dict(specification_source_trace.get("fields") or {})
    bridge_trace = dict(fields.get("required_bridge") or {})
    source_state = str(bridge_trace.get("raw_source_match_state") or "")
    accepted = list(bridge_trace.get("accepted_exact_matches") or [])
    accepted_quote = (
        " ".join(str(dict(accepted[0]).get("quote") or "").split()).strip()
        if len(accepted) == 1
        else ""
    )

    eligible = bool(
        bridge
        and exact_classification == "NO_RECOMPILE_NEEDED"
        and reasons == ["atomic_claim_scope_qualifier_not_preserved"]
        and source_state == "EXACT_UNIQUE"
        and len(accepted) == 1
        and accepted_quote == bridge
        and len(binding_scopes) == 1
        and binding_source_scope_match
        and claim_scope
        and source_scope
        and scope_match.get("matched") is True
        and not list(
            semantic_fidelity_shadow.get(
                "bridge_missing_relation_endpoint_anchors"
            )
            or []
        )
        and not list(
            semantic_fidelity_shadow.get("bridge_missing_scope_qualifiers")
            or []
        )
    )

    if not bridge:
        classification = "NOT_ELIGIBLE_REQUIRED_BRIDGE_ABSENT"
    elif exact_classification != "NO_RECOMPILE_NEEDED":
        classification = "NOT_ELIGIBLE_EXACT_SOURCE_PATH_NOT_STABLE"
    elif reasons != ["atomic_claim_scope_qualifier_not_preserved"]:
        classification = "NOT_ELIGIBLE_SEMANTIC_REASON_SET"
    elif source_state != "EXACT_UNIQUE":
        classification = "NOT_ELIGIBLE_REQUIRED_BRIDGE_NOT_EXACT_UNIQUE"
    elif len(accepted) != 1 or accepted_quote != bridge:
        classification = "NOT_ELIGIBLE_REQUIRED_BRIDGE_TRACE_IDENTITY_MISMATCH"
    elif len(binding_scopes) != 1:
        classification = "BINDING_UNUSABLE_SCOPE_CARDINALITY"
    elif not binding_source_scope_match:
        classification = "BINDING_SCOPE_NOT_NORMALIZED_EXACT_SOURCE_SCOPE"
    elif not claim_scope:
        classification = "CLAIM_SCOPE_WRAPPER_NOT_FOUND"
    elif not source_scope:
        classification = "SOURCE_SCOPE_WRAPPER_NOT_FOUND"
    elif scope_match.get("matched") is not True:
        classification = "NO_EXISTING_BRIDGE_SCOPE_ALIGNMENT_CANDIDATE"
    elif eligible:
        classification = "EXISTING_BRIDGE_SCOPE_ALIGNMENT_CANDIDATE"
    else:
        classification = "EXISTING_BRIDGE_SCOPE_ALIGNMENT_GUARD_FAILED"

    return {
        "schema_version": (
            "novelty-claim-existing-bridge-scope-alignment-shadow-v1"
        ),
        "diagnostic_only": True,
        "production_authority": False,
        "production_recompile_enabled": False,
        "recompile_performed": False,
        "semantic_scope_synonymy_allowed": False,
        "claim_local_id": claim.local_id,
        "classification": classification,
        "fallback_from_exact_source_classification": exact_classification,
        "semantic_reason_codes": reasons,
        "required_bridge_source_match_state": source_state,
        "binding_scope_qualifier_spans": binding_scopes,
        "binding_source_scope_match": {
            "matched": binding_source_scope_match,
            "binding_wrapper": binding_wrapper,
            "source_wrapper": source_wrapper,
            "normalized_binding_core": normalized_binding_core,
            "normalized_source_core": normalized_source_core,
        },
        "claim_scope": claim_scope,
        "source_scope": source_scope,
        "scope_match": scope_match,
        "candidate_count": 1 if eligible else 0,
        "candidate": (
            {
                "exact_source_text": bridge,
                "claim_scope": claim_scope,
                "source_scope": source_scope,
                "scope_match": scope_match,
                "accepted_exact_match": dict(accepted[0]),
            }
            if eligible
            else None
        ),
    }


def _preview_existing_bridge_scope_alignment_shadow(
    hypothesis: HypothesisCard,
    claim: object,
    *,
    alignment_plan: dict[str, object],
) -> dict[str, object]:
    """Preview exact-source claim-text alignment only."""

    base = {
        "schema_version": (
            "novelty-claim-existing-bridge-scope-alignment-preview-shadow-v1"
        ),
        "diagnostic_only": True,
        "production_authority": False,
        "production_recompile_enabled": False,
        "recompile_performed": False,
        "semantic_scope_synonymy_allowed": False,
        "claim_local_id": claim.local_id,
        "planner_classification": alignment_plan.get("classification"),
        "preview_status": "NOT_ELIGIBLE",
        "ready_for_bounded_recompile": False,
        "exact_source_text": "",
        "claim_scope": "",
        "source_scope": "",
        "preview_claim": None,
        "sanitized_required_bridge": "",
        "sanitized_predicted_observation": "",
        "sanitized_falsification_condition": "",
        "semantic_fidelity_shadow": None,
        "semantic_fidelity_taxonomy_shadow": None,
        "mutation_audit": {
            "allowed_changed_fields": ["text"],
            "allowed_binding_changed_fields": [],
            "changed_fields": [],
            "unexpected_changed_fields": [],
            "changed_binding_fields": [],
            "unexpected_binding_changed_fields": [],
            "all_other_fields_preserved": True,
        },
        "preview_reason_codes": [],
    }

    if (
        alignment_plan.get("classification")
        != "EXISTING_BRIDGE_SCOPE_ALIGNMENT_CANDIDATE"
        or int(alignment_plan.get("candidate_count") or 0) != 1
        or not isinstance(alignment_plan.get("candidate"), dict)
    ):
        return base

    candidate = dict(alignment_plan["candidate"])
    exact_source = " ".join(
        str(candidate.get("exact_source_text") or "").split()
    ).strip()
    claim_scope = " ".join(
        str(candidate.get("claim_scope") or "").split()
    ).strip()
    source_scope = " ".join(
        str(candidate.get("source_scope") or "").split()
    ).strip()

    before = claim.model_dump(mode="json")
    after = json.loads(json.dumps(before))
    after["text"] = exact_source
    reason_codes: list[str] = []

    try:
        preview_claim = type(claim).model_validate(after)
    except Exception:
        base["preview_status"] = "REVIEW_REQUIRED_PREVIEW"
        base["preview_reason_codes"] = [
            "existing_bridge_scope_alignment_preview_claim_invalid"
        ]
        return base

    payload = preview_claim.model_dump(mode="json")
    changed = sorted(
        key
        for key in set(before) | set(payload)
        if before.get(key) != payload.get(key)
    )
    unexpected = [key for key in changed if key != "text"]
    before_binding = dict(before.get("semantic_fidelity_binding") or {})
    after_binding = dict(payload.get("semantic_fidelity_binding") or {})
    binding_changed = sorted(
        key
        for key in set(before_binding) | set(after_binding)
        if before_binding.get(key) != after_binding.get(key)
    )

    if changed != ["text"]:
        reason_codes.append(
            "existing_bridge_scope_alignment_preview_text_only_contract_failed"
        )
    if unexpected:
        reason_codes.append(
            "existing_bridge_scope_alignment_preview_unexpected_field_mutation"
        )
    if binding_changed:
        reason_codes.append(
            "existing_bridge_scope_alignment_preview_binding_mutation"
        )

    sources = [hypothesis.inferential_bridge, *hypothesis.assumptions]
    identities = list(claim.prior_art_identity_terms)
    old_bridge = _clean_branch_specific_bridge(
        claim.required_bridge, identities, sources
    )
    new_bridge = _clean_branch_specific_bridge(
        preview_claim.required_bridge, identities, sources
    )
    old_prediction = _clean_branch_specific_specification(
        claim.predicted_observation, identities
    )
    new_prediction = _clean_branch_specific_specification(
        preview_claim.predicted_observation, identities
    )
    old_falsifier = _clean_branch_specific_specification(
        claim.falsification_condition, identities
    )
    new_falsifier = _clean_branch_specific_specification(
        preview_claim.falsification_condition, identities
    )

    if old_bridge != exact_source or new_bridge != exact_source:
        reason_codes.append(
            "existing_bridge_scope_alignment_preview_bridge_not_preserved"
        )
    if new_prediction != old_prediction:
        reason_codes.append(
            "existing_bridge_scope_alignment_preview_prediction_changed"
        )
    if new_falsifier != old_falsifier:
        reason_codes.append(
            "existing_bridge_scope_alignment_preview_falsifier_changed"
        )

    semantic_shadow = assess_atomic_semantic_fidelity(hypothesis, preview_claim)
    taxonomy = compile_atomic_semantic_fidelity_taxonomy_shadow(
        preview_claim,
        semantic_fidelity_shadow=semantic_shadow,
        sanitized_required_bridge=new_bridge,
        sanitized_predicted_observation=new_prediction,
        sanitized_falsification_condition=new_falsifier,
    )

    if list(semantic_shadow.get("reason_codes") or []):
        reason_codes.append(
            "existing_bridge_scope_alignment_preview_semantic_fidelity_not_clean"
        )

    taxonomy_clean = (
        taxonomy.get("binding_contract", {}).get("status") == "VALID"
        and taxonomy.get("claim_fidelity", {}).get("status") == "NO_FLAG"
        and taxonomy.get("specification_completeness", {}).get("status")
        == "NO_FLAG"
        and taxonomy.get("specification_self_containment", {}).get("status")
        == "NO_FLAG"
        and taxonomy.get("overall_review_status") == "PASS_SHADOW"
    )
    if not taxonomy_clean:
        reason_codes.append(
            "existing_bridge_scope_alignment_preview_taxonomy_not_clean"
        )

    unique_reasons = list(dict.fromkeys(reason_codes))
    ready = not unique_reasons
    base.update(
        {
            "preview_status": (
                "PASS_SHADOW_PREVIEW"
                if ready
                else "REVIEW_REQUIRED_PREVIEW"
            ),
            "ready_for_bounded_recompile": ready,
            "exact_source_text": exact_source,
            "claim_scope": claim_scope,
            "source_scope": source_scope,
            "preview_claim": payload,
            "sanitized_required_bridge": new_bridge,
            "sanitized_predicted_observation": new_prediction,
            "sanitized_falsification_condition": new_falsifier,
            "semantic_fidelity_shadow": semantic_shadow,
            "semantic_fidelity_taxonomy_shadow": taxonomy,
            "mutation_audit": {
                "allowed_changed_fields": ["text"],
                "allowed_binding_changed_fields": [],
                "changed_fields": changed,
                "unexpected_changed_fields": unexpected,
                "changed_binding_fields": binding_changed,
                "unexpected_binding_changed_fields": binding_changed,
                "all_other_fields_preserved": (
                    not unexpected and not binding_changed
                ),
            },
            "preview_reason_codes": unique_reasons,
        }
    )
    return base


def _authorize_existing_bridge_scope_alignment_shadow(
    claim: object,
    *,
    alignment_plan: dict[str, object],
    alignment_preview: dict[str, object],
) -> dict[str, object]:
    """Re-check the text-only alignment preview without granting authority."""

    reasons: list[str] = []
    candidate = (
        dict(alignment_plan.get("candidate") or {})
        if isinstance(alignment_plan.get("candidate"), dict)
        else {}
    )
    exact_source = " ".join(
        str(candidate.get("exact_source_text") or "").split()
    ).strip()
    claim_scope = " ".join(
        str(candidate.get("claim_scope") or "").split()
    ).strip()
    source_scope = " ".join(
        str(candidate.get("source_scope") or "").split()
    ).strip()
    scope_match = dict(candidate.get("scope_match") or {})

    if (
        alignment_plan.get("classification")
        != "EXISTING_BRIDGE_SCOPE_ALIGNMENT_CANDIDATE"
        or int(alignment_plan.get("candidate_count") or 0) != 1
    ):
        reasons.append(
            "existing_bridge_scope_alignment_authority_plan_not_candidate"
        )
    if (
        not exact_source
        or not claim_scope
        or not source_scope
        or scope_match.get("matched") is not True
        or scope_match.get("normalized_binding_core")
        != scope_match.get("normalized_source_core")
    ):
        reasons.append(
            "existing_bridge_scope_alignment_authority_candidate_invalid"
        )
    if (
        alignment_preview.get("preview_status") != "PASS_SHADOW_PREVIEW"
        or alignment_preview.get("ready_for_bounded_recompile") is not True
    ):
        reasons.append(
            "existing_bridge_scope_alignment_authority_preview_not_ready"
        )
    if (
        alignment_preview.get("diagnostic_only") is not True
        or alignment_preview.get("production_authority") is not False
        or alignment_preview.get("production_recompile_enabled") is not False
        or alignment_preview.get("recompile_performed") is not False
        or alignment_preview.get("semantic_scope_synonymy_allowed") is not False
    ):
        reasons.append(
            "existing_bridge_scope_alignment_authority_metadata_guard_failed"
        )
    if list(alignment_preview.get("preview_reason_codes") or []):
        reasons.append(
            "existing_bridge_scope_alignment_authority_preview_reason_codes_present"
        )

    before = claim.model_dump(mode="json")
    raw_preview = alignment_preview.get("preview_claim")
    preview = dict(raw_preview) if isinstance(raw_preview, dict) else None
    changed: list[str] = []
    unexpected: list[str] = []
    binding_changed: list[str] = []

    if preview is None:
        reasons.append(
            "existing_bridge_scope_alignment_authority_preview_claim_missing"
        )
    else:
        try:
            preview = type(claim).model_validate(preview).model_dump(mode="json")
        except Exception:
            preview = None
            reasons.append(
                "existing_bridge_scope_alignment_authority_preview_claim_invalid"
            )

    if preview is not None:
        changed = sorted(
            key
            for key in set(before) | set(preview)
            if before.get(key) != preview.get(key)
        )
        unexpected = [key for key in changed if key != "text"]
        before_binding = dict(before.get("semantic_fidelity_binding") or {})
        preview_binding = dict(preview.get("semantic_fidelity_binding") or {})
        binding_changed = sorted(
            key
            for key in set(before_binding) | set(preview_binding)
            if before_binding.get(key) != preview_binding.get(key)
        )
        if changed != ["text"] or unexpected:
            reasons.append(
                "existing_bridge_scope_alignment_authority_text_only_contract_failed"
            )
        if binding_changed:
            reasons.append(
                "existing_bridge_scope_alignment_authority_binding_mutation"
            )
        if " ".join(str(preview.get("text") or "").split()).strip() != exact_source:
            reasons.append(
                "existing_bridge_scope_alignment_authority_text_not_exact_source"
            )
        if " ".join(
            str(alignment_preview.get("sanitized_required_bridge") or "").split()
        ).strip() != exact_source:
            reasons.append(
                "existing_bridge_scope_alignment_authority_bridge_not_preserved"
            )

    audit = dict(alignment_preview.get("mutation_audit") or {})
    if (
        sorted(audit.get("changed_fields") or []) != changed
        or list(audit.get("unexpected_changed_fields") or []) != unexpected
        or list(audit.get("changed_binding_fields") or []) != binding_changed
        or list(audit.get("unexpected_binding_changed_fields") or [])
        != binding_changed
        or audit.get("all_other_fields_preserved") is not True
    ):
        reasons.append(
            "existing_bridge_scope_alignment_authority_mutation_audit_mismatch"
        )

    semantic_shadow = alignment_preview.get("semantic_fidelity_shadow")
    if (
        not isinstance(semantic_shadow, dict)
        or list(semantic_shadow.get("reason_codes") or [])
    ):
        reasons.append(
            "existing_bridge_scope_alignment_authority_semantic_fidelity_not_clean"
        )
    taxonomy = alignment_preview.get("semantic_fidelity_taxonomy_shadow")
    taxonomy_clean = isinstance(taxonomy, dict) and (
        taxonomy.get("binding_contract", {}).get("status") == "VALID"
        and taxonomy.get("claim_fidelity", {}).get("status") == "NO_FLAG"
        and taxonomy.get("specification_completeness", {}).get("status")
        == "NO_FLAG"
        and taxonomy.get("specification_self_containment", {}).get("status")
        == "NO_FLAG"
        and taxonomy.get("overall_review_status") == "PASS_SHADOW"
    )
    if not taxonomy_clean:
        reasons.append(
            "existing_bridge_scope_alignment_authority_taxonomy_not_clean"
        )

    unique_reasons = list(dict.fromkeys(reasons))
    satisfied = not unique_reasons
    return {
        "schema_version": (
            "novelty-claim-existing-bridge-scope-alignment-authority-shadow-v1"
        ),
        "diagnostic_only": True,
        "production_authority": False,
        "production_recompile_enabled": False,
        "recompile_performed": False,
        "semantic_scope_synonymy_allowed": False,
        "claim_local_id": claim.local_id,
        "authority_status": (
            "AUTHORIZED_SHADOW" if satisfied else "DENIED_SHADOW"
        ),
        "bounded_recompile_contract_satisfied": satisfied,
        "exact_source_text": exact_source,
        "claim_scope": claim_scope,
        "source_scope": source_scope,
        "changed_fields": changed,
        "unexpected_changed_fields": unexpected,
        "nested_binding_changed_fields": binding_changed,
        "authority_reason_codes": unique_reasons,
    }



def _apply_existing_bridge_scope_alignment_canonical_action_opt_in(
    hypothesis: HypothesisCard,
    decomposition: HypothesisNoveltyClaims,
    source_claim: object,
    *,
    claim_id: str,
) -> tuple[HypothesisNoveltyClaims, dict[str, object]]:
    """Explicit opt-in canonical text action for validated alignment only.

    This helper is intentionally NOT called by NoveltyClaimDecomposer.decompose().
    It recomputes the complete existing-bridge alignment shadow chain from the
    supplied source draft and hypothesis, then applies only the authorized preview
    text to exactly one canonical NoveltyClaim.

    It does not enable exact-source or empty-bridge scope-wrapper canonical actions,
    does not grant novelty/scientific authority, and does not mutate its inputs.
    """

    if not isinstance(decomposition, HypothesisNoveltyClaims):
        raise ValueError(
            "alignment canonical action requires HypothesisNoveltyClaims"
        )
    if not str(claim_id or "").strip():
        raise ValueError(
            "alignment canonical action requires nonempty claim_id"
        )
    if not hasattr(source_claim, "model_dump"):
        raise ValueError(
            "alignment canonical action requires model-backed source claim"
        )

    source_before = source_claim.model_dump(mode="json")
    container_before = decomposition.model_dump(mode="json")

    if decomposition.hypothesis_id != hypothesis.hypothesis_id:
        raise ValueError(
            "alignment canonical action hypothesis identity mismatch"
        )

    matches = [
        row
        for row in decomposition.claims
        if row.claim_id == claim_id
    ]
    if len(matches) != 1:
        raise ValueError(
            "alignment canonical action requires exactly one canonical target"
        )

    canonical_before = matches[0]
    canonical_before_payload = canonical_before.model_dump(mode="json")

    if canonical_before.hypothesis_id != hypothesis.hypothesis_id:
        raise ValueError(
            "alignment canonical action target hypothesis mismatch"
        )
    if canonical_before.text != str(source_claim.text):
        raise ValueError(
            "alignment canonical action stale source/canonical text mismatch"
        )
    if canonical_before.kind != source_claim.kind:
        raise ValueError(
            "alignment canonical action kind mismatch"
        )
    if canonical_before.importance != source_claim.importance:
        raise ValueError(
            "alignment canonical action importance mismatch"
        )
    if (
        canonical_before.novelty_selection_role
        != source_claim.novelty_selection_role
    ):
        raise ValueError(
            "alignment canonical action role mismatch"
        )
    if canonical_before.rationale != source_claim.rationale:
        raise ValueError(
            "alignment canonical action rationale mismatch"
        )

    identities = list(source_claim.prior_art_identity_terms)
    bridge_sources = [
        hypothesis.inferential_bridge,
        *hypothesis.assumptions,
    ]

    sanitized_required_bridge = _clean_branch_specific_bridge(
        source_claim.required_bridge,
        identities,
        bridge_sources,
    )
    sanitized_predicted_observation = (
        _clean_branch_specific_specification(
            source_claim.predicted_observation,
            identities,
        )
    )
    sanitized_falsification_condition = (
        _clean_branch_specific_specification(
            source_claim.falsification_condition,
            identities,
        )
    )

    if canonical_before.required_bridge != sanitized_required_bridge:
        raise ValueError(
            "alignment canonical action canonical bridge does not match "
            "current sanitizer output"
        )
    if (
        canonical_before.predicted_observation
        != sanitized_predicted_observation
    ):
        raise ValueError(
            "alignment canonical action canonical prediction does not match "
            "current sanitizer output"
        )
    if (
        canonical_before.falsification_condition
        != sanitized_falsification_condition
    ):
        raise ValueError(
            "alignment canonical action canonical falsifier does not match "
            "current sanitizer output"
        )

    specification_source_trace = trace_specification_sources(
        hypothesis,
        {
            "required_bridge": source_claim.required_bridge,
            "predicted_observation": source_claim.predicted_observation,
            "falsification_condition": source_claim.falsification_condition,
        },
        {
            "required_bridge": sanitized_required_bridge,
            "predicted_observation": sanitized_predicted_observation,
            "falsification_condition": (
                sanitized_falsification_condition
            ),
        },
    )

    semantic_fidelity_shadow = assess_atomic_semantic_fidelity(
        hypothesis,
        source_claim,
    )
    exact_source_plan = _plan_exact_source_atomic_recompile_shadow(
        hypothesis,
        source_claim,
        sanitized_required_bridge=sanitized_required_bridge,
    )
    alignment_plan = _plan_existing_bridge_scope_alignment_shadow(
        source_claim,
        sanitized_required_bridge=sanitized_required_bridge,
        exact_source_plan=exact_source_plan,
        semantic_fidelity_shadow=semantic_fidelity_shadow,
        specification_source_trace=specification_source_trace,
    )
    alignment_preview = (
        _preview_existing_bridge_scope_alignment_shadow(
            hypothesis,
            source_claim,
            alignment_plan=alignment_plan,
        )
    )
    alignment_authority = (
        _authorize_existing_bridge_scope_alignment_shadow(
            source_claim,
            alignment_plan=alignment_plan,
            alignment_preview=alignment_preview,
        )
    )

    if (
        exact_source_plan.get("classification")
        != "NO_RECOMPILE_NEEDED"
    ):
        raise ValueError(
            "alignment canonical action requires stable existing exact bridge"
        )
    if (
        alignment_plan.get("classification")
        != "EXISTING_BRIDGE_SCOPE_ALIGNMENT_CANDIDATE"
        or int(alignment_plan.get("candidate_count") or 0) != 1
    ):
        raise ValueError(
            "alignment canonical action planner did not produce unique candidate"
        )
    if (
        alignment_preview.get("preview_status")
        != "PASS_SHADOW_PREVIEW"
        or alignment_preview.get("ready_for_bounded_recompile")
        is not True
    ):
        raise ValueError(
            "alignment canonical action preview is not ready"
        )
    if (
        alignment_authority.get("authority_status")
        != "AUTHORIZED_SHADOW"
        or alignment_authority.get(
            "bounded_recompile_contract_satisfied"
        )
        is not True
        or list(
            alignment_authority.get("authority_reason_codes")
            or []
        )
    ):
        raise ValueError(
            "alignment canonical action shadow authority denied"
        )

    # Shadow metadata must remain explicitly non-production.
    for label, payload in (
        ("plan", alignment_plan),
        ("preview", alignment_preview),
        ("authority", alignment_authority),
    ):
        if payload.get("diagnostic_only") is not True:
            raise ValueError(
                f"alignment canonical action {label} diagnostic metadata invalid"
            )
        if payload.get("production_authority") is not False:
            raise ValueError(
                f"alignment canonical action {label} production authority invalid"
            )
        if payload.get("production_recompile_enabled") is not False:
            raise ValueError(
                f"alignment canonical action {label} production flag invalid"
            )
        if payload.get("recompile_performed") is not False:
            raise ValueError(
                f"alignment canonical action {label} recompile flag invalid"
            )
        if payload.get("semantic_scope_synonymy_allowed") is not False:
            raise ValueError(
                f"alignment canonical action {label} synonymy flag invalid"
            )

    local_id = str(source_claim.local_id)
    for label, payload in (
        ("plan", alignment_plan),
        ("preview", alignment_preview),
        ("authority", alignment_authority),
    ):
        if str(payload.get("claim_local_id") or "") != local_id:
            raise ValueError(
                f"alignment canonical action {label} local-id mismatch"
            )

    candidate = dict(alignment_plan.get("candidate") or {})
    exact_source_text = " ".join(
        str(candidate.get("exact_source_text") or "").split()
    ).strip()

    if not exact_source_text:
        raise ValueError(
            "alignment canonical action exact source text missing"
        )
    if (
        " ".join(
            str(alignment_preview.get("exact_source_text") or "").split()
        ).strip()
        != exact_source_text
    ):
        raise ValueError(
            "alignment canonical action preview/source identity mismatch"
        )
    if (
        " ".join(
            str(alignment_authority.get("exact_source_text") or "").split()
        ).strip()
        != exact_source_text
    ):
        raise ValueError(
            "alignment canonical action authority/source identity mismatch"
        )

    raw_preview_claim = alignment_preview.get("preview_claim")
    if not isinstance(raw_preview_claim, dict):
        raise ValueError(
            "alignment canonical action preview claim missing"
        )

    try:
        preview_claim = type(source_claim).model_validate(
            raw_preview_claim
        )
    except Exception as exc:
        raise ValueError(
            "alignment canonical action preview claim invalid"
        ) from exc

    preview_payload = preview_claim.model_dump(mode="json")
    source_changed_fields = sorted(
        key
        for key in set(source_before) | set(preview_payload)
        if source_before.get(key) != preview_payload.get(key)
    )
    if source_changed_fields != ["text"]:
        raise ValueError(
            "alignment canonical action source preview is not text-only"
        )

    before_binding = dict(
        source_before.get("semantic_fidelity_binding") or {}
    )
    preview_binding = dict(
        preview_payload.get("semantic_fidelity_binding") or {}
    )
    if before_binding != preview_binding:
        raise ValueError(
            "alignment canonical action preview binding changed"
        )

    if preview_claim.text != exact_source_text:
        raise ValueError(
            "alignment canonical action preview text is not exact source"
        )
    if preview_claim.required_bridge != canonical_before.required_bridge:
        raise ValueError(
            "alignment canonical action preview bridge mismatch"
        )
    if (
        preview_claim.predicted_observation
        != canonical_before.predicted_observation
    ):
        raise ValueError(
            "alignment canonical action preview prediction mismatch"
        )
    if (
        preview_claim.falsification_condition
        != canonical_before.falsification_condition
    ):
        raise ValueError(
            "alignment canonical action preview falsifier mismatch"
        )

    audit = dict(alignment_preview.get("mutation_audit") or {})
    if (
        list(audit.get("allowed_changed_fields") or []) != ["text"]
        or list(audit.get("allowed_binding_changed_fields") or [])
        != []
        or list(audit.get("changed_fields") or []) != ["text"]
        or list(audit.get("unexpected_changed_fields") or []) != []
        or list(audit.get("changed_binding_fields") or []) != []
        or list(
            audit.get("unexpected_binding_changed_fields")
            or []
        )
        != []
        or audit.get("all_other_fields_preserved") is not True
    ):
        raise ValueError(
            "alignment canonical action mutation audit invalid"
        )

    semantic_after = alignment_preview.get(
        "semantic_fidelity_shadow"
    )
    if (
        not isinstance(semantic_after, dict)
        or list(semantic_after.get("reason_codes") or [])
    ):
        raise ValueError(
            "alignment canonical action preview semantic fidelity not clean"
        )

    taxonomy_after = alignment_preview.get(
        "semantic_fidelity_taxonomy_shadow"
    )
    if not isinstance(taxonomy_after, dict):
        raise ValueError(
            "alignment canonical action preview taxonomy missing"
        )
    if not (
        taxonomy_after.get("binding_contract", {}).get("status")
        == "VALID"
        and taxonomy_after.get("claim_fidelity", {}).get("status")
        == "NO_FLAG"
        and taxonomy_after.get(
            "specification_completeness", {}
        ).get("status")
        == "NO_FLAG"
        and taxonomy_after.get(
            "specification_self_containment", {}
        ).get("status")
        == "NO_FLAG"
        and taxonomy_after.get("overall_review_status")
        == "PASS_SHADOW"
    ):
        raise ValueError(
            "alignment canonical action preview taxonomy not clean"
        )

    canonical_after_payload = json.loads(
        json.dumps(canonical_before_payload)
    )
    canonical_after_payload["text"] = exact_source_text

    try:
        canonical_after = NoveltyClaim.model_validate(
            canonical_after_payload
        )
    except Exception as exc:
        raise ValueError(
            "alignment canonical action canonical claim invalid"
        ) from exc

    canonical_after_dump = canonical_after.model_dump(mode="json")
    canonical_changed_fields = sorted(
        key
        for key in set(canonical_before_payload)
        | set(canonical_after_dump)
        if canonical_before_payload.get(key)
        != canonical_after_dump.get(key)
    )
    if canonical_changed_fields != ["text"]:
        raise ValueError(
            "alignment canonical action canonical mutation is not text-only"
        )

    after_container_payload = json.loads(
        json.dumps(container_before)
    )
    replaced = 0
    for index, row in enumerate(
        after_container_payload["claims"]
    ):
        if row["claim_id"] == claim_id:
            after_container_payload["claims"][index] = (
                canonical_after_dump
            )
            replaced += 1

    if replaced != 1:
        raise ValueError(
            "alignment canonical action target replacement count invalid"
        )

    try:
        result = HypothesisNoveltyClaims.model_validate(
            after_container_payload
        )
    except Exception as exc:
        raise ValueError(
            "alignment canonical action output container invalid"
        ) from exc

    result_dump = result.model_dump(mode="json")
    before_claims = {
        row["claim_id"]: row
        for row in container_before["claims"]
    }
    after_claims = {
        row["claim_id"]: row
        for row in result_dump["claims"]
    }

    if set(before_claims) != set(after_claims):
        raise ValueError(
            "alignment canonical action claim identity set changed"
        )

    for current_claim_id, before_row in before_claims.items():
        after_row = after_claims[current_claim_id]
        changed = sorted(
            key
            for key in set(before_row) | set(after_row)
            if before_row.get(key) != after_row.get(key)
        )
        expected = ["text"] if current_claim_id == claim_id else []
        if changed != expected:
            raise ValueError(
                "alignment canonical action unexpected container mutation"
            )

    # Input objects must remain unchanged.
    if source_claim.model_dump(mode="json") != source_before:
        raise ValueError(
            "alignment canonical action mutated source claim"
        )
    if decomposition.model_dump(mode="json") != container_before:
        raise ValueError(
            "alignment canonical action mutated input decomposition"
        )

    action_record: dict[str, object] = {
        "schema_version": (
            "novelty-claim-existing-bridge-scope-alignment-"
            "canonical-action-opt-in-v1"
        ),
        "action_mode": "EXPLICIT_OPT_IN_ONLY",
        "explicit_opt_in_required": True,
        "automatic_runtime_enabled": False,
        "canonical_action_performed": True,
        "source_recovery_path": (
            "EXISTING_BRIDGE_SCOPE_ALIGNMENT"
        ),
        "exact_source_canonical_action_enabled": False,
        "scope_wrapper_canonical_action_enabled": False,
        "scientific_truth_assessed": False,
        "novelty_assessed": False,
        "semantic_synonymy_allowed": False,
        "hypothesis_id": decomposition.hypothesis_id,
        "claim_id": claim_id,
        "claim_local_id": local_id,
        "claim_rank": canonical_before.claim_rank,
        "novelty_selection_role": (
            canonical_before.novelty_selection_role
        ),
        "shadow_authority_status": (
            alignment_authority.get("authority_status")
        ),
        "shadow_authority_reason_codes": list(
            alignment_authority.get(
                "authority_reason_codes"
            )
            or []
        ),
        "before_text": canonical_before.text,
        "after_text": canonical_after.text,
        "canonical_changed_fields": (
            canonical_changed_fields
        ),
        "all_non_text_canonical_fields_preserved": True,
        "claim_count_preserved": (
            len(result.claims) == len(decomposition.claims)
        ),
        "input_objects_mutated": False,
    }

    return result, action_record


def _plan_exact_source_atomic_recompile_shadow(
    hypothesis: HypothesisCard,
    claim: object,
    *,
    sanitized_required_bridge: str,
) -> dict[str, object]:
    """Plan, but never perform, bounded exact-source recompilation.

    A candidate must be one contiguous proposition from inferential_bridge or
    assumptions, contain every model-declared relation/scope/direction facet
    literally after lexical normalization, and independently pass the current
    required_bridge sanitizer. The planner never chooses between multiple
    candidates and never mutates the atomic claim.
    """

    binding = claim.semantic_fidelity_binding
    anchors = [
        " ".join(str(value or "").split())
        for value in binding.relation_endpoint_anchors
        if " ".join(str(value or "").split())
    ]
    scope_qualifiers = [
        " ".join(str(value or "").split())
        for value in binding.scope_qualifier_spans
        if " ".join(str(value or "").split())
    ]
    directional_qualifiers = [
        " ".join(str(value or "").split())
        for value in binding.directional_qualifier_spans
        if " ".join(str(value or "").split())
    ]

    sources: list[tuple[str, str]] = [
        ("inferential_bridge", hypothesis.inferential_bridge),
    ]
    sources.extend(
        (f"assumptions[{index}]", value)
        for index, value in enumerate(hypothesis.assumptions)
    )

    candidates: list[dict[str, object]] = []
    seen: set[str] = set()

    if not sanitized_required_bridge and anchors:
        for parent_path, source_text in sources:
            for unit_index, unit in enumerate(
                _exact_source_recompile_proposition_units(source_text)
            ):
                anchor_hits = {
                    value: _literal_facet_present(unit, value)
                    for value in anchors
                }
                scope_hits = {
                    value: _literal_facet_present(unit, value)
                    for value in scope_qualifiers
                }
                direction_hits = {
                    value: _literal_facet_present(unit, value)
                    for value in directional_qualifiers
                }

                if not all(anchor_hits.values()):
                    continue
                if not all(scope_hits.values()):
                    continue
                if not all(direction_hits.values()):
                    continue

                sanitized_candidate = _clean_branch_specific_bridge(
                    unit,
                    list(claim.prior_art_identity_terms),
                    [unit],
                )
                if not sanitized_candidate:
                    continue

                key = _clean_query(
                    sanitized_candidate,
                    limit=8000,
                ).casefold()
                if key in seen:
                    continue
                seen.add(key)

                candidates.append(
                    {
                        "source_path": (
                            f"{parent_path}.unit[{unit_index}]"
                        ),
                        "parent_path": parent_path,
                        "exact_source_text": unit,
                        "anchor_hits": anchor_hits,
                        "scope_hits": scope_hits,
                        "direction_hits": direction_hits,
                        "sanitized_candidate": sanitized_candidate,
                    }
                )

    if sanitized_required_bridge:
        classification = "NO_RECOMPILE_NEEDED"
    elif not anchors:
        classification = "BINDING_UNUSABLE_NO_RELATION_ENDPOINTS"
    elif len(candidates) == 1:
        classification = "EXACT_SOURCE_RECOMPILE_CANDIDATE"
    elif len(candidates) > 1:
        classification = "AMBIGUOUS_EXACT_SOURCE_CANDIDATES"
    else:
        classification = "NO_EXACT_SOURCE_RECOMPILE_CANDIDATE"

    return {
        "schema_version": (
            "novelty-claim-exact-source-recompile-shadow-v1"
        ),
        "diagnostic_only": True,
        "production_authority": False,
        "recompile_performed": False,
        "claim_local_id": claim.local_id,
        "classification": classification,
        "relation_endpoint_anchors": anchors,
        "scope_qualifier_spans": scope_qualifiers,
        "directional_qualifier_spans": directional_qualifiers,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def _preview_exact_source_atomic_recompile_shadow(
    hypothesis: HypothesisCard,
    claim: object,
    *,
    recompile_plan: dict[str, object],
) -> dict[str, object]:
    """Compile a unique exact-source candidate into a diagnostic preview.

    This function never mutates the supplied claim and never returns production
    authority. It applies only the bounded mutation demonstrated in S14-J/M:
    claim.text and required_bridge become the unique exact source proposition,
    while semantic_fidelity_binding.proposition_basis is rebound to the same
    proposition. All other draft fields must remain unchanged. The resulting
    preview is then re-sanitized and re-evaluated by the existing fidelity
    shadow/taxonomy.
    """

    base = {
        "schema_version": (
            "novelty-claim-exact-source-recompile-preview-shadow-v1"
        ),
        "diagnostic_only": True,
        "production_authority": False,
        "production_recompile_enabled": False,
        "recompile_performed": False,
        "claim_local_id": claim.local_id,
        "planner_classification": recompile_plan.get("classification"),
        "preview_status": "NOT_ELIGIBLE",
        "ready_for_bounded_recompile": False,
        "candidate_source_path": None,
        "exact_source_text": "",
        "preview_claim": None,
        "sanitized_required_bridge": "",
        "sanitized_predicted_observation": "",
        "sanitized_falsification_condition": "",
        "semantic_fidelity_shadow": None,
        "semantic_fidelity_taxonomy_shadow": None,
        "mutation_audit": {
            "allowed_changed_fields": [
                "required_bridge",
                "semantic_fidelity_binding",
                "text",
            ],
            "changed_fields": [],
            "unexpected_changed_fields": [],
            "all_other_fields_preserved": True,
        },
        "preview_reason_codes": [],
    }

    if (
        recompile_plan.get("classification")
        != "EXACT_SOURCE_RECOMPILE_CANDIDATE"
        or int(recompile_plan.get("candidate_count") or 0) != 1
    ):
        return base

    candidates = list(recompile_plan.get("candidates") or [])
    if len(candidates) != 1:
        base["preview_status"] = "REVIEW_REQUIRED_PREVIEW"
        base["preview_reason_codes"] = [
            "recompile_preview_candidate_cardinality_unexpected"
        ]
        return base

    candidate = candidates[0]
    exact_source = " ".join(
        str(candidate.get("exact_source_text") or "").split()
    ).strip()
    source_path = str(candidate.get("source_path") or "").strip()

    base["candidate_source_path"] = source_path or None
    base["exact_source_text"] = exact_source

    if not exact_source:
        base["preview_status"] = "REVIEW_REQUIRED_PREVIEW"
        base["preview_reason_codes"] = [
            "recompile_preview_exact_source_empty"
        ]
        return base

    before_payload = claim.model_dump(mode="json")
    after_payload = dict(before_payload)
    after_payload["text"] = exact_source
    after_payload["required_bridge"] = exact_source

    binding_payload = dict(
        before_payload.get("semantic_fidelity_binding") or {}
    )
    binding_payload["proposition_basis"] = exact_source
    after_payload["semantic_fidelity_binding"] = binding_payload

    preview_claim = type(claim).model_validate(after_payload)
    preview_payload = preview_claim.model_dump(mode="json")

    allowed_changed_fields = {
        "text",
        "required_bridge",
        "semantic_fidelity_binding",
    }
    changed_fields = sorted(
        key
        for key in set(before_payload) | set(preview_payload)
        if before_payload.get(key) != preview_payload.get(key)
    )
    unexpected_changed_fields = [
        key
        for key in changed_fields
        if key not in allowed_changed_fields
    ]

    bridge_source_texts = [
        hypothesis.inferential_bridge,
        *hypothesis.assumptions,
    ]
    sanitized_required_bridge = _clean_branch_specific_bridge(
        preview_claim.required_bridge,
        list(preview_claim.prior_art_identity_terms),
        bridge_source_texts,
    )
    sanitized_predicted_observation = (
        _clean_branch_specific_specification(
            preview_claim.predicted_observation,
            list(preview_claim.prior_art_identity_terms),
        )
    )
    sanitized_falsification_condition = (
        _clean_branch_specific_specification(
            preview_claim.falsification_condition,
            list(preview_claim.prior_art_identity_terms),
        )
    )

    semantic_fidelity_shadow = assess_atomic_semantic_fidelity(
        hypothesis,
        preview_claim,
    )
    taxonomy = compile_atomic_semantic_fidelity_taxonomy_shadow(
        preview_claim,
        semantic_fidelity_shadow=semantic_fidelity_shadow,
        sanitized_required_bridge=sanitized_required_bridge,
        sanitized_predicted_observation=sanitized_predicted_observation,
        sanitized_falsification_condition=sanitized_falsification_condition,
    )

    preview_reason_codes: list[str] = []
    if sanitized_required_bridge != exact_source:
        preview_reason_codes.append(
            "recompile_preview_required_bridge_not_preserved"
        )
    if unexpected_changed_fields:
        preview_reason_codes.append(
            "recompile_preview_unexpected_field_mutation"
        )
    if semantic_fidelity_shadow.get("reason_codes"):
        preview_reason_codes.append(
            "recompile_preview_semantic_fidelity_review"
        )
    if taxonomy["binding_contract"]["status"] != "VALID":
        preview_reason_codes.append(
            "recompile_preview_binding_contract_not_valid"
        )
    if taxonomy["claim_fidelity"]["status"] != "NO_FLAG":
        preview_reason_codes.append(
            "recompile_preview_claim_fidelity_not_clean"
        )
    if taxonomy["specification_completeness"]["status"] != "NO_FLAG":
        preview_reason_codes.append(
            "recompile_preview_specification_incomplete"
        )
    if taxonomy["specification_self_containment"]["status"] != "NO_FLAG":
        preview_reason_codes.append(
            "recompile_preview_self_containment_not_clean"
        )
    if taxonomy["overall_review_status"] != "PASS_SHADOW":
        preview_reason_codes.append(
            "recompile_preview_overall_not_pass_shadow"
        )

    ready = not preview_reason_codes

    base.update(
        {
            "preview_status": (
                "PASS_SHADOW_PREVIEW"
                if ready
                else "REVIEW_REQUIRED_PREVIEW"
            ),
            "ready_for_bounded_recompile": ready,
            "preview_claim": preview_payload,
            "sanitized_required_bridge": sanitized_required_bridge,
            "sanitized_predicted_observation": (
                sanitized_predicted_observation
            ),
            "sanitized_falsification_condition": (
                sanitized_falsification_condition
            ),
            "semantic_fidelity_shadow": semantic_fidelity_shadow,
            "semantic_fidelity_taxonomy_shadow": taxonomy,
            "mutation_audit": {
                "allowed_changed_fields": sorted(allowed_changed_fields),
                "changed_fields": changed_fields,
                "unexpected_changed_fields": unexpected_changed_fields,
                "all_other_fields_preserved": (
                    not unexpected_changed_fields
                ),
            },
            "preview_reason_codes": preview_reason_codes,
        }
    )
    return base


def _authorize_exact_source_atomic_recompile_shadow(
    claim: object,
    *,
    recompile_plan: dict[str, object],
    recompile_preview: dict[str, object],
) -> dict[str, object]:
    """Validate the bounded mutation boundary without granting authority.

    This gate intentionally re-checks the planner and preview rather than
    trusting ``ready_for_bounded_recompile``. It verifies source identity,
    mutation scope (including nested semantic-fidelity binding fields),
    sanitizer preservation, and the post-preview fidelity taxonomy. The
    result is diagnostic only; production recompilation remains disabled.
    """

    reason_codes: list[str] = []

    candidates = list(recompile_plan.get("candidates") or [])
    if (
        recompile_plan.get("classification")
        != "EXACT_SOURCE_RECOMPILE_CANDIDATE"
        or int(recompile_plan.get("candidate_count") or 0) != 1
        or len(candidates) != 1
    ):
        reason_codes.append(
            "recompile_authority_plan_not_unique_candidate"
        )

    candidate = candidates[0] if len(candidates) == 1 else {}
    candidate_source_path = str(
        candidate.get("source_path") or ""
    ).strip()
    candidate_exact_source = " ".join(
        str(candidate.get("exact_source_text") or "").split()
    ).strip()
    candidate_sanitized = " ".join(
        str(candidate.get("sanitized_candidate") or "").split()
    ).strip()

    if (
        not candidate_source_path
        or not (
            candidate_source_path.startswith("inferential_bridge.unit[")
            or candidate_source_path.startswith("assumptions[")
        )
        or not candidate_exact_source
        or candidate_sanitized != candidate_exact_source
    ):
        reason_codes.append(
            "recompile_authority_candidate_source_invalid"
        )

    for facet_key in ("anchor_hits", "scope_hits", "direction_hits"):
        hits = dict(candidate.get(facet_key) or {})
        if hits and not all(bool(value) for value in hits.values()):
            reason_codes.append(
                "recompile_authority_candidate_facet_guard_failed"
            )
            break

    if (
        recompile_preview.get("preview_status")
        != "PASS_SHADOW_PREVIEW"
        or recompile_preview.get("ready_for_bounded_recompile") is not True
    ):
        reason_codes.append(
            "recompile_authority_preview_not_ready"
        )

    if (
        recompile_preview.get("diagnostic_only") is not True
        or recompile_preview.get("production_authority") is not False
        or recompile_preview.get("production_recompile_enabled") is not False
        or recompile_preview.get("recompile_performed") is not False
    ):
        reason_codes.append(
            "recompile_authority_preview_metadata_guard_failed"
        )

    preview_source_path = str(
        recompile_preview.get("candidate_source_path") or ""
    ).strip()
    preview_exact_source = " ".join(
        str(recompile_preview.get("exact_source_text") or "").split()
    ).strip()
    if (
        preview_source_path != candidate_source_path
        or preview_exact_source != candidate_exact_source
    ):
        reason_codes.append(
            "recompile_authority_source_identity_mismatch"
        )

    if list(recompile_preview.get("preview_reason_codes") or []):
        reason_codes.append(
            "recompile_authority_preview_reason_codes_present"
        )

    before_payload = claim.model_dump(mode="json")
    preview_payload_raw = recompile_preview.get("preview_claim")
    preview_payload = (
        dict(preview_payload_raw)
        if isinstance(preview_payload_raw, dict)
        else None
    )

    changed_fields: list[str] = []
    unexpected_changed_fields: list[str] = []
    nested_binding_changed_fields: list[str] = []

    if preview_payload is None:
        reason_codes.append(
            "recompile_authority_preview_claim_missing"
        )
    else:
        try:
            validated_preview = type(claim).model_validate(preview_payload)
            preview_payload = validated_preview.model_dump(mode="json")
        except Exception:
            reason_codes.append(
                "recompile_authority_preview_claim_invalid"
            )
            preview_payload = None

    if preview_payload is not None:
        allowed_changed_fields = {
            "text",
            "required_bridge",
            "semantic_fidelity_binding",
        }
        changed_fields = sorted(
            key
            for key in set(before_payload) | set(preview_payload)
            if before_payload.get(key) != preview_payload.get(key)
        )
        unexpected_changed_fields = [
            key
            for key in changed_fields
            if key not in allowed_changed_fields
        ]
        if unexpected_changed_fields:
            reason_codes.append(
                "recompile_authority_unexpected_field_mutation"
            )

        before_binding = dict(
            before_payload.get("semantic_fidelity_binding") or {}
        )
        preview_binding = dict(
            preview_payload.get("semantic_fidelity_binding") or {}
        )
        nested_binding_changed_fields = sorted(
            key
            for key in set(before_binding) | set(preview_binding)
            if before_binding.get(key) != preview_binding.get(key)
        )
        if any(
            key != "proposition_basis"
            for key in nested_binding_changed_fields
        ):
            reason_codes.append(
                "recompile_authority_binding_mutation_out_of_bounds"
            )

        if (
            " ".join(str(preview_payload.get("text") or "").split()).strip()
            != candidate_exact_source
            or " ".join(
                str(preview_payload.get("required_bridge") or "").split()
            ).strip()
            != candidate_exact_source
            or " ".join(
                str(preview_binding.get("proposition_basis") or "").split()
            ).strip()
            != candidate_exact_source
        ):
            reason_codes.append(
                "recompile_authority_preview_payload_not_exact_source"
            )

        sanitized_bridge = " ".join(
            str(
                recompile_preview.get("sanitized_required_bridge")
                or ""
            ).split()
        ).strip()
        if sanitized_bridge != candidate_exact_source:
            reason_codes.append(
                "recompile_authority_required_bridge_not_preserved"
            )

        sanitized_prediction = " ".join(
            str(
                recompile_preview.get("sanitized_predicted_observation")
                or ""
            ).split()
        ).strip()
        preview_prediction = " ".join(
            str(preview_payload.get("predicted_observation") or "").split()
        ).strip()
        if sanitized_prediction != preview_prediction:
            reason_codes.append(
                "recompile_authority_prediction_not_preserved"
            )

        sanitized_falsifier = " ".join(
            str(
                recompile_preview.get("sanitized_falsification_condition")
                or ""
            ).split()
        ).strip()
        preview_falsifier = " ".join(
            str(
                preview_payload.get("falsification_condition") or ""
            ).split()
        ).strip()
        if sanitized_falsifier != preview_falsifier:
            reason_codes.append(
                "recompile_authority_falsifier_not_preserved"
            )

    audit = dict(recompile_preview.get("mutation_audit") or {})
    if (
        list(audit.get("unexpected_changed_fields") or [])
        or audit.get("all_other_fields_preserved") is not True
        or sorted(audit.get("changed_fields") or []) != changed_fields
    ):
        reason_codes.append(
            "recompile_authority_mutation_audit_mismatch"
        )

    semantic_shadow = recompile_preview.get("semantic_fidelity_shadow")
    if (
        not isinstance(semantic_shadow, dict)
        or list(semantic_shadow.get("reason_codes") or [])
    ):
        reason_codes.append(
            "recompile_authority_semantic_fidelity_not_clean"
        )

    taxonomy = recompile_preview.get(
        "semantic_fidelity_taxonomy_shadow"
    )
    if not isinstance(taxonomy, dict):
        reason_codes.append(
            "recompile_authority_taxonomy_missing"
        )
    else:
        taxonomy_clean = (
            taxonomy.get("binding_contract", {}).get("status") == "VALID"
            and taxonomy.get("claim_fidelity", {}).get("status") == "NO_FLAG"
            and taxonomy.get("specification_completeness", {}).get("status")
            == "NO_FLAG"
            and taxonomy.get("specification_self_containment", {}).get("status")
            == "NO_FLAG"
            and taxonomy.get("overall_review_status") == "PASS_SHADOW"
        )
        if not taxonomy_clean:
            reason_codes.append(
                "recompile_authority_taxonomy_not_clean"
            )

    # Stable de-duplication preserves first-failure order for diagnostics.
    unique_reason_codes = list(dict.fromkeys(reason_codes))
    contract_satisfied = not unique_reason_codes

    return {
        "schema_version": (
            "novelty-claim-exact-source-recompile-authority-shadow-v1"
        ),
        "diagnostic_only": True,
        "production_authority": False,
        "production_recompile_enabled": False,
        "recompile_performed": False,
        "claim_local_id": claim.local_id,
        "authority_status": (
            "AUTHORIZED_SHADOW"
            if contract_satisfied
            else "DENIED_SHADOW"
        ),
        "bounded_recompile_contract_satisfied": contract_satisfied,
        "candidate_source_path": candidate_source_path or None,
        "exact_source_text": candidate_exact_source,
        "changed_fields": changed_fields,
        "unexpected_changed_fields": unexpected_changed_fields,
        "nested_binding_changed_fields": nested_binding_changed_fields,
        "authority_reason_codes": unique_reason_codes,
    }


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

    # A required bridge must be self-contained as a standalone atomic
    # specification. Exact source provenance does not make demonstrative
    # references such as "this relationship" safe for downstream use.
    if has_anaphoric_relation_reference(
        cleaned
    ):
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
            if has_anaphoric_relation_reference(
                raw_bridge
            ):
                codes.append(
                    "required_bridge_rejected_"
                    "anaphoric_relation_reference"
                )

            elif not identities:
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
        enable_existing_bridge_scope_alignment_canonical_action: bool = False,
    ) -> None:
        self.backend = backend
        self.max_claims = int(max_claims_per_hypothesis)
        self.max_queries = int(max_queries_per_claim)
        self.enable_existing_bridge_scope_alignment_canonical_action = (
            bool(
                enable_existing_bridge_scope_alignment_canonical_action
            )
        )
        self.canonical_action_records: list[dict[str, object]] = []

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
        self.canonical_action_records = []
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

            specification_source_trace = trace_specification_sources(
                hypothesis,
                {
                    "required_bridge": raw_required_bridge,
                    "predicted_observation": row.predicted_observation,
                    "falsification_condition": row.falsification_condition,
                },
                {
                    "required_bridge": sanitized_required_bridge,
                    "predicted_observation": sanitized_predicted_observation,
                    "falsification_condition": sanitized_falsification_condition,
                },
            )

            semantic_fidelity_shadow = (
                assess_atomic_semantic_fidelity(
                    hypothesis,
                    row,
                )
            )

            ordered_marker_source_channel_shadow = (
                compile_ordered_marker_source_channel_shadow(
                    hypothesis,
                    row,
                    semantic_fidelity_shadow=(
                        semantic_fidelity_shadow
                    ),
                )
            )

            epistemic_modality_fidelity_shadow = (
                compile_epistemic_modality_fidelity_shadow(
                    row,
                    semantic_fidelity_shadow=(
                        semantic_fidelity_shadow
                    ),
                    sanitized_required_bridge=(
                        sanitized_required_bridge
                    ),
                )
            )

            modal_preserving_claim_text_repair_plan_shadow = (
                plan_modal_preserving_claim_text_repair_shadow(
                    row,
                    epistemic_modality_fidelity_shadow=(
                        epistemic_modality_fidelity_shadow
                    ),
                )
            )

            modal_preserving_claim_text_repair_preview_shadow = (
                preview_modal_preserving_claim_text_repair_shadow(
                    row,
                    semantic_fidelity_shadow=(
                        semantic_fidelity_shadow
                    ),
                    sanitized_required_bridge=(
                        sanitized_required_bridge
                    ),
                    repair_plan_shadow=(
                        modal_preserving_claim_text_repair_plan_shadow
                    ),
                )
            )

            semantic_fidelity_taxonomy_shadow = (
                compile_atomic_semantic_fidelity_taxonomy_shadow(
                    row,
                    semantic_fidelity_shadow=(
                        semantic_fidelity_shadow
                    ),
                    sanitized_required_bridge=(
                        sanitized_required_bridge
                    ),
                    sanitized_predicted_observation=(
                        sanitized_predicted_observation
                    ),
                    sanitized_falsification_condition=(
                        sanitized_falsification_condition
                    ),
                )
            )

            exact_source_recompile_shadow = (
                _plan_exact_source_atomic_recompile_shadow(
                    hypothesis,
                    row,
                    sanitized_required_bridge=(
                        sanitized_required_bridge
                    ),
                )
            )

            existing_bridge_scope_alignment_shadow = (
                _plan_existing_bridge_scope_alignment_shadow(
                    row,
                    sanitized_required_bridge=(
                        sanitized_required_bridge
                    ),
                    exact_source_plan=(
                        exact_source_recompile_shadow
                    ),
                    semantic_fidelity_shadow=(
                        semantic_fidelity_shadow
                    ),
                    specification_source_trace=(
                        specification_source_trace
                    ),
                )
            )

            existing_bridge_scope_alignment_preview_shadow = (
                _preview_existing_bridge_scope_alignment_shadow(
                    hypothesis,
                    row,
                    alignment_plan=(
                        existing_bridge_scope_alignment_shadow
                    ),
                )
            )

            existing_bridge_scope_alignment_authority_shadow = (
                _authorize_existing_bridge_scope_alignment_shadow(
                    row,
                    alignment_plan=(
                        existing_bridge_scope_alignment_shadow
                    ),
                    alignment_preview=(
                        existing_bridge_scope_alignment_preview_shadow
                    ),
                )
            )

            scope_wrapper_recompile_shadow = (
                _plan_scope_wrapper_exact_source_atomic_recompile_shadow(
                    hypothesis,
                    row,
                    sanitized_required_bridge=(
                        sanitized_required_bridge
                    ),
                    exact_source_plan=(
                        exact_source_recompile_shadow
                    ),
                )
            )

            scope_wrapper_recompile_preview_shadow = (
                _preview_scope_wrapper_exact_source_atomic_recompile_shadow(
                    hypothesis,
                    row,
                    recompile_plan=(
                        scope_wrapper_recompile_shadow
                    ),
                )
            )

            scope_wrapper_recompile_authority_shadow = (
                _authorize_scope_wrapper_exact_source_atomic_recompile_shadow(
                    row,
                    recompile_plan=(
                        scope_wrapper_recompile_shadow
                    ),
                    recompile_preview=(
                        scope_wrapper_recompile_preview_shadow
                    ),
                )
            )

            exact_source_recompile_preview_shadow = (
                _preview_exact_source_atomic_recompile_shadow(
                    hypothesis,
                    row,
                    recompile_plan=(
                        exact_source_recompile_shadow
                    ),
                )
            )

            exact_source_recompile_authority_shadow = (
                _authorize_exact_source_atomic_recompile_shadow(
                    row,
                    recompile_plan=(
                        exact_source_recompile_shadow
                    ),
                    recompile_preview=(
                        exact_source_recompile_preview_shadow
                    ),
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
                    "raw_prior_art_identity_terms": list(
                        row.prior_art_identity_terms
                    ),
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
                    "semantic_fidelity_shadow": (
                        semantic_fidelity_shadow
                    ),
                    "ordered_marker_source_channel_shadow": (
                        ordered_marker_source_channel_shadow
                    ),
                    "epistemic_modality_fidelity_shadow": (
                        epistemic_modality_fidelity_shadow
                    ),
                    "modal_preserving_claim_text_repair_plan_shadow": (
                        modal_preserving_claim_text_repair_plan_shadow
                    ),
                    "modal_preserving_claim_text_repair_preview_shadow": (
                        modal_preserving_claim_text_repair_preview_shadow
                    ),
                    "semantic_fidelity_taxonomy_shadow": (
                        semantic_fidelity_taxonomy_shadow
                    ),
                    "exact_source_recompile_shadow": (
                        exact_source_recompile_shadow
                    ),
                    "existing_bridge_scope_alignment_shadow": (
                        existing_bridge_scope_alignment_shadow
                    ),
                    "existing_bridge_scope_alignment_preview_shadow": (
                        existing_bridge_scope_alignment_preview_shadow
                    ),
                    "existing_bridge_scope_alignment_authority_shadow": (
                        existing_bridge_scope_alignment_authority_shadow
                    ),
                    "scope_wrapper_exact_source_recompile_shadow": (
                        scope_wrapper_recompile_shadow
                    ),
                    "scope_wrapper_exact_source_recompile_preview_shadow": (
                        scope_wrapper_recompile_preview_shadow
                    ),
                    "scope_wrapper_exact_source_recompile_authority_shadow": (
                        scope_wrapper_recompile_authority_shadow
                    ),
                    "exact_source_recompile_preview_shadow": (
                        exact_source_recompile_preview_shadow
                    ),
                    "exact_source_recompile_authority_shadow": (
                        exact_source_recompile_authority_shadow
                    ),
                    "source_trace": specification_source_trace,
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
        result = HypothesisNoveltyClaims(
            hypothesis_id=hypothesis.hypothesis_id,
            title=hypothesis.title,
            claims=rows,
            decomposition_notes=draft.decomposition_notes,
        )

        if not (
            self.enable_existing_bridge_scope_alignment_canonical_action
        ):
            return result

        bounded_noneligibility_messages = {
            "alignment canonical action requires stable existing exact bridge",
            "alignment canonical action planner did not produce unique candidate",
            "alignment canonical action preview is not ready",
            "alignment canonical action shadow authority denied",
        }

        for source_row in draft_rows:
            claim_id = claim_id_by_local_id[source_row.local_id]
            try:
                result, action_record = (
                    _apply_existing_bridge_scope_alignment_canonical_action_opt_in(
                        hypothesis,
                        result,
                        source_row,
                        claim_id=claim_id,
                    )
                )
            except ValueError as exc:
                if str(exc) in bounded_noneligibility_messages:
                    continue
                raise

            self.canonical_action_records.append(action_record)

        return result


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
