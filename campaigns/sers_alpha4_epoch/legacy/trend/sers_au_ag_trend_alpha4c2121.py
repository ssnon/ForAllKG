from __future__ import annotations

import re
from dataclasses import replace
from typing import Any, Mapping

import networkx as nx

from campaigns.sers_alpha4_epoch.legacy.trend.sers_au_ag_trend_alpha4c212 import (
    SERS_AU_AG_TREND_ADAPTER as _ALPHA4C212_ADAPTER,
)
from dac_her.trend_domain import (
    TrendDomainAdapter,
    TrendEvidence,
    TrendEvidenceSource,
)
from dac_her.trend_evidence import stable_trend_id


SERS_AU_AG_TREND_SEMANTICS_ID = "sers_au_ag_trend_v5_alpha4c2121"

_CLAIM_TEXT_FIELDS = (
    "statement",
    "description",
    "label",
    "source_expression",
    "node_text",
)


def _match_norm(value: object) -> str:
    """
    Normalization used only for semantic cue matching.

    Hyphenated canonical phrases such as
    "surface-plasmon-resonance" must match the same concept as
    "surface plasmon resonance". This does not rewrite stored provenance.
    """
    text = str(value or "").casefold()
    text = (
        text.replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("‐", "-")
        .replace("‑", "-")
    )
    text = re.sub(r"[-_/]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _canonical_claim_text_bundle(
    item: TrendEvidence,
    graph: nx.Graph,
) -> str:
    """
    Build a matching-only text bundle from the evidence plus its grounded
    canonical Claim node(s).

    The evidence source_expression is preserved unchanged in the returned
    TrendEvidence; this bundle exists only for semantic re-grounding.
    """
    values: list[str] = []

    if str(item.source_expression or "").strip():
        values.append(str(item.source_expression).strip())

    for value in item.source_expressions:
        value = str(value).strip()
        if value:
            values.append(value)

    for claim_id in item.source_claim_ids:
        if claim_id not in graph:
            continue
        attrs: Mapping[str, Any] = graph.nodes[claim_id]
        for field in _CLAIM_TEXT_FIELDS:
            value = str(attrs.get(field, "")).strip()
            if value:
                values.append(value)

    # Deterministic stable dedupe.
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    return "\n".join(deduped)


def _is_spectral_alignment_claim(text: str) -> bool:
    normalized = _match_norm(text)
    has_resonance = bool(re.search(
        r"\b(?:spr|lspr|surface\s+plasmon\s+resonance|"
        r"localized\s+surface\s+plasmon\s+resonance|"
        r"plasmon\s+resonance)\b",
        normalized,
    ))
    has_excitation = bool(re.search(
        r"\b(?:excitation|exciting|laser)\b",
        normalized,
    ))
    has_alignment = bool(re.search(
        r"\b(?:closer|closest|close\s+to|matching|matched|match|"
        r"overlap|overlapping|aligned|alignment|detuning|"
        r"spectral\s+proximity|proximity)\b",
        normalized,
    ))
    return has_resonance and has_excitation and has_alignment


def _formal_ef_language(text: str) -> bool:
    normalized = _match_norm(text)
    return bool(re.search(
        r"\b(?:sers\s+)?enhancement\s+factor\b|\bsers\s+ef\b|"
        r"\bef\s+(?:value|coefficient)\b",
        normalized,
    ))


def _spectral_detuning_direction(
    text: str,
) -> tuple[str, str] | None:
    """
    Canonical direction is defined with respect to INCREASING detuning.

    Source relation:
        closer SPR/excitation match -> higher enhancement

    Canonical relation:
        SPR-excitation detuning increases -> enhancement decreases
    """
    normalized = _match_norm(text)

    closer_higher = bool(
        re.search(
            r"\b(?:closer|closest|better\s+matched|"
            r"more\s+closely\s+matched|greater\s+overlap|"
            r"better\s+overlap|smaller\s+(?:spectral\s+)?detuning|"
            r"lower\s+(?:spectral\s+)?detuning)\b"
            r".{0,180}\b(?:higher|stronger|increase\w*|enhanc\w*)\b",
            normalized,
        )
        or re.search(
            r"\b(?:higher|stronger|increase\w*|enhanc\w*)\b"
            r".{0,180}\b(?:closer|closest|better\s+matched|"
            r"more\s+closely\s+matched|greater\s+overlap|"
            r"better\s+overlap|smaller\s+(?:spectral\s+)?detuning|"
            r"lower\s+(?:spectral\s+)?detuning)\b",
            normalized,
        )
    )
    if closer_higher:
        return "negative", "monotonic"

    return None


def _reground_item(
    item: TrendEvidence,
    graph: nx.Graph,
) -> TrendEvidence:
    """
    Re-ground an already-emitted claim through its canonical Claim node.

    This specifically repairs the case where alpha4c211/212 emitted:
        excitation_wavelength -> Raman intensity

    while the canonical grounded Claim actually states:
        SPR closer to excitation wavelength -> higher enhancement factor.

    No new evidence mention is synthesized here.
    """
    updates: dict[str, object] = {
        "trend_semantics_id": SERS_AU_AG_TREND_SEMANTICS_ID,
    }

    if (
        item.evidence_basis
        not in {"reported_directional_claim", "reported_correlation"}
        or not item.source_claim_ids
    ):
        return replace(item, **updates)

    bundle = _canonical_claim_text_bundle(item, graph)
    if not bundle or not _is_spectral_alignment_claim(bundle):
        return replace(item, **updates)

    direction_shape = _spectral_detuning_direction(bundle)
    if direction_shape is None:
        return replace(item, **updates)

    direction, shape = direction_shape
    dependent_key = item.dependent_observable_key
    dependent_label = item.dependent_observable_label

    # Promote to formal EF only when the canonical Claim itself preserves
    # enhancement-factor wording (statement/description/label/etc.).
    if _formal_ef_language(bundle):
        dependent_key = "sers_enhancement_factor"
        dependent_label = "SERS enhancement factor"

    trend_id = stable_trend_id(
        paper_id=item.paper_id,
        independent_variable_key="spr_excitation_detuning",
        dependent_observable_key=dependent_key,
        evidence_basis=item.evidence_basis,
        source_node_ids=item.source_node_ids,
    )

    return replace(
        item,
        trend_id=trend_id,
        trend_semantics_id=SERS_AU_AG_TREND_SEMANTICS_ID,
        independent_variable_key="spr_excitation_detuning",
        independent_variable_label="SPR–excitation spectral detuning",
        dependent_observable_key=dependent_key,
        dependent_observable_label=dependent_label,
        direction=direction,
        shape=shape,
        causal_status="not_asserted",
        varied_dimension="spr_excitation_detuning",
    )


def extract_sers_au_ag_trend_evidence(
    source: TrendEvidenceSource,
) -> list[TrendEvidence]:
    base = _ALPHA4C212_ADAPTER.extract_evidence(source)

    regrounded = [
        _reground_item(item, source.graph)
        for item in base
    ]

    by_id: dict[str, TrendEvidence] = {}
    for item in regrounded:
        existing = by_id.get(item.trend_id)
        if existing is not None and existing != item:
            raise ValueError(
                "alpha4c2121 canonical re-grounding produced conflicting "
                f"TrendEvidence rows for trend_id={item.trend_id!r}."
            )
        by_id[item.trend_id] = item

    return sorted(
        by_id.values(),
        key=lambda item: (
            item.paper_id,
            item.independent_variable_key,
            item.dependent_observable_key,
            item.evidence_basis,
            item.trend_id,
        ),
    )


SERS_AU_AG_TREND_ADAPTER = TrendDomainAdapter(
    adapter_id=_ALPHA4C212_ADAPTER.adapter_id,
    domain_profile_id=_ALPHA4C212_ADAPTER.domain_profile_id,
    semantics_id=SERS_AU_AG_TREND_SEMANTICS_ID,
    supported_evidence_bases=
        _ALPHA4C212_ADAPTER.supported_evidence_bases,
    required_inputs=_ALPHA4C212_ADAPTER.required_inputs,
    extract_evidence_fn=extract_sers_au_ag_trend_evidence,
)
