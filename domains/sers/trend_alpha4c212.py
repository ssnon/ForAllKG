from __future__ import annotations

import re
from dataclasses import replace
from typing import Any, Mapping

import networkx as nx

from domains.sers.trend_alpha4c211 import (
    SERS_AU_AG_TREND_ADAPTER as _ALPHA4C211_ADAPTER,
)
from dac_her.trend_domain import (
    TrendDomainAdapter,
    TrendEvidence,
    TrendEvidenceSource,
)
from dac_her.trend_evidence import stable_trend_id


SERS_AU_AG_TREND_SEMANTICS_ID = "sers_au_ag_trend_v4_alpha4c212"

_CLAIM_TYPES = frozenset({"ObservationClaim", "MechanismClaim"})
_SUBJECT_TYPES = frozenset({
    "PlasmonicSubstrate",
    "Nanostructure",
    "Metal",
    "Material",
    "Support",
    "StructuralMotif",
    "Morphology",
    "SynthesisMethod",
})


def _norm(value: object) -> str:
    text = str(value or "").casefold()
    text = (
        text.replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("μ", "µ")
    )
    return re.sub(r"\s+", " ", text).strip()


def _relation(attrs: Mapping[str, Any]) -> str:
    return str(attrs.get("relation", "")).strip()


def _outgoing(
    graph: nx.Graph,
    node_id: str,
    relation: str,
) -> tuple[str, ...]:
    if node_id not in graph or not graph.is_directed():
        return ()
    if graph.is_multigraph():
        iterator = graph.out_edges(
            node_id,
            keys=True,
            data=True,
        )
        values = {
            str(right)
            for _left, right, _key, attrs in iterator
            if _relation(attrs) == relation
        }
    else:
        iterator = graph.out_edges(node_id, data=True)
        values = {
            str(right)
            for _left, right, attrs in iterator
            if _relation(attrs) == relation
        }
    return tuple(sorted(values))


def _claim_text(attrs: Mapping[str, Any]) -> str:
    for key in (
        "statement",
        "source_expression",
        "description",
        "label",
        "node_text",
    ):
        value = str(attrs.get(key, "")).strip()
        if value:
            return value
    return ""


def _claim_subjects(
    graph: nx.Graph,
    claim_id: str,
) -> tuple[str, ...]:
    subjects: set[str] = set()
    attrs = graph.nodes[claim_id]

    explicit = str(attrs.get("subject_id", "")).strip()
    if explicit and explicit in graph:
        subjects.add(explicit)

    for node_id in _outgoing(graph, claim_id, "APPLIES_TO"):
        if (
            node_id in graph
            and str(graph.nodes[node_id].get("type", ""))
            in _SUBJECT_TYPES
        ):
            subjects.add(node_id)

    return tuple(sorted(subjects))


def _requires_verification(attrs: Mapping[str, Any]) -> bool:
    value = attrs.get("requires_verification", False)
    if isinstance(value, bool):
        return value
    if _norm(value) in {"true", "1", "yes"}:
        return True
    return _norm(attrs.get("evidence_status", "")) in {
        "candidate",
        "requires_verification",
        "unconfirmed",
    }


def _is_spectral_alignment_claim(text: str) -> bool:
    normalized = _norm(text)
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
        r"spectral\s+proximity)\b",
        normalized,
    ))
    return has_resonance and has_excitation and has_alignment


def _formal_ef_language(text: str) -> bool:
    normalized = _norm(text)
    return bool(re.search(
        r"\b(?:sers\s+)?enhancement[-\s]+factor\b|\bsers\s+ef\b|"
        r"\bef\s+(?:value|coefficient)\b",
        normalized,
    ))


def _spectral_detuning_direction(
    text: str,
) -> tuple[str, str] | None:
    """
    Canonical independent variable is spectral detuning.

    Source: closer/more matched resonance -> stronger SERS
    Canonical: detuning increases -> SERS decreases.
    """
    normalized = _norm(text)

    closer_higher = bool(
        re.search(
            r"\b(?:closer|closest|better\s+matched|"
            r"more\s+closely\s+matched|greater\s+overlap|"
            r"better\s+overlap)\b"
            r".{0,140}\b(?:higher|stronger|increase\w*|enhanc\w*)\b",
            normalized,
        )
        or re.search(
            r"\b(?:higher|stronger|increase\w*|enhanc\w*)\b"
            r".{0,140}\b(?:closer|closest|better\s+matched|"
            r"greater\s+overlap|better\s+overlap)\b",
            normalized,
        )
    )
    if closer_higher:
        return "negative", "monotonic"

    explicit_detuning = bool(
        re.search(
            r"\b(?:smaller|lower|decreasing|reduced)\s+"
            r"(?:spectral\s+)?detuning\b"
            r".{0,120}\b(?:higher|stronger|increase\w*|enhanc\w*)\b",
            normalized,
        )
        or re.search(
            r"\b(?:higher|stronger|increase\w*|enhanc\w*)\b"
            r".{0,120}\b(?:smaller|lower|decreasing|reduced)\s+"
            r"(?:spectral\s+)?detuning\b",
            normalized,
        )
    )
    if explicit_detuning:
        return "negative", "monotonic"

    # "Matching matters" without an explicit directional relation is not
    # enough to synthesize a signed trend.
    return None


def _spectral_claim_evidence(
    *,
    graph: nx.Graph,
    paper_id: str,
    claim_id: str,
    attrs: Mapping[str, Any],
) -> TrendEvidence | None:
    text = _claim_text(attrs)
    if not text or not _is_spectral_alignment_claim(text):
        return None

    direction_shape = _spectral_detuning_direction(text)
    if direction_shape is None:
        return None
    direction, shape = direction_shape

    # alpha4c.2.1.2 is deliberately narrow: the direct supplementation lane
    # only emits a spectral-detuning relation when the source explicitly
    # names an enhancement factor. Generic "performance" matching claims
    # remain fail-closed to the historical parser.
    if not _formal_ef_language(text):
        return None

    trend_id = stable_trend_id(
        paper_id=paper_id,
        independent_variable_key="spr_excitation_detuning",
        dependent_observable_key="sers_enhancement_factor",
        evidence_basis="reported_directional_claim",
        source_node_ids=(str(claim_id),),
    )

    return TrendEvidence(
        trend_id=trend_id,
        domain_profile_id="sers_au_ag",
        trend_semantics_id=SERS_AU_AG_TREND_SEMANTICS_ID,
        paper_id=paper_id,
        independent_variable_key="spr_excitation_detuning",
        independent_variable_label=
            "SPR–excitation spectral detuning",
        dependent_observable_key="sers_enhancement_factor",
        dependent_observable_label="SERS enhancement factor",
        direction=direction,
        shape=shape,
        evidence_basis="reported_directional_claim",
        causal_status="not_asserted",
        varied_dimension="spr_excitation_detuning",
        subject_ids=_claim_subjects(graph, str(claim_id)),
        source_expression=text,
        source_expressions=(text,),
        source_claim_ids=(str(claim_id),),
        source_node_ids=(str(claim_id),),
        requires_verification=_requires_verification(attrs),
    )


def _refine(item: TrendEvidence) -> TrendEvidence:
    updates: dict[str, object] = {
        "trend_semantics_id": SERS_AU_AG_TREND_SEMANTICS_ID,
    }

    text = str(item.source_expression or "")
    if (
        item.evidence_basis
        in {"reported_directional_claim", "reported_correlation"}
        and _is_spectral_alignment_claim(text)
    ):
        direction_shape = _spectral_detuning_direction(text)
        if direction_shape is not None:
            direction, shape = direction_shape
            dependent_key = item.dependent_observable_key
            dependent_label = item.dependent_observable_label
            if _formal_ef_language(text):
                dependent_key = "sers_enhancement_factor"
                dependent_label = "SERS enhancement factor"

            trend_id = stable_trend_id(
                paper_id=item.paper_id,
                independent_variable_key="spr_excitation_detuning",
                dependent_observable_key=dependent_key,
                evidence_basis=item.evidence_basis,
                source_node_ids=item.source_node_ids,
            )
            updates.update({
                "trend_id": trend_id,
                "independent_variable_key":
                    "spr_excitation_detuning",
                "independent_variable_label":
                    "SPR–excitation spectral detuning",
                "dependent_observable_key": dependent_key,
                "dependent_observable_label": dependent_label,
                "direction": direction,
                "shape": shape,
                "varied_dimension": "spr_excitation_detuning",
            })

    return replace(item, **updates)


def extract_sers_au_ag_trend_evidence(
    source: TrendEvidenceSource,
) -> list[TrendEvidence]:
    # Preserve the full alpha4c211 extraction lane first.
    refined = [
        _refine(item)
        for item in _ALPHA4C211_ADAPTER.extract_evidence(source)
    ]

    # A narrow supplementation lane handles explicit SPR/excitation matching
    # claims even when the alpha4c211 control parser does not emit them.
    # If alpha4c211 already emitted the same source claim, do not duplicate it.
    emitted_claim_ids = {
        claim_id
        for item in refined
        for claim_id in item.source_claim_ids
    }

    graph = source.graph
    for claim_id, attrs in sorted(
        graph.nodes(data=True),
        key=lambda item: str(item[0]),
    ):
        if str(attrs.get("type", "")) not in _CLAIM_TYPES:
            continue
        claim_id = str(claim_id)
        if claim_id in emitted_claim_ids:
            continue

        supplemental = _spectral_claim_evidence(
            graph=graph,
            paper_id=source.paper_id,
            claim_id=claim_id,
            attrs=attrs,
        )
        if supplemental is not None:
            refined.append(supplemental)
            emitted_claim_ids.add(claim_id)

    # Final exact trend-id dedupe is deterministic and fail-closed.
    by_id: dict[str, TrendEvidence] = {}
    for item in refined:
        existing = by_id.get(item.trend_id)
        if existing is not None and existing != item:
            raise ValueError(
                "alpha4c212 produced conflicting TrendEvidence rows "
                f"for trend_id={item.trend_id!r}."
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
    adapter_id=_ALPHA4C211_ADAPTER.adapter_id,
    domain_profile_id=_ALPHA4C211_ADAPTER.domain_profile_id,
    semantics_id=SERS_AU_AG_TREND_SEMANTICS_ID,
    supported_evidence_bases=
        _ALPHA4C211_ADAPTER.supported_evidence_bases,
    required_inputs=_ALPHA4C211_ADAPTER.required_inputs,
    extract_evidence_fn=extract_sers_au_ag_trend_evidence,
)
