from __future__ import annotations

import re
from dataclasses import replace

from dac_her.domains import sers_au_ag_trend as v1
from dac_her.domains import sers_au_ag_trend_alpha4c211 as v3
from domains.sers.trend_alpha4c2121 import (
    SERS_AU_AG_TREND_ADAPTER as _V5_ADAPTER,
)
from domains.sers.trend_alpha4c5g2 import (
    resolve_measurement_local_method_contexts,
    _comparative_gap_direction,
)
from dac_her.trend_domain import (
    TrendDomainAdapter,
    TrendEvidence,
    TrendEvidenceSource,
)
from dac_her.trend_evidence import stable_trend_id


SERS_AU_AG_TREND_SEMANTICS_ID = (
    "sers_au_ag_trend_v6r1_alpha4c5g2r1"
)

_EXPLICIT_DIMENSION_RE = re.compile(
    r"\b(?:interior\s+)?(?:nano\s*)?gap\s+"
    r"(?:sizes?|widths?|distances?)\b",
    re.I,
)

_NUMERIC_GAP_RE = re.compile(
    r"(?:"
    r"\b\d+(?:\.\d+)?\s*[- ]?\s*(?:nm|µm|um)\b"
    r".{0,35}\bgap\b"
    r"|"
    r"\bgap\b.{0,35}"
    r"\b\d+(?:\.\d+)?\s*[- ]?\s*(?:nm|µm|um)\b"
    r")",
    re.I,
)

_ADJECTIVE_GAP_RE = re.compile(
    r"(?:"
    r"\b(?:smaller|larger|narrower|wider|small|large)\b"
    r".{0,40}\b(?:interior\s+)?(?:nano\s*)?gap\b"
    r"|"
    r"\b(?:interior\s+)?(?:nano\s*)?gap\b"
    r".{0,40}\b(?:smaller|larger|narrower|wider)\b"
    r")",
    re.I,
)

_DYNAMIC_GAP_RE = re.compile(
    r"\b(?:interior\s+)?(?:nano\s*)?gap\b"
    r".{0,45}\b(?:decreas\w*|increas\w*|"
    r"narrow\w*|widen\w*|shrink\w*|grow\w*)\b",
    re.I,
)


def _norm(value: object) -> str:
    return v3._norm(value)


def _nanogap_size_cue(text: str) -> bool:
    normalized = _norm(text)
    if "gap" not in normalized:
        return False
    return bool(
        _EXPLICIT_DIMENSION_RE.search(normalized)
        or _NUMERIC_GAP_RE.search(normalized)
        or _ADJECTIVE_GAP_RE.search(normalized)
        or _DYNAMIC_GAP_RE.search(normalized)
    )


def _resolved_claim_control(
    text: str,
) -> tuple[str, str] | None:
    current = v3._claim_control(text)

    # A quantitative size cue takes precedence over the categorical
    # nanogap-presence family. This includes plural "gap sizes", explicit
    # numeric gap comparisons, and directional size-change language such as
    # "large gap ... gap decreases".
    if _nanogap_size_cue(text):
        if (
            current is None
            or current[0] == "nanogap_presence"
        ):
            return "nanogap_size", "nanogap size"

    return current


def _resolved_direction_shape(
    *,
    text: str,
    control_key: str,
) -> tuple[str, str] | None:
    current = v3._direction_shape(text, control_key)
    if current is not None:
        return current

    if control_key != "nanogap_size":
        return None

    # alpha4c.5g.1b proved that three grounded nanogap-size claims were
    # accepted by the historical v1 parser and lost in alpha4c211. Reuse
    # that already-established narrow parser only as a fallback for the
    # same quantitative nanogap control.
    historical = v1._claim_direction_shape(
        "nanogap_size",
        text,
    )
    if historical is not None:
        return historical

    # Finally handle explicit ordered numeric comparisons such as
    # "greater for the 2-nm gap than for the 8-nm gap".
    return _comparative_gap_direction(text)


def _supplemental_nanogap_claims(
    source: TrendEvidenceSource,
    *,
    already_emitted_claim_ids: set[str],
) -> list[TrendEvidence]:
    evidence: list[TrendEvidence] = []
    graph = source.graph

    for claim_id, attrs in sorted(
        graph.nodes(data=True),
        key=lambda item: str(item[0]),
    ):
        claim_id = str(claim_id)
        if claim_id in already_emitted_claim_ids:
            continue
        if str(attrs.get("type", "")) not in v3._CLAIM_TYPES:
            continue

        text = v1._claim_text(attrs)
        if not text:
            continue

        control = _resolved_claim_control(text)
        if control is None or control[0] != "nanogap_size":
            continue

        response = v3._claim_response(
            text,
            control_key=control[0],
        )
        if response is None:
            continue

        direction_shape = _resolved_direction_shape(
            text=text,
            control_key=control[0],
        )
        if direction_shape is None:
            continue
        direction, shape = direction_shape

        normalized = _norm(text)
        basis = (
            "reported_correlation"
            if (
                re.search(r"\bcorrelat\w*\b", normalized)
                or "linear relationship" in normalized
            )
            else "reported_directional_claim"
        )
        causal_status = (
            "source_asserted"
            if (
                basis == "reported_directional_claim"
                and v1._explicit_causal_language(text)
            )
            else "not_asserted"
        )

        trend_id = stable_trend_id(
            paper_id=source.paper_id,
            independent_variable_key=control[0],
            dependent_observable_key=response[0],
            evidence_basis=basis,
            source_node_ids=(claim_id,),
        )
        evidence.append(
            TrendEvidence(
                trend_id=trend_id,
                domain_profile_id="sers_au_ag",
                trend_semantics_id=SERS_AU_AG_TREND_SEMANTICS_ID,
                paper_id=source.paper_id,
                independent_variable_key=control[0],
                independent_variable_label=control[1],
                dependent_observable_key=response[0],
                dependent_observable_label=response[1],
                direction=direction,
                shape=shape,
                evidence_basis=basis,
                causal_status=causal_status,
                varied_dimension=control[0],
                subject_ids=v1._claim_subjects(
                    graph,
                    claim_id,
                ),
                source_expression=text,
                source_expressions=(text,),
                source_claim_ids=(claim_id,),
                source_node_ids=(claim_id,),
                requires_verification=v1._requires_verification(
                    attrs
                ),
            )
        )

    return evidence


def extract_sers_au_ag_trend_evidence(
    source: TrendEvidenceSource,
) -> list[TrendEvidence]:
    resolved_source, _locality_audit = (
        resolve_measurement_local_method_contexts(source)
    )

    base = _V5_ADAPTER.extract_evidence(
        resolved_source
    )
    updated_base = [
        replace(
            item,
            trend_semantics_id=SERS_AU_AG_TREND_SEMANTICS_ID,
        )
        for item in base
    ]
    emitted_claim_ids = {
        str(claim_id)
        for item in updated_base
        for claim_id in item.source_claim_ids
    }

    combined = [
        *updated_base,
        *_supplemental_nanogap_claims(
            resolved_source,
            already_emitted_claim_ids=emitted_claim_ids,
        ),
    ]

    by_id: dict[str, TrendEvidence] = {}
    for item in combined:
        existing = by_id.get(item.trend_id)
        if existing is not None and existing != item:
            raise ValueError(
                "alpha4c5g2r1 produced conflicting TrendEvidence "
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
    adapter_id=_V5_ADAPTER.adapter_id,
    domain_profile_id=_V5_ADAPTER.domain_profile_id,
    semantics_id=SERS_AU_AG_TREND_SEMANTICS_ID,
    supported_evidence_bases=_V5_ADAPTER.supported_evidence_bases,
    required_inputs=_V5_ADAPTER.required_inputs,
    extract_evidence_fn=extract_sers_au_ag_trend_evidence,
)
