from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from dac_her.discovery_axis_contracts import AxisFidelityReview, DiscoveryAxis
from pipeline_core.discovery.hypothesis_contracts import HypothesisCard


class EncoderProtocol(Protocol):
    def encode_query(self, text: str) -> np.ndarray: ...


_GENERIC_TOKENS = {
    "a", "an", "and", "as", "at", "by", "for", "from", "in", "into", "is", "of",
    "on", "or", "the", "through", "to", "with", "within", "between", "across",
    "catalyst", "catalysts", "activity", "performance", "effect", "effects", "relation",
    "relationship", "correlation", "modulation", "regulation", "dependent", "dependence",
    "hydrogen", "her", "evolution", "reaction", "metal", "metals", "dual", "atom",
    "nitrogen", "coordination", "site", "sites", "model", "models",
}


def _normalize(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text or "")).lower()
    value = value.replace("–", "-").replace("—", "-")
    value = re.sub(r"[^a-z0-9α-ωδΔ+*/._-]+", " ", value)
    return " ".join(value.split())


def _tokens(text: str) -> list[str]:
    rows = re.findall(r"[a-z0-9α-ωδΔ]+", _normalize(text))
    return [row for row in rows if len(row) >= 3 and row not in _GENERIC_TOKENS]


def _unit(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(value))
    if norm <= 0.0:
        return value
    return value / norm


def _similarity(encoder: EncoderProtocol, left: str, right: str) -> float:
    a = _unit(encoder.encode_query(left))
    b = _unit(encoder.encode_query(right))
    if a.shape != b.shape or a.size == 0:
        return 0.0
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


def _axis_text(axis: DiscoveryAxis) -> str:
    return "\n".join(
        value
        for value in [
            axis.label,
            axis.proposed_subject,
            axis.proposed_relation,
            axis.proposed_object,
            axis.entry_anchor_label,
            axis.exit_anchor_label,
        ]
        if str(value).strip()
    )


def _proposal_text(card: HypothesisCard) -> str:
    parts = [card.title, card.hypothesis_statement, card.inferential_bridge]
    for row in card.predicted_observations:
        parts.extend([row.observable, row.rationale])
    for row in card.falsification_criteria:
        parts.extend([row.observable, row.falsifying_outcome])
    return "\n".join(parts)


@dataclass(frozen=True)
class AxisFidelityPolicy:
    fail_bridge_similarity: float = 0.32
    pass_bridge_similarity: float = 0.45
    pass_secondary_similarity: float = 0.42
    pass_signature_coverage: float = 0.20


class DiscoveryAxisFidelityCritic:
    """Deterministic guard against decorative/unused discovery lineage.

    The check intentionally combines semantic similarity and axis-distinctive
    token coverage. It does not decide scientific truth or external novelty.
    """

    def __init__(self, policy: AxisFidelityPolicy | None = None) -> None:
        self.policy = policy or AxisFidelityPolicy()

    def review(
        self,
        axis: DiscoveryAxis,
        card: HypothesisCard,
        encoder: EncoderProtocol,
    ) -> AxisFidelityReview:
        axis_text = _axis_text(axis)
        proposal_text = _proposal_text(card)
        prediction_text = "\n".join(
            f"{row.observable} {row.rationale}" for row in card.predicted_observations
        )

        hypothesis_similarity = _similarity(encoder, axis_text, card.hypothesis_statement)
        bridge_similarity = _similarity(encoder, axis_text, card.inferential_bridge)
        prediction_similarity = _similarity(encoder, axis_text, prediction_text)
        combined_similarity = _similarity(encoder, axis_text, proposal_text)

        signature_source = " ".join(
            value
            for value in [
                axis.label,
                axis.proposed_subject,
                axis.proposed_relation,
                axis.proposed_object,
            ]
            if str(value).strip()
        )
        signature = sorted(set(_tokens(signature_source)))
        proposal_tokens = set(_tokens(proposal_text))
        matched = sorted(token for token in signature if token in proposal_tokens)
        coverage = len(matched) / len(signature) if signature else 1.0

        reasons: list[str] = []
        if bridge_similarity < self.policy.fail_bridge_similarity:
            reasons.append("axis_bridge_semantic_mismatch")
        if signature and not matched:
            reasons.append("no_axis_distinctive_terms_used")
        if prediction_similarity < self.policy.pass_secondary_similarity:
            reasons.append("axis_not_clearly_tested_in_prediction")
        if coverage < self.policy.pass_signature_coverage:
            reasons.append("low_axis_signature_coverage")

        if (
            bridge_similarity < self.policy.fail_bridge_similarity
            or (
                signature
                and coverage == 0.0
                and bridge_similarity < 0.55
            )
        ):
            status = "fail"
            interpretation = (
                "The assigned discovery axis does not appear essential to the inferential bridge. "
                "Treat the lineage as decorative until the proposal is repaired or abstained."
            )
        elif (
            bridge_similarity >= self.policy.pass_bridge_similarity
            and max(hypothesis_similarity, prediction_similarity)
            >= self.policy.pass_secondary_similarity
            and coverage >= self.policy.pass_signature_coverage
        ):
            status = "pass"
            interpretation = (
                "The proposal substantively reflects the assigned discovery axis in its bridge and testable content."
            )
        else:
            status = "warning"
            interpretation = (
                "The proposal is related to the assigned axis, but axis dependence is weaker than the preferred alpha4 threshold."
            )

        return AxisFidelityReview(
            axis_id=axis.axis_id,
            hypothesis_id=card.hypothesis_id,
            status=status,
            axis_signature_tokens=signature,
            matched_signature_tokens=matched,
            signature_coverage=float(coverage),
            hypothesis_similarity=float(hypothesis_similarity),
            bridge_similarity=float(bridge_similarity),
            prediction_similarity=float(prediction_similarity),
            combined_similarity=float(combined_similarity),
            reason_codes=sorted(set(reasons)),
            interpretation=interpretation,
        )
