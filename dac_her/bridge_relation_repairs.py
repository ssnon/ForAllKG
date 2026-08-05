from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any

from dac_her.bridge_schemas import (
    BridgeChunkGraph,
    BridgeConcept,
)


@dataclass(frozen=True)
class BridgeRelationRepair:
    concept_id: str
    old_relation: str
    new_relation: str
    rule_id: str
    evidence: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_COMPARATIVE_CUE = re.compile(
    r"""
    \b(?:
        (?:
            better
            | worse
            | higher
            | lower
            | greater
            | smaller
            | stronger
            | weaker
        )
        (?:\s+[\w-]+){0,3}
        \s+than

        |

        (?:
            more
            | less
        )
        \s+
        (?:[\w-]+\s+){0,2}
        than

        |

        compared\s+(?:with|to)

        |

        in\s+contrast\s+to

        |

        whereas
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

def _repair_concept(
    concept: BridgeConcept,
) -> tuple[
    BridgeConcept,
    BridgeRelationRepair | None,
]:
    evidence_text = " ".join(
        value
        for value in (
            concept.relation_evidence_phrase,
            concept.source_phrase,
        )
        if value
    )

    if (
        concept.pattern_relation
        == "VARIES_WITH"
        and concept.pattern_support_mode
        == "explicit_single_span"
        and _COMPARATIVE_CUE.search(
            evidence_text
        )
    ):
        updated = concept.model_copy(
            update={
                "pattern_relation": (
                    "CONTRASTS_WITH"
                ),
            }
        )

        return updated, BridgeRelationRepair(
            concept_id=concept.id,
            old_relation="VARIES_WITH",
            new_relation="CONTRASTS_WITH",
            rule_id=(
                "PAIRWISE_COMPARISON_TO_CONTRAST"
            ),
            evidence=concept.source_phrase,
        )

    return concept, None


def apply_deterministic_relation_repairs(
    result: BridgeChunkGraph,
) -> tuple[
    BridgeChunkGraph,
    list[BridgeRelationRepair],
]:
    concepts: list[BridgeConcept] = []
    repairs: list[BridgeRelationRepair] = []

    for concept in result.concepts:
        updated, repair = _repair_concept(
            concept
        )
        concepts.append(updated)

        if repair is not None:
            repairs.append(repair)

    repaired = result.model_copy(
        update={"concepts": concepts}
    )

    return (
        BridgeChunkGraph.model_validate(
            repaired.model_dump()
        ),
        repairs,
    )