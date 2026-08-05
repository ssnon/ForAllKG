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
    # No deterministic relation relabel is enabled
    # in the frozen three-paper baseline.
    #
    # Comparative wording alone does not distinguish:
    #   peer A CONTRASTS_WITH peer B
    # from:
    #   property VARIES_WITH condition/composition.
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