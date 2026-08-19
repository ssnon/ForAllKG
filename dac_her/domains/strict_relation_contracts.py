"""Compatibility/composition facade for historical strict relation policies."""

from __future__ import annotations

from domains.dac_her.relation_constraints import (
    DAC_LEGACY_STRICT_RELATION_CONSTRAINTS,
)
from pipeline_core.evidence_relation_constraints import (
    COMMON_EVIDENCE_STRICT_RELATION_CONSTRAINTS,
)
from pipeline_core.graph_domain import RelationConstraint


DAC_HER_STRICT_RELATION_CONSTRAINTS = (
    DAC_LEGACY_STRICT_RELATION_CONSTRAINTS
)

CATALYSIS_MECHANISM_STRICT_RELATION_CONSTRAINTS = (
    DAC_LEGACY_STRICT_RELATION_CONSTRAINTS
)

SERS_AU_AG_STRICT_RELATION_CONSTRAINTS = (
    COMMON_EVIDENCE_STRICT_RELATION_CONSTRAINTS
)
