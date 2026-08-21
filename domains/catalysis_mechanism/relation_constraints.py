"""Strict relation contract owned by the catalysis-mechanism domain.

This module intentionally preserves the frozen pre-H3 relation semantics while
removing cross-domain ownership. Shared evidence-topology constraints continue
to reuse the canonical pipeline-core objects.
"""

from __future__ import annotations

from pipeline_core.corpus.extraction.evidence_relation_constraints import (
    COMMON_EVIDENCE_STRICT_RELATION_CONSTRAINTS,
)
from pipeline_core.corpus.graph.graph_domain import RelationConstraint


CATALYSIS_MECHANISM_STRICT_RELATION_CONSTRAINTS = (
    RelationConstraint(
        "EVALUATED_IN",
        source_types=frozenset({"Catalyst", "CatalystModel", "Material"}),
        target_types=frozenset({"Experiment"}),
    ),
    RelationConstraint(
        "CHARACTERIZED_BY",
        source_types=frozenset(
            {"Catalyst", "Support", "Material", "CoordinationMotif"}
        ),
        target_types=frozenset({"Experiment"}),
    ),
    RelationConstraint(
        "MODELED_BY",
        source_types=frozenset({"CatalystModel"}),
        target_types=frozenset({"Calculation"}),
    ),
    RelationConstraint(
        "SYNTHESIZED_BY",
        source_types=frozenset({"Catalyst"}),
        target_types=frozenset({"SynthesisMethod"}),
    ),
    RelationConstraint(
        "USES_PRECURSOR",
        source_types=frozenset({"SynthesisMethod"}),
        target_types=frozenset({"Precursor"}),
    ),
    *COMMON_EVIDENCE_STRICT_RELATION_CONSTRAINTS,
    RelationConstraint(
        "MODEL_OF",
        source_types=frozenset({"CatalystModel"}),
        target_types=frozenset({"Catalyst"}),
    ),
    RelationConstraint(
        "HAS_METAL",
        source_types=frozenset({"Catalyst", "CatalystModel"}),
        target_types=frozenset({"Metal"}),
    ),
    RelationConstraint(
        "SUPPORTED_ON",
        source_types=frozenset({"Catalyst", "CatalystModel"}),
        target_types=frozenset({"Support"}),
    ),
    RelationConstraint(
        "CATALYZES",
        source_types=frozenset({"Catalyst"}),
        target_types=frozenset({"Reaction"}),
    ),
)
