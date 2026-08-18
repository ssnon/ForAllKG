from __future__ import annotations

from pipeline_core.domain_profile import (
    ProjectionBacktraceRule,
    ProjectionSemantics,
)


# Historical DAC-HER projection compatibility payload.
#
# This is intentionally separate from the shared projection
# engine. None-valued projection semantics still resolve to
# this policy for compatibility with existing callers,
# including the current catalysis_mechanism profile.
_MECHANISM_NODE_TYPES = {
    "Catalyst",
    "CatalystModel",
    "Metal",
    "Support",
    "CoordinationMotif",
    "SynthesisMethod",
    "Precursor",
    "Reaction",
    "ReactionStep",
    "Intermediate",
    "Material",
    "ObservationClaim",
    "MechanismClaim",
    "BridgeConcept",
}

_ORIGIN_NODE_TYPES = {
    "Catalyst",
    "CatalystModel",
    "Support",
    "CoordinationMotif",
    "Material",
    "Reaction",
    "ReactionStep",
    "Intermediate",
}

_BACKTRACE_RELATIONS = {
    "HAS_MEASUREMENT",
    "EVALUATED_IN",
    "CHARACTERIZED_BY",
    "MODELED_BY",
    "APPLIES_TO",
    "SUPPORTS_CLAIM",
}


LEGACY_DAC_HER_PROJECTION_SEMANTICS = (
    ProjectionSemantics(
        semantics_id=(
            "dac_her_legacy_projection_v1"
        ),
        mechanism_node_types=frozenset(
            _MECHANISM_NODE_TYPES
        ),
        origin_node_types=frozenset(
            _ORIGIN_NODE_TYPES
        ),
        backtrace_rules=tuple(
            ProjectionBacktraceRule(
                relation,
                "incoming",
            )
            for relation in sorted(
                _BACKTRACE_RELATIONS
            )
        ),
        max_backtrace_depth=3,
    )
)
