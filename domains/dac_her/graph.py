from __future__ import annotations

from pipeline_core.graph_domain import GraphDomainAdapter
from domains.dac_her.semantic_roles import normalize_measurement_subject_roles


DAC_HER_GRAPH_ADAPTER = GraphDomainAdapter(
    adapter_id="dac_her",
    domain_profile_id="dac_her",
    semantic_role_policy=(
        "Material/Support mentions may be inferred as Catalyst only when "
        "EVALUATED_IN and MEASURED_FOR/HAS_MEASUREMENT structure jointly "
        "establishes a catalytic role."
    ),
    semantic_role_normalizer=normalize_measurement_subject_roles,
)
