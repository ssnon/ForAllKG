from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dac_her.broad_compact_schema import BroadMechanismGraphDraft
from dac_her.domains.extraction_registry import get_extraction_adapter
from pipeline_core.draft_schema import KnowledgeGraphDraft
from dac_her.strict_recovery import (
    _domain_gate_recovery_response_model,
)


def test_broad_adapter_owns_compact_domain_gate_recovery_schema():
    adapter = get_extraction_adapter("catalysis_mechanism")

    assert (
        adapter.domain_gate_recovery_response_model(compact=True)
        is BroadMechanismGraphDraft
    )
    assert (
        _domain_gate_recovery_response_model(
            adapter,
            compact=True,
        )
        is BroadMechanismGraphDraft
    )
    assert (
        adapter.domain_gate_recovery_response_model(compact=False)
        is KnowledgeGraphDraft
    )


@pytest.mark.parametrize(
    "domain_profile_id",
    ["dac_her", "sers_au_ag"],
)
def test_other_domains_do_not_silently_gain_compact_recovery(
    domain_profile_id: str,
):
    adapter = get_extraction_adapter(domain_profile_id)

    assert (
        adapter.domain_gate_recovery_response_model(compact=False)
        is KnowledgeGraphDraft
    )
    with pytest.raises(
        ValueError,
        match="compact domain-gate recovery schema is not configured",
    ):
        adapter.domain_gate_recovery_response_model(compact=True)


def test_compact_recovery_is_separate_adapter_capability():
    adapter = get_extraction_adapter("catalysis_mechanism")

    assert (
        adapter.compact_generation_response_model
        is BroadMechanismGraphDraft
    )
    assert (
        adapter.compact_domain_gate_recovery_response_model
        is BroadMechanismGraphDraft
    )


