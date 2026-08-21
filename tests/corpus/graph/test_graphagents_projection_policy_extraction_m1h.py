from __future__ import annotations

import pipeline_core.corpus.graph.graphagents_adapter as adapter

from pipeline_core.corpus.graph.legacy_dac_projection_policy import (
    LEGACY_DAC_HER_PROJECTION_SEMANTICS,
)


def test_graphagents_fallback_uses_extracted_legacy_policy_object():
    resolved = (
        adapter._resolve_projection_semantics(
            None
        )
    )

    assert (
        resolved
        is LEGACY_DAC_HER_PROJECTION_SEMANTICS
    )


def test_adapter_compatibility_binding_preserves_object_identity():
    assert (
        adapter._LEGACY_DAC_HER_PROJECTION_SEMANTICS
        is LEGACY_DAC_HER_PROJECTION_SEMANTICS
    )


def test_extracted_legacy_projection_identity_is_preserved():
    semantics = (
        LEGACY_DAC_HER_PROJECTION_SEMANTICS
    )

    assert (
        semantics.semantics_id
        == "dac_her_legacy_projection_v1"
    )

    assert semantics.max_backtrace_depth == 3
