from __future__ import annotations

import pipeline_core.chunking as core


def test_legacy_chunking_reexports_filter_parent_assets():
    import dac_her.chunking as legacy

    assert (
        legacy._filter_parent_assets
        is core._filter_parent_assets
    )


def test_chunking_recovery_imports_through_legacy_surface():
    import dac_her.chunking_recovery as recovery

    assert (
        recovery._filter_parent_assets
        is core._filter_parent_assets
    )
