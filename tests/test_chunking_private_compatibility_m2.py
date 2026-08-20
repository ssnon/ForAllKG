from __future__ import annotations

import pipeline_core.chunking as core




def test_chunking_recovery_imports_through_legacy_surface():
    import pipeline_core.corpus.chunking_recovery as recovery

    assert (
        recovery._filter_parent_assets
        is core._filter_parent_assets
    )
