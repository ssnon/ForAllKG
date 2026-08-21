from __future__ import annotations

from pathlib import Path


RETIRED_MODULE_FILES = (
    "pipeline_core/literature/discovery/query_plan.py",
    "pipeline_core/literature/discovery/registry.py",
    "pipeline_core/literature/discovery/relevance.py",
    "pipeline_core/literature/discovery/runtime.py",
    "pipeline_core/literature/discovery/selection.py",
    "pipeline_core/literature/discovery/selection_plan.py",
)


def test_superseded_literature_discovery_stack_is_retired():
    for relative in RETIRED_MODULE_FILES:
        assert not Path(relative).exists()


def test_active_openalex_provider_island_remains_importable():
    from pipeline_core.literature.discovery.providers import (
        LiteratureSearchRequest,
        OpenAlexProvider,
    )

    assert LiteratureSearchRequest is not None
    assert OpenAlexProvider is not None


def test_canonical_openalex_adapters_remain_importable():
    from pipeline_core.literature.acquisition.openalex_access import (
        OpenAlexAccessResolver,
    )
    from pipeline_core.literature.acquisition.openalex_catalog_adapter import (
        OpenAlexCatalogProvider,
    )

    assert OpenAlexAccessResolver is not None
    assert OpenAlexCatalogProvider is not None


def test_discovery_package_no_longer_reexports_old_runtime_api():
    import pipeline_core.literature.discovery as discovery

    forbidden = (
        "run_discovery",
        "select_pilot_requests",
        "LiteratureRegistry",
        "LiteratureQueryPlan",
        "LiteratureSelectionPlan",
        "select_literature",
    )

    for name in forbidden:
        assert not hasattr(discovery, name)
