from __future__ import annotations

import inspect

from pipeline_core.corpus.corpus_graph import (
    load_projection_bundle,
)


def test_projection_bundle_loader_requires_explicit_data_root():
    parameter = inspect.signature(
        load_projection_bundle
    ).parameters["data_root"]

    assert parameter.default is inspect.Parameter.empty
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
