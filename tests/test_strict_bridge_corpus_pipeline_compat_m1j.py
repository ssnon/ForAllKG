from __future__ import annotations

import dac_her.strict_bridge_corpus_pipeline as legacy
import pipeline_core.strict_bridge_corpus_pipeline as core


def test_strict_corpus_hash_helpers_preserve_legacy_import_identity():
    assert (
        legacy._sha256_file
        is core._sha256_file
    )

    assert (
        legacy._sha256_source_tree
        is core._sha256_source_tree
    )
