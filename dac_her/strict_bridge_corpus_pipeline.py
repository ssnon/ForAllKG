"""Compatibility shim for the extracted strict corpus orchestration runtime."""

from pipeline_core.strict_bridge_corpus_pipeline import *  # noqa: F401,F403

# Historical compatibility for callers that imported
# these implementation fingerprint helpers directly
# from dac_her.strict_bridge_corpus_pipeline.
from pipeline_core.strict_bridge_corpus_pipeline import (
    _sha256_file,
    _sha256_source_tree,
)

