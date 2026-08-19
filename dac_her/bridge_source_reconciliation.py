"""Compatibility facade for shared Bridge source reconciliation."""

from pipeline_core.bridge_source_reconciliation import *  # noqa: F401,F403

from pipeline_core.bridge_source_reconciliation import (  # noqa: F401
    _drop_html_tags_with_map,
    _normalized_with_map,
    _unique_occurrence,
    _visible_markdown_text_with_map,
)
