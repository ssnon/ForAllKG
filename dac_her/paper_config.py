"""Compatibility re-export for the multi-document paper configuration API."""

from dac_her.config import (  # noqa: F401
    DocumentConfig,
    DocumentSelection,
    FigureProcessingConfig,
    PaperConfig,
    get_paper_config,
    load_paper_configs,
    paper_config_fingerprint_payload,
)
