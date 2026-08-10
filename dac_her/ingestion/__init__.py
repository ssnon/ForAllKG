"""Drive -> Marker -> Markdown ingestion for GraphAgentsDAC."""

from .contracts import (
    ArticleRow,
    DiscoveredPaper,
    DriveFile,
    IngestionIssue,
    MarkerResult,
    PaperRegistryEntry,
)
from .naming import ParsedPdfName, parse_pdf_name, paper_id_for
from .registry import PaperRegistry

__all__ = [
    "ArticleRow",
    "DiscoveredPaper",
    "DriveFile",
    "IngestionIssue",
    "MarkerResult",
    "PaperRegistryEntry",
    "ParsedPdfName",
    "PaperRegistry",
    "parse_pdf_name",
    "paper_id_for",
]
