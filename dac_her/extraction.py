"""Compatibility entrypoint for strict extraction v2.3.5.

The recovery implementation lives in strict_recovery.py so validation,
normalization, patching, and rechunk routing remain independently testable.
"""

from dac_her.strict_recovery import (
    chunk_output_path,
    extract_one_chunk,
    is_truncation_error,
    load_existing_result,
)

__all__ = [
    "chunk_output_path",
    "extract_one_chunk",
    "is_truncation_error",
    "load_existing_result",
]
