from __future__ import annotations

from dac_her.corpus_acquisition.progress import (
    progress_prefix,
)


def test_progress_prefix_zero_pads_to_total_width():
    assert progress_prefix("M1", 7, 32) == "[M1 07/32]"
    assert progress_prefix("M2 assess", 7, 527) == "[M2 assess 007/527]"


def test_progress_prefix_handles_single_item():
    assert progress_prefix("M1", 1, 1) == "[M1 1/1]"
