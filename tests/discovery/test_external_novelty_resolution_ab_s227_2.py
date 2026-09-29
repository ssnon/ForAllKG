from pathlib import Path

import pytest

from pipeline_core.discovery.external_novelty_contracts import (
    ExternalNoveltyReport,
    PriorArtPacket,
)
from scripts.discovery.run_s227_2_external_novelty_resolution_ab import (
    source_prefix_from_prior_art,
)


def test_source_prefix_from_prior_art():
    path = Path("/tmp/external_novelty.prior_art.json")
    assert source_prefix_from_prior_art(path) == Path(
        "/tmp/external_novelty"
    )


def test_source_prefix_rejects_non_prior_art_name():
    with pytest.raises(RuntimeError):
        source_prefix_from_prior_art(
            Path("/tmp/report.json")
        )
