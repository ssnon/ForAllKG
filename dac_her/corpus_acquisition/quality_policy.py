from __future__ import annotations

from pathlib import Path

import yaml

from dac_her.corpus_acquisition.quality_contracts import (
    CorpusQualityPolicy,
)


def load_corpus_quality_policy(
    path: Path,
) -> CorpusQualityPolicy:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(
            f"Corpus quality policy must be a mapping: {path}"
        )
    return CorpusQualityPolicy.model_validate(loaded)
