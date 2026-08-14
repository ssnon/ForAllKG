from __future__ import annotations

from pathlib import Path

import yaml

from dac_her.corpus_acquisition.access_contracts import (
    SourceAcquisitionPolicy,
)


def load_source_acquisition_policy(
    path: Path,
) -> SourceAcquisitionPolicy:
    loaded = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )
    if not isinstance(loaded, dict):
        raise ValueError(
            f"Source acquisition policy must be a mapping: {path}"
        )
    return SourceAcquisitionPolicy.model_validate(loaded)
