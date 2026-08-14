from __future__ import annotations

from pathlib import Path

import yaml

from dac_her.corpus_acquisition.backfill_contracts import (
    AcquisitionAwareBackfillPolicy,
)


def load_acquisition_backfill_policy(
    path: Path,
) -> AcquisitionAwareBackfillPolicy:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(
            f"Acquisition-aware backfill policy must be a mapping: {path}"
        )
    return AcquisitionAwareBackfillPolicy.model_validate(loaded)
