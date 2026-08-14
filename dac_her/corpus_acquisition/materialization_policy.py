from __future__ import annotations

from pathlib import Path

import yaml

from dac_her.corpus_acquisition.materialization_contracts import (
    MaterializationPolicy,
)


def load_materialization_policy(
    path: Path,
) -> MaterializationPolicy:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(
            f"Materialization policy must be a mapping: {path}"
        )
    return MaterializationPolicy.model_validate(loaded)
