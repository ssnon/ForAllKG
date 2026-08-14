from __future__ import annotations

from pathlib import Path

import yaml

from dac_her.corpus_acquisition.supplementary_contracts import (
    SupplementaryDiscoveryPolicy,
)


def load_supplementary_discovery_policy(
    path: Path,
) -> SupplementaryDiscoveryPolicy:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(
            f"Supplementary discovery policy must be a mapping: {path}"
        )
    return SupplementaryDiscoveryPolicy.model_validate(loaded)
