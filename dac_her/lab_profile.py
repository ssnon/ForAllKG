from __future__ import annotations

from pathlib import Path

import yaml

from dac_her.experimental_contracts import LabProfile


def load_lab_profile(path: str | Path) -> LabProfile:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("lab profile YAML must contain a mapping")
    return LabProfile.model_validate(payload)
