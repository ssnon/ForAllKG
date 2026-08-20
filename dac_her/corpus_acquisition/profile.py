from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from dac_her.corpus_acquisition.contracts import (
    AcquisitionProfile,
)
from pipeline_core.literature.catalog_contracts import CatalogQuery


def load_acquisition_profile(
    path: Path,
) -> AcquisitionProfile:
    loaded = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )
    if not isinstance(loaded, dict):
        raise ValueError(
            f"Acquisition profile must be a mapping: {path}"
        )
    return AcquisitionProfile.model_validate(loaded)


def _query_id(
    *,
    profile_id: str,
    axis_id: str,
    index: int,
    text: str,
) -> str:
    payload = (
        f"{profile_id}|{axis_id}|{index}|{text}"
    )
    digest = hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()[:20]
    return f"catalog_query:{digest}"


def build_catalog_queries(
    profile: AcquisitionProfile,
) -> list[CatalogQuery]:
    rows: list[CatalogQuery] = []
    for axis in profile.axes:
        for index, text in enumerate(
            axis.queries,
            start=1,
        ):
            query_text = str(text).strip()
            if not query_text:
                raise ValueError(
                    f"Empty query in axis {axis.axis_id!r}"
                )
            rows.append(
                CatalogQuery(
                    query_id=_query_id(
                        profile_id=profile.profile_id,
                        axis_id=axis.axis_id,
                        index=index,
                        text=query_text,
                    ),
                    profile_id=profile.profile_id,
                    axis_id=axis.axis_id,
                    query_text=query_text,
                )
            )
    return rows
