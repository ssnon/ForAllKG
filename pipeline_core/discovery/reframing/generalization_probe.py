from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


_REQUIRED_ARTIFACTS = (
    "explorer.packet.json",
    "explorer.report.json",
    "hypothesis.context.json",
)
_DOMAIN_KEYS = {
    "domain",
    "domain_id",
    "domain_profile",
    "domain_profile_id",
    "scientific_domain",
}
_DEFAULT_SOURCE_DOMAIN_MARKERS = (
    "sers",
    "raman",
    "lspr",
    "plasmon",
    "hotspot",
    "electromagnetic",
)

# Metadata-only aliases. These normalize domain identity for validation cohort
# bookkeeping; they do not change trigger/generation/critic semantics.
_DOMAIN_FAMILY_ALIASES = {
    "sers": "sers",
    "sers_au_ag": "sers",
    "dac_her": "dac_her",
    "catalysis_mechanism": "catalysis_mechanism",
}


class SourceDomainMarkerHit(StrictModel):
    relative_path: str = Field(min_length=1)
    marker: str = Field(min_length=1)
    line_number: int = Field(ge=1)
    line_text: str


class ValidationTaskInventoryRow(StrictModel):
    canonical_dir: str = Field(min_length=1)
    inferred_benchmark_root: str = Field(min_length=1)
    task_key: str = Field(min_length=1)
    case_key: str = Field(min_length=1)
    replicate_key: str = Field(min_length=1)

    artifact_presence: dict[str, bool]
    complete_triplet: bool
    task_id: str | None = None
    inferred_domain_labels: list[str] = Field(default_factory=list)
    canonical_domain_labels: list[str] = Field(default_factory=list)

    frozen_task_id_overlap: bool = False
    frozen_case_key_overlap: bool = False
    any_frozen_overlap: bool = False


class ValidationBenchmarkInventory(StrictModel):
    benchmark_root: str = Field(min_length=1)
    complete_task_count: int = Field(ge=0)
    incomplete_task_count: int = Field(ge=0)
    case_count: int = Field(ge=0)
    inferred_domain_labels: list[str] = Field(default_factory=list)
    canonical_domain_labels: list[str] = Field(default_factory=list)
    frozen_overlap_detected: bool = False
    domain_classification: Literal[
        "source_domain",
        "cross_domain_candidate",
        "mixed_or_ambiguous_domain",
        "domain_unknown",
    ]


class CrossDomainCandidateCohort(StrictModel):
    canonical_domain_label: str = Field(min_length=1)
    task_count: int = Field(ge=1)
    case_count: int = Field(ge=1)
    task_keys: list[str] = Field(default_factory=list)
    canonical_dirs: list[str] = Field(default_factory=list)
    inferred_benchmark_roots: list[str] = Field(default_factory=list)
    raw_domain_labels: list[str] = Field(default_factory=list)
    calibration_overlap_detected: Literal[False] = False
    complete_triplets_only: Literal[True] = True
    cohort_is_path_independent: Literal[True] = True


class ScientificReframeGeneralizationPreflight(StrictModel):
    schema_version: Literal[
        "scientific-reframe-generalization-preflight-v1"
    ] = "scientific-reframe-generalization-preflight-v1"

    preflight_id: str = Field(min_length=1)
    source_freeze_id: str = Field(min_length=1)
    calibration_domain_label: str = Field(min_length=1)
    search_root: str = Field(min_length=1)

    frozen_semantics_fingerprint: str = Field(min_length=64, max_length=64)
    current_semantics_fingerprint: str = Field(min_length=64, max_length=64)
    semantics_unchanged: bool

    source_domain_markers: list[str] = Field(default_factory=list)
    source_domain_marker_hits: list[SourceDomainMarkerHit] = Field(default_factory=list)
    source_domain_marker_hit_count: int = Field(ge=0)
    marker_hit_counts: dict[str, int] = Field(default_factory=dict)
    marker_hits_are_static_diagnostics_only: Literal[True] = True
    marker_hits_prove_runtime_dependency: Literal[False] = False

    discovered_task_materialization_count: int = Field(ge=0)
    complete_triplet_count: int = Field(ge=0)
    incomplete_materialization_count: int = Field(ge=0)
    task_rows: list[ValidationTaskInventoryRow] = Field(default_factory=list)
    benchmark_inventories: list[ValidationBenchmarkInventory] = Field(default_factory=list)
    cross_domain_candidate_roots: list[str] = Field(default_factory=list)
    cross_domain_candidate_cohorts: list[CrossDomainCandidateCohort] = Field(default_factory=list)
    unclassified_candidate_roots: list[str] = Field(default_factory=list)

    calibration_overlap_detected: bool = False
    llm_calls_performed: Literal[0] = 0
    frozen_semantics_modified: Literal[False] = False
    scientific_authority: Literal[False] = False
    generalization_claim_established: Literal[False] = False
    threshold_tuning_performed: Literal[False] = False
    prompt_tuning_performed: Literal[False] = False
    production_selection_changed: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(self) -> "ScientificReframeGeneralizationPreflight":
        if self.discovered_task_materialization_count != len(self.task_rows):
            raise ValueError("discovered task count must equal task row count")
        if self.complete_triplet_count != sum(row.complete_triplet for row in self.task_rows):
            raise ValueError("complete triplet count does not match task rows")
        if self.incomplete_materialization_count != sum(
            not row.complete_triplet for row in self.task_rows
        ):
            raise ValueError("incomplete materialization count does not match task rows")
        if self.source_domain_marker_hit_count != len(self.source_domain_marker_hits):
            raise ValueError("source-domain marker hit count does not match hit rows")
        return self


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _stable_id(prefix: str, payload: object) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]
    return f"{prefix}:{digest}"


def _walk_values(value: Any, *, key_name: str | None = None) -> list[tuple[str | None, Any]]:
    out: list[tuple[str | None, Any]] = [(key_name, value)]
    if isinstance(value, dict):
        for key, child in value.items():
            out.extend(_walk_values(child, key_name=str(key)))
    elif isinstance(value, list):
        for child in value:
            out.extend(_walk_values(child, key_name=key_name))
    return out


def _extract_task_id(objects: list[dict[str, Any]]) -> str | None:
    values: set[str] = set()
    for obj in objects:
        for key, value in _walk_values(obj):
            if key == "task_id" and isinstance(value, str) and value.strip():
                values.add(value.strip())
    if len(values) == 1:
        return next(iter(values))
    return None


def _extract_domain_labels(objects: list[dict[str, Any]]) -> list[str]:
    labels: set[str] = set()
    for obj in objects:
        for key, value in _walk_values(obj):
            if key in _DOMAIN_KEYS and isinstance(value, str) and value.strip():
                labels.add(value.strip())
    return sorted(labels)


def _normalize_domain_label(value: str) -> str:
    token = value.strip().casefold().replace("-", "_").replace(" ", "_")
    while "__" in token:
        token = token.replace("__", "_")
    return _DOMAIN_FAMILY_ALIASES.get(token, token)


def _canonical_domain_labels(labels: list[str]) -> list[str]:
    return sorted({_normalize_domain_label(value) for value in labels if value.strip()})


def _infer_path_identity(search_root: Path, canonical_dir: Path) -> tuple[Path, str, str, str]:
    # Some legacy/cross-domain evaluation materializations place the canonical
    # packet/report/context triplet directly in a task directory rather than in
    # a trailing ``canonical/`` directory. Treat that directory itself as the
    # task identity instead of accidentally collapsing it to its parent root.
    if canonical_dir.name != "canonical":
        benchmark_root = canonical_dir.parent
        task_key = canonical_dir.name
        return benchmark_root, task_key, canonical_dir.name, "default"

    parent = canonical_dir.parent
    if parent.name.startswith("replicate_") and parent.parent != canonical_dir:
        case_dir = parent.parent
        benchmark_root = case_dir.parent
        task_key = f"{case_dir.name}/{parent.name}"
        return benchmark_root, task_key, case_dir.name, parent.name

    relative = parent.relative_to(search_root) if parent != search_root else Path(parent.name)
    parts = relative.parts
    case_key = parts[-1] if parts else parent.name
    task_key = relative.as_posix() if parts else case_key
    return search_root, task_key, case_key, "default"


def _discover_canonical_dirs(search_root: Path) -> list[Path]:
    parents: set[Path] = set()
    for name in _REQUIRED_ARTIFACTS:
        for path in search_root.rglob(name):
            if path.is_file():
                parents.add(path.parent.resolve())
    return sorted(parents, key=lambda path: str(path))


def _current_semantics_from_freeze(
    *,
    freeze: dict[str, Any],
    repository_root: Path,
) -> tuple[str, dict[str, str]]:
    frozen_hashes = freeze.get("semantics_file_sha256")
    if not isinstance(frozen_hashes, dict) or not frozen_hashes:
        raise ValueError("freeze is missing semantics_file_sha256")

    current: dict[str, str] = {}
    for relative in sorted(str(key) for key in frozen_hashes):
        path = repository_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"frozen semantics file is missing: {path}")
        current[relative] = _sha256_bytes(path.read_bytes())
    payload = json.dumps(current, sort_keys=True, separators=(",", ":"))
    return _sha256_bytes(payload.encode("utf-8")), current


def _scan_source_markers(
    *,
    repository_root: Path,
    relative_paths: list[str],
    markers: list[str],
) -> list[SourceDomainMarkerHit]:
    hits: list[SourceDomainMarkerHit] = []
    normalized = [(marker, marker.casefold()) for marker in markers if marker.strip()]
    for relative in sorted(relative_paths):
        path = repository_root / relative
        text = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            folded = line.casefold()
            for marker, marker_folded in normalized:
                if marker_folded in folded:
                    hits.append(
                        SourceDomainMarkerHit(
                            relative_path=relative,
                            marker=marker,
                            line_number=line_number,
                            line_text=line.strip()[:500],
                        )
                    )
    return hits


def build_generalization_preflight(
    *,
    freeze_path: str | Path,
    search_root: str | Path,
    repository_root: str | Path,
    source_domain_markers: list[str] | None = None,
) -> ScientificReframeGeneralizationPreflight:
    freeze_path = Path(freeze_path).resolve()
    search_root = Path(search_root).resolve()
    repository_root = Path(repository_root).resolve()
    freeze = _load_object(freeze_path)

    if freeze.get("schema_version") != "scientific-reframe-calibration-freeze-v1":
        raise ValueError("unsupported calibration freeze schema")
    freeze_id = str(freeze.get("freeze_id") or "").strip()
    calibration_domain_raw = str(freeze.get("calibration_domain_label") or "").strip()
    calibration_domain = _normalize_domain_label(calibration_domain_raw) if calibration_domain_raw else ""
    frozen_fingerprint = str(freeze.get("semantics_fingerprint") or "").strip()
    if not freeze_id or not calibration_domain or len(frozen_fingerprint) != 64:
        raise ValueError("freeze is missing required identity/fingerprint fields")

    current_fingerprint, current_hashes = _current_semantics_from_freeze(
        freeze=freeze,
        repository_root=repository_root,
    )
    markers = list(source_domain_markers or _DEFAULT_SOURCE_DOMAIN_MARKERS)
    marker_hits = _scan_source_markers(
        repository_root=repository_root,
        relative_paths=list(current_hashes),
        markers=markers,
    )

    frozen_task_ids = set(str(value) for value in freeze.get("task_ids", []) if value)
    frozen_case_keys = set(str(value) for value in freeze.get("case_keys", []) if value)

    rows: list[ValidationTaskInventoryRow] = []
    for canonical_dir in _discover_canonical_dirs(search_root):
        benchmark_root, task_key, case_key, replicate_key = _infer_path_identity(
            search_root, canonical_dir
        )
        presence = {
            name: (canonical_dir / name).is_file()
            for name in _REQUIRED_ARTIFACTS
        }
        objects: list[dict[str, Any]] = []
        for name, present in presence.items():
            if present:
                try:
                    objects.append(_load_object(canonical_dir / name))
                except (OSError, ValueError, json.JSONDecodeError):
                    pass
        task_id = _extract_task_id(objects)
        domains = _extract_domain_labels(objects)
        canonical_domains = _canonical_domain_labels(domains)
        overlap_task = bool(task_id and task_id in frozen_task_ids)
        overlap_case = case_key in frozen_case_keys
        rows.append(
            ValidationTaskInventoryRow(
                canonical_dir=str(canonical_dir),
                inferred_benchmark_root=str(benchmark_root.resolve()),
                task_key=task_key,
                case_key=case_key,
                replicate_key=replicate_key,
                artifact_presence=presence,
                complete_triplet=all(presence.values()),
                task_id=task_id,
                inferred_domain_labels=domains,
                canonical_domain_labels=canonical_domains,
                frozen_task_id_overlap=overlap_task,
                frozen_case_key_overlap=overlap_case,
                any_frozen_overlap=overlap_task or overlap_case,
            )
        )

    grouped: dict[str, list[ValidationTaskInventoryRow]] = defaultdict(list)
    for row in rows:
        grouped[row.inferred_benchmark_root].append(row)

    inventories: list[ValidationBenchmarkInventory] = []
    cross_domain_roots: list[str] = []
    unclassified_roots: list[str] = []
    for root, group_rows in sorted(grouped.items()):
        complete_count = sum(row.complete_triplet for row in group_rows)
        incomplete_count = len(group_rows) - complete_count
        domains = sorted(
            {
                domain
                for row in group_rows
                for domain in row.inferred_domain_labels
            }
        )
        canonical_domains = sorted(
            {
                domain
                for row in group_rows
                for domain in row.canonical_domain_labels
            }
        )
        overlap = any(row.any_frozen_overlap for row in group_rows)
        if not canonical_domains:
            classification = "domain_unknown"
            if complete_count and not overlap:
                unclassified_roots.append(root)
        elif len(canonical_domains) > 1:
            classification = "mixed_or_ambiguous_domain"
        elif canonical_domains[0] == calibration_domain:
            classification = "source_domain"
        else:
            classification = "cross_domain_candidate"
            if complete_count and not overlap:
                cross_domain_roots.append(root)
        inventories.append(
            ValidationBenchmarkInventory(
                benchmark_root=root,
                complete_task_count=complete_count,
                incomplete_task_count=incomplete_count,
                case_count=len({row.case_key for row in group_rows}),
                inferred_domain_labels=domains,
                canonical_domain_labels=canonical_domains,
                frozen_overlap_detected=overlap,
                domain_classification=classification,
            )
        )

    # Cross-domain cohorts are selected by canonical domain identity rather than
    # directory layout. This prevents mixed evaluation roots from hiding usable
    # validation tasks and prevents source-domain aliases from appearing novel.
    cohort_rows: dict[str, list[ValidationTaskInventoryRow]] = defaultdict(list)
    for row in rows:
        if not row.complete_triplet or row.any_frozen_overlap:
            continue
        if len(row.canonical_domain_labels) != 1:
            continue
        domain = row.canonical_domain_labels[0]
        if domain == calibration_domain:
            continue
        cohort_rows[domain].append(row)

    cross_domain_cohorts: list[CrossDomainCandidateCohort] = []
    for domain, domain_rows in sorted(cohort_rows.items()):
        cross_domain_cohorts.append(
            CrossDomainCandidateCohort(
                canonical_domain_label=domain,
                task_count=len(domain_rows),
                case_count=len({row.case_key for row in domain_rows}),
                task_keys=sorted(row.task_key for row in domain_rows),
                canonical_dirs=sorted(row.canonical_dir for row in domain_rows),
                inferred_benchmark_roots=sorted({row.inferred_benchmark_root for row in domain_rows}),
                raw_domain_labels=sorted(
                    {label for row in domain_rows for label in row.inferred_domain_labels}
                ),
            )
        )

    marker_counts = Counter(hit.marker for hit in marker_hits)
    payload = {
        "freeze_id": freeze_id,
        "search_root": str(search_root),
        "current_semantics_fingerprint": current_fingerprint,
        "markers": markers,
        "task_rows": [row.model_dump(mode="json") for row in rows],
    }
    return ScientificReframeGeneralizationPreflight(
        preflight_id=_stable_id("scientific_reframe_generalization_preflight", payload),
        source_freeze_id=freeze_id,
        calibration_domain_label=calibration_domain,
        search_root=str(search_root),
        frozen_semantics_fingerprint=frozen_fingerprint,
        current_semantics_fingerprint=current_fingerprint,
        semantics_unchanged=(current_fingerprint == frozen_fingerprint),
        source_domain_markers=markers,
        source_domain_marker_hits=marker_hits,
        source_domain_marker_hit_count=len(marker_hits),
        marker_hit_counts=dict(sorted(marker_counts.items())),
        discovered_task_materialization_count=len(rows),
        complete_triplet_count=sum(row.complete_triplet for row in rows),
        incomplete_materialization_count=sum(not row.complete_triplet for row in rows),
        task_rows=rows,
        benchmark_inventories=inventories,
        cross_domain_candidate_roots=sorted(cross_domain_roots),
        cross_domain_candidate_cohorts=cross_domain_cohorts,
        unclassified_candidate_roots=sorted(unclassified_roots),
        calibration_overlap_detected=any(row.any_frozen_overlap for row in rows),
    )
