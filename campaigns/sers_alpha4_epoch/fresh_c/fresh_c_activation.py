from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_acquisition import (
    HistoricalExclusionLedger,
    HistoricalLedgerSource,
    canonical_identity_from_fields,
    canonical_json,
    make_historical_exclusion_ledger,
    sha256_file,
    sha256_json,
    validate_historical_exclusion_ledger,
)


C01B_SEMANTICS_ID = "sers_fresh_c_activation_readiness_v1"
HISTORICAL_SWEEP_SEMANTICS_ID = (
    "sers_fresh_c_historical_identity_sweep_v1"
)
C00_DISPOSITION = "NEW_FRESH_ACQUISITION_REQUIRED"

EXPECTED_I0_FREEZE_ID = (
    "sers_i0_integrated_orchestration_freeze_v1:"
    "11a5fc254379f718a679"
)
EXPECTED_I0_MANIFEST_SHA256 = (
    "11a5fc254379f718a679cc8b61c168a704979d86e94ccb11617e2fa8e9d48a62"
)

EXPECTED_C01A_PROTOCOL_ID = (
    "sers_fresh_c_acquisition_protocol_preregistration_v1:"
    "c44b473a98541ac8beeb"
)
EXPECTED_C01A_PROTOCOL_SHA256 = (
    "248575dfe8b6c5510933bd5b68154561388ba1a158acd92a2faa096343210cb8"
)
EXPECTED_C01A_FREEZE_ID = (
    "sers_fresh_c_acquisition_protocol_preregistration_freeze_v1:"
    "f8423322c602383bb317"
)
EXPECTED_C01A_FREEZE_MANIFEST_SHA256 = (
    "40aa2d94ebb6155874178b668d2557fd0cfe3346d7cc31e071d76ff04b686872"
)
EXPECTED_C01A_SOURCE_COMMIT = (
    "5f367c98f1d529bd21ce2f3a5840eb8c7f5ba732"
)
EXPECTED_C01A_FREEZE_COMMIT = (
    "6f02c92b84a7e36e335b79f812b1e8803645fe12"
)

FRESH_C_TARGET_COUNT = 25
RESULTS_PER_QUERY = 100
QUERY_COUNT = 4
PROVIDER_COUNT = 2
PROVIDER_QUERY_EXECUTIONS = QUERY_COUNT * PROVIDER_COUNT
MAX_RAW_METADATA_ROWS = (
    RESULTS_PER_QUERY * PROVIDER_QUERY_EXECUTIONS
)

EXPECTED_PROVIDERS = ["semantic_scholar", "crossref"]
EXPECTED_BROAD_QUERIES = [
    "surface enhanced Raman spectroscopy gold silver",
    "SERS gold silver",
    "surface enhanced Raman spectroscopy Au Ag",
    "SERS Au Ag",
]

SWEEP_ROOTS = (
    "configs",
    "data",
    "evaluation",
    "dac_her",
    "scripts",
    "tests",
)
TEXT_EXTENSIONS = frozenset(
    {".json", ".jsonl", ".yaml", ".yml", ".py", ".md", ".txt"}
)
STRUCTURED_TITLE_KEYS = frozenset(
    {"title", "paper_title", "article_title", "source_title"}
)
STRUCTURED_DOI_KEYS = frozenset(
    {"doi", "primary_doi", "canonical_doi", "paper_doi", "source_doi"}
)
_DOI_RE = re.compile(
    r"10\.\d{4,9}/[-._;()/:A-Z0-9]+",
    re.IGNORECASE,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ScannedHistoricalFile(StrictModel):
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    identity_count: int = Field(ge=0)
    tracked_in_source_commit: bool


class HistoricalIdentitySweepManifest(StrictModel):
    schema_version: Literal[
        "sers-fresh-c-historical-identity-sweep-v1"
    ] = "sers-fresh-c-historical-identity-sweep-v1"
    semantics_id: Literal[
        "sers_fresh_c_historical_identity_sweep_v1"
    ] = HISTORICAL_SWEEP_SEMANTICS_ID
    sweep_id: str
    sweep_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    scanned_roots: list[str]
    scanned_extensions: list[str]
    fresh_c_paths_excluded: Literal[True]
    scanned_files: list[ScannedHistoricalFile]
    scanned_file_count: int = Field(ge=0)
    files_with_identities: int = Field(ge=0)
    canonical_identity_count: int = Field(ge=1)
    conservative_overexclusion_allowed: Literal[True]
    scientific_content_retained: Literal[False]
    llm_calls: Literal[0]
    network_calls: Literal[0]

    @model_validator(mode="after")
    def _consistent(self) -> "HistoricalIdentitySweepManifest":
        if self.scanned_roots != list(SWEEP_ROOTS):
            raise ValueError("Historical sweep roots drifted.")
        if self.scanned_extensions != sorted(TEXT_EXTENSIONS):
            raise ValueError("Historical sweep extensions drifted.")
        if self.scanned_file_count != len(self.scanned_files):
            raise ValueError("Historical sweep file count drifted.")
        if [row.path for row in self.scanned_files] != sorted(
            row.path for row in self.scanned_files
        ):
            raise ValueError(
                "Historical sweep files must be sorted by path."
            )
        return self


class SearchBudget(StrictModel):
    providers: list[str]
    broad_queries: list[str]
    results_per_query: Literal[100]
    provider_query_executions: Literal[8]
    max_raw_metadata_rows: Literal[800]
    budget_basis: Literal[
        "existing_provider_code_clamp_max_100_per_query"
    ]
    budget_is_scientific_acceptance_threshold: Literal[False]
    expansion_after_observing_results_allowed: Literal[False]
    insufficient_candidate_behavior: Literal[
        "fail_closed_new_protocol_epoch_required"
    ]


class TargetCountPolicy(StrictModel):
    target_acquired_papers: Literal[25]
    basis: Literal[
        "match_preexisting_reserve_a_and_reserve_b_cardinality_25"
    ]
    target_is_scientific_acceptance_threshold: Literal[False]
    target_must_not_change_after_live_discovery: Literal[True]
    inaccessible_candidate_behavior: Literal[
        "continue_next_identity_in_frozen_blind_order"
    ]
    insufficient_acquired_papers_behavior: Literal[
        "fail_closed_new_protocol_epoch_required"
    ]


class ActivationReadinessLock(StrictModel):
    schema_version: Literal[
        "sers-fresh-c-activation-readiness-lock-v1"
    ] = "sers-fresh-c-activation-readiness-lock-v1"
    semantics_id: Literal[
        "sers_fresh_c_activation_readiness_v1"
    ] = C01B_SEMANTICS_ID
    lock_id: str
    lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")

    i0_freeze_id: Literal[
        "sers_i0_integrated_orchestration_freeze_v1:"
        "11a5fc254379f718a679"
    ]
    i0_manifest_sha256: Literal[
        "11a5fc254379f718a679cc8b61c168a704979d86e94ccb11617e2fa8e9d48a62"
    ]
    c0_0_disposition: Literal["NEW_FRESH_ACQUISITION_REQUIRED"]
    c0_0_operator_attested: Literal[True]

    c0_1a_protocol_id: Literal[
        "sers_fresh_c_acquisition_protocol_preregistration_v1:"
        "c44b473a98541ac8beeb"
    ]
    c0_1a_protocol_sha256: Literal[
        "248575dfe8b6c5510933bd5b68154561388ba1a158acd92a2faa096343210cb8"
    ]
    c0_1a_freeze_id: Literal[
        "sers_fresh_c_acquisition_protocol_preregistration_freeze_v1:"
        "f8423322c602383bb317"
    ]
    c0_1a_freeze_manifest_sha256: Literal[
        "40aa2d94ebb6155874178b668d2557fd0cfe3346d7cc31e071d76ff04b686872"
    ]
    c0_1a_source_commit: Literal[
        "5f367c98f1d529bd21ce2f3a5840eb8c7f5ba732"
    ]
    c0_1a_freeze_commit: Literal[
        "6f02c92b84a7e36e335b79f812b1e8803645fe12"
    ]

    historical_sweep_manifest_path: str
    historical_sweep_manifest_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )
    historical_exclusion_ledger_path: str
    historical_exclusion_ledger_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )
    historical_canonical_identity_count: int = Field(ge=1)

    search_budget: SearchBudget
    target_count_policy: TargetCountPolicy

    fresh_c_stage_activated: Literal[False]
    live_discovery_ready: Literal[True]
    live_discovery_authorized: Literal[False]
    live_discovery_started: Literal[False]
    live_selection_started: Literal[False]
    live_acquisition_started: Literal[False]
    fresh_reserve_c_consumed: Literal[False]
    semantic_read_performed: Literal[False]
    network_calls_during_lock: Literal[0]
    llm_calls_during_lock: Literal[0]
    automatic_next_stage_authorized: Literal[False]
    stop: Literal[True]

    critical_component_sha256: dict[str, str]

    @model_validator(mode="after")
    def _exact_budget(self) -> "ActivationReadinessLock":
        if self.search_budget.providers != EXPECTED_PROVIDERS:
            raise ValueError("Fresh-C provider set drifted.")
        if self.search_budget.broad_queries != EXPECTED_BROAD_QUERIES:
            raise ValueError("Fresh-C broad queries drifted.")
        if (
            self.target_count_policy.target_acquired_papers
            != FRESH_C_TARGET_COUNT
        ):
            raise ValueError("Fresh-C target count drifted.")
        return self


def _payload_sha(
    payload: Mapping[str, Any],
    sha_field: str,
) -> str:
    value = dict(payload)
    value.pop(sha_field, None)
    return sha256_json(value)


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=root,
        text=True,
    ).strip()


def _tracked_paths_at_commit(root: Path, commit: str) -> set[str]:
    raw = _git(root, "ls-tree", "-r", "--name-only", commit)
    return set(line for line in raw.splitlines() if line)


def _is_fresh_c_path(relative: str) -> bool:
    normalized = relative.replace("\\", "/").casefold()
    return (
        "/sers_fresh_c/" in "/" + normalized
        or "fresh_c" in Path(normalized).name
        or "fresh-c" in Path(normalized).name
    )


def _iter_sweep_files(root: Path) -> Iterable[Path]:
    for raw_root in SWEEP_ROOTS:
        base = root / raw_root
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            if _is_fresh_c_path(relative):
                continue
            if path.suffix.casefold() not in TEXT_EXTENSIONS:
                continue
            yield path


def _clean_doi_match(value: str) -> str:
    return value.rstrip(".,;:)]}>'\"")


def _identity_from_doi(value: str) -> str | None:
    try:
        canonical_id, method = canonical_identity_from_fields(
            doi=value,
            title=None,
        )
    except ValueError:
        return None
    return canonical_id if method == "doi_family" else None


def _structured_identities(value: Any) -> set[str]:
    identities: set[str] = set()
    if isinstance(value, dict):
        doi_values: list[str] = []
        for key, item in value.items():
            if str(key).casefold() in STRUCTURED_DOI_KEYS:
                if isinstance(item, str) and item.strip():
                    doi_values.append(item)
        valid_doi_ids = {
            row
            for raw in doi_values
            if (row := _identity_from_doi(raw)) is not None
        }
        identities.update(valid_doi_ids)

        if not valid_doi_ids:
            for key, item in value.items():
                if (
                    str(key).casefold() in STRUCTURED_TITLE_KEYS
                    and isinstance(item, str)
                    and item.strip()
                ):
                    try:
                        canonical_id, method = (
                            canonical_identity_from_fields(
                                doi=None,
                                title=item,
                            )
                        )
                    except ValueError:
                        continue
                    if method == "normalized_title_sha256":
                        identities.add(canonical_id)

        for item in value.values():
            identities.update(_structured_identities(item))
    elif isinstance(value, list):
        for item in value:
            identities.update(_structured_identities(item))
    return identities


def extract_historical_identities(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    identities = {
        row
        for match in _DOI_RE.finditer(text)
        if (row := _identity_from_doi(
            _clean_doi_match(match.group(0))
        )) is not None
    }

    suffix = path.suffix.casefold()
    if suffix == ".json":
        try:
            identities.update(
                _structured_identities(json.loads(text))
            )
        except json.JSONDecodeError:
            pass
    elif suffix == ".jsonl":
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                identities.update(
                    _structured_identities(json.loads(line))
                )
            except json.JSONDecodeError:
                continue
    return identities


def build_historical_identity_sweep(
    *,
    root: Path,
    source_commit: str,
) -> tuple[
    HistoricalIdentitySweepManifest,
    HistoricalExclusionLedger,
]:
    tracked = _tracked_paths_at_commit(root, source_commit)
    all_ids: set[str] = set()
    file_rows: list[ScannedHistoricalFile] = []
    ledger_sources: list[HistoricalLedgerSource] = []

    for path in sorted(
        _iter_sweep_files(root),
        key=lambda row: row.relative_to(root).as_posix(),
    ):
        relative = path.relative_to(root).as_posix()
        identities = extract_historical_identities(path)
        all_ids.update(identities)
        digest = sha256_file(path)
        file_rows.append(
            ScannedHistoricalFile(
                path=relative,
                sha256=digest,
                identity_count=len(identities),
                tracked_in_source_commit=(relative in tracked),
            )
        )
        if identities:
            ledger_sources.append(
                HistoricalLedgerSource(
                    source_id=relative,
                    source_sha256=digest,
                    canonical_identity_count=len(identities),
                )
            )

    if not all_ids:
        raise ValueError(
            "Historical identity sweep found zero canonical identities; "
            "fail closed."
        )

    ledger = make_historical_exclusion_ledger(
        canonical_ids=sorted(all_ids),
        sources=ledger_sources,
    )
    manifest_body: dict[str, Any] = {
        "schema_version": (
            "sers-fresh-c-historical-identity-sweep-v1"
        ),
        "semantics_id": HISTORICAL_SWEEP_SEMANTICS_ID,
        "source_commit": source_commit,
        "scanned_roots": list(SWEEP_ROOTS),
        "scanned_extensions": sorted(TEXT_EXTENSIONS),
        "fresh_c_paths_excluded": True,
        "scanned_files": [
            row.model_dump(mode="json")
            for row in file_rows
        ],
        "scanned_file_count": len(file_rows),
        "files_with_identities": sum(
            1 for row in file_rows if row.identity_count > 0
        ),
        "canonical_identity_count": len(all_ids),
        "conservative_overexclusion_allowed": True,
        "scientific_content_retained": False,
        "llm_calls": 0,
        "network_calls": 0,
    }
    identity_sha = sha256_json(manifest_body)
    manifest_body["sweep_id"] = (
        "sers_fresh_c_historical_identity_sweep_v1:"
        + identity_sha[:20]
    )
    manifest_body["sweep_sha256"] = _payload_sha(
        manifest_body,
        "sweep_sha256",
    )
    manifest = HistoricalIdentitySweepManifest.model_validate(
        manifest_body
    )
    return manifest, ledger


def validate_historical_sweep_artifacts(
    *,
    root: Path,
    manifest_path: Path,
    ledger_path: Path,
) -> tuple[
    HistoricalIdentitySweepManifest,
    HistoricalExclusionLedger,
]:
    manifest_raw = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    manifest = HistoricalIdentitySweepManifest.model_validate(
        manifest_raw
    )
    if manifest.sweep_sha256 != _payload_sha(
        manifest_raw,
        "sweep_sha256",
    ):
        raise ValueError("Historical sweep manifest SHA drifted.")

    ledger_raw = json.loads(
        ledger_path.read_text(encoding="utf-8")
    )
    ledger = validate_historical_exclusion_ledger(ledger_raw)
    if len(ledger.canonical_ids) != manifest.canonical_identity_count:
        raise ValueError(
            "Historical ledger identity count does not match sweep."
        )

    # Validate every recorded source against the frozen source commit when
    # tracked there, otherwise against the current immutable local artifact.
    for row in manifest.scanned_files:
        if row.tracked_in_source_commit:
            try:
                content = subprocess.check_output(
                    [
                        "git",
                        "show",
                        f"{manifest.source_commit}:{row.path}",
                    ],
                    cwd=root,
                )
            except subprocess.CalledProcessError as exc:
                raise ValueError(
                    f"Historical tracked source missing: {row.path}"
                ) from exc
            observed = hashlib.sha256(content).hexdigest()
        else:
            path = root / row.path
            if not path.exists():
                raise FileNotFoundError(path)
            observed = sha256_file(path)
        if observed != row.sha256:
            raise ValueError(
                "Historical source drifted: "
                f"{row.path}: {observed} != {row.sha256}"
            )
    return manifest, ledger


def make_search_budget() -> SearchBudget:
    return SearchBudget(
        providers=EXPECTED_PROVIDERS,
        broad_queries=EXPECTED_BROAD_QUERIES,
        results_per_query=RESULTS_PER_QUERY,
        provider_query_executions=PROVIDER_QUERY_EXECUTIONS,
        max_raw_metadata_rows=MAX_RAW_METADATA_ROWS,
        budget_basis=(
            "existing_provider_code_clamp_max_100_per_query"
        ),
        budget_is_scientific_acceptance_threshold=False,
        expansion_after_observing_results_allowed=False,
        insufficient_candidate_behavior=(
            "fail_closed_new_protocol_epoch_required"
        ),
    )


def make_target_count_policy() -> TargetCountPolicy:
    return TargetCountPolicy(
        target_acquired_papers=FRESH_C_TARGET_COUNT,
        basis=(
            "match_preexisting_reserve_a_and_reserve_b_cardinality_25"
        ),
        target_is_scientific_acceptance_threshold=False,
        target_must_not_change_after_live_discovery=True,
        inaccessible_candidate_behavior=(
            "continue_next_identity_in_frozen_blind_order"
        ),
        insufficient_acquired_papers_behavior=(
            "fail_closed_new_protocol_epoch_required"
        ),
    )
