from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from dac_her.external_novelty_contracts import (
    LiteratureQueryPlan,
    PriorArtPacket,
)


PROVIDER_FAILURE_TAXONOMY_SEMANTICS_ID = (
    "sers_provider_failure_taxonomy_v1"
)
EXPECTED_CLEAN_BRANCH = "feat/SERS-clean-next"
EXPECTED_CLEAN_HEAD = (
    "cdbba2eff2d9f59bfdddd0b28373adf6e0904b00"
)
EXPECTED_DIAGNOSTIC_BRANCH = (
    "diag/SERS-downstream-audit-5l-20260816"
)
EXPECTED_DIAGNOSTIC_COMMIT = (
    "7d69807"
)

DEFAULT_DIAGNOSTIC_ROOT = Path.home() / "GraphAgentsDAC"
DEFAULT_OUTPUT_ROOT = Path(
    "evaluation/sers_provider_reliability/"
    "provider_failure_taxonomy_v1"
)

BASELINE_QUERY_PLAN = Path(
    "evaluation/sers_alpha4c5k/dev_e2e_v2/"
    "external_novelty.claims_queries.json"
)
BASELINE_PRIOR_ART = Path(
    "evaluation/sers_alpha4c5k/dev_e2e_v2/"
    "external_novelty.prior_art.json"
)
PROBE_SUMMARY = Path(
    "evaluation/sers_alpha4c5k5d/one_shot_probe_v1/"
    "probe_summary.json"
)
PROBE_RUN_MANIFEST = Path(
    "evaluation/sers_alpha4c5k5d/one_shot_probe_v1/"
    "run_manifest.json"
)

_HTTP_STATUS_RE = re.compile(
    r"(?:HTTP(?:Error)?|HTTP Error)\s*(?:[: ]\s*)?(\d{3})",
    re.IGNORECASE,
)
_STATUS_CODE_RE = re.compile(
    r"\b(?:status|code)\s*[=: ]+\s*(\d{3})\b",
    re.IGNORECASE,
)


def canonical_json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        canonical_json(value).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(
            dict(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(value, encoding="utf-8")
    tmp.replace(path)


def extract_http_status(error: str | None) -> int | None:
    text = str(error or "")
    for pattern in (_HTTP_STATUS_RE, _STATUS_CODE_RE):
        match = pattern.search(text)
        if match:
            code = int(match.group(1))
            if 100 <= code <= 599:
                return code
    return None


def classify_failure(error: str | None) -> dict[str, Any]:
    text = str(error or "").strip()
    lower = text.lower()
    status = extract_http_status(text)

    if status == 429:
        category = "HTTP_429_RATE_LIMIT"
        retryability = "TRANSIENT_OR_QUOTA_DEPENDENT"
    elif status == 401:
        category = "HTTP_401_AUTHENTICATION"
        retryability = "CONFIGURATION_OR_CREDENTIAL"
    elif status == 403:
        category = "HTTP_403_AUTHORIZATION"
        retryability = "CONFIGURATION_OR_PROVIDER_POLICY"
    elif status is not None and 500 <= status <= 599:
        category = "HTTP_5XX_UPSTREAM"
        retryability = "TRANSIENT_PROVIDER"
    elif status is not None:
        category = "HTTP_OTHER"
        retryability = "STATUS_SPECIFIC"
    elif (
        "timed out" in lower
        or "timeout" in lower
        or "timeouterror" in lower
    ):
        category = "TIMEOUT"
        retryability = "TRANSIENT_TRANSPORT"
    elif (
        "urlerror" in lower
        or "connection reset" in lower
        or "connection refused" in lower
        or "temporary failure" in lower
        or "name or service not known" in lower
        or "remote end closed" in lower
    ):
        category = "TRANSPORT_ERROR"
        retryability = "TRANSIENT_OR_NETWORK"
    elif (
        "jsondecodeerror" in lower
        or "expecting value" in lower
        or "invalid json" in lower
    ):
        category = "PAYLOAD_JSON_DECODE_ERROR"
        retryability = "PROVIDER_OR_PARSER"
    elif not text:
        category = "MISSING_ERROR_DETAIL"
        retryability = "UNKNOWN"
    else:
        category = "OTHER_EXCEPTION"
        retryability = "UNKNOWN"

    exception_type = (
        text.split(":", 1)[0].strip()
        if ":" in text
        else text.split(" ", 1)[0].strip()
    )
    return {
        "category": category,
        "http_status": status,
        "exception_type": exception_type or None,
        "retryability_class": retryability,
        "error_text_sha256": hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest(),
        "error_text": text,
    }


def _source_paths(
    diagnostic_root: Path,
) -> dict[str, Path]:
    result = {
        "baseline_query_plan":
            diagnostic_root / BASELINE_QUERY_PLAN,
        "baseline_prior_art":
            diagnostic_root / BASELINE_PRIOR_ART,
        "probe_summary":
            diagnostic_root / PROBE_SUMMARY,
        "probe_run_manifest":
            diagnostic_root / PROBE_RUN_MANIFEST,
    }
    for name, path in result.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"Diagnostic source missing ({name}): {path}"
            )
    return result


def source_bindings(
    diagnostic_root: Path,
) -> dict[str, dict[str, Any]]:
    result = {}
    for name, path in sorted(
        _source_paths(diagnostic_root).items()
    ):
        result[name] = {
            "path": str(
                path.resolve().relative_to(
                    diagnostic_root.resolve()
                )
            ),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    return result


def verify_source_bindings(
    diagnostic_root: Path,
    bindings: Mapping[str, Any],
) -> list[str]:
    issues = []
    for name, row in bindings.items():
        if not isinstance(row, Mapping):
            issues.append(f"invalid source binding: {name}")
            continue
        path = diagnostic_root / str(
            row.get("path") or ""
        )
        if not path.is_file():
            issues.append(f"source missing: {name}")
            continue
        if sha256_file(path) != str(
            row.get("sha256") or ""
        ):
            issues.append(f"source drift: {name}")
    return issues


def build_baseline_taxonomy(
    *,
    plan: LiteratureQueryPlan,
    packet: PriorArtPacket,
) -> dict[str, Any]:
    if packet.source_query_plan_id != plan.plan_id:
        raise ValueError(
            "Baseline query-plan / prior-art mismatch."
        )

    query_index = {
        query.query_id: query
        for query in plan.queries
    }

    provider_rows: dict[
        str,
        dict[str, Any],
    ] = {}
    by_provider_counts: dict[
        str,
        Counter,
    ] = defaultdict(Counter)
    by_provider_categories: dict[
        str,
        Counter,
    ] = defaultdict(Counter)
    by_provider_exception_types: dict[
        str,
        Counter,
    ] = defaultdict(Counter)
    by_query_kind: dict[
        str,
        Counter,
    ] = defaultdict(Counter)
    by_hypothesis: dict[
        str,
        Counter,
    ] = defaultdict(Counter)

    failures = []
    successful_pairs = set()

    for execution in packet.executions:
        provider = execution.provider
        counts = by_provider_counts[provider]
        counts["execution_count"] += 1
        counts["raw_result_count"] += int(
            execution.result_count
        )

        query = query_index.get(
            execution.query_id
        )
        query_kind = (
            query.query_kind
            if query is not None
            else "UNKNOWN_QUERY_KIND"
        )
        hypothesis_id = (
            query.hypothesis_id
            if query is not None
            else "UNKNOWN_HYPOTHESIS"
        )

        if execution.success:
            counts[
                "successful_execution_count"
            ] += 1
            successful_pairs.add(
                (
                    execution.query_id,
                    provider,
                )
            )
            by_query_kind[query_kind][
                f"{provider}:success"
            ] += 1
            by_hypothesis[hypothesis_id][
                f"{provider}:success"
            ] += 1
            continue

        counts[
            "failed_execution_count"
        ] += 1
        classified = classify_failure(
            execution.error
        )
        category = classified["category"]
        exception_type = (
            classified[
                "exception_type"
            ]
            or "UNKNOWN"
        )
        by_provider_categories[
            provider
        ][category] += 1
        by_provider_exception_types[
            provider
        ][exception_type] += 1
        by_query_kind[query_kind][
            f"{provider}:failure"
        ] += 1
        by_hypothesis[hypothesis_id][
            f"{provider}:failure"
        ] += 1

        failures.append({
            "query_id":
                execution.query_id,
            "query_kind":
                query_kind,
            "hypothesis_id":
                hypothesis_id,
            "claim_id":
                (
                    query.claim_id
                    if query is not None
                    else None
                ),
            "provider":
                provider,
            "elapsed_seconds":
                execution.elapsed_seconds,
            **classified,
        })

    for provider in sorted(
        by_provider_counts
    ):
        counts = by_provider_counts[
            provider
        ]
        executions = int(
            counts["execution_count"]
        )
        failures_count = int(
            counts["failed_execution_count"]
        )
        provider_rows[provider] = {
            **dict(
                sorted(counts.items())
            ),
            "failure_rate":
                (
                    failures_count
                    / executions
                    if executions
                    else None
                ),
            "failure_categories":
                dict(
                    sorted(
                        by_provider_categories[
                            provider
                        ].items()
                    )
                ),
            "exception_types":
                dict(
                    sorted(
                        by_provider_exception_types[
                            provider
                        ].items()
                    )
                ),
        }

    failure_hypotheses = defaultdict(set)
    failure_queries = defaultdict(set)
    for row in failures:
        failure_hypotheses[
            row["provider"]
        ].add(row["hypothesis_id"])
        failure_queries[
            row["provider"]
        ].add(row["query_id"])

    return {
        "provider_summary":
            provider_rows,
        "failure_count":
            len(failures),
        "failures":
            failures,
        "provider_failure_recurrence": {
            provider: {
                "affected_hypothesis_count":
                    len(
                        failure_hypotheses[
                            provider
                        ]
                    ),
                "affected_hypothesis_ids":
                    sorted(
                        failure_hypotheses[
                            provider
                        ]
                    ),
                "affected_query_count":
                    len(
                        failure_queries[
                            provider
                        ]
                    ),
                "affected_query_ids":
                    sorted(
                        failure_queries[
                            provider
                        ]
                    ),
            }
            for provider in sorted(
                failure_hypotheses
            )
        },
        "by_query_kind": {
            kind: dict(
                sorted(counts.items())
            )
            for kind, counts in sorted(
                by_query_kind.items()
            )
        },
        "by_hypothesis": {
            hypothesis: dict(
                sorted(counts.items())
            )
            for hypothesis, counts in sorted(
                by_hypothesis.items()
            )
        },
        "successful_query_provider_pair_count":
            len(successful_pairs),
    }


def _probe_context(
    diagnostic_root: Path,
) -> dict[str, Any]:
    paths = _source_paths(
        diagnostic_root
    )
    summary = read_json(
        paths["probe_summary"]
    )
    run = read_json(
        paths["probe_run_manifest"]
    )

    failures = []
    for row in run.get(
        "request_log",
        []
    ):
        if row.get("success"):
            continue
        message = (
            f"{row.get('error_type') or ''}: "
            f"{row.get('error_message') or ''}"
        ).strip(": ")
        classified = classify_failure(
            message
        )
        # Structured status wins if present.
        status = row.get("http_status")
        if isinstance(status, int):
            synthetic = classify_failure(
                f"HTTP Error {status}: "
                f"{row.get('error_message') or ''}"
            )
            classified["category"] = (
                synthetic["category"]
            )
            classified["http_status"] = status
            classified["retryability_class"] = (
                synthetic[
                    "retryability_class"
                ]
            )

        failures.append({
            "query_id": row.get(
                "query_id"
            ),
            "provider": row.get(
                "provider"
            ),
            "http_status": classified[
                "http_status"
            ],
            "category":
                classified["category"],
            "exception_type":
                classified[
                    "exception_type"
                ],
            "retryability_class":
                classified[
                    "retryability_class"
                ],
            "error_text_sha256":
                classified[
                    "error_text_sha256"
                ],
        })

    return {
        "denominator_status":
            "CONDITIONAL_FOLLOWUP_NOT_BASELINE_PREVALENCE",
        "primary_diagnosis":
            summary.get(
                "primary_diagnosis"
            ),
        "successful_logical_execution_count":
            summary.get(
                "successful_logical_execution_count"
            ),
        "expected_logical_execution_count":
            summary.get(
                "expected_logical_execution_count"
            ),
        "failures": failures,
        "failure_category_counts":
            dict(
                sorted(
                    Counter(
                        row["category"]
                        for row in failures
                    ).items()
                )
            ),
    }


def _semantic_scholar_conclusion(
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    provider = baseline[
        "provider_summary"
    ].get(
        "semantic_scholar",
        {},
    )
    categories = provider.get(
        "failure_categories",
        {},
    )
    failures = int(
        provider.get(
            "failed_execution_count",
            0,
        )
    )
    executions = int(
        provider.get(
            "execution_count",
            0,
        )
    )

    dominant_category = None
    dominant_count = 0
    if categories:
        dominant_category, dominant_count = max(
            categories.items(),
            key=lambda item: (
                int(item[1]),
                item[0],
            ),
        )

    dominance_fraction = (
        dominant_count / failures
        if failures
        else None
    )
    recurrence = baseline[
        "provider_failure_recurrence"
    ].get(
        "semantic_scholar",
        {},
    )

    if (
        failures > 0
        and recurrence.get(
            "affected_hypothesis_count",
            0,
        ) >= 2
    ):
        generality = (
            "CROSS_HYPOTHESIS_PROVIDER_FAILURE"
        )
    elif failures > 0:
        generality = (
            "LOCAL_PROVIDER_FAILURE"
        )
    else:
        generality = (
            "NO_PROVIDER_FAILURE"
        )

    if (
        dominance_fraction is not None
        and dominance_fraction >= 0.8
    ):
        failure_shape = (
            "DOMINANT_SINGLE_FAILURE_CLASS"
        )
    elif failures:
        failure_shape = (
            "MIXED_FAILURE_CLASSES"
        )
    else:
        failure_shape = (
            "NO_FAILURE"
        )

    return {
        "execution_count": executions,
        "failed_execution_count":
            failures,
        "failure_rate":
            (
                failures / executions
                if executions
                else None
            ),
        "generality":
            generality,
        "dominant_failure_category":
            dominant_category,
        "dominant_failure_count":
            dominant_count,
        "dominant_failure_fraction":
            dominance_fraction,
        "failure_shape":
            failure_shape,
        "generic_provider_hardening_justified":
            (
                generality
                == "CROSS_HYPOTHESIS_PROVIDER_FAILURE"
            ),
        "specific_retry_policy_change_authorized":
            False,
        "specific_auth_policy_change_authorized":
            False,
        "specific_rate_limit_policy_change_authorized":
            False,
        "reason":
            (
                "This audit identifies the recurrent failure class. "
                "A behavioral provider-layer change should be selected "
                "only after reviewing this taxonomy."
            ),
    }


def build_audit(
    *,
    diagnostic_root: Path,
) -> dict[str, Any]:
    paths = _source_paths(
        diagnostic_root
    )
    plan = (
        LiteratureQueryPlan.
        model_validate_json(
            paths[
                "baseline_query_plan"
            ].read_text(
                encoding="utf-8"
            )
        )
    )
    packet = (
        PriorArtPacket.
        model_validate_json(
            paths[
                "baseline_prior_art"
            ].read_text(
                encoding="utf-8"
            )
        )
    )

    baseline = build_baseline_taxonomy(
        plan=plan,
        packet=packet,
    )
    probe = _probe_context(
        diagnostic_root
    )
    s2 = _semantic_scholar_conclusion(
        baseline
    )

    body: dict[str, Any] = {
        "schema_version":
            "sers-provider-failure-taxonomy-v1",
        "semantics_id":
            PROVIDER_FAILURE_TAXONOMY_SEMANTICS_ID,
        "scope": {
            "primary_prevalence_source":
                "alpha4c.5k initial external-novelty retrieval",
            "conditional_probe_used_as_primary_prevalence":
                False,
            "diagnostic_root":
                str(
                    diagnostic_root.resolve()
                ),
        },
        "source_artifact_bindings":
            source_bindings(
                diagnostic_root
            ),
        "baseline_taxonomy":
            baseline,
        "conditional_probe_context":
            probe,
        "semantic_scholar_conclusion":
            s2,
        "behavioral_changes": {
            "retrieval_code_modified":
                False,
            "retry_policy_modified":
                False,
            "authentication_policy_modified":
                False,
            "request_pacing_modified":
                False,
            "query_policy_modified":
                False,
        },
        "safety": {
            "network_searches": 0,
            "llm_calls": 0,
            "source_artifacts_modified":
                False,
            "scientific_semantics_modified":
                False,
            "fresh_reserve_used": False,
        },
    }
    body["audit_sha256"] = (
        sha256_json(body)
    )
    body["audit_id"] = (
        "sers_provider_failure_taxonomy:"
        + body["audit_sha256"][:20]
    )
    return body


def render_markdown(
    audit: Mapping[str, Any],
) -> str:
    baseline = audit[
        "baseline_taxonomy"
    ]
    conclusion = audit[
        "semantic_scholar_conclusion"
    ]
    lines = [
        "# SERS Provider Failure Taxonomy",
        "",
        f"- Audit ID: `{audit['audit_id']}`",
        f"- Audit SHA256: `{audit['audit_sha256']}`",
        "- Network searches: `0`",
        "- LLM calls: `0`",
        "",
        "## Baseline provider census",
        "",
        "```json",
        json.dumps(
            baseline[
                "provider_summary"
            ],
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        "```",
        "",
        "## Semantic Scholar conclusion",
        "",
        (
            "- Executions / failures: "
            f"{conclusion['execution_count']} / "
            f"{conclusion['failed_execution_count']}"
        ),
        (
            "- Failure rate: "
            f"{conclusion['failure_rate']}"
        ),
        (
            "- Generality: "
            f"`{conclusion['generality']}`"
        ),
        (
            "- Dominant category: "
            f"`{conclusion['dominant_failure_category']}` "
            f"({conclusion['dominant_failure_count']})"
        ),
        (
            "- Failure shape: "
            f"`{conclusion['failure_shape']}`"
        ),
        "",
        "## Conditional 5k.5d probe",
        "",
        (
            "This is retained as corroborating context only and "
            "does not increase the baseline prevalence denominator."
        ),
        "",
        "```json",
        json.dumps(
            audit[
                "conditional_probe_context"
            ],
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        "```",
        "",
        "## Behavioral status",
        "",
        "- Retrieval behavior changed: `False`",
        "- Retry policy changed: `False`",
        "- Authentication policy changed: `False`",
        "- Request pacing changed: `False`",
        "",
    ]
    return "\n".join(lines)


def verify_audit(
    *,
    audit_path: Path,
    diagnostic_root: Path,
) -> tuple[list[str], dict[str, Any]]:
    if not audit_path.is_file():
        return ["audit artifact missing"], {}

    stored = read_json(audit_path)
    issues = []

    if (
        stored.get("semantics_id")
        != PROVIDER_FAILURE_TAXONOMY_SEMANTICS_ID
    ):
        issues.append(
            "semantics mismatch"
        )

    body = dict(stored)
    stored_id = body.pop(
        "audit_id",
        None,
    )
    stored_sha = body.pop(
        "audit_sha256",
        None,
    )
    observed_sha = sha256_json(
        body
    )
    if stored_sha != observed_sha:
        issues.append(
            "audit SHA mismatch"
        )
    if stored_id != (
        "sers_provider_failure_taxonomy:"
        + observed_sha[:20]
    ):
        issues.append(
            "audit ID mismatch"
        )

    bindings = stored.get(
        "source_artifact_bindings",
        {},
    )
    if isinstance(bindings, Mapping):
        issues.extend(
            verify_source_bindings(
                diagnostic_root,
                bindings,
            )
        )
    else:
        issues.append(
            "source bindings missing"
        )

    try:
        recomputed = build_audit(
            diagnostic_root=
                diagnostic_root
        )
        left = dict(stored)
        right = dict(recomputed)
        for key in (
            "audit_id",
            "audit_sha256",
        ):
            left.pop(key, None)
            right.pop(key, None)
        if canonical_json(
            left
        ) != canonical_json(
            right
        ):
            issues.append(
                "deterministic recomputation mismatch"
            )
    except Exception as exc:
        issues.append(
            "recomputation failed: "
            f"{type(exc).__name__}: {exc}"
        )

    return sorted(set(issues)), stored
