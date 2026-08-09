from __future__ import annotations

import hashlib
import os
from pathlib import Path

from dac_her.hypothesis_contracts import HypothesisContext, HypothesisPortfolio
from dac_her.hypothesis_gold_contracts import SemanticGoldCase, SemanticGoldSuite
from dac_her.hypothesis_real_gold_contracts import (
    HypothesisRealGoldArtifactLineage,
    HypothesisRealGoldCase,
    HypothesisRealGoldLineageCaseCheck,
    HypothesisRealGoldLineageIssue,
    HypothesisRealGoldLineagePreflightReport,
    HypothesisRealGoldSpec,
    HypothesisRealGoldSuite,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_from(base: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()


def _relative_to_output(path: Path, output_path: Path) -> str:
    return Path(
        os.path.relpath(path.resolve(), start=output_path.resolve().parent)
    ).as_posix()


def _assert_cross_artifact_lineage(
    context: HypothesisContext,
    portfolio: HypothesisPortfolio,
    *,
    case_id: str,
) -> None:
    errors: list[str] = []
    if portfolio.source_context_id != context.context_id:
        errors.append("portfolio.source_context_id does not match context.context_id")
    if portfolio.source_context_sha256 != context.context_sha256:
        errors.append(
            "portfolio.source_context_sha256 does not match context.context_sha256"
        )
    if portfolio.source_report_id != context.source_report_id:
        errors.append("portfolio.source_report_id does not match context.source_report_id")
    if portfolio.source_report_sha256 != context.source_report_sha256:
        errors.append(
            "portfolio.source_report_sha256 does not match context.source_report_sha256"
        )
    if errors:
        raise ValueError(
            f"real gold case {case_id} has inconsistent source lineage: "
            + "; ".join(errors)
        )


def build_real_gold_suite(
    spec: HypothesisRealGoldSpec,
    *,
    repo_root: Path,
    output_path: Path,
) -> HypothesisRealGoldSuite:
    repo_root = repo_root.resolve()
    output_path = output_path.resolve()
    cases: list[HypothesisRealGoldCase] = []

    for row in spec.cases:
        context_path = _resolve_from(repo_root, row.context_path)
        portfolio_path = _resolve_from(repo_root, row.portfolio_path)
        if not context_path.exists():
            raise FileNotFoundError(
                f"{row.case_id}: context file does not exist: {context_path}"
            )
        if not portfolio_path.exists():
            raise FileNotFoundError(
                f"{row.case_id}: portfolio file does not exist: {portfolio_path}"
            )

        context = HypothesisContext.model_validate_json(
            context_path.read_text(encoding="utf-8")
        )
        portfolio = HypothesisPortfolio.model_validate_json(
            portfolio_path.read_text(encoding="utf-8")
        )
        _assert_cross_artifact_lineage(context, portfolio, case_id=row.case_id)

        lineage = HypothesisRealGoldArtifactLineage(
            context_file_sha256=sha256_file(context_path),
            portfolio_file_sha256=sha256_file(portfolio_path),
            context_id=context.context_id,
            context_declared_sha256=context.context_sha256,
            portfolio_id=portfolio.portfolio_id,
            portfolio_source_context_id=portfolio.source_context_id,
            portfolio_source_context_sha256=portfolio.source_context_sha256,
            context_source_report_id=context.source_report_id,
            context_source_report_sha256=context.source_report_sha256,
            portfolio_source_report_id=portfolio.source_report_id,
            portfolio_source_report_sha256=portfolio.source_report_sha256,
        )
        cases.append(
            HypothesisRealGoldCase(
                case_id=row.case_id,
                description=row.description,
                context_path=_relative_to_output(context_path, output_path),
                portfolio_path=_relative_to_output(portfolio_path, output_path),
                lineage=lineage,
                expectations=row.expectations,
                forbid_unexpected_failures=row.forbid_unexpected_failures,
                allowed_additional_fail_dimensions=(
                    row.allowed_additional_fail_dimensions
                ),
                generator_version=row.generator_version,
                note=row.note,
            )
        )

    return HypothesisRealGoldSuite(suite_id=spec.suite_id, cases=cases)


def _issue(code: str, message: str) -> HypothesisRealGoldLineageIssue:
    return HypothesisRealGoldLineageIssue(
        code=code,  # type: ignore[arg-type]
        message=message,
    )


def validate_real_gold_lineage(
    suite: HypothesisRealGoldSuite,
    *,
    suite_path: Path,
) -> HypothesisRealGoldLineagePreflightReport:
    base = suite_path.resolve().parent
    results: list[HypothesisRealGoldLineageCaseCheck] = []

    for row in suite.cases:
        issues: list[HypothesisRealGoldLineageIssue] = []
        context_path = _resolve_from(base, row.context_path)
        portfolio_path = _resolve_from(base, row.portfolio_path)

        if not context_path.exists():
            issues.append(
                _issue(
                    "MISSING_CONTEXT_FILE",
                    f"context file does not exist: {context_path}",
                )
            )
        if not portfolio_path.exists():
            issues.append(
                _issue(
                    "MISSING_PORTFOLIO_FILE",
                    f"portfolio file does not exist: {portfolio_path}",
                )
            )
        if issues:
            results.append(
                HypothesisRealGoldLineageCaseCheck(
                    case_id=row.case_id,
                    passed=False,
                    issues=issues,
                )
            )
            continue

        if sha256_file(context_path) != row.lineage.context_file_sha256:
            issues.append(
                _issue(
                    "CONTEXT_FILE_SHA_MISMATCH",
                    "context file bytes changed after gold was built",
                )
            )
        if sha256_file(portfolio_path) != row.lineage.portfolio_file_sha256:
            issues.append(
                _issue(
                    "PORTFOLIO_FILE_SHA_MISMATCH",
                    "portfolio file bytes changed after gold was built",
                )
            )

        try:
            context = HypothesisContext.model_validate_json(
                context_path.read_text(encoding="utf-8")
            )
        except Exception as exc:
            context = None
            issues.append(
                _issue("INVALID_CONTEXT_FILE", f"context validation failed: {exc}")
            )

        try:
            portfolio = HypothesisPortfolio.model_validate_json(
                portfolio_path.read_text(encoding="utf-8")
            )
        except Exception as exc:
            portfolio = None
            issues.append(
                _issue(
                    "INVALID_PORTFOLIO_FILE",
                    f"portfolio validation failed: {exc}",
                )
            )

        if context is not None:
            checks = [
                (
                    context.context_id == row.lineage.context_id,
                    "CONTEXT_ID_MISMATCH",
                    "context.context_id differs from frozen real-gold lineage",
                ),
                (
                    context.context_sha256 == row.lineage.context_declared_sha256,
                    "CONTEXT_DECLARED_SHA_MISMATCH",
                    "context.context_sha256 differs from frozen real-gold lineage",
                ),
                (
                    context.source_report_id == row.lineage.context_source_report_id,
                    "CONTEXT_REPORT_ID_MISMATCH",
                    "context.source_report_id differs from frozen real-gold lineage",
                ),
                (
                    context.source_report_sha256
                    == row.lineage.context_source_report_sha256,
                    "CONTEXT_REPORT_SHA_MISMATCH",
                    "context.source_report_sha256 differs from frozen real-gold lineage",
                ),
            ]
            for passed, code, message in checks:
                if not passed:
                    issues.append(_issue(code, message))

        if portfolio is not None:
            checks = [
                (
                    portfolio.portfolio_id == row.lineage.portfolio_id,
                    "PORTFOLIO_ID_MISMATCH",
                    "portfolio.portfolio_id differs from frozen real-gold lineage",
                ),
                (
                    portfolio.source_context_id
                    == row.lineage.portfolio_source_context_id,
                    "PORTFOLIO_CONTEXT_ID_MISMATCH",
                    "portfolio.source_context_id differs from frozen real-gold lineage",
                ),
                (
                    portfolio.source_context_sha256
                    == row.lineage.portfolio_source_context_sha256,
                    "PORTFOLIO_CONTEXT_SHA_MISMATCH",
                    "portfolio.source_context_sha256 differs from frozen real-gold lineage",
                ),
                (
                    portfolio.source_report_id
                    == row.lineage.portfolio_source_report_id,
                    "PORTFOLIO_REPORT_ID_MISMATCH",
                    "portfolio.source_report_id differs from frozen real-gold lineage",
                ),
                (
                    portfolio.source_report_sha256
                    == row.lineage.portfolio_source_report_sha256,
                    "PORTFOLIO_REPORT_SHA_MISMATCH",
                    "portfolio.source_report_sha256 differs from frozen real-gold lineage",
                ),
            ]
            for passed, code, message in checks:
                if not passed:
                    issues.append(_issue(code, message))

        if context is not None and portfolio is not None:
            if portfolio.source_context_id != context.context_id:
                issues.append(
                    _issue(
                        "CROSS_ARTIFACT_CONTEXT_ID_MISMATCH",
                        "portfolio source_context_id does not match loaded context.context_id",
                    )
                )
            if portfolio.source_context_sha256 != context.context_sha256:
                issues.append(
                    _issue(
                        "CROSS_ARTIFACT_CONTEXT_SHA_MISMATCH",
                        "portfolio source_context_sha256 does not match loaded context.context_sha256",
                    )
                )
            if portfolio.source_report_id != context.source_report_id:
                issues.append(
                    _issue(
                        "CROSS_ARTIFACT_REPORT_ID_MISMATCH",
                        "portfolio source_report_id does not match loaded context.source_report_id",
                    )
                )
            if portfolio.source_report_sha256 != context.source_report_sha256:
                issues.append(
                    _issue(
                        "CROSS_ARTIFACT_REPORT_SHA_MISMATCH",
                        "portfolio source_report_sha256 does not match loaded context.source_report_sha256",
                    )
                )

        results.append(
            HypothesisRealGoldLineageCaseCheck(
                case_id=row.case_id,
                passed=not issues,
                issues=issues,
            )
        )

    failed = sum(not row.passed for row in results)
    return HypothesisRealGoldLineagePreflightReport(
        suite_id=suite.suite_id,
        passed=failed == 0,
        case_count=len(results),
        failed_cases=failed,
        case_results=results,
    )


def to_semantic_gold_suite(
    suite: HypothesisRealGoldSuite,
) -> SemanticGoldSuite:
    return SemanticGoldSuite(
        suite_id=suite.suite_id,
        cases=[
            SemanticGoldCase(
                case_id=row.case_id,
                description=row.description,
                context_path=row.context_path,
                portfolio_path=row.portfolio_path,
                expectations=row.expectations,
                forbid_unexpected_failures=row.forbid_unexpected_failures,
                allowed_additional_fail_dimensions=(
                    row.allowed_additional_fail_dimensions
                ),
            )
            for row in suite.cases
        ],
    )
