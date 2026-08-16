from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from dac_her.alpha4c5i_dev_compatibility import (
    EXPECTED_DEV_COUNT,
    EXPECTED_TREND_SEMANTICS_ID,
    load_exact_dev_paper_ids,
    verify_5i_component_hashes,
    verify_frozen_postmortem,
    verify_h1_dev_summary,
)
from dac_her.explorer_contracts import (
    ExplorationReport,
    GraphExplorerPacket,
)
from dac_her.hypothesis_context import HypothesisContextBuilder
from dac_her.hypothesis_contracts import HypothesisContext
from dac_her.hypothesis_trend_grounding import (
    HypothesisTrendGroundingBundle,
)
from dac_her.hypothesis_trend_input import (
    TrendAwareHypothesisInput,
    build_trend_aware_hypothesis_input,
    load_trend_corpus_binding,
    validate_hypothesis_context_sha,
    validate_trend_grounding_bundle_sha,
    verify_trend_aware_input_sources,
)


ALPHA4C5I_DEV_INPUT_BUILDER_SEMANTICS_ID = (
    "sers_alpha4c5i_dev_trend_input_builder_v1"
)
ALPHA4C5I_DEV_INPUT_SEMANTICS_ID = (
    "sers_alpha4c5i_dev_trend_aware_input_v1"
)

DEFAULT_H1_DEV_ROOT = Path(
    "evaluation/sers_alpha4c5h1/dev_compat_v1"
)
DEFAULT_GROUNDING = (
    DEFAULT_H1_DEV_ROOT / "trend_hypothesis_grounding.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "evaluation/sers_alpha4c5i/dev_compat_v1/input"
)

DISCOVERY_ROOTS = (
    Path("evaluation"),
)

_FORBIDDEN_PATH_TOKENS = (
    "reserve_a",
    "reserve_b",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _repo_relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def path_is_closed_reserve(path: Path) -> bool:
    parts = [
        str(part).lower().replace("-", "_")
        for part in path.parts
    ]
    return any(
        token in part
        for part in parts
        for token in _FORBIDDEN_PATH_TOKENS
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def load_dev_grounding(
    *,
    root: Path,
    path: Path | None = None,
) -> HypothesisTrendGroundingBundle:
    source_path = path or DEFAULT_GROUNDING
    source_path = (
        source_path
        if source_path.is_absolute()
        else root / source_path
    )
    if path_is_closed_reserve(source_path):
        raise ValueError(
            "Refusing grounding from a closed Reserve A/B path."
        )
    if not source_path.is_file():
        raise FileNotFoundError(
            f"DEV Trend grounding missing: {source_path}"
        )

    bundle = HypothesisTrendGroundingBundle.model_validate_json(
        source_path.read_text(encoding="utf-8")
    )
    validate_trend_grounding_bundle_sha(bundle)
    binding = load_trend_corpus_binding(bundle)

    if binding.domain_profile_id != "sers_au_ag":
        raise ValueError(
            f"Unexpected DEV grounding domain: "
            f"{binding.domain_profile_id!r}"
        )
    if binding.trend_semantics_id != EXPECTED_TREND_SEMANTICS_ID:
        raise ValueError(
            "DEV grounding Trend semantics drifted: "
            f"{binding.trend_semantics_id!r}"
        )

    exact_dev = load_exact_dev_paper_ids(root)
    if binding.paper_ids != exact_dev:
        raise ValueError(
            "SHA-locked Trend corpus paper IDs do not equal exact DEV53."
        )
    return bundle


@dataclass(frozen=True)
class ContextCandidate:
    context: HypothesisContext
    source_path: Path
    source_kind: str
    packet_path: Path | None = None
    report_path: Path | None = None


@dataclass(frozen=True)
class PacketReportCandidate:
    packet: GraphExplorerPacket
    report: ExplorationReport
    packet_path: Path
    report_path: Path


def _candidate_json_paths(
    *,
    root: Path,
    name_tokens: Iterable[str],
) -> list[Path]:
    tokens = tuple(token.lower() for token in name_tokens)
    found: list[Path] = []
    for relative_root in DISCOVERY_ROOTS:
        base = root / relative_root
        if not base.exists():
            continue
        for path in base.rglob("*.json"):
            if path_is_closed_reserve(path):
                continue
            lowered = path.name.lower()
            if any(token in lowered for token in tokens):
                found.append(path)
    return sorted(set(found))


def _parse_context(path: Path) -> HypothesisContext | None:
    try:
        value = HypothesisContext.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        validate_hypothesis_context_sha(value)
        return value
    except Exception:
        return None


def _parse_packet(path: Path) -> GraphExplorerPacket | None:
    try:
        return GraphExplorerPacket.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except Exception:
        return None


def _parse_report(path: Path) -> ExplorationReport | None:
    try:
        return ExplorationReport.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except Exception:
        return None


def _packet_exact_dev(
    *,
    packet: GraphExplorerPacket,
    binding: Any,
    exact_dev: list[str],
) -> bool:
    if packet.domain_profile_id != binding.domain_profile_id:
        return False
    if packet.corpus.corpus_id != binding.corpus_id:
        return False
    packet_papers = sorted(
        str(row.paper_id) for row in packet.corpus.papers
    )
    return packet_papers == exact_dev


def discover_existing_contexts(
    *,
    root: Path,
    binding: Any,
) -> list[ContextCandidate]:
    candidates: list[ContextCandidate] = []
    for path in _candidate_json_paths(
        root=root,
        name_tokens=("hypothesis_context", "context"),
    ):
        value = _parse_context(path)
        if value is None:
            continue
        if value.domain_profile_id != binding.domain_profile_id:
            continue
        if value.corpus_id != binding.corpus_id:
            continue
        candidates.append(
            ContextCandidate(
                context=value,
                source_path=path,
                source_kind="existing_context",
            )
        )
    return candidates


def discover_packet_report_pairs(
    *,
    root: Path,
    binding: Any,
    exact_dev: list[str],
) -> list[PacketReportCandidate]:
    packets: list[tuple[Path, GraphExplorerPacket]] = []
    reports: list[tuple[Path, ExplorationReport]] = []

    for path in _candidate_json_paths(
        root=root,
        name_tokens=("packet", "explorer"),
    ):
        value = _parse_packet(path)
        if value is None:
            continue
        if not _packet_exact_dev(
            packet=value,
            binding=binding,
            exact_dev=exact_dev,
        ):
            continue
        packets.append((path, value))

    for path in _candidate_json_paths(
        root=root,
        name_tokens=("report", "explorer"),
    ):
        value = _parse_report(path)
        if value is not None:
            reports.append((path, value))

    pairs: list[PacketReportCandidate] = []
    for packet_path, packet in packets:
        for report_path, report in reports:
            if report.source_packet_sha256 != packet.packet_sha256:
                continue
            if report.task_id != packet.task.task_id:
                continue
            pairs.append(
                PacketReportCandidate(
                    packet=packet,
                    report=report,
                    packet_path=packet_path,
                    report_path=report_path,
                )
            )
    return pairs


def _load_explicit_context(
    *,
    root: Path,
    path: Path,
    binding: Any,
) -> ContextCandidate:
    source_path = path if path.is_absolute() else root / path
    if path_is_closed_reserve(source_path):
        raise ValueError(
            "Refusing HypothesisContext from a closed Reserve path."
        )
    value = _parse_context(source_path)
    if value is None:
        raise ValueError(
            f"Not a valid SHA-verified HypothesisContext: {source_path}"
        )
    if (
        value.domain_profile_id != binding.domain_profile_id
        or value.corpus_id != binding.corpus_id
    ):
        raise ValueError(
            "Explicit HypothesisContext does not match DEV Trend corpus."
        )
    return ContextCandidate(
        context=value,
        source_path=source_path,
        source_kind="existing_context",
    )


def _load_explicit_pair(
    *,
    root: Path,
    packet_path: Path,
    report_path: Path,
    binding: Any,
    exact_dev: list[str],
) -> PacketReportCandidate:
    pp = packet_path if packet_path.is_absolute() else root / packet_path
    rp = report_path if report_path.is_absolute() else root / report_path
    if path_is_closed_reserve(pp) or path_is_closed_reserve(rp):
        raise ValueError(
            "Refusing Explorer packet/report from a closed Reserve path."
        )
    packet = _parse_packet(pp)
    report = _parse_report(rp)
    if packet is None:
        raise ValueError(f"Invalid GraphExplorerPacket: {pp}")
    if report is None:
        raise ValueError(f"Invalid ExplorationReport: {rp}")
    if not _packet_exact_dev(
        packet=packet,
        binding=binding,
        exact_dev=exact_dev,
    ):
        raise ValueError(
            "Explicit GraphExplorerPacket is not exact DEV53 / corpus-bound."
        )
    if report.source_packet_sha256 != packet.packet_sha256:
        raise ValueError(
            "ExplorationReport source_packet_sha256 mismatch."
        )
    if report.task_id != packet.task.task_id:
        raise ValueError("Explorer task ID mismatch.")
    return PacketReportCandidate(
        packet=packet,
        report=report,
        packet_path=pp,
        report_path=rp,
    )


def select_or_build_context(
    *,
    root: Path,
    binding: Any,
    exact_dev: list[str],
    explicit_context: Path | None = None,
    explicit_packet: Path | None = None,
    explicit_report: Path | None = None,
) -> ContextCandidate:
    if explicit_context is not None and (
        explicit_packet is not None or explicit_report is not None
    ):
        raise ValueError(
            "--context cannot be combined with --packet/--report."
        )

    if explicit_context is not None:
        return _load_explicit_context(
            root=root,
            path=explicit_context,
            binding=binding,
        )

    if (explicit_packet is None) != (explicit_report is None):
        raise ValueError(
            "--packet and --report must be supplied together."
        )

    if explicit_packet is not None and explicit_report is not None:
        pair = _load_explicit_pair(
            root=root,
            packet_path=explicit_packet,
            report_path=explicit_report,
            binding=binding,
            exact_dev=exact_dev,
        )
        context = HypothesisContextBuilder().build(
            pair.packet,
            pair.report,
            require_valid_report=True,
        )
        validate_hypothesis_context_sha(context)
        return ContextCandidate(
            context=context,
            source_path=pair.report_path,
            source_kind="derived_from_packet_report",
            packet_path=pair.packet_path,
            report_path=pair.report_path,
        )

    contexts = discover_existing_contexts(
        root=root,
        binding=binding,
    )
    # Deduplicate by exact context SHA, not filename.
    by_sha: dict[str, ContextCandidate] = {}
    for candidate in contexts:
        by_sha.setdefault(
            candidate.context.context_sha256,
            candidate,
        )
    contexts = list(by_sha.values())

    if len(contexts) == 1:
        return contexts[0]
    if len(contexts) > 1:
        rendered = ", ".join(
            _repo_relative(root, row.source_path)
            for row in sorted(
                contexts,
                key=lambda item: str(item.source_path),
            )
        )
        raise ValueError(
            "Multiple distinct DEV HypothesisContext candidates found; "
            "use --context explicitly: "
            + rendered
        )

    pairs = discover_packet_report_pairs(
        root=root,
        binding=binding,
        exact_dev=exact_dev,
    )
    pair_by_lineage: dict[
        tuple[str, str], PacketReportCandidate
    ] = {}
    for pair in pairs:
        key = (
            pair.packet.packet_sha256,
            pair.report.report_id,
        )
        pair_by_lineage.setdefault(key, pair)
    pairs = list(pair_by_lineage.values())

    if len(pairs) == 0:
        raise ValueError(
            "No matching DEV53 HypothesisContext and no valid "
            "GraphExplorerPacket + ExplorationReport pair were found. "
            "No context was fabricated."
        )
    if len(pairs) > 1:
        rendered = " | ".join(
            (
                _repo_relative(root, pair.packet_path)
                + " + "
                + _repo_relative(root, pair.report_path)
            )
            for pair in sorted(
                pairs,
                key=lambda item: (
                    str(item.packet_path),
                    str(item.report_path),
                ),
            )
        )
        raise ValueError(
            "Multiple valid DEV Explorer packet/report pairs found; "
            "use --packet and --report explicitly: "
            + rendered
        )

    pair = pairs[0]
    context = HypothesisContextBuilder().build(
        pair.packet,
        pair.report,
        require_valid_report=True,
    )
    validate_hypothesis_context_sha(context)
    return ContextCandidate(
        context=context,
        source_path=pair.report_path,
        source_kind="derived_from_packet_report",
        packet_path=pair.packet_path,
        report_path=pair.report_path,
    )


def build_dev_trend_input(
    *,
    root: Path,
    grounding_path: Path | None = None,
    explicit_context: Path | None = None,
    explicit_packet: Path | None = None,
    explicit_report: Path | None = None,
) -> tuple[
    TrendAwareHypothesisInput,
    ContextCandidate,
    HypothesisTrendGroundingBundle,
]:
    errors: list[str] = []
    errors.extend(verify_frozen_postmortem(root))
    errors.extend(verify_5i_component_hashes(root))
    errors.extend(verify_h1_dev_summary(root))
    if errors:
        raise ValueError(
            "Frozen/DEV prerequisites failed:\n- "
            + "\n- ".join(errors)
        )

    exact_dev = load_exact_dev_paper_ids(root)
    if len(exact_dev) != EXPECTED_DEV_COUNT:
        raise ValueError("Exact DEV partition is not 53 papers.")

    grounding = load_dev_grounding(
        root=root,
        path=grounding_path,
    )
    binding = load_trend_corpus_binding(grounding)

    context_candidate = select_or_build_context(
        root=root,
        binding=binding,
        exact_dev=exact_dev,
        explicit_context=explicit_context,
        explicit_packet=explicit_packet,
        explicit_report=explicit_report,
    )

    value = build_trend_aware_hypothesis_input(
        grounded_context=context_candidate.context,
        trend_grounding=grounding,
        input_semantics_id=ALPHA4C5I_DEV_INPUT_SEMANTICS_ID,
    )
    verify_trend_aware_input_sources(value)

    if value.trend_corpus_binding.paper_ids != exact_dev:
        raise ValueError(
            "Built TrendAwareHypothesisInput is not exact DEV53."
        )
    if value.domain_profile_id != "sers_au_ag":
        raise ValueError("Built input domain profile mismatch.")
    if (
        value.trend_corpus_binding.trend_semantics_id
        != EXPECTED_TREND_SEMANTICS_ID
    ):
        raise ValueError("Built input Trend semantics mismatch.")

    return value, context_candidate, grounding


def build_manifest(
    *,
    root: Path,
    value: TrendAwareHypothesisInput,
    context_candidate: ContextCandidate,
    grounding_path: Path,
) -> dict[str, Any]:
    context_source_path = context_candidate.source_path
    payload: dict[str, Any] = {
        "schema_version":
            "sers-alpha4c5i-dev-trend-input-build-v1",
        "semantics_id": ALPHA4C5I_DEV_INPUT_BUILDER_SEMANTICS_ID,
        "development_only": True,
        "input_semantics_id": ALPHA4C5I_DEV_INPUT_SEMANTICS_ID,
        "input_id": value.input_id,
        "input_sha256": value.input_sha256,
        "dev_paper_count": len(
            value.trend_corpus_binding.paper_ids
        ),
        "trend_semantics_id":
            value.trend_corpus_binding.trend_semantics_id,
        "grounding_bundle_id":
            value.trend_grounding.bundle_id,
        "grounding_bundle_sha256":
            value.trend_grounding.bundle_sha256,
        "grounding_path": _repo_relative(root, grounding_path),
        "grounding_file_sha256": sha256_file(grounding_path),
        "context_id": value.grounded_context.context_id,
        "context_sha256": value.grounded_context.context_sha256,
        "context_source_kind": context_candidate.source_kind,
        "context_source_path":
            _repo_relative(root, context_source_path),
        "context_source_file_sha256":
            sha256_file(context_source_path),
        "packet_path": (
            None
            if context_candidate.packet_path is None
            else _repo_relative(root, context_candidate.packet_path)
        ),
        "packet_file_sha256": (
            None
            if context_candidate.packet_path is None
            else sha256_file(context_candidate.packet_path)
        ),
        "report_path": (
            None
            if context_candidate.report_path is None
            else _repo_relative(root, context_candidate.report_path)
        ),
        "report_file_sha256": (
            None
            if context_candidate.report_path is None
            else sha256_file(context_candidate.report_path)
        ),
        "reserve_a_used": False,
        "reserve_b_used": False,
        "reserve_b_rerun": False,
        "scientific_semantics_modified": False,
        "new_scientific_extraction": False,
        "llm_calls": 0,
        "count_thresholds_used_for_acceptance": False,
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    payload["manifest_sha256"] = hashlib.sha256(
        canonical
    ).hexdigest()
    return payload
