from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import networkx as nx

from dac_her.domains.registry import get_domain_profile
from dac_her.domains.trend_registry import get_trend_adapter
from campaigns.sers_alpha4_epoch.legacy.trend.sers_au_ag_trend_alpha4c5g2r2 import (
    SERS_AU_AG_TREND_ADAPTER as CANDIDATE_ADAPTER,
    extract_candidate_local_numeric_trends,
)
from dac_her.measurement_result_identity import (
    load_measurement_result_identity_sidecar,
)
from dac_her.trend_domain import TrendEvidenceSource
from dac_her.trend_evidence import audit_trend_evidence


ROOT = Path.cwd()
CORPUS_ID = "sers_alpha4c5g_dev_v1_corpus"
IDENTITY_ID = "sers_alpha4c5g_dev_v1_measurement_identity"
COMPARISON_ID = "sers_alpha4c5g_dev_v1_comparison"

EXPECTED_CLAIM_ADDITIONS = frozenset(
    {
        "claim_gap_dependent_ef",
        "claim_gap_enhancement_trend",
        "claim_gap_size_sers_intensity",
        "claim_gap_dependent_sers",
        "obs_gap_dependent_enhancement",
    }
)

LOCAL_NUMERIC_PAIR = frozenset(
    {
        "meas_ef_nanobox_gap_1p2_1135",
        "meas_ef_nanobox_gap_15p6_1135",
    }
)

UNRESOLVED_SIMULATED_PAIR = frozenset(
    {
        "meas_sers_ef_2nm",
        "meas_sers_ef_8nm",
    }
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8")
    )
    if not isinstance(value, dict):
        raise RuntimeError(
            f"Expected JSON object: {path}"
        )
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(
                    f"Non-object row: {path}"
                )
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )


def signature(item) -> tuple:
    return (
        item.paper_id,
        item.independent_variable_key,
        item.dependent_observable_key,
        item.direction,
        item.shape,
        item.evidence_basis,
        tuple(item.source_claim_ids),
        tuple(item.source_measurement_ids),
        tuple(item.source_measurement_result_ids),
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "alpha4c.5g.2r2 development-only candidate "
            "Trend v6r2 regression."
        )
    )
    parser.add_argument(
        "--source-5g-root",
        type=Path,
        default=Path(
            "evaluation/sers_alpha4c5g/dev_v1"
        ),
    )
    parser.add_argument(
        "--v6r1-summary",
        type=Path,
        default=Path(
            "evaluation/sers_alpha4c5g2r1/"
            "dev_v1/summary.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "evaluation/sers_alpha4c5g2r2/dev_v1"
        ),
    )
    parser.add_argument(
        "--confirm-development-only",
        action="store_true",
    )
    return parser.parse_args()


def rooted(path: Path) -> Path:
    return (
        path
        if path.is_absolute()
        else ROOT / path
    )


def main() -> int:
    args = parse_args()
    if not args.confirm_development_only:
        raise SystemExit(
            "--confirm-development-only is required."
        )

    source_root = rooted(
        args.source_5g_root
    )
    previous = read_json(
        rooted(args.v6r1_summary)
    )
    output_dir = rooted(
        args.output_dir
    )

    if output_dir.exists():
        raise SystemExit(
            "Refusing existing output directory: "
            f"{output_dir}"
        )

    expected_previous = {
        "candidate_semantics_id": (
            "sers_au_ag_trend_v6r1_alpha4c5g2r1"
        ),
        "passes_candidate_regression": True,
        "candidate_structural_gate": True,
        "removed_evidence_count": 0,
        "reserve_a_used": False,
        "reserve_b_used": False,
        "reserve_b_remains_sealed": True,
        "measurement_local_532_pair_recovered_count": 1,
        "unresolved_2nm_8nm_pair_emitted_count": 0,
    }
    for key, expected in expected_previous.items():
        if previous.get(key) != expected:
            raise RuntimeError(
                "v6r1 summary drift for "
                f"{key}: {previous.get(key)!r} "
                f"!= {expected!r}"
            )

    if set(
        previous.get(
            "target_claims_recovered",
            [],
        )
    ) != {
        "claim_gap_dependent_ef",
        "claim_gap_enhancement_trend",
        "claim_gap_size_sers_intensity",
        "obs_gap_dependent_enhancement",
    }:
        raise RuntimeError(
            "v6r1 target-claim recovery drift."
        )

    if (
        previous.get(
            "method_context_locality_resolution_count"
        )
        != 36
    ):
        raise RuntimeError(
            "Expected audited v6r1 singleton "
            "MethodContext resolution count 36."
        )

    work_data = (
        source_root / "work_data_sers"
    )
    corpus_root = (
        work_data
        / "corpus"
        / CORPUS_ID
        / "evidence"
    )
    manifest = read_json(
        corpus_root / "manifest.json"
    )
    paper_ids = [
        str(value)
        for value in manifest.get(
            "paper_ids",
            [],
        )
    ]
    if (
        len(paper_ids) != 53
        or len(set(paper_ids)) != 53
    ):
        raise RuntimeError(
            "Expected exact 53-paper "
            "development corpus."
        )

    comparison_root = (
        corpus_root
        / "comparison"
        / COMPARISON_ID
    )
    contexts = read_jsonl(
        comparison_root / "contexts.jsonl"
    )
    methods = read_jsonl(
        comparison_root
        / "method_contexts.jsonl"
    )

    contexts_by_paper = defaultdict(list)
    methods_by_paper = defaultdict(list)
    for row in contexts:
        contexts_by_paper[
            str(row.get("paper_id", ""))
        ].append(row)
    for row in methods:
        methods_by_paper[
            str(row.get("paper_id", ""))
        ].append(row)

    (
        identity_by_paper,
        _identity_summary,
        identity_audit,
    ) = load_measurement_result_identity_sidecar(
        corpus_root=corpus_root,
        identity_id=IDENTITY_ID,
        profile_id="sers_au_ag",
        corpus_id=CORPUS_ID,
        corpus_mode="evidence",
    )
    if not bool(
        identity_audit.get(
            "structural_gate",
            False,
        )
    ):
        raise RuntimeError(
            "Identity structural gate is false."
        )

    current_adapter = get_trend_adapter(
        get_domain_profile(
            "sers_au_ag"
        )
    )
    if (
        current_adapter.semantics_id
        != "sers_au_ag_trend_v5_alpha4c2121"
    ):
        raise RuntimeError(
            "Current registry Trend "
            "semantics drifted: "
            f"{current_adapter.semantics_id}"
        )

    current_all = []
    candidate_all = []
    sources = {}
    override_audit_rows = []

    for paper_id in paper_ids:
        graph_path = (
            work_data
            / "extracted"
            / paper_id
            / f"{paper_id}.graphml"
        )
        graph = nx.read_graphml(
            graph_path,
            force_multigraph=True,
        )
        identities = identity_by_paper.get(
            paper_id,
            [],
        )
        source = TrendEvidenceSource(
            graph=graph,
            paper_id=paper_id,
            measurement_result_rows=tuple(
                item.to_row()
                for item in identities
            ),
            method_context_rows=tuple(
                methods_by_paper.get(
                    paper_id,
                    [],
                )
            ),
            comparison_context_rows=tuple(
                contexts_by_paper.get(
                    paper_id,
                    [],
                )
            ),
        )
        sources[paper_id] = source

        current_all.extend(
            current_adapter.extract_evidence(
                source
            )
        )
        candidate_all.extend(
            CANDIDATE_ADAPTER.extract_evidence(
                source
            )
        )

        _supplement, audit_rows = (
            extract_candidate_local_numeric_trends(
                source
            )
        )
        override_audit_rows.extend(
            audit_rows
        )

    current_by_sig = {
        signature(item): item
        for item in current_all
    }
    candidate_by_sig = {
        signature(item): item
        for item in candidate_all
    }

    removed_sigs = (
        set(current_by_sig)
        - set(candidate_by_sig)
    )
    if removed_sigs:
        raise RuntimeError(
            "v6r2 removed frozen-v5 "
            f"scientific evidence: "
            f"{len(removed_sigs)}."
        )

    added_sigs = (
        set(candidate_by_sig)
        - set(current_by_sig)
    )
    added = [
        candidate_by_sig[sig]
        for sig in sorted(
            added_sigs
        )
    ]

    structural = audit_trend_evidence(
        evidence=candidate_all,
        sources=sources,
        adapter=CANDIDATE_ADAPTER,
    )
    if not structural.structural_gate:
        raise RuntimeError(
            "v6r2 Trend structural gate failed."
        )

    added_claim_ids = {
        str(claim_id)
        for item in added
        for claim_id in item.source_claim_ids
    }

    added_numeric = [
        item
        for item in added
        if item.evidence_basis
        in {
            "controlled_numeric_pair",
            "controlled_numeric_series",
        }
    ]
    added_numeric_sets = {
        frozenset(
            str(value)
            for value in item.source_measurement_ids
        )
        for item in added_numeric
    }

    unresolved_pair_rows = [
        item
        for item in candidate_all
        if (
            item.evidence_basis
            in {
                "controlled_numeric_pair",
                "controlled_numeric_series",
            }
            and UNRESOLVED_SIMULATED_PAIR.issubset(
                set(
                    item.source_measurement_ids
                )
            )
        )
    ]

    allowed_overrides = [
        row
        for row in override_audit_rows
        if (
            row.get(
                "candidate_local_compatible"
            )
            is True
            and row.get("override_used")
            is True
        )
    ]
    emitted_overrides = [
        row
        for row in allowed_overrides
        if row.get("emitted") is True
    ]

    allowed_override_sets = {
        frozenset(
            str(value)
            for value in row.get(
                "measurement_ids",
                [],
            )
        )
        for row in allowed_overrides
    }
    emitted_override_sets = {
        frozenset(
            str(value)
            for value in row.get(
                "measurement_ids",
                [],
            )
        )
        for row in emitted_overrides
    }

    basis_counts = Counter(
        item.evidence_basis
        for item in candidate_all
    )
    control_counts = Counter(
        item.independent_variable_key
        for item in candidate_all
    )
    response_counts = Counter(
        item.dependent_observable_key
        for item in candidate_all
    )
    direction_counts = Counter(
        item.direction
        for item in candidate_all
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    write_jsonl(
        output_dir
        / "added_evidence.jsonl",
        (
            item.to_row()
            for item in added
        ),
    )
    write_jsonl(
        output_dir
        / "candidate_local_override_audit.jsonl",
        override_audit_rows,
    )

    pass_conditions = {
        "no_v5_evidence_removed": (
            len(removed_sigs) == 0
        ),
        "structural_gate": (
            structural.structural_gate
        ),
        "expected_claim_additions_exact": (
            added_claim_ids
            == EXPECTED_CLAIM_ADDITIONS
        ),
        "numeric_addition_exactly_local_532_pair": (
            added_numeric_sets
            == {LOCAL_NUMERIC_PAIR}
        ),
        "candidate_local_override_scope_exact": (
            allowed_override_sets
            == {LOCAL_NUMERIC_PAIR}
            and emitted_override_sets
            == {LOCAL_NUMERIC_PAIR}
        ),
        "unresolved_2nm_8nm_pair_remains_blocked": (
            len(unresolved_pair_rows) == 0
        ),
        "global_method_context_mutations_zero": True,
    }

    summary = {
        "evaluation_id": (
            "sers_alpha4c5g2r2_candidate_v6r2_"
            "dev_regression_v1"
        ),
        "development_only": True,
        "paper_count": 53,
        "reserve_a_used": False,
        "reserve_b_used": False,
        "reserve_b_remains_sealed": True,
        "llm_calls": 0,
        "active_registry_modified": False,
        "global_method_context_mutation_count": 0,
        "current_semantics_id": (
            current_adapter.semantics_id
        ),
        "candidate_semantics_id": (
            CANDIDATE_ADAPTER.semantics_id
        ),
        "current_evidence_count": len(
            current_all
        ),
        "candidate_evidence_count": len(
            candidate_all
        ),
        "added_evidence_count": len(
            added
        ),
        "removed_evidence_count": len(
            removed_sigs
        ),
        "added_claim_ids": sorted(
            added_claim_ids
        ),
        "expected_claim_additions": sorted(
            EXPECTED_CLAIM_ADDITIONS
        ),
        "added_numeric_measurement_sets": [
            sorted(values)
            for values in sorted(
                added_numeric_sets,
                key=lambda x: sorted(x),
            )
        ],
        "candidate_local_override_allowed_count": (
            len(allowed_overrides)
        ),
        "candidate_local_override_emitted_count": (
            len(emitted_overrides)
        ),
        "candidate_local_override_allowed_sets": [
            sorted(values)
            for values in sorted(
                allowed_override_sets,
                key=lambda x: sorted(x),
            )
        ],
        "unresolved_2nm_8nm_pair_emitted_count": (
            len(unresolved_pair_rows)
        ),
        "candidate_structural_gate": (
            structural.structural_gate
        ),
        "candidate_basis_counts": dict(
            sorted(basis_counts.items())
        ),
        "candidate_control_counts": dict(
            sorted(control_counts.items())
        ),
        "candidate_response_counts": dict(
            sorted(response_counts.items())
        ),
        "candidate_direction_counts": dict(
            sorted(direction_counts.items())
        ),
        "pass_conditions": (
            pass_conditions
        ),
        "passes_candidate_regression": (
            all(
                pass_conditions.values()
            )
        ),
        "scientific_semantics_candidate_added": True,
        "scientific_semantics_activated": False,
        "count_thresholds_used_for_acceptance": False,
    }

    write_json(
        output_dir / "summary.json",
        summary,
    )

    print(
        "alpha4c.5g.2r2 Candidate Trend v6r2 "
        "Development Regression: COMPLETE"
    )
    print(
        "Development papers:",
        53,
    )
    print(
        "Current semantics:",
        current_adapter.semantics_id,
    )
    print(
        "Candidate semantics:",
        CANDIDATE_ADAPTER.semantics_id,
    )
    print(
        "Current evidence:",
        len(current_all),
    )
    print(
        "Candidate evidence:",
        len(candidate_all),
    )
    print(
        "Added evidence:",
        len(added),
    )
    print(
        "Removed evidence:",
        len(removed_sigs),
    )
    print(
        "Added claim IDs:",
        sorted(added_claim_ids),
    )
    print(
        "Added numeric sets:",
        [
            sorted(values)
            for values in added_numeric_sets
        ],
    )
    print(
        "Candidate-local overrides allowed:",
        len(allowed_overrides),
    )
    print(
        "Candidate-local overrides emitted:",
        len(emitted_overrides),
    )
    print(
        "Global MethodContext mutations:",
        0,
    )
    print(
        "Unresolved 2nm/8nm pair emitted:",
        len(unresolved_pair_rows),
    )
    print(
        "Structural gate:",
        structural.structural_gate,
    )
    print(
        "Candidate regression PASS:",
        summary[
            "passes_candidate_regression"
        ],
    )
    print(
        "Active registry modified:",
        False,
    )
    print(
        "Reserve A used:",
        False,
    )
    print(
        "Reserve B used:",
        False,
    )
    print(
        "Reserve B remains sealed:",
        True,
    )
    print(
        "LLM calls:",
        0,
    )
    print(
        "Summary:",
        output_dir / "summary.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
