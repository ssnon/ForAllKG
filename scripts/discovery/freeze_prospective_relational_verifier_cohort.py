from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from pipeline_core.discovery.prospective_relational_verifier_cohort import (
    ProspectiveRelationalVerifierPoolSpec,
    build_prospective_relational_verifier_cohort_freeze,
)
from pipeline_core.discovery.relational_scientific_verifier_shadow import (
    write_json_exclusive,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze an outcome-blind P06+ relational scientific verifier "
            "cohort from pre-verifier structural artifacts only. Prior-art, "
            "external-novelty, positive-nonobviousness, and certification "
            "artifacts are not accepted as selector inputs."
        )
    )
    parser.add_argument("--pool-spec", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    spec_path = args.pool_spec.expanduser().resolve()
    if not spec_path.is_file():
        raise ValueError("missing pool spec: " + str(spec_path))
    if args.output.expanduser().resolve().exists():
        raise ValueError(
            "prospective cohort freeze is write-once; use a fresh output path"
        )

    spec = ProspectiveRelationalVerifierPoolSpec.model_validate_json(
        spec_path.read_text(encoding="utf-8")
    )
    freeze = build_prospective_relational_verifier_cohort_freeze(
        pool_spec=spec,
        pool_spec_sha256=_sha256_file(spec_path),
        base_dir=spec_path.parent,
    )
    write_json_exclusive(args.output.expanduser().resolve(), freeze)

    print("Prospective relational verifier cohort frozen")
    print("Freeze:", freeze.freeze_id)
    print("Eligible hypotheses:", freeze.eligible_candidate_count)
    print("Excluded hypotheses:", freeze.excluded_candidate_count)
    print("Frozen cases:", freeze.frozen_case_count)
    print("Lane counts:", freeze.frozen_lane_profile_counts)
    print()
    for row in freeze.frozen_cases:
        print(
            row.prospective_case_id,
            "| source_case=", row.pool_case_key,
            "| lane=", row.primary_lane_profile,
            "| synthesis=", row.synthesis_kind,
            "| final=", row.final_hypothesis_id,
            "| claims=", row.novelty_bearing_bound_claim_ids,
        )
    print()
    print("Prior-art retrieval used for selection: false")
    print("External novelty outcomes used for selection: false")
    print("Positive non-obviousness used for selection: false")
    print("Final certification used for selection: false")
    print("New verifier result observed before freeze: false")
    print("Post-freeze selection change allowed: false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
