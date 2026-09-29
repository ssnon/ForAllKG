from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

from pipeline_core.discovery.prospective_novelty_validation_cohort import (
    ProspectiveNoveltyValidationCohortFreeze,
)
from pipeline_core.discovery.prospective_novelty_validation_execution import (
    build_prospective_novelty_execution_plan,
)


def _data_root(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "--data-root must use DOMAIN=PATH"
        )
    domain, path = value.split("=", 1)
    domain = domain.strip()
    path = path.strip()
    if not domain or not path:
        raise argparse.ArgumentTypeError(
            "--data-root must use non-empty DOMAIN=PATH"
        )
    return domain, path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build an immutable execution plan from a frozen prospective "
            "novelty-validation cohort. The plan freezes model names and all "
            "scientific runner arguments but never stores API credentials."
        )
    )
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--model",
        default=(
            os.getenv("OPENROUTER_AGENT_MODEL")
            or os.getenv("GRAPHAGENTS_HYPOTHESIS_MODEL")
            or ""
        ),
    )
    parser.add_argument(
        "--critic-model",
        default=(
            os.getenv("OPENROUTER_CRITIC_MODEL")
            or os.getenv("OPENROUTER_AGENT_MODEL")
            or ""
        ),
    )
    parser.add_argument(
        "--base-url",
        default=(
            os.getenv("BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or None
        ),
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENROUTER_API_KEY",
    )
    parser.add_argument(
        "--results-per-query",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--data-root",
        action="append",
        default=[],
        type=_data_root,
        metavar="DOMAIN=PATH",
        help=(
            "Freeze an explicit scientific-domain data root into the execution "
            "plan. Repeat for multiple domains. The root must contain "
            "corpus/<corpus-id>/..., not be the corpus directory itself."
        ),
    )
    args = parser.parse_args()

    freeze_path = args.freeze.expanduser().resolve()
    output = args.output.expanduser().resolve()

    if not freeze_path.is_file():
        raise ValueError("missing freeze: " + str(freeze_path))
    if output.exists():
        raise ValueError(
            "execution plan is write-once; use a fresh output path"
        )

    freeze = ProspectiveNoveltyValidationCohortFreeze.model_validate_json(
        freeze_path.read_text(encoding="utf-8")
    )

    data_roots: dict[str, str] = {}
    for domain, path in args.data_root:
        if domain in data_roots:
            raise ValueError(
                "duplicate --data-root domain: " + domain
            )
        data_roots[domain] = path

    plan = build_prospective_novelty_execution_plan(
        freeze=freeze,
        source_freeze_file_sha256=_sha256_file(freeze_path),
        run_root=str(args.run_root.expanduser().resolve()),
        model_name=args.model,
        critic_model_name=args.critic_model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        results_per_query=args.results_per_query,
        domain_data_roots=data_roots,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        plan.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    print("Prospective novelty-validation execution plan frozen")
    print("Plan:", plan.plan_id)
    print("Source freeze:", plan.source_freeze_id)
    print("Model:", plan.model_name)
    print("Critic:", plan.critic_model_name)
    print("Data roots:", plan.domain_data_roots)
    print("Cases:", len(plan.cases))
    print()
    for row in plan.cases:
        print(row.source_case_id)
        print(" run:", row.run_dir)
        print(" argv:", "python " + " ".join(row.command_argv))
    print()
    print("Scientific case selection changed: false")
    print("Result-conditioned route changes allowed: false")
    print("Overwrite existing case runs allowed: false")
    print("Production selection authority: false")
    print("Artifact:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
