from __future__ import annotations

import argparse
from pathlib import Path

from dac_her.kg_config_adapter import load_and_generate_paper_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a current-GraphAgentsDAC-compatible papers.yaml from a "
            "frozen Drive-ingestion corpus while preserving Marker asset packages."
        )
    )
    parser.add_argument("--frozen-manifest", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--validate-with-repo",
        action="store_true",
        help="Load the generated YAML through dac_her.config.load_paper_configs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frozen = Path(args.frozen_manifest)
    if args.output:
        output = Path(args.output)
    else:
        # Keep generated config out of the hand-maintained configs/papers.yaml.
        corpus_id = frozen.parent.name
        output = Path("data_dac") / "generated_configs" / corpus_id / "papers.yaml"

    generated = load_and_generate_paper_config(
        frozen,
        output,
        project_root=args.project_root,
    )

    if args.validate_with_repo:
        from dac_her.config import load_paper_configs

        loaded = load_paper_configs(
            generated.papers_yaml,
            project_root=args.project_root,
        )
        if set(loaded) != set(generated.paper_ids):
            raise RuntimeError("Repository config loader returned a different paper-id set")

    print("[kg-config] generated")
    print("[kg-config] papers:", len(generated.paper_ids))
    print("[kg-config] yaml:", generated.papers_yaml)
    print("[kg-config] adapter manifest:", generated.adapter_manifest)


if __name__ == "__main__":
    main()
