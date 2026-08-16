from __future__ import annotations

import argparse
import json

from dac_her.literature_provider_plan import (
    build_literature_providers,
    resolve_literature_provider_plan,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve the literature provider set from the current "
            "environment without making network calls."
        )
    )
    parser.add_argument(
        "--require-standard-or-full",
        action="store_true",
        help=(
            "Fail unless the resolved mode is STANDARD_2_PROVIDER "
            "or FULL_3_PROVIDER."
        ),
    )
    args = parser.parse_args()

    plan = resolve_literature_provider_plan()
    print("Literature provider plan preflight")
    print("Plan ID:", plan.plan_id)
    print("Plan SHA256:", plan.plan_sha256)
    print("Requested mode:", plan.requested_mode)
    print("Provider mode:", plan.mode)
    print(
        "Active providers:",
        ", ".join(plan.active_providers)
        if plan.active_providers
        else "(none)",
    )
    print(
        "OpenAlex API key configured:",
        plan.openalex_api_key_configured,
    )
    print(
        "Crossref mailto configured:",
        plan.crossref_mailto_configured,
    )
    print(
        "Semantic Scholar API key configured:",
        plan.semantic_scholar_api_key_configured,
    )
    print(
        "Scientific equivalence to FULL_3_PROVIDER established:",
        plan.scientific_equivalence_to_full_3_provider_established,
    )
    print("Secret values persisted:", plan.secret_values_persisted)
    print("Network calls:", 0)
    print("LLM calls:", 0)

    # Construction validates that the frozen environment can instantiate
    # the selected providers, but it performs no network access.
    if plan.active_providers:
        providers = build_literature_providers(plan)
        print(
            "Provider instances:",
            [row.provider_name for row in providers],
        )

    if (
        args.require_standard_or_full
        and plan.mode
        not in {
            "STANDARD_2_PROVIDER",
            "FULL_3_PROVIDER",
        }
    ):
        print("Preflight: FAIL")
        return 2

    print("Preflight: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
