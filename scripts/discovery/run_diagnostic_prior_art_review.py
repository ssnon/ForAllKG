from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from domains.registry import (
    get_domain_profile,
)
from pipeline_core.discovery.diagnostic_prior_art_report import (
    build_diagnostic_review_report,
)
from pipeline_core.discovery.diagnostic_prior_art_review import (
    compile_diagnostic_prior_art_review,
)
from pipeline_core.discovery.external_novelty_contracts import (
    LiteratureQueryPlan,
    PriorArtPacket,
)
from pipeline_core.discovery.external_novelty_llm import (
    InstructorOpenAICompatibleExternalNoveltyBackend,
)
from pipeline_core.discovery.node_mapping import (
    DEFAULT_EMBED_MODEL,
    SentenceTransformerEncoder,
)
from pipeline_core.discovery.prior_art_matching import (
    PriorArtRanker,
)
from pipeline_core.discovery.prior_art_review_audit import (
    prior_art_review_audit_scope,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Review an already-retrieved bounded "
            "diagnostic prior-art packet. "
            "The output is diagnostic-only and "
            "cannot alter ordinary full-claim "
            "novelty status."
        )
    )

    parser.add_argument(
        "--query-plan",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--prior-art",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--domain-profile",
        required=True,
    )

    parser.add_argument(
        "--model",
        required=True,
    )

    parser.add_argument(
        "--base-url",
        default=None,
    )

    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
    )

    parser.add_argument(
        "--device",
        default=None,
    )

    parser.add_argument(
        "--embed-model",
        default=DEFAULT_EMBED_MODEL,
    )

    parser.add_argument(
        "--max-ranked-works",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--save-prompts",
        action="store_true",
    )

    return parser.parse_args()


def write_json(
    path: Path,
    value: object,
) -> None:
    if hasattr(
        value,
        "model_dump",
    ):
        value = value.model_dump(
            mode="json"
        )

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


def main() -> int:
    args = parse_args()

    plan = (
        LiteratureQueryPlan
        .model_validate_json(
            args.query_plan.read_text(
                encoding="utf-8"
            )
        )
    )

    packet = (
        PriorArtPacket
        .model_validate_json(
            args.prior_art.read_text(
                encoding="utf-8"
            )
        )
    )

    if (
        packet.source_portfolio_id
        != plan.source_portfolio_id
    ):
        raise ValueError(
            "diagnostic plan/packet "
            "source_portfolio_id mismatch"
        )

    if (
        packet.source_query_plan_id
        != plan.plan_id
    ):
        raise ValueError(
            "diagnostic plan/packet "
            "query-plan provenance mismatch"
        )

    if any(
        row.query_kind
        != "claim_diagnostic"
        for row in plan.queries
    ):
        raise ValueError(
            "diagnostic review runner accepts "
            "claim_diagnostic queries only"
        )

    domain = get_domain_profile(
        args.domain_profile
    )

    encoder = SentenceTransformerEncoder(
        args.embed_model,
        device=args.device,
    )

    ranker = PriorArtRanker(
        encoder,
        max_ranked_works_per_claim=(
            args.max_ranked_works
        ),
        domain_profile=domain,
    )

    backend = (
        InstructorOpenAICompatibleExternalNoveltyBackend(
            model=args.model,
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            temperature=0.0,
            parse_retries=1,
            capture_prompts=(
                args.save_prompts
            ),
        )
    )

    claim_ids = {
        row.claim_id
        for row in plan.queries
        if row.claim_id
    }

    claims = [
        claim
        for group in plan.claims
        for claim in group.claims
        if claim.claim_id
        in claim_ids
    ]

    work_index = {
        row.work_id: row
        for row in packet.works
    }

    reviews = []

    with prior_art_review_audit_scope(
        assessment_kind=(
            "diagnostic_prior_art_review"
        ),
        source_portfolio_id=(
            plan.source_portfolio_id
        ),
        query_plan_id=(
            plan.plan_id
        ),
        prior_art_packet_id=(
            packet.packet_id
        ),
    ):
        for claim in claims:
            candidates = ranker.rank(
                claim,
                packet,
                plan,
            )

            review_input = []

            for ranked in (
                candidates.ranked_works
            ):
                work = work_index[
                    ranked.work_id
                ]

                review_input.append(
                    {
                        "work_id":
                            work.work_id,
                        "title":
                            work.title,
                        "year":
                            work.year,
                        "doi":
                            work.doi,
                        "abstract":
                            work.abstract,
                        "semantic_similarity":
                            ranked.semantic_similarity,
                        "lexical_coverage":
                            ranked.lexical_coverage,
                        "reaction_domain_relevance":
                            ranked.reaction_domain_relevance,
                        "catalyst_scope_relevance":
                            ranked.catalyst_scope_relevance,
                        "relevance_score":
                            ranked.relevance_score,
                    }
                )

            draft = (
                backend
                .review_diagnostic_claim(
                    claim,
                    review_input,
                )
            )

            reviews.append(
                compile_diagnostic_prior_art_review(
                    claim=claim,
                    candidates=candidates,
                    draft=draft,
                    packet=packet,
                )
            )

    report = (
        build_diagnostic_review_report(
            source_portfolio_id=(
                plan.source_portfolio_id
            ),
            source_query_plan_id=(
                plan.plan_id
            ),
            source_query_plan_sha256=(
                plan.plan_sha256
            ),
            source_prior_art_packet_id=(
                packet.packet_id
            ),
            source_prior_art_packet_sha256=(
                packet.packet_sha256
            ),
            reviews=reviews,
        )
    )

    write_json(
        args.output,
        report,
    )

    if args.save_prompts:
        prompt_dir = (
            args.output.parent
            / (
                args.output.stem
                + ".prompts"
            )
        )

        prompt_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for index, row in enumerate(
            backend.prompt_records,
            start=1,
        ):
            safe = "".join(
                character
                if (
                    character.isalnum()
                    or character
                    in "_-"
                )
                else "_"
                for character
                in row.name
            )

            target = (
                prompt_dir
                / (
                    f"{index:02d}_"
                    f"{safe}.txt"
                )
            )

            target.write_text(
                "\n".join(
                    [
                        (
                            "prompt_sha256: "
                            f"{row.prompt_sha256}"
                        ),
                        "",
                        "SYSTEM",
                        "======",
                        row.system_prompt,
                        "",
                        "USER",
                        "====",
                        row.user_prompt,
                        "",
                    ]
                ),
                encoding="utf-8",
            )

    relationships = Counter(
        match.relationship
        for review in reviews
        for match in review.matches
    )

    print(
        "Diagnostic prior-art review complete"
    )

    print(
        "Reviewed claims:",
        len(reviews),
    )

    print(
        "Signal claims:",
        report.signal_claim_count,
    )

    print(
        "Unique signal works:",
        report.signal_work_count,
    )

    print(
        "Relationships:",
        dict(
            sorted(
                relationships.items()
            )
        ),
    )

    for review in reviews:
        if not review.signal_work_ids:
            continue

        print()
        print(
            review.claim_id
        )

        for match in review.matches:
            if (
                match.work_id
                not in review.signal_work_ids
            ):
                continue

            print(
                " ",
                match.relationship,
                "|",
                match.doi,
                "|",
                match.title[:120],
            )

    print()
    print(
        "SHADOW ONLY: ordinary external "
        "novelty / N9 / N10 unchanged."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
