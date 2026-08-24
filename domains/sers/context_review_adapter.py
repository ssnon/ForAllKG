from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from domains.sers.context_comparator import (
    SERSHypothesisContextComparator,
)
from domains.sers.context_compiler import (
    SERSContextCompiler,
)
from domains.sers.hypothesis_context_interpreter import (
    SERSHypothesisContextInterpreter,
)
from domains.sers.hypothesis_context_llm import (
    InstructorOpenAICompatibleHypothesisContextBackend,
)
from pipeline_core.discovery.discovery_axis_contracts import (
    DiscoveryAxis,
)
from pipeline_core.discovery.dual_hypothesis_context import (
    DualHypothesisContext,
)


SERS_AU_AG_CONTEXT_REVIEW_ADAPTER_ID = "sers_au_ag"


class SERSDiscoveryAxisContextReviewer:
    """Claim-local SERS scientific-context review.

    The reviewer compiles context only from:
      * the hypothesis's selected positive premises; and
      * its assigned discovery inspiration.

    It intentionally does not inherit context from an entire paper or
    from the whole DiscoveryBundle.
    """

    adapter_id = SERS_AU_AG_CONTEXT_REVIEW_ADAPTER_ID
    domain_profile_id = "sers_au_ag"

    def __init__(
        self,
        *,
        compiler: Any,
        interpreter: Any,
        comparator: Any,
    ) -> None:
        self.compiler = compiler
        self.interpreter = interpreter
        self.comparator = comparator

    @classmethod
    def build(
        cls,
        *,
        graph: Any,
        backend: Any,
        domain_profile_id: str = "sers_au_ag",
    ) -> "SERSDiscoveryAxisContextReviewer":
        if domain_profile_id != cls.domain_profile_id:
            raise ValueError(
                "SERS context reviewer/domain mismatch: "
                f"expected={cls.domain_profile_id!r}, "
                f"actual={domain_profile_id!r}"
            )

        return cls(
            compiler=SERSContextCompiler(
                graph=graph,
                domain_profile_id=domain_profile_id,
            ),
            interpreter=SERSHypothesisContextInterpreter(
                backend
            ),
            comparator=SERSHypothesisContextComparator(),
        )

    @staticmethod
    def _index_unique(
        rows: list[Any],
        *,
        attribute: str,
        label: str,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}

        for row in rows:
            key = str(
                getattr(
                    row,
                    attribute,
                    "",
                )
            )

            if not key:
                raise RuntimeError(
                    f"{label} without {attribute}"
                )

            if key in result:
                raise RuntimeError(
                    f"duplicate {label} {attribute}: {key}"
                )

            result[key] = row

        return result

    def review(
        self,
        *,
        dual: DualHypothesisContext,
        axis: DiscoveryAxis,
        card: Any,
    ) -> Any:
        context = dual.grounded_context

        if (
            context.domain_profile_id
            != self.domain_profile_id
        ):
            raise RuntimeError(
                "SERS context reviewer received wrong domain: "
                f"{context.domain_profile_id!r}"
            )

        inspirations = self._index_unique(
            list(
                dual.discovery_bundle.inspirations
            ),
            attribute="inspiration_id",
            label="discovery inspiration",
        )

        try:
            inspiration = inspirations[
                axis.inspiration_id
            ]
        except KeyError as exc:
            raise RuntimeError(
                "assigned discovery inspiration is absent "
                f"from DualHypothesisContext: {axis.inspiration_id}"
            ) from exc

        evidence = self._index_unique(
            list(
                context.evidence_statements
            ),
            attribute="statement_id",
            label="grounded evidence statement",
        )

        premise_ids = [
            str(value)
            for value
            in card.premise_statement_ids
        ]

        if not premise_ids:
            raise RuntimeError(
                "SERS context review requires at least one "
                "selected positive premise"
            )

        if len(
            premise_ids
        ) != len(
            set(premise_ids)
        ):
            raise RuntimeError(
                "duplicate premise_statement_ids in hypothesis"
            )

        missing = [
            statement_id
            for statement_id
            in premise_ids
            if statement_id not in evidence
        ]

        if missing:
            raise RuntimeError(
                "selected premise statements are absent from "
                "grounded context: "
                + ", ".join(missing)
            )

        source_signatures = [
            self.compiler
            .compile_grounded_statement(
                evidence[statement_id]
            )
            for statement_id
            in premise_ids
        ]

        # Inspiration remains inspiration-only. It is compiled separately
        # from positive-premise context and appended as exactly one axis
        # signature.
        source_signatures.append(
            self.compiler
            .compile_axis_inspiration(
                inspiration
            )
        )

        interpretation_outcome = (
            self.interpreter.interpret(
                card=card,
                source_signatures=source_signatures,
            )
        )

        interpretation = getattr(
            interpretation_outcome,
            "interpretation",
            None,
        )

        if interpretation is None:
            raise RuntimeError(
                "SERS hypothesis-context interpreter returned "
                "no validated interpretation"
            )

        review = self.comparator.compare(
            interpretation=interpretation,
            source_signatures=source_signatures,
            domain_profile_id=(
                self.domain_profile_id
            ),
        )

        if review.hypothesis_id != card.hypothesis_id:
            raise RuntimeError(
                "SERS context comparator hypothesis mismatch"
            )

        return review


@dataclass(frozen=True)
class SERSContextReviewAdapter:
    adapter_id: str = (
        SERS_AU_AG_CONTEXT_REVIEW_ADAPTER_ID
    )
    domain_profile_id: str = "sers_au_ag"

    def build(
        self,
        *,
        graph: Any,
        backend: Any,
    ) -> SERSDiscoveryAxisContextReviewer:
        return (
            SERSDiscoveryAxisContextReviewer
            .build(
                graph=graph,
                backend=backend,
                domain_profile_id=(
                    self.domain_profile_id
                ),
            )
        )


    def build_openai_compatible(
        self,
        *,
        graph: Any,
        model: str,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str | None = None,
        instructor_mode: str = "JSON",
        temperature: float = 0.0,
        parse_retries: int = 1,
        timeout: float | None = 180.0,
        extra_headers: dict[str, str] | None = None,
    ) -> SERSDiscoveryAxisContextReviewer:
        backend = (
            InstructorOpenAICompatibleHypothesisContextBackend(
                model=model,
                api_key_env=api_key_env,
                base_url=base_url,
                instructor_mode=instructor_mode,
                temperature=temperature,
                parse_retries=parse_retries,
                timeout=timeout,
                extra_headers=extra_headers,
            )
        )

        return self.build(
            graph=graph,
            backend=backend,
        )


SERS_AU_AG_CONTEXT_REVIEW_ADAPTER = (
    SERSContextReviewAdapter()
)
