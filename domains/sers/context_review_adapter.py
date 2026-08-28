from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from domains.sers.context_comparator import (
    SERSHypothesisContextComparator,
)
from domains.sers.context_compiler import (
    SERSContextCompilationError,
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
from pipeline_core.discovery.discovery_axis_context_runtime import (
    AxisContextReviewUnavailableError,
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
        compiler: Any | None = None,
        grounded_compiler: Any | None = None,
        axis_compiler: Any | None = None,
        interpreter: Any,
        comparator: Any,
    ) -> None:
        if compiler is not None:
            if (
                grounded_compiler is not None
                or axis_compiler is not None
            ):
                raise ValueError(
                    "compiler cannot be combined with "
                    "grounded_compiler/axis_compiler"
                )

            grounded_compiler = compiler
            axis_compiler = compiler

        if (
            grounded_compiler is None
            or axis_compiler is None
        ):
            raise ValueError(
                "SERS context reviewer requires both "
                "grounded_compiler and axis_compiler"
            )

        self.grounded_compiler = (
            grounded_compiler
        )
        self.axis_compiler = (
            axis_compiler
        )

        # Compatibility alias only when both lanes intentionally share
        # one compiler. Production dual-lane wiring leaves this None.
        self.compiler = (
            grounded_compiler
            if grounded_compiler
            is axis_compiler
            else None
        )

        self.interpreter = interpreter
        self.comparator = comparator

    @classmethod
    def build(
        cls,
        *,
        backend: Any,
        graph: Any | None = None,
        grounded_graph: Any | None = None,
        axis_graph: Any | None = None,
        domain_profile_id: str = "sers_au_ag",
    ) -> "SERSDiscoveryAxisContextReviewer":
        if domain_profile_id != cls.domain_profile_id:
            raise ValueError(
                "SERS context reviewer/domain mismatch: "
                f"expected={cls.domain_profile_id!r}, "
                f"actual={domain_profile_id!r}"
            )

        if graph is not None:
            if (
                grounded_graph is not None
                or axis_graph is not None
            ):
                raise ValueError(
                    "graph cannot be combined with "
                    "grounded_graph/axis_graph"
                )

            # Explicit shared-graph compatibility/diagnostic mode:
            # both context lanes intentionally use the SAME compiler.
            shared_compiler = SERSContextCompiler(
                graph=graph,
                domain_profile_id=domain_profile_id,
            )

            return cls(
                compiler=shared_compiler,
                interpreter=SERSHypothesisContextInterpreter(
                    backend
                ),
                comparator=SERSHypothesisContextComparator(),
            )

        if (
            grounded_graph is None
            or axis_graph is None
        ):
            raise ValueError(
                "SERS context reviewer requires both "
                "grounded_graph and axis_graph"
            )

        # Production scientific path:
        # grounded premises and axis inspirations retain independent
        # source-graph provenance.
        return cls(
            grounded_compiler=SERSContextCompiler(
                graph=grounded_graph,
                domain_profile_id=domain_profile_id,
            ),
            axis_compiler=SERSContextCompiler(
                graph=axis_graph,
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
            self.grounded_compiler
            .compile_grounded_statement(
                evidence[statement_id]
            )
            for statement_id
            in premise_ids
        ]

        # Inspiration remains inspiration-only. It is compiled separately
        # from positive-premise context and appended as exactly one axis
        # signature.
        try:
            axis_signature = (
                self.axis_compiler
                .compile_axis_inspiration(
                    inspiration
                )
            )
        except SERSContextCompilationError as exc:
            raise AxisContextReviewUnavailableError(
                "assigned discovery inspiration cannot produce "
                "claim-local SERS scientific context: "
                f"{axis.inspiration_id}: {exc}"
            ) from exc

        source_signatures.append(
            axis_signature
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
        backend: Any,
        graph: Any | None = None,
        grounded_graph: Any | None = None,
        axis_graph: Any | None = None,
    ) -> SERSDiscoveryAxisContextReviewer:
        return (
            SERSDiscoveryAxisContextReviewer
            .build(
                backend=backend,
                graph=graph,
                grounded_graph=grounded_graph,
                axis_graph=axis_graph,
                domain_profile_id=(
                    self.domain_profile_id
                ),
            )
        )


    def build_openai_compatible(
        self,
        *,
        model: str,
        graph: Any | None = None,
        grounded_graph: Any | None = None,
        axis_graph: Any | None = None,
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
            backend=backend,
            graph=graph,
            grounded_graph=grounded_graph,
            axis_graph=axis_graph,
        )


SERS_AU_AG_CONTEXT_REVIEW_ADAPTER = (
    SERSContextReviewAdapter()
)
