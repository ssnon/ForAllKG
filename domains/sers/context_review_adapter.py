from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
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
    HypothesisContextInterpreterValidationError,
)
from domains.sers.hypothesis_context_llm import (
    InstructorOpenAICompatibleHypothesisContextBackend,
)
from domains.sers.context_contracts import (
    SERSContextFact,
    SERSContextProvenance,
    SERSContextSignature,
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
from pipeline_core.discovery.open_world_discovery_axis import (
    ExternalAxisProvenance,
    OpenWorldExternalAxisBundle,
)


SERS_AU_AG_CONTEXT_REVIEW_ADAPTER_ID = "sers_au_ag"

_EXTERNAL_UNKNOWN_CONTEXT_DIMENSIONS = (
    ("substrate", "plasmonic_substrate"),
    ("material_identity", "component"),
    ("material_state", "material_state"),
    ("support", "support"),
    ("morphology", "morphology"),
    ("architecture", "architecture"),
    ("structural_motif", "structural_motif"),
    ("gap_regime", "gap_regime"),
    ("optical_condition", "optical_condition"),
    ("analyte", "analyte"),
    ("reporter", "reporter"),
    ("measurement_geometry", "measurement_geometry"),
    ("environment", "environment"),
)


def _stable_external_context_id(prefix: str, payload: object) -> str:
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:20]}"


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
        self.external_axis_bundle: OpenWorldExternalAxisBundle | None = None

    def bind_external_axis_bundle(
        self,
        bundle: OpenWorldExternalAxisBundle,
    ) -> None:
        # Provenance binding only. This does not mutate DualHypothesisContext
        # and does not promote external literature to positive-premise or
        # novelty authority.
        if not isinstance(bundle, OpenWorldExternalAxisBundle):
            raise TypeError(
                "external axis context provenance must be an "
                "OpenWorldExternalAxisBundle"
            )
        self.external_axis_bundle = bundle

    def _external_axis_provenance(
        self,
        *,
        dual: DualHypothesisContext,
        axis: DiscoveryAxis,
    ) -> ExternalAxisProvenance:
        bundle = self.external_axis_bundle
        if bundle is None:
            raise AxisContextReviewUnavailableError(
                "external_open_world axis requires a bound, source-validated "
                "external-axis provenance bundle"
            )

        if bundle.source_dual_context_id != dual.dual_context_id:
            raise RuntimeError(
                "external-axis provenance dual_context_id mismatch"
            )
        if bundle.source_dual_context_sha256 != dual.dual_context_sha256:
            raise RuntimeError(
                "external-axis provenance dual_context_sha256 mismatch"
            )

        rows = [
            row
            for row in bundle.provenance
            if row.axis_id == axis.axis_id
        ]
        if len(rows) != 1:
            raise RuntimeError(
                "external axis requires exactly one provenance row: "
                f"axis={axis.axis_id}, found={len(rows)}"
            )

        row = rows[0]
        if not row.source_work_ids or not row.source_evidence_spans:
            raise RuntimeError(
                "external axis provenance lacks source works/evidence spans: "
                f"axis={axis.axis_id}"
            )
        return row

    @staticmethod
    def _external_axis_unknown_signature(
        *,
        axis: DiscoveryAxis,
        provenance: ExternalAxisProvenance,
    ) -> SERSContextSignature:
        # External literature is source-validated for the discovery axis, but
        # it is not graph-typed SERS context. Preserve that uncertainty rather
        # than inferring morphology/material/optical values from prose.
        excerpt = "\n\n".join(
            str(value).strip()
            for value in provenance.source_evidence_spans
            if str(value).strip()
        )

        facts = []
        for dimension, role in _EXTERNAL_UNKNOWN_CONTEXT_DIMENSIONS:
            fact_id = _stable_external_context_id(
                "sers_context_fact",
                {
                    "kind": "external_open_world_unknown_context",
                    "axis_id": axis.axis_id,
                    "inspiration_id": axis.inspiration_id,
                    "dimension": dimension,
                    "role": role,
                    "source_work_ids": sorted(provenance.source_work_ids),
                },
            )
            facts.append(
                SERSContextFact(
                    fact_id=fact_id,
                    dimension=dimension,
                    scientific_role=role,
                    knowledge_state="unknown",
                    value=None,
                    normalized_value=None,
                    binding=None,
                    provenance=[
                        SERSContextProvenance(
                            kind="external_axis_source_span",
                            paper_ids=sorted(provenance.source_work_ids),
                            statement_ids=sorted(
                                provenance.compatible_grounded_statement_ids
                            ),
                            excerpt=excerpt,
                        )
                    ],
                    tags=[
                        "external_open_world",
                        "inspiration_only",
                        "typed_context_unknown",
                    ],
                )
            )

        signature_id = _stable_external_context_id(
            "sers_context_signature",
            {
                "kind": "external_open_world_unknown_context",
                "axis_id": axis.axis_id,
                "inspiration_id": axis.inspiration_id,
                "fact_ids": [row.fact_id for row in facts],
            },
        )

        return SERSContextSignature(
            signature_id=signature_id,
            domain_profile_id="sers_au_ag",
            scope="axis_inspiration",
            source_ref_id=axis.inspiration_id,
            facts=facts,
        )

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
            [
                *dual.discovery_bundle.inspirations,
                *dual.task_lane_inspirations,
            ],
            attribute="inspiration_id",
            label="discovery inspiration",
        )

        inspiration = inspirations.get(
            axis.inspiration_id
        )
        external_axis_provenance = None

        if inspiration is None:
            if (
                getattr(
                    axis,
                    "source_mode",
                    "",
                )
                != "external_open_world"
            ):
                raise RuntimeError(
                    "assigned discovery inspiration is absent "
                    f"from DualHypothesisContext: {axis.inspiration_id}"
                )
            external_axis_provenance = self._external_axis_provenance(
                dual=dual,
                axis=axis,
            )

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

        # Inspiration remains inspiration-only. Persistent-KG inspirations
        # keep the original graph-compiled path. Source-validated open-world
        # axes use a traceable UNKNOWN typed-context projection rather than
        # inventing graph context or positive premises.
        if inspiration is not None:
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
        else:
            if external_axis_provenance is None:
                raise RuntimeError(
                    "external axis context provenance was not resolved"
                )

            missing_compatible = sorted(
                set(
                    external_axis_provenance
                    .compatible_grounded_statement_ids
                )
                - set(evidence)
            )
            if missing_compatible:
                raise RuntimeError(
                    "external axis provenance references grounded statements "
                    "absent from the active context: "
                    + ", ".join(missing_compatible)
                )

            axis_signature = self._external_axis_unknown_signature(
                axis=axis,
                provenance=external_axis_provenance,
            )

        source_signatures.append(
            axis_signature
        )

        try:
            interpretation_outcome = (
                self.interpreter.interpret(
                    card=card,
                    source_signatures=source_signatures,
                )
            )
        except HypothesisContextInterpreterValidationError as exc:
            # Final interpreter validation failure is axis-local.
            # Keep strict context validation intact; the discovery
            # runtime already converts this unavailable review into
            # context_rejected and continues to the next axis.
            raise AxisContextReviewUnavailableError(
                "SERS hypothesis-context interpretation unavailable: "
                + str(exc)
            ) from exc

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
