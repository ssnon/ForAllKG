from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.atomic_scientific_source_provenance import (
    AtomicScientificSourceBindingBundle,
)
from pipeline_core.discovery.atomic_scientific_specification_bundle import (
    AtomicScientificSpecificationBundle,
    build_atomic_scientific_specification_bundle_from_source_bindings,
)
from pipeline_core.discovery.external_novelty_contracts import LiteratureQueryPlan
from pipeline_core.discovery.hypothesis_contracts import HypothesisPortfolio
from pipeline_core.discovery.pre_n10_downstream_handoff_v1 import (
    PreN10DownstreamHandoffReportV1,
)
from pipeline_core.discovery.pre_n10_external_n10_shadow_v1 import (
    PreN10ExternalN10ShadowReportV1,
    classify_pre_n10_n10_production_gate_v1,
)
from pipeline_core.discovery.relational_atomic_binding_plan import (
    RelationalAtomicBindingHypothesisPlan,
    RelationalAtomicBindingPlan,
    assess_claim_binding_readiness,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _canonical_json(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pretty_json_bytes(value: object) -> bytes:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _write_exact_or_validate(path: Path, payload: object) -> str:
    path = path.expanduser().resolve()
    expected = _pretty_json_bytes(payload)
    if path.exists():
        observed = path.read_bytes()
        if observed != expected:
            raise ValueError(
                "existing write-once relational binding bridge artifact differs: "
                + str(path)
            )
        return hashlib.sha256(observed).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(expected)
    return hashlib.sha256(expected).hexdigest()


def _slug(value: str) -> str:
    result = str(value).replace(":", "_").replace("/", "_")
    if not result or result in {".", ".."}:
        raise ValueError("invalid downstream lineage ID")
    return result


def _hypothesis_binding_status(
    *,
    ready_count: int,
    novelty_ready_count: int,
) -> Literal[
    "READY_FOR_LITERAL_ENDPOINT_BINDING",
    "NO_BINDABLE_CLAIMS",
    "NO_NOVELTY_BEARING_BINDABLE_CLAIM",
]:
    if ready_count == 0:
        return "NO_BINDABLE_CLAIMS"
    if novelty_ready_count == 0:
        return "NO_NOVELTY_BEARING_BINDABLE_CLAIM"
    return "READY_FOR_LITERAL_ENDPOINT_BINDING"


class PreN10RelationalBindingBridgeLineageV1(StrictModel):
    lineage_id: str
    origin: Literal["INITIAL_PRIMARY_READY", "REGENERATED_REENTRY_READY"]
    source_hypothesis_id: str
    downstream_hypothesis_id: str

    n10_certification_status: Literal[
        "NOVELTY_CERTIFIED",
        "NOVELTY_UNRESOLVED",
        "NOVELTY_REJECTED",
    ]
    n10_selection_class: Literal["ELIGIBLE", "CONDITIONAL", "INELIGIBLE"]

    source_external_report_path: str
    source_external_report_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_n10_production_gate_path: str
    source_n10_production_gate_file_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )

    binding_plan_path: str
    binding_plan_id: str
    binding_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    binding_plan_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    binding_status: str
    claim_count: int = Field(ge=0)
    binding_ready_claim_count: int = Field(ge=0)
    novelty_bearing_binding_ready_claim_count: int = Field(ge=0)

    candidate_final_identity_mode: Literal[
        "EXACT_DOWNSTREAM_IDENTITY_NO_POST_N10_MUTATION"
    ] = "EXACT_DOWNSTREAM_IDENTITY_NO_POST_N10_MUTATION"
    candidate_final_authority_equivalent: Literal[True] = True
    n10_certification_status_filters_binding_reachability: Literal[False] = False
    endpoint_binding_performed: Literal[False] = False
    verifier_performed: Literal[False] = False
    production_selection_changed: Literal[False] = False


class PreN10RelationalBindingBridgeReportV1(StrictModel):
    schema_version: Literal[
        "pre-n10-relational-binding-bridge-report-v1"
    ] = "pre-n10-relational-binding-bridge-report-v1"

    report_id: str
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    source_handoff_report_id: str
    source_handoff_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_external_n10_report_id: str
    source_external_n10_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_external_n10_report_path: str
    source_external_n10_report_file_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )

    canonical_spec_bundle_path: str | None = None
    canonical_spec_bundle_id: str | None = None
    canonical_spec_bundle_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    canonical_spec_bundle_file_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )

    lineages: list[PreN10RelationalBindingBridgeLineageV1]
    lineage_count: int = Field(ge=0)
    binding_ready_lineage_count: int = Field(ge=0)
    not_binding_ready_lineage_count: int = Field(ge=0)
    claim_count: int = Field(ge=0)
    binding_ready_claim_count: int = Field(ge=0)
    novelty_bearing_binding_ready_claim_count: int = Field(ge=0)
    n10_certification_status_counts: dict[str, int]
    binding_status_counts: dict[str, int]

    exact_handoff_population_consumed: Literal[True] = True
    stable_source_ids_used_for_relational_input: bool = False
    exact_text_reconstruction_required_for_relational_input: bool = True
    n10_population_consumed_without_survival_filter: Literal[True] = True
    conditional_may_reach_binding_when_contract_ready: Literal[True] = True
    rejected_may_reach_binding_for_diagnostic_verification: Literal[True] = True
    candidate_final_exact_identity_required: Literal[True] = True
    endpoint_binding_performed: Literal[False] = False
    verifier_performed: Literal[False] = False
    production_selection_authority: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_report(self) -> "PreN10RelationalBindingBridgeReportV1":
        if self.lineage_count != len(self.lineages):
            raise ValueError("binding bridge lineage_count mismatch")
        ready = sum(
            row.binding_status == "READY_FOR_LITERAL_ENDPOINT_BINDING"
            for row in self.lineages
        )
        if self.binding_ready_lineage_count != ready:
            raise ValueError("binding-ready lineage count mismatch")
        if self.not_binding_ready_lineage_count != len(self.lineages) - ready:
            raise ValueError("not-binding-ready lineage count mismatch")
        if self.claim_count != sum(row.claim_count for row in self.lineages):
            raise ValueError("binding bridge claim_count mismatch")
        if self.binding_ready_claim_count != sum(
            row.binding_ready_claim_count for row in self.lineages
        ):
            raise ValueError("binding-ready claim count mismatch")
        if self.novelty_bearing_binding_ready_claim_count != sum(
            row.novelty_bearing_binding_ready_claim_count
            for row in self.lineages
        ):
            raise ValueError("novelty-bearing binding-ready claim count mismatch")

        canonical_fields = (
            self.canonical_spec_bundle_path,
            self.canonical_spec_bundle_id,
            self.canonical_spec_bundle_sha256,
            self.canonical_spec_bundle_file_sha256,
        )
        if self.stable_source_ids_used_for_relational_input:
            if any(value is None for value in canonical_fields):
                raise ValueError(
                    "stable-ID relational input requires canonical "
                    "specification bundle"
                )
            if self.exact_text_reconstruction_required_for_relational_input:
                raise ValueError(
                    "stable-ID relational input cannot require exact-text "
                    "reconstruction"
                )
        else:
            if any(value is not None for value in canonical_fields):
                raise ValueError(
                    "legacy relational input cannot carry canonical "
                    "specification bundle"
                )
            if not self.exact_text_reconstruction_required_for_relational_input:
                raise ValueError(
                    "legacy relational input must declare exact-text "
                    "reconstruction"
                )

        expected_n10 = Counter(row.n10_certification_status for row in self.lineages)
        if dict(sorted(expected_n10.items())) != dict(
            sorted(self.n10_certification_status_counts.items())
        ):
            raise ValueError("N10 certification status counts mismatch")
        expected_binding = Counter(row.binding_status for row in self.lineages)
        if dict(sorted(expected_binding.items())) != dict(
            sorted(self.binding_status_counts.items())
        ):
            raise ValueError("binding status counts mismatch")

        body = self.model_dump(mode="json")
        observed_id = body.pop("report_id")
        observed_sha = body.pop("report_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("relational binding bridge SHA mismatch")
        if observed_id != (
            "pre_n10_relational_binding_bridge_v1:" + expected_sha[:20]
        ):
            raise ValueError("relational binding bridge ID mismatch")
        return self


def _load_json_object(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("expected JSON object: " + str(path))
    return payload


def _build_lineage_plan(
    *,
    handoff_row,
    n10_row,
    external_n10_report_path: Path,
    lineage_root: Path,
    canonical_relation_endpoints_by_claim: (
        dict[str, list[str]] | None
    ) = None,
) -> RelationalAtomicBindingPlan:
    for path, expected_sha, label in (
        (
            handoff_row.portfolio_path,
            handoff_row.portfolio_file_sha256,
            "handoff portfolio",
        ),
        (
            handoff_row.query_plan_path,
            handoff_row.query_plan_file_sha256,
            "handoff query plan",
        ),
        (
            n10_row.external_report_path,
            n10_row.external_report_file_sha256,
            "external novelty report",
        ),
        (
            n10_row.n10_production_gate_path,
            n10_row.n10_production_gate_file_sha256,
            "N10 production gate",
        ),
    ):
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise ValueError("missing " + label + ": " + str(resolved))
        if _sha256_file(resolved) != expected_sha:
            raise ValueError(label + " changed after upstream freeze")

    if handoff_row.lineage_id != n10_row.lineage_id:
        raise ValueError("handoff/N10 lineage ID mismatch")
    if handoff_row.origin != n10_row.origin:
        raise ValueError("handoff/N10 origin mismatch")
    if handoff_row.source_hypothesis_id != n10_row.source_hypothesis_id:
        raise ValueError("handoff/N10 source-hypothesis mismatch")
    if handoff_row.downstream_hypothesis_id != n10_row.downstream_hypothesis_id:
        raise ValueError("handoff/N10 downstream-hypothesis mismatch")
    if handoff_row.portfolio_id != n10_row.portfolio_id:
        raise ValueError("handoff/N10 portfolio ID mismatch")
    if handoff_row.query_plan_id != n10_row.query_plan_id:
        raise ValueError("handoff/N10 query-plan ID mismatch")

    portfolio_path = Path(handoff_row.portfolio_path).expanduser().resolve()
    query_plan_path = Path(handoff_row.query_plan_path).expanduser().resolve()
    portfolio = HypothesisPortfolio.model_validate_json(
        portfolio_path.read_text(encoding="utf-8")
    )
    query_plan = LiteratureQueryPlan.model_validate_json(
        query_plan_path.read_text(encoding="utf-8")
    )

    if portfolio.portfolio_id != handoff_row.portfolio_id:
        raise ValueError("handoff portfolio ID drift")
    if query_plan.plan_id != handoff_row.query_plan_id:
        raise ValueError("handoff query-plan ID drift")
    if query_plan.source_portfolio_id != portfolio.portfolio_id:
        raise ValueError("binding bridge query-plan/portfolio provenance mismatch")

    cards = [
        row
        for row in portfolio.hypotheses
        if row.hypothesis_id == handoff_row.downstream_hypothesis_id
    ]
    if len(cards) != 1 or len(portfolio.hypotheses) != 1:
        raise ValueError("binding bridge requires one exact downstream hypothesis")

    groups = [
        row
        for row in query_plan.claims
        if row.hypothesis_id == handoff_row.downstream_hypothesis_id
    ]
    if len(groups) != 1:
        raise ValueError("binding bridge requires one exact downstream claim group")

    claim_ids = {
        claim.claim_id
        for claim in groups[0].claims
    }
    if canonical_relation_endpoints_by_claim is not None:
        if set(canonical_relation_endpoints_by_claim) != claim_ids:
            raise ValueError(
                "canonical endpoint population differs from downstream "
                "claim population"
            )

    production_gate = _load_json_object(n10_row.n10_production_gate_path)
    status, gate_row = classify_pre_n10_n10_production_gate_v1(
        origin=n10_row.origin,
        downstream_hypothesis_id=n10_row.downstream_hypothesis_id,
        portfolio_id=n10_row.portfolio_id,
        query_plan_id=n10_row.query_plan_id,
        gate=production_gate,
    )
    if status != n10_row.certification_status:
        raise ValueError("N10 report/production-gate certification mismatch")
    if str(gate_row.get("selection_class") or "") != n10_row.n10_selection_class:
        raise ValueError("N10 report/production-gate selection mismatch")
    if (
        gate_row.get("positive_nonobviousness_authority")
        is not n10_row.positive_nonobviousness_authority
    ):
        raise ValueError("N10 report/production-gate positive-authority mismatch")
    if gate_row.get("fallback_allowed") is not n10_row.fallback_allowed:
        raise ValueError("N10 report/production-gate fallback mismatch")

    candidate_id = handoff_row.downstream_hypothesis_id
    final_id = candidate_id
    claim_plans = [
        assess_claim_binding_readiness(
            claim=claim,
            candidate_hypothesis_id=candidate_id,
            final_hypothesis_id=final_id,
            relation_endpoint_anchors=(
                canonical_relation_endpoints_by_claim[claim.claim_id]
                if canonical_relation_endpoints_by_claim is not None
                else None
            ),
        )
        for claim in groups[0].claims
    ]
    ready_count = sum(
        row.binding_status == "READY_FOR_LITERAL_ENDPOINT_BINDING"
        for row in claim_plans
    )
    novelty_ready_count = sum(
        row.binding_status == "READY_FOR_LITERAL_ENDPOINT_BINDING"
        and row.novelty_selection_role == "NOVELTY_BEARING"
        for row in claim_plans
    )
    binding_status = _hypothesis_binding_status(
        ready_count=ready_count,
        novelty_ready_count=novelty_ready_count,
    )

    hypothesis_plan = RelationalAtomicBindingHypothesisPlan(
        original_hypothesis_id=handoff_row.source_hypothesis_id,
        candidate_hypothesis_id=candidate_id,
        final_hypothesis_id=final_id,
        alpha6_decision=handoff_row.origin,
        certification_status=n10_row.certification_status,
        n10_selection_class=n10_row.n10_selection_class,
        source_candidate_portfolio=str(portfolio_path),
        source_candidate_portfolio_sha256=_sha256_file(portfolio_path),
        source_query_plan=str(query_plan_path),
        source_query_plan_sha256=_sha256_file(query_plan_path),
        claim_count=len(claim_plans),
        binding_ready_claim_count=ready_count,
        novelty_bearing_binding_ready_claim_count=novelty_ready_count,
        binding_status=binding_status,
        claims=claim_plans,
    )

    h_counts = Counter([hypothesis_plan.binding_status])
    c_counts = Counter(row.binding_status for row in claim_plans)
    body = {
        "schema_version": "relational-atomic-binding-plan-v1",
        "run_dir": str(lineage_root.resolve()),
        "source_alpha6_candidate_portfolio": str(portfolio_path),
        "source_alpha6_candidate_portfolio_sha256": _sha256_file(portfolio_path),
        "source_certification_report": str(
            external_n10_report_path.expanduser().resolve()
        ),
        "source_certification_report_sha256": _sha256_file(
            external_n10_report_path
        ),
        "hypotheses": [hypothesis_plan.model_dump(mode="json")],
        "hypothesis_count": 1,
        "ready_hypothesis_count": int(
            binding_status == "READY_FOR_LITERAL_ENDPOINT_BINDING"
        ),
        "not_ready_hypothesis_count": int(
            binding_status != "READY_FOR_LITERAL_ENDPOINT_BINDING"
        ),
        "claim_count": len(claim_plans),
        "binding_ready_claim_count": ready_count,
        "novelty_bearing_binding_ready_claim_count": novelty_ready_count,
        "hypothesis_status_counts": dict(sorted(h_counts.items())),
        "claim_status_counts": dict(sorted(c_counts.items())),
        "certification_only_source_required": True,
        "candidate_final_authority_equivalence_required": True,
        "endpoint_binding_performed": False,
        "verifier_result_observed": False,
        "production_selection_authority": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
    }
    digest = _sha256_json(body)
    return RelationalAtomicBindingPlan(
        **body,
        plan_id="relational_atomic_binding_plan:" + digest[:20],
        plan_sha256=digest,
    )


def build_pre_n10_relational_binding_bridge_v1(
    *,
    handoff: PreN10DownstreamHandoffReportV1,
    external_n10_report_path: Path,
    output_root: Path,
) -> PreN10RelationalBindingBridgeReportV1:
    external_path = external_n10_report_path.expanduser().resolve()
    if not external_path.is_file():
        raise ValueError(
            "missing unified external/N9/N10 shadow report: " + str(external_path)
        )
    external = PreN10ExternalN10ShadowReportV1.model_validate_json(
        external_path.read_text(encoding="utf-8")
    )
    if external.source_handoff_report_id != handoff.report_id:
        raise ValueError("external-N10/handoff report ID mismatch")
    if external.source_handoff_report_sha256 != handoff.report_sha256:
        raise ValueError("external-N10/handoff report SHA mismatch")

    handoff_by_id = {row.lineage_id: row for row in handoff.lineages}
    external_ids = [row.lineage_id for row in external.lineages]
    handoff_ids = [row.lineage_id for row in handoff.lineages]
    if external_ids != handoff_ids:
        raise ValueError(
            "binding bridge population/order differs from frozen handoff"
        )

    root = output_root.expanduser().resolve()

    canonical_count = sum(
        row.stable_source_ids_used_for_pre_n10_authority
        for row in handoff.lineages
    )
    if canonical_count not in {0, len(handoff.lineages)}:
        raise ValueError(
            "binding bridge cannot mix canonical and legacy handoff lineages"
        )

    canonical_spec_bundle = None
    canonical_spec_bundle_path = None
    canonical_spec_bundle_file_sha = None
    canonical_relation_endpoints_by_hypothesis: dict[
        str,
        dict[str, list[str]],
    ] = {}
    canonical_mode = bool(handoff.lineages) and (
        canonical_count == len(handoff.lineages)
    )
    if canonical_mode:
        canonical_inputs = []
        for handoff_row in handoff.lineages:
            if handoff_row.source_binding_bundle_path is None:
                raise ValueError(
                    "canonical handoff lineage lacks source-binding bundle"
                )
            portfolio_path = Path(
                handoff_row.portfolio_path
            ).expanduser().resolve()
            query_plan_path = Path(
                handoff_row.query_plan_path
            ).expanduser().resolve()
            source_bundle_path = Path(
                handoff_row.source_binding_bundle_path
            ).expanduser().resolve()
            for path, expected_sha, label in (
                (
                    portfolio_path,
                    handoff_row.portfolio_file_sha256,
                    "canonical handoff portfolio",
                ),
                (
                    query_plan_path,
                    handoff_row.query_plan_file_sha256,
                    "canonical handoff query plan",
                ),
                (
                    source_bundle_path,
                    handoff_row.source_binding_bundle_file_sha256,
                    "canonical handoff source-binding bundle",
                ),
            ):
                if not path.is_file():
                    raise ValueError(
                        "missing " + label + ": " + str(path)
                    )
                if _sha256_file(path) != expected_sha:
                    raise ValueError(
                        label + " changed after handoff freeze"
                    )

            portfolio = HypothesisPortfolio.model_validate_json(
                portfolio_path.read_text(encoding="utf-8")
            )
            query_plan = LiteratureQueryPlan.model_validate_json(
                query_plan_path.read_text(encoding="utf-8")
            )
            source_bundle = (
                AtomicScientificSourceBindingBundle.model_validate_json(
                    source_bundle_path.read_text(encoding="utf-8")
                )
            )
            if source_bundle.bundle_id != (
                handoff_row.source_binding_bundle_id
            ):
                raise ValueError(
                    "handoff/source-binding bundle ID mismatch at bridge"
                )
            canonical_inputs.append(
                (portfolio, query_plan, source_bundle)
            )

        canonical_spec_bundle = (
            build_atomic_scientific_specification_bundle_from_source_bindings(
                source_report_id=handoff.report_id,
                source_contract=handoff.schema_version,
                inputs=canonical_inputs,
            )
        )
        canonical_spec_bundle_path = (
            root / "canonical_atomic_specification.bundle.json"
        )
        canonical_spec_bundle_file_sha = _write_exact_or_validate(
            canonical_spec_bundle_path,
            canonical_spec_bundle,
        )

        for hypothesis in canonical_spec_bundle.hypotheses:
            if hypothesis.hypothesis_id in (
                canonical_relation_endpoints_by_hypothesis
            ):
                raise ValueError(
                    "duplicate canonical hypothesis in relational bridge"
                )
            per_claim = {
                specification.claim_id: list(
                    specification.relation_endpoint_anchors
                )
                for specification in hypothesis.specifications
            }
            if len(per_claim) != len(hypothesis.specifications):
                raise ValueError(
                    "duplicate canonical claim in relational bridge"
                )
            canonical_relation_endpoints_by_hypothesis[
                hypothesis.hypothesis_id
            ] = per_claim

    rows: list[PreN10RelationalBindingBridgeLineageV1] = []
    for n10_row in external.lineages:
        handoff_row = handoff_by_id[n10_row.lineage_id]
        lineage_root = root / "lineage" / _slug(n10_row.lineage_id)

        canonical_endpoints = None
        if canonical_mode:
            canonical_endpoints = (
                canonical_relation_endpoints_by_hypothesis.get(
                    handoff_row.downstream_hypothesis_id
                )
            )
            if canonical_endpoints is None:
                raise ValueError(
                    "canonical relational bridge lacks downstream "
                    "hypothesis specification"
                )

        plan = _build_lineage_plan(
            handoff_row=handoff_row,
            n10_row=n10_row,
            external_n10_report_path=external_path,
            lineage_root=lineage_root,
            canonical_relation_endpoints_by_claim=canonical_endpoints,
        )
        plan_path = lineage_root / "relational_atomic_binding_plan.json"
        file_sha = _write_exact_or_validate(plan_path, plan)
        hp = plan.hypotheses[0]
        rows.append(
            PreN10RelationalBindingBridgeLineageV1(
                lineage_id=n10_row.lineage_id,
                origin=n10_row.origin,
                source_hypothesis_id=n10_row.source_hypothesis_id,
                downstream_hypothesis_id=n10_row.downstream_hypothesis_id,
                n10_certification_status=n10_row.certification_status,
                n10_selection_class=n10_row.n10_selection_class,
                source_external_report_path=str(
                    Path(n10_row.external_report_path).expanduser().resolve()
                ),
                source_external_report_file_sha256=(
                    n10_row.external_report_file_sha256
                ),
                source_n10_production_gate_path=str(
                    Path(n10_row.n10_production_gate_path).expanduser().resolve()
                ),
                source_n10_production_gate_file_sha256=(
                    n10_row.n10_production_gate_file_sha256
                ),
                binding_plan_path=str(plan_path.resolve()),
                binding_plan_id=plan.plan_id,
                binding_plan_sha256=plan.plan_sha256,
                binding_plan_file_sha256=file_sha,
                binding_status=hp.binding_status,
                claim_count=hp.claim_count,
                binding_ready_claim_count=hp.binding_ready_claim_count,
                novelty_bearing_binding_ready_claim_count=(
                    hp.novelty_bearing_binding_ready_claim_count
                ),
            )
        )

    n10_counts = Counter(row.n10_certification_status for row in rows)
    binding_counts = Counter(row.binding_status for row in rows)
    body = {
        "schema_version": "pre-n10-relational-binding-bridge-report-v1",
        "source_handoff_report_id": handoff.report_id,
        "source_handoff_report_sha256": handoff.report_sha256,
        "source_external_n10_report_id": external.report_id,
        "source_external_n10_report_sha256": external.report_sha256,
        "source_external_n10_report_path": str(external_path),
        "source_external_n10_report_file_sha256": _sha256_file(external_path),
        "canonical_spec_bundle_path": (
            str(canonical_spec_bundle_path.resolve())
            if canonical_spec_bundle_path is not None
            else None
        ),
        "canonical_spec_bundle_id": (
            canonical_spec_bundle.bundle_id
            if canonical_spec_bundle is not None
            else None
        ),
        "canonical_spec_bundle_sha256": (
            canonical_spec_bundle.bundle_sha256
            if canonical_spec_bundle is not None
            else None
        ),
        "canonical_spec_bundle_file_sha256": (
            canonical_spec_bundle_file_sha
        ),
        "lineages": [row.model_dump(mode="json") for row in rows],
        "lineage_count": len(rows),
        "binding_ready_lineage_count": sum(
            row.binding_status == "READY_FOR_LITERAL_ENDPOINT_BINDING"
            for row in rows
        ),
        "not_binding_ready_lineage_count": sum(
            row.binding_status != "READY_FOR_LITERAL_ENDPOINT_BINDING"
            for row in rows
        ),
        "claim_count": sum(row.claim_count for row in rows),
        "binding_ready_claim_count": sum(
            row.binding_ready_claim_count for row in rows
        ),
        "novelty_bearing_binding_ready_claim_count": sum(
            row.novelty_bearing_binding_ready_claim_count for row in rows
        ),
        "n10_certification_status_counts": dict(sorted(n10_counts.items())),
        "binding_status_counts": dict(sorted(binding_counts.items())),
        "exact_handoff_population_consumed": True,
        "stable_source_ids_used_for_relational_input": canonical_mode,
        "exact_text_reconstruction_required_for_relational_input": (
            not canonical_mode
        ),
        "n10_population_consumed_without_survival_filter": True,
        "conditional_may_reach_binding_when_contract_ready": True,
        "rejected_may_reach_binding_for_diagnostic_verification": True,
        "candidate_final_exact_identity_required": True,
        "endpoint_binding_performed": False,
        "verifier_performed": False,
        "production_selection_authority": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
    }
    digest = _sha256_json(body)
    report = PreN10RelationalBindingBridgeReportV1(
        **body,
        report_id="pre_n10_relational_binding_bridge_v1:" + digest[:20],
        report_sha256=digest,
    )
    _write_exact_or_validate(root / "relational_binding_bridge.report.json", report)
    return report


__all__ = [
    "PreN10RelationalBindingBridgeLineageV1",
    "PreN10RelationalBindingBridgeReportV1",
    "build_pre_n10_relational_binding_bridge_v1",
]
