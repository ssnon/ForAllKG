from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.external_novelty_contracts import (
    LiteratureQueryPlan,
    NoveltyClaim,
)
from pipeline_core.discovery.relational_atomic_binding_plan import (
    RelationalAtomicBindingClaimPlan,
    RelationalAtomicBindingHypothesisPlan,
    RelationalAtomicBindingPlan,
)
from pipeline_core.discovery.relational_atomic_endpoint_binding import (
    CompiledLiteralEndpointBinding,
    RelationalAtomicEndpointBindingReport,
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
    return hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize(value: object) -> str:
    text = str(value or "").casefold()
    text = re.sub(r"[‐-‒–—−-]+", " ", text)
    text = re.sub(r"[^\w\s+*/().,]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def _surface_contains(text: str, phrase: str) -> bool:
    needle = _normalize(phrase)
    haystack = _normalize(text)
    return bool(needle and needle in haystack)


class RepairedVerifierProvenanceRebindManifest(StrictModel):
    schema_version: Literal[
        "repaired-verifier-provenance-rebind-manifest-v1"
    ] = "repaired-verifier-provenance-rebind-manifest-v1"

    manifest_id: str
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    case_id: Literal["P06", "P07", "P08", "P09", "P10"]

    source_binding_plan_id: str
    source_binding_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rebound_binding_plan_id: str
    rebound_binding_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    source_endpoint_report_id: str
    rebound_endpoint_report_id: str

    source_query_plan_paths: list[str]
    rebound_query_plan_paths: list[str]
    rebound_query_plan_file_sha256s: dict[str, str]

    rebound_claim_ids: list[str]
    rebound_claim_count: int = Field(ge=0)

    endpoint_anchors_preserved_exactly: Literal[True] = True
    endpoint_abstentions_preserved_exactly: Literal[True] = True
    endpoint_llm_recalled: Literal[False] = False
    specification_repair_recalled: Literal[False] = False
    literature_retrieval_performed: Literal[False] = False
    verifier_result_observed_during_rebind: Literal[False] = False
    scientific_content_added: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_manifest(
        self,
    ) -> "RepairedVerifierProvenanceRebindManifest":
        if self.rebound_claim_count != len(self.rebound_claim_ids):
            raise ValueError("rebound_claim_count mismatch")
        if len(self.rebound_claim_ids) != len(set(self.rebound_claim_ids)):
            raise ValueError("rebound claim IDs must be unique")

        body = self.model_dump(mode="json")
        observed_id = body.pop("manifest_id")
        observed_sha = body.pop("manifest_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("provenance rebind manifest SHA mismatch")
        if observed_id != (
            "repaired_verifier_provenance_rebind_manifest:"
            + expected_sha[:20]
        ):
            raise ValueError("provenance rebind manifest ID mismatch")
        return self


def _assert_binding_vs_source_claim(
    binding_claim: RelationalAtomicBindingClaimPlan,
    source_claim: NoveltyClaim,
) -> None:
    exact_pairs = (
        ("claim_id", binding_claim.claim_id, source_claim.claim_id),
        ("claim_rank", binding_claim.claim_rank, source_claim.claim_rank),
        ("kind", binding_claim.kind, source_claim.kind),
        ("importance", binding_claim.importance, source_claim.importance),
        (
            "novelty_selection_role",
            binding_claim.novelty_selection_role,
            source_claim.novelty_selection_role,
        ),
        ("claim_text", binding_claim.claim_text, source_claim.text),
        ("rationale", binding_claim.rationale, source_claim.rationale),
        (
            "prior_art_identity_terms",
            binding_claim.prior_art_identity_terms,
            source_claim.prior_art_identity_terms,
        ),
        (
            "relation_nucleus_terms",
            binding_claim.relation_nucleus_terms,
            source_claim.relation_nucleus_terms,
        ),
        (
            "search_concepts",
            binding_claim.search_concepts,
            source_claim.search_concepts,
        ),
        (
            "search_queries",
            binding_claim.search_queries,
            source_claim.search_queries,
        ),
    )
    mismatches = [
        label
        for label, left, right in exact_pairs
        if left != right
    ]
    if mismatches:
        raise ValueError(
            binding_claim.claim_id
            + ": repaired binding claim differs from canonical source "
            "outside permitted specification fields: "
            + repr(mismatches)
        )


def _replace_novelty_claim(
    *,
    source_claim: NoveltyClaim,
    binding_claim: RelationalAtomicBindingClaimPlan,
) -> NoveltyClaim:
    _assert_binding_vs_source_claim(binding_claim, source_claim)
    payload = source_claim.model_dump(mode="json")
    payload["required_bridge"] = binding_claim.required_bridge
    payload["predicted_observation"] = binding_claim.predicted_observation
    payload["falsification_condition"] = (
        binding_claim.falsification_condition
    )
    repaired = NoveltyClaim.model_validate(payload)
    return repaired


def _build_query_plan_sidecar(
    *,
    source_plan: LiteratureQueryPlan,
    candidate_hypothesis_id: str,
    repaired_claims: dict[str, RelationalAtomicBindingClaimPlan],
) -> tuple[LiteratureQueryPlan, dict[str, str]]:
    groups = []
    observed: set[str] = set()
    repaired_hashes: dict[str, str] = {}

    for group in source_plan.claims:
        payload = group.model_dump(mode="json")
        new_claims = []
        for claim in group.claims:
            repair = (
                repaired_claims.get(claim.claim_id)
                if group.hypothesis_id == candidate_hypothesis_id
                else None
            )
            if repair is None:
                new_claims.append(claim.model_dump(mode="json"))
                continue

            repaired = _replace_novelty_claim(
                source_claim=claim,
                binding_claim=repair,
            )
            repaired_payload = repaired.model_dump(mode="json")
            repaired_hashes[repair.claim_id] = _sha256_json(
                repaired_payload
            )
            new_claims.append(repaired_payload)
            observed.add(repair.claim_id)

        payload["claims"] = new_claims
        groups.append(payload)

    missing = sorted(set(repaired_claims) - observed)
    if missing:
        raise ValueError(
            "repaired claims absent from source query plan: "
            + repr(missing)
        )

    body = {
        "schema_version": source_plan.schema_version,
        "source_portfolio_id": source_plan.source_portfolio_id,
        "queries": [
            row.model_dump(mode="json")
            for row in source_plan.queries
        ],
        "claims": groups,
        "policy_version": source_plan.policy_version,
    }
    digest = _sha256_json(body)
    rebound = LiteratureQueryPlan(
        **body,
        plan_id="literature_query_plan:r1:" + digest[:20],
        plan_sha256=digest,
    )
    return rebound, repaired_hashes


def _rebuild_binding_plan(
    *,
    source_plan: RelationalAtomicBindingPlan,
    sidecar_paths: dict[str, Path],
    sidecar_file_hashes: dict[str, str],
    repaired_claim_hashes: dict[str, str],
) -> RelationalAtomicBindingPlan:
    hypotheses: list[RelationalAtomicBindingHypothesisPlan] = []

    for hypothesis in source_plan.hypotheses:
        sidecar = sidecar_paths.get(hypothesis.final_hypothesis_id)
        if sidecar is None:
            raise ValueError(
                hypothesis.final_hypothesis_id
                + ": missing rebound query-plan sidecar"
            )
        claims = []
        for claim in hypothesis.claims:
            payload = claim.model_dump(mode="json")
            if claim.claim_id in repaired_claim_hashes:
                payload["source_claim_sha256"] = (
                    repaired_claim_hashes[claim.claim_id]
                )
            claims.append(
                RelationalAtomicBindingClaimPlan.model_validate(payload)
            )

        hp = hypothesis.model_dump(mode="json")
        hp["source_query_plan"] = str(sidecar)
        hp["source_query_plan_sha256"] = sidecar_file_hashes[
            hypothesis.final_hypothesis_id
        ]
        hp["claims"] = [
            row.model_dump(mode="json")
            for row in claims
        ]
        hypotheses.append(
            RelationalAtomicBindingHypothesisPlan.model_validate(hp)
        )

    h_counts = Counter(row.binding_status for row in hypotheses)
    c_counts = Counter(
        claim.binding_status
        for row in hypotheses
        for claim in row.claims
    )
    body = source_plan.model_dump(mode="json")
    body.pop("plan_id")
    body.pop("plan_sha256")
    body["hypotheses"] = [
        row.model_dump(mode="json")
        for row in hypotheses
    ]
    body["hypothesis_status_counts"] = dict(sorted(h_counts.items()))
    body["claim_status_counts"] = dict(sorted(c_counts.items()))

    digest = _sha256_json(body)
    return RelationalAtomicBindingPlan(
        **body,
        plan_id="relational_atomic_binding_plan:" + digest[:20],
        plan_sha256=digest,
    )


def _validate_existing_binding_against_rebound_claim(
    *,
    claim: RelationalAtomicBindingClaimPlan,
    binding: CompiledLiteralEndpointBinding,
) -> None:
    if binding.claim_id != claim.claim_id:
        raise ValueError("endpoint/claim ID mismatch during rebind")
    if binding.outcome != "BOUND_LITERAL_ENDPOINTS":
        return

    for index, endpoint in enumerate(binding.relation_endpoint_anchors):
        if not _surface_contains(claim.claim_text, endpoint):
            raise ValueError(
                claim.claim_id
                + f": preserved endpoint[{index}] no longer literal in claim"
            )
        if not _surface_contains(claim.required_bridge, endpoint):
            raise ValueError(
                claim.claim_id
                + f": preserved endpoint[{index}] no longer literal in bridge"
            )
        endpoint_norm = _normalize(endpoint)
        for identity in claim.prior_art_identity_terms:
            identity_norm = _normalize(identity)
            if identity_norm and (
                endpoint_norm == identity_norm
                or identity_norm in endpoint_norm
            ):
                raise ValueError(
                    claim.claim_id
                    + ": preserved endpoint leaks branch identity"
                )


def _rebind_endpoint_report(
    *,
    source: RelationalAtomicEndpointBindingReport,
    rebound_plan: RelationalAtomicBindingPlan,
) -> RelationalAtomicEndpointBindingReport:
    claim_by_id = {
        claim.claim_id: claim
        for hypothesis in rebound_plan.hypotheses
        for claim in hypothesis.claims
    }
    rebound_bindings: list[CompiledLiteralEndpointBinding] = []

    for binding in source.bindings:
        claim = claim_by_id.get(binding.claim_id)
        if claim is None:
            raise ValueError(
                "endpoint binding claim absent from rebound plan: "
                + binding.claim_id
            )
        _validate_existing_binding_against_rebound_claim(
            claim=claim,
            binding=binding,
        )
        payload = binding.model_dump(mode="json")
        payload["source_claim_sha256"] = claim.source_claim_sha256
        rebound_bindings.append(
            CompiledLiteralEndpointBinding.model_validate(payload)
        )

    report_id = (
        "relational_atomic_endpoint_binding:"
        + _sha256_json(
            {
                "source_binding_plan_id": rebound_plan.plan_id,
                "bindings": [
                    row.model_dump(mode="json")
                    for row in rebound_bindings
                ],
                "backend_name": source.backend_name,
                "model_name": source.model_name,
            }
        )[:20]
    )
    return RelationalAtomicEndpointBindingReport(
        report_id=report_id,
        source_binding_plan_id=rebound_plan.plan_id,
        source_binding_plan_sha256=rebound_plan.plan_sha256,
        backend_name=source.backend_name,
        model_name=source.model_name,
        bindings=rebound_bindings,
        selected_hypothesis_count=source.selected_hypothesis_count,
        selected_claim_count=source.selected_claim_count,
        bound_claim_count=source.bound_claim_count,
        abstained_claim_count=source.abstained_claim_count,
        novelty_bearing_bound_claim_count=(
            source.novelty_bearing_bound_claim_count
        ),
        llm_calls_performed=source.llm_calls_performed,
        input_tokens=source.input_tokens,
        output_tokens=source.output_tokens,
    )


def build_provenance_rebound_inputs(
    *,
    case_id: str,
    source_binding_plan: RelationalAtomicBindingPlan,
    source_endpoint_report: RelationalAtomicEndpointBindingReport,
    sidecar_directory: Path,
    write_json_exclusive,
) -> tuple[
    RelationalAtomicBindingPlan,
    RelationalAtomicEndpointBindingReport,
    RepairedVerifierProvenanceRebindManifest,
    Path,
    Path,
]:
    if source_endpoint_report.source_binding_plan_id != (
        source_binding_plan.plan_id
    ):
        raise ValueError("source endpoint/binding-plan ID mismatch")
    if source_endpoint_report.source_binding_plan_sha256 != (
        source_binding_plan.plan_sha256
    ):
        raise ValueError("source endpoint/binding-plan SHA mismatch")

    root = sidecar_directory.expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError(
            "provenance-rebind directory must be fresh: " + str(root)
        )
    root.mkdir(parents=True, exist_ok=True)

    sidecar_paths: dict[str, Path] = {}
    sidecar_hashes: dict[str, str] = {}
    repaired_claim_hashes: dict[str, str] = {}
    source_query_paths: list[str] = []
    rebound_query_paths: list[str] = []

    for index, hypothesis in enumerate(source_binding_plan.hypotheses, start=1):
        source_query_path = Path(
            hypothesis.source_query_plan
        ).expanduser().resolve()
        if not source_query_path.is_file():
            raise ValueError(
                "source query plan missing: " + str(source_query_path)
            )
        if sha256_file(source_query_path) != hypothesis.source_query_plan_sha256:
            raise ValueError(
                "source query plan changed after binding-plan freeze"
            )
        source_query = LiteratureQueryPlan.model_validate_json(
            source_query_path.read_text(encoding="utf-8")
        )
        repaired_claims = {
            claim.claim_id: claim
            for claim in hypothesis.claims
        }
        rebound_query, hashes = _build_query_plan_sidecar(
            source_plan=source_query,
            candidate_hypothesis_id=hypothesis.candidate_hypothesis_id,
            repaired_claims=repaired_claims,
        )

        sidecar_path = root / (
            f"{index:02d}_{hypothesis.candidate_hypothesis_id.replace(':', '_')}"
            ".query_plan.r1_rebound.json"
        )
        write_json_exclusive(sidecar_path, rebound_query)
        file_hash = sha256_file(sidecar_path)

        sidecar_paths[hypothesis.final_hypothesis_id] = sidecar_path
        sidecar_hashes[hypothesis.final_hypothesis_id] = file_hash
        repaired_claim_hashes.update(hashes)
        source_query_paths.append(str(source_query_path))
        rebound_query_paths.append(str(sidecar_path))

    rebound_plan = _rebuild_binding_plan(
        source_plan=source_binding_plan,
        sidecar_paths=sidecar_paths,
        sidecar_file_hashes=sidecar_hashes,
        repaired_claim_hashes=repaired_claim_hashes,
    )
    rebound_endpoint = _rebind_endpoint_report(
        source=source_endpoint_report,
        rebound_plan=rebound_plan,
    )

    rebound_plan_path = root / "relational_atomic_binding_plan.r1_rebound.json"
    rebound_endpoint_path = (
        root / "relational_atomic_endpoint_binding.r1_rebound.json"
    )
    write_json_exclusive(rebound_plan_path, rebound_plan)
    write_json_exclusive(rebound_endpoint_path, rebound_endpoint)

    rebound_claim_ids = sorted(repaired_claim_hashes)
    body = {
        "schema_version":
            "repaired-verifier-provenance-rebind-manifest-v1",
        "case_id": case_id,
        "source_binding_plan_id": source_binding_plan.plan_id,
        "source_binding_plan_sha256": source_binding_plan.plan_sha256,
        "rebound_binding_plan_id": rebound_plan.plan_id,
        "rebound_binding_plan_sha256": rebound_plan.plan_sha256,
        "source_endpoint_report_id": source_endpoint_report.report_id,
        "rebound_endpoint_report_id": rebound_endpoint.report_id,
        "source_query_plan_paths": source_query_paths,
        "rebound_query_plan_paths": rebound_query_paths,
        "rebound_query_plan_file_sha256s": {
            final_id: sidecar_hashes[final_id]
            for final_id in sorted(sidecar_hashes)
        },
        "rebound_claim_ids": rebound_claim_ids,
        "rebound_claim_count": len(rebound_claim_ids),
        "endpoint_anchors_preserved_exactly": True,
        "endpoint_abstentions_preserved_exactly": True,
        "endpoint_llm_recalled": False,
        "specification_repair_recalled": False,
        "literature_retrieval_performed": False,
        "verifier_result_observed_during_rebind": False,
        "scientific_content_added": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
    }
    digest = _sha256_json(body)
    manifest = RepairedVerifierProvenanceRebindManifest(
        **body,
        manifest_id=(
            "repaired_verifier_provenance_rebind_manifest:"
            + digest[:20]
        ),
        manifest_sha256=digest,
    )
    manifest_path = root / "provenance_rebind_manifest.json"
    write_json_exclusive(manifest_path, manifest)

    return (
        rebound_plan,
        rebound_endpoint,
        manifest,
        rebound_plan_path,
        rebound_endpoint_path,
    )


__all__ = [
    "RepairedVerifierProvenanceRebindManifest",
    "build_provenance_rebound_inputs",
    "sha256_file",
]
