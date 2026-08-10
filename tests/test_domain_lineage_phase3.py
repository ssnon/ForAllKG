from __future__ import annotations

import networkx as nx
import pytest

from dac_her.discovery_bundle import DiscoveryBundleBuilder, DiscoveryPolicy
from dac_her.discovery_contracts import DiscoveryBundle
from dac_her.domains import get_domain_profile
from dac_her.dual_hypothesis_context import DualHypothesisContext
from dac_her.hypothesis_compiler import HypothesisCompiler
from dac_her.hypothesis_contracts import (
    HypothesisContext,
    HypothesisPortfolioDraft,
)


def _context(domain: str, *, corpus: str = "shared-corpus") -> HypothesisContext:
    return HypothesisContext.model_validate(
        {
            "context_id": "context:same",
            "context_sha256": "same-context-sha",
            "source_packet_id": "packet:same",
            "source_packet_sha256": "packet-sha",
            "source_report_id": "report:same",
            "source_report_sha256": "report-sha",
            "task_id": "task:same",
            "question": "bounded scientific question",
            "corpus_id": corpus,
            "domain_profile_id": domain,
            "evidence_statements": [
                {
                    "statement_id": "stmt:1",
                    "text": "A reported observation.",
                    "epistemic_role": "reported",
                    "claim_kind": "observation",
                    "paper_ids": ["paper:1"],
                    "scientific_support_node_ids": ["node:1"],
                    "eligible_as_premise": True,
                    "eligible_as_gap": False,
                }
            ],
        }
    )


def _bundle(domain: str, *, corpus: str = "shared-corpus") -> DiscoveryBundle:
    return DiscoveryBundle.model_validate(
        {
            "bundle_id": f"bundle:{domain}",
            "bundle_sha256": f"bundle-sha:{domain}",
            "corpus_id": corpus,
            "domain_profile_id": domain,
            "query_signature": "same query",
            "inspirations": [],
        }
    )


def _draft() -> HypothesisPortfolioDraft:
    return HypothesisPortfolioDraft.model_validate(
        {
            "hypotheses": [
                {
                    "local_id": "h1",
                    "title": "Bounded hypothesis",
                    "hypothesis_statement": "A may influence B.",
                    "hypothesis_type": "mechanistic_extension",
                    "premise_statement_ids": ["stmt:1"],
                    "inferential_bridge": "bounded bridge",
                    "predicted_observations": [
                        {
                            "local_id": "p1",
                            "observable": "B",
                            "expected_direction": "qualitative_change",
                            "rationale": "tests the proposed relation",
                        }
                    ],
                    "falsification_criteria": [
                        {
                            "local_id": "f1",
                            "observable": "B",
                            "falsifying_outcome": "no reproducible relation",
                        }
                    ],
                }
            ]
        }
    )


def _empty_discovery_bundle_id(domain: str) -> str:
    profile = get_domain_profile(domain)
    payload = {
        "corpus_id": "shared-corpus",
        "domain_profile_id": domain,
        "mode": "mechanism",
        "source_query": "same source",
        "semantic_stop_query": None,
        "target_query": "same target",
        "candidate_paths": [],
        "paths": [],
    }
    bundle = DiscoveryBundleBuilder(
        DiscoveryPolicy(
            top_k=1,
            semantic_diversity_enabled=False,
            min_exploration_score=0.0,
        ),
        domain_profile=profile,
    ).build([("same.json", payload, nx.DiGraph())])
    assert bundle.domain_profile_id == domain
    return bundle.bundle_id


def test_discovery_bundle_domain_is_explicit_and_id_is_domain_sensitive():
    dac_id = _empty_discovery_bundle_id("dac_her")
    sers_id = _empty_discovery_bundle_id("sers_au_ag")
    assert dac_id != sers_id


def test_dual_context_rejects_same_corpus_but_different_domain():
    with pytest.raises(ValueError, match="domain_profile_id"):
        DualHypothesisContext.build(
            _context("dac_her"),
            _bundle("sers_au_ag"),
        )


def test_dual_context_parse_time_guard_rejects_tampered_domain():
    valid = DualHypothesisContext.build(
        _context("dac_her"),
        _bundle("dac_her"),
    )
    payload = valid.model_dump(mode="json")
    payload["domain_profile_id"] = "sers_au_ag"
    with pytest.raises(ValueError, match="domain profile mismatch"):
        DualHypothesisContext.model_validate(payload)


def test_dual_context_id_is_domain_sensitive_even_with_same_source_hashes():
    dac = DualHypothesisContext.build(
        _context("dac_her"),
        _bundle("dac_her"),
    )
    sers = DualHypothesisContext.build(
        _context("sers_au_ag"),
        _bundle("sers_au_ag"),
    )
    assert dac.domain_profile_id == "dac_her"
    assert sers.domain_profile_id == "sers_au_ag"
    assert dac.dual_context_id != sers.dual_context_id


def test_hypothesis_compiler_propagates_domain_and_ids_are_domain_sensitive():
    compiler = HypothesisCompiler()
    dac = compiler.compile(_context("dac_her"), _draft())
    sers = compiler.compile(_context("sers_au_ag"), _draft())

    assert dac.domain_profile_id == "dac_her"
    assert sers.domain_profile_id == "sers_au_ag"
    assert dac.hypotheses[0].domain_profile_id == "dac_her"
    assert sers.hypotheses[0].domain_profile_id == "sers_au_ag"
    assert dac.portfolio_id != sers.portfolio_id
    assert dac.hypotheses[0].hypothesis_id != sers.hypotheses[0].hypothesis_id


def test_legacy_contracts_default_to_dac_her_only():
    context_payload = _context("dac_her").model_dump(mode="json")
    bundle_payload = _bundle("dac_her").model_dump(mode="json")
    context_payload.pop("domain_profile_id")
    bundle_payload.pop("domain_profile_id")

    assert HypothesisContext.model_validate(context_payload).domain_profile_id == "dac_her"
    assert DiscoveryBundle.model_validate(bundle_payload).domain_profile_id == "dac_her"
