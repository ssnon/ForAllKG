from __future__ import annotations

from pathlib import Path

import pytest

from pipeline_core.discovery.pre_n10_downstream_handoff_v1 import (
    _materialize_ready_lineage,
)
from pipeline_core.discovery.pre_n10_scientific_contract_v2 import (
    PreN10ScientificContractReportV2,
)
from tests.discovery.test_pre_n10_downstream_canonical_authority_s181c2 import (
    _canonical_initial,
)


def _inputs(tmp_path: Path):
    (
        portfolio_path,
        routed_path,
        gate,
        _primary,
        bundle_path,
        canonical_path,
        contract_v2_path,
    ) = _canonical_initial(tmp_path)

    from pipeline_core.discovery.atomic_scientific_source_provenance import (
        AtomicScientificSourceBindingBundle,
    )
    from pipeline_core.discovery.external_novelty_contracts import (
        LiteratureQueryPlan,
    )
    from pipeline_core.discovery.hypothesis_contracts import (
        HypothesisPortfolio,
    )
    from pipeline_core.discovery.pre_n10_canonical_source_reference_v1 import (
        PreN10CanonicalSourceReferenceReportV1,
    )

    portfolio = HypothesisPortfolio.model_validate_json(
        portfolio_path.read_text(encoding="utf-8")
    )
    plan = LiteratureQueryPlan.model_validate_json(
        routed_path.read_text(encoding="utf-8")
    )
    bundle = AtomicScientificSourceBindingBundle.model_validate_json(
        bundle_path.read_text(encoding="utf-8")
    )
    canonical = PreN10CanonicalSourceReferenceReportV1.model_validate_json(
        canonical_path.read_text(encoding="utf-8")
    )
    contract = PreN10ScientificContractReportV2.model_validate_json(
        contract_v2_path.read_text(encoding="utf-8")
    )
    disposition_path = Path(str(gate.semantic_disposition_path))
    hypothesis_id = portfolio.hypotheses[0].hypothesis_id
    return (
        portfolio,
        plan,
        bundle,
        canonical,
        contract,
        disposition_path,
        str(gate.semantic_disposition_id),
        hypothesis_id,
    )


def test_ready_lineage_uses_hypothesis_local_v2_not_global_disposition(
    tmp_path: Path,
) -> None:
    (
        portfolio,
        plan,
        bundle,
        canonical,
        contract,
        disposition_path,
        disposition_id,
        hypothesis_id,
    ) = _inputs(tmp_path)

    assert contract.hypotheses[0].contract_status == "READY_FOR_N10"

    mixed_report = contract.model_copy(
        update={"disposition": "INTERVENTION_REQUIRED"}
    )

    row = _materialize_ready_lineage(
        source_portfolio=portfolio,
        source_plan=plan,
        hypothesis_id=hypothesis_id,
        source_hypothesis_id=hypothesis_id,
        origin="INITIAL_PRIMARY_READY",
        semantic_disposition_path=disposition_path,
        semantic_disposition_id=disposition_id,
        output_root=tmp_path / "handoff",
        lineage_key="initial:" + hypothesis_id,
        source_binding_bundle=bundle,
        source_canonical_reference=canonical,
        source_contract_v2=mixed_report,
    )

    assert row.downstream_hypothesis_id == hypothesis_id
    assert row.stable_source_ids_used_for_pre_n10_authority is True


def test_ready_lineage_still_fails_closed_when_local_v2_not_ready(
    tmp_path: Path,
) -> None:
    (
        portfolio,
        plan,
        bundle,
        canonical,
        contract,
        disposition_path,
        disposition_id,
        hypothesis_id,
    ) = _inputs(tmp_path)

    not_ready_hypothesis = contract.hypotheses[0].model_copy(
        update={"contract_status": "REQUIRES_PRE_N10_INTERVENTION"}
    )
    mixed_report = contract.model_copy(
        update={
            "disposition": "INTERVENTION_REQUIRED",
            "hypotheses": [not_ready_hypothesis],
        }
    )

    with pytest.raises(
        ValueError,
        match="source hypothesis V2 is not READY_FOR_N10",
    ):
        _materialize_ready_lineage(
            source_portfolio=portfolio,
            source_plan=plan,
            hypothesis_id=hypothesis_id,
            source_hypothesis_id=hypothesis_id,
            origin="INITIAL_PRIMARY_READY",
            semantic_disposition_path=disposition_path,
            semantic_disposition_id=disposition_id,
            output_root=tmp_path / "handoff",
            lineage_key="initial:" + hypothesis_id,
            source_binding_bundle=bundle,
            source_canonical_reference=canonical,
            source_contract_v2=mixed_report,
        )


def test_ready_lineage_fails_closed_when_local_v2_row_missing(
    tmp_path: Path,
) -> None:
    (
        portfolio,
        plan,
        bundle,
        canonical,
        contract,
        disposition_path,
        disposition_id,
        hypothesis_id,
    ) = _inputs(tmp_path)

    missing_report = contract.model_copy(
        update={
            "disposition": "INTERVENTION_REQUIRED",
            "hypotheses": [],
        }
    )

    with pytest.raises(
        ValueError,
        match="hypothesis must resolve exactly once",
    ):
        _materialize_ready_lineage(
            source_portfolio=portfolio,
            source_plan=plan,
            hypothesis_id=hypothesis_id,
            source_hypothesis_id=hypothesis_id,
            origin="INITIAL_PRIMARY_READY",
            semantic_disposition_path=disposition_path,
            semantic_disposition_id=disposition_id,
            output_root=tmp_path / "handoff",
            lineage_key="initial:" + hypothesis_id,
            source_binding_bundle=bundle,
            source_canonical_reference=canonical,
            source_contract_v2=missing_report,
        )
