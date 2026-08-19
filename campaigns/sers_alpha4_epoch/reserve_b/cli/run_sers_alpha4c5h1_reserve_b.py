from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from campaigns.sers_alpha4_epoch.reserve_b.alpha4c5h1_reserve_b import (
    Alpha4c5h1Protocol,
    load_h1_protocol,
    verify_h1_protocol,
)
from campaigns.sers_alpha4_epoch.readiness.canonical_readiness import (
    guarded_write_consumption_marker,
    load_and_verify_readiness_lock,
)
from campaigns.sers_alpha4_epoch.alpha4.cli.run_sers_alpha4c5f_reserve import (
    CampaignFailure,
    Runner as Frozen5fScientificRunner,
    now_iso,
    write_json,
)


ROOT = Path.cwd()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "alpha4c.5h.1 readiness-locked one-shot Reserve-B "
            "confirmation runner. Preflight is metadata/readiness only. "
            "Execution irreversibly consumes Reserve B immediately before "
            "the first scientific transformation."
        )
    )
    parser.add_argument("--protocol", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--execute-reserve-b", action="store_true")
    parser.add_argument(
        "--confirm-consume-reserve-b",
        action="store_true",
    )
    return parser.parse_args()


class ReserveBRunner(Frozen5fScientificRunner):
    def __init__(self, protocol: Alpha4c5h1Protocol) -> None:
        super().__init__(protocol)  # duck-typed frozen scientific runner
        self.protocol = protocol
        self.readiness_lock = (
            ROOT / protocol.canonical_readiness_lock_path
        )

    def verify(self) -> list[str]:
        return verify_h1_protocol(
            root=ROOT,
            protocol=self.protocol,
        )

    def status(self) -> int:
        print("alpha4c.5h.1 Reserve-B status")
        print("Protocol ID:", self.protocol.protocol_id)
        print("Campaign:", self.protocol.campaign_id)
        print("Papers:", len(self.protocol.reserve_paper_ids))
        print("Trend semantics:", self.protocol.trend_semantics_id)
        print("Readiness lock:", self.readiness_lock)
        print("Consumption marker:", self.marker.exists())
        print("PASS marker:", self.pass_marker.exists())
        print("FAIL marker:", self.fail_marker.exists())
        return 0

    def preflight(self) -> int:
        if self.marker.exists():
            raise CampaignFailure(
                "Reserve B is already consumed; this epoch cannot reopen it."
            )
        issues = self.verify()
        if self.work_data.exists():
            issues.append(
                "campaign work_data already exists before consumption"
            )
        if self.pass_marker.exists() or self.fail_marker.exists():
            issues.append(
                "campaign terminal marker exists before consumption"
            )

        print("alpha4c.5h.1 guarded Reserve-B preflight")
        print("Protocol ID:", self.protocol.protocol_id)
        print("Protocol SHA256:", self.protocol.protocol_sha256)
        print("Campaign:", self.protocol.campaign_id)
        print("Partition:", self.protocol.reserve_partition)
        print("Papers:", len(self.protocol.reserve_paper_ids))
        print("5h freeze ID:", self.protocol.five_h_freeze_id)
        print(
            "5h confirmation ID:",
            self.protocol.five_h_confirmation_protocol_id,
        )
        print(
            "Readiness lock SHA256:",
            self.protocol.canonical_readiness_lock_payload_sha256,
        )
        print("Trend semantics:", self.protocol.trend_semantics_id)
        print("Precision semantics:", self.protocol.precision_semantics_id)
        print("Scientific mode: evidence")
        print("Bridge required: False")
        print("New extraction LLM allowed: False")
        print("Direct consumption marker write allowed: False")
        print("Count thresholds used: False")
        print("Reserve B consumed:", False)
        print("LLM calls:", 0)

        if issues:
            print("Preflight: FAIL")
            for issue in issues:
                print(" -", issue)
            return 2
        print("5h freeze: CURRENT")
        print("Execution component hashes: CURRENT")
        print("Canonical readiness: LOCKED + CURRENT")
        print("Preflight: PASS")
        return 0

    def _build_evidence_substrate(self) -> None:
        ids = self.protocol.artifact_ids
        for paper_id in self.protocol.reserve_paper_ids:
            self._python(
                f"projection:{paper_id}",
                "scripts.build_graphagents_projection",
                "--paper-id", paper_id,
                "--domain-profile", "sers_au_ag",
                "--data-root", str(self.work_data),
                "--mode", "evidence",
            )

        self._python(
            "corpus",
            "scripts.build_corpus_graph",
            "--corpus-id", ids.corpus,
            "--domain-profile", "sers_au_ag",
            "--data-root", str(self.work_data),
            "--paper-ids",
            *self.protocol.reserve_paper_ids,
            "--mode", "evidence",
            "--allow-critical-partial",
        )
        self._python(
            "measurement_result_identity",
            "scripts.build_measurement_result_identities",
            "--domain-profile", "sers_au_ag",
            "--data-root", str(self.work_data),
            "--corpus-id", ids.corpus,
            "--mode", "evidence",
            "--measurement-result-identity-id",
            ids.measurement_result_identity,
        )
        self._python(
            "metric_definition",
            "scripts.build_metric_definition_contexts",
            "--domain-profile", "sers_au_ag",
            "--data-root", str(self.work_data),
            "--corpus-id", ids.corpus,
            "--mode", "evidence",
            "--metric-definition-id", ids.metric_definition,
            "--measurement-result-identity-id",
            ids.measurement_result_identity,
        )
        self._python(
            "comparison",
            "scripts.build_comparison_contexts",
            "--domain-profile", "sers_au_ag",
            "--data-root", str(self.work_data),
            "--corpus-id", ids.corpus,
            "--mode", "evidence",
            "--comparison-id", ids.comparison,
            "--metric-definition-id", ids.metric_definition,
            "--measurement-result-identity-id",
            ids.measurement_result_identity,
        )
        self._python(
            "trend_v6r2",
            "scripts.build_trend_evidence_alpha4c5h1",
            "--domain-profile", "sers_au_ag",
            "--data-root", str(self.work_data),
            "--corpus-id", ids.corpus,
            "--mode", "evidence",
            "--trend-id", ids.trend,
            "--measurement-result-identity-id",
            ids.measurement_result_identity,
            "--comparison-id", ids.comparison,
        )
        self._python(
            "trend_precision",
            "scripts.build_trend_precision_alpha4c5h1",
            "--domain-profile", "sers_au_ag",
            "--data-root", str(self.work_data),
            "--corpus-id", ids.corpus,
            "--mode", "evidence",
            "--trend-id", ids.trend,
            "--precision-id", ids.precision,
        )

        trend_summary = json.loads(
            (self.trend_root / "summary.json").read_text(
                encoding="utf-8"
            )
        )
        if (
            trend_summary.get("trend_semantics_id")
            != self.protocol.trend_semantics_id
        ):
            raise CampaignFailure(
                "Reserve-B Trend output did not use frozen v6r2 semantics."
            )

        local_results = self.precision_root / "local_results.jsonl"
        if not local_results.exists():
            raise CampaignFailure(
                f"precision local_results missing: {local_results}"
            )
        local_count = sum(
            bool(line.strip())
            for line in local_results.read_text(
                encoding="utf-8"
            ).splitlines()
        )
        write_json(
            self.eval_root / "trend_yield.json",
            {
                "local_result_count": local_count,
                "count_thresholds_used_for_acceptance": False,
                "zero_is_valid": True,
            },
        )

        if local_count > 0:
            self._python(
                "cross_context_profiles",
                "scripts.build_cross_context_profiles_alpha4c5h1",
                "--domain-profile", "sers_au_ag",
                "--data-root", str(self.work_data),
                "--corpus-id", ids.corpus,
                "--mode", "evidence",
                "--trend-id", ids.trend,
                "--precision-id", ids.precision,
                "--context-id", ids.context,
            )
            self._python(
                "cross_context_assessments",
                "scripts.build_cross_context_assessments_alpha4c5h1",
                "--domain-profile", "sers_au_ag",
                "--data-root", str(self.work_data),
                "--corpus-id", ids.corpus,
                "--mode", "evidence",
                "--trend-id", ids.trend,
                "--precision-id", ids.precision,
                "--context-id", ids.context,
                "--assessment-id", ids.assessment,
            )

    def execute(self) -> int:
        if self.marker.exists():
            raise CampaignFailure(
                "Reserve B already consumed. alpha4c.5h.1 is single-shot."
            )

        issues = self.verify()
        if issues:
            raise CampaignFailure(
                "Pre-execution frozen verification failed:\n- "
                + "\n- ".join(issues)
            )
        if self.work_data.exists():
            raise CampaignFailure(
                "Campaign work_data already exists before consumption. "
                "Refusing ambiguous restart."
            )
        if self.pass_marker.exists() or self.fail_marker.exists():
            raise CampaignFailure(
                "Campaign terminal marker exists before consumption."
            )

        self.eval_root.mkdir(parents=True, exist_ok=True)

        marker_payload = {
            "campaign_id": self.protocol.campaign_id,
            "protocol_id": self.protocol.protocol_id,
            "protocol_sha256": self.protocol.protocol_sha256,
            "five_h_freeze_id": self.protocol.five_h_freeze_id,
            "five_h_confirmation_protocol_id":
                self.protocol.five_h_confirmation_protocol_id,
            "evaluation_protocol_id":
                self.protocol.evaluation_protocol_id,
            "reserve_manifest_id":
                self.protocol.reserve_manifest_id,
            "reserve_manifest_sha256":
                self.protocol.reserve_manifest_sha256,
            "reserve_partition": "reserve_b",
            "paper_ids": self.protocol.reserve_paper_ids,
            "trend_semantics_id": self.protocol.trend_semantics_id,
            "started_at": now_iso(),
            "reserve_consumed": True,
            "reason": (
                "single-shot alpha4c.5h.1 Reserve-B confirmation began "
                "after immediate 5h freeze, execution-hash, and canonical "
                "readiness revalidation"
            ),
            "direct_marker_write_used": False,
        }

        # No canonical scientific graph is parsed/copied by this runner before
        # this call. verify() performs frozen metadata/hash/readiness checks.
        guarded = guarded_write_consumption_marker(
            root=ROOT,
            lock_path=self.readiness_lock,
            marker_path=self.marker,
            expected_paper_ids=self.protocol.reserve_paper_ids,
            expected_domain_profile_id=self.protocol.domain_profile_id,
            marker_payload=marker_payload,
        )
        print("[alpha4c.5h.1] RESERVE B CONSUMED:", self.marker)
        print(
            "[alpha4c.5h.1] canonical readiness revalidated:",
            guarded[
                "canonical_readiness_verified_immediately_before_consumption"
            ],
        )

        # Defensive post-boundary check. Any failure from here onward leaves
        # Reserve B consumed and forbids rerun/tuning under this epoch.
        load_and_verify_readiness_lock(
            root=ROOT,
            lock_path=self.readiness_lock,
            expected_paper_ids=self.protocol.reserve_paper_ids,
            expected_domain_profile_id=self.protocol.domain_profile_id,
        )

        try:
            write_json(
                self.eval_root / "campaign_manifest.json",
                {
                    "campaign_id": self.protocol.campaign_id,
                    "protocol_id": self.protocol.protocol_id,
                    "protocol_sha256": self.protocol.protocol_sha256,
                    "five_h_freeze_id":
                        self.protocol.five_h_freeze_id,
                    "reserve_manifest_id":
                        self.protocol.reserve_manifest_id,
                    "reserve_manifest_sha256":
                        self.protocol.reserve_manifest_sha256,
                    "canonical_readiness_lock_sha256":
                        self.protocol.canonical_readiness_lock_payload_sha256,
                    "canonical_readiness_verified_before_consumption":
                        True,
                    "trend_semantics_id":
                        self.protocol.trend_semantics_id,
                    "paper_ids": self.protocol.reserve_paper_ids,
                    "state": "running",
                    "started_at": marker_payload["started_at"],
                    "count_thresholds_used_for_acceptance": False,
                },
            )

            self._freeze_canonical_sources()
            self._build_evidence_substrate()
            context_path = self._build_explorer_context()
            input_path = self._build_trend_input(context_path)
            evaluation_path = self._run_maker_and_evaluate(input_path)
            evaluation = json.loads(
                evaluation_path.read_text(encoding="utf-8")
            )
            if evaluation.get("accepted") is not True:
                raise CampaignFailure(
                    "5e Reserve-B evaluation did not accept campaign."
                )

            success = {
                "campaign_id": self.protocol.campaign_id,
                "protocol_id": self.protocol.protocol_id,
                "protocol_sha256": self.protocol.protocol_sha256,
                "five_h_freeze_id": self.protocol.five_h_freeze_id,
                "reserve_manifest_sha256":
                    self.protocol.reserve_manifest_sha256,
                "canonical_readiness_verified_before_consumption": True,
                "completed_at": now_iso(),
                "reserve_consumed": True,
                "accepted": True,
                "fatal_issue_count":
                    evaluation.get("fatal_issue_count"),
                "hypothesis_count":
                    evaluation.get("hypothesis_count"),
                "abstained": evaluation.get("abstained"),
                "count_thresholds_used_for_acceptance": False,
                "evaluation_path": str(evaluation_path),
                "scientific_semantics_patch_authorized": False,
            }
            write_json(self.pass_marker, success)
            write_json(
                self.eval_root / "campaign_manifest.json",
                {**success, "state": "pass"},
            )
            print("\nalpha4c.5h.1 Reserve-B campaign PASS")
            print("Campaign:", self.protocol.campaign_id)
            print("Reserve B consumed:", True)
            print("5e accepted:", evaluation.get("accepted"))
            print("Fatal issues:", evaluation.get("fatal_issue_count"))
            print("Hypotheses:", evaluation.get("hypothesis_count"))
            print("Abstained:", evaluation.get("abstained"))
            print("Count thresholds used:", False)
            print("PASS marker:", self.pass_marker)
            return 0

        except Exception as exc:
            failure = {
                "campaign_id": self.protocol.campaign_id,
                "protocol_id": self.protocol.protocol_id,
                "protocol_sha256": self.protocol.protocol_sha256,
                "five_h_freeze_id": self.protocol.five_h_freeze_id,
                "failed_at": now_iso(),
                "reserve_consumed": True,
                "canonical_readiness_verified_before_consumption": True,
                "accepted": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "rerun_allowed": False,
                "reserve_b_failure_authorizes_tuning": False,
                "automatic_scientific_output_rollback": False,
                "count_thresholds_used_for_acceptance": False,
            }
            write_json(self.fail_marker, failure)
            write_json(
                self.eval_root / "campaign_manifest.json",
                {**failure, "state": "fail"},
            )
            print(
                "\n[alpha4c.5h.1] CAMPAIGN FAIL; Reserve B remains "
                "consumed and may not be rerun or used for semantic tuning "
                "under this epoch.",
                file=sys.stderr,
            )
            print(
                "[alpha4c.5h.1] failure marker:",
                self.fail_marker,
                file=sys.stderr,
            )
            raise


def main() -> int:
    args = parse_args()
    protocol_path = (
        args.protocol
        if args.protocol.is_absolute()
        else ROOT / args.protocol
    )
    protocol = load_h1_protocol(protocol_path)
    runner = ReserveBRunner(protocol)

    if args.status:
        return runner.status()
    if args.preflight:
        return runner.preflight()
    if args.execute_reserve_b:
        if not args.confirm_consume_reserve_b:
            raise SystemExit(
                "--confirm-consume-reserve-b is required for the real "
                "irreversible one-shot Reserve-B execution."
            )
        return runner.execute()
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
