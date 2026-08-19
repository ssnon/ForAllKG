from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from campaigns.sers_alpha4_epoch.alpha4.alpha4c5f2_reserve import (
    Alpha4c5f2Protocol,
    load_5f2_protocol,
    verify_5f2_protocol,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "alpha4c.5f.2 readiness-locked single-shot Reserve-A "
            "orchestrator. Preflight performs zero LLM calls and does not "
            "consume the reserve. Execution revalidates the frozen "
            "canonical-readiness lock immediately before the consumption "
            "marker, then reuses the frozen 5f scientific pipeline."
        )
    )
    parser.add_argument("--protocol", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--execute-reserve", action="store_true")
    parser.add_argument(
        "--confirm-consume-reserve",
        action="store_true",
    )
    return parser.parse_args()


class ReadinessLockedRunner(Frozen5fScientificRunner):
    def __init__(self, protocol: Alpha4c5f2Protocol) -> None:
        # The inherited methods are the already-frozen alpha4c.5f
        # scientific pipeline. This epoch overrides only verification and
        # the reserve-consumption boundary.
        super().__init__(protocol)  # type: ignore[arg-type]
        self.protocol = protocol
        self.readiness_lock = (
            ROOT / protocol.canonical_readiness_lock_path
            if not Path(
                protocol.canonical_readiness_lock_path
            ).is_absolute()
            else Path(protocol.canonical_readiness_lock_path)
        )

    def verify(self) -> list[str]:
        return verify_5f2_protocol(
            root=ROOT,
            protocol=self.protocol,
            verify_source_manifest=True,
        )

    def status(self) -> int:
        print("alpha4c.5f.2 Reserve-A status")
        print("Protocol ID:", self.protocol.protocol_id)
        print("Protocol SHA256:", self.protocol.protocol_sha256)
        print("Campaign:", self.protocol.campaign_id)
        print("Reserve papers:", len(self.protocol.reserve_paper_ids))
        print("Readiness lock:", self.readiness_lock)
        print("Consumption marker:", self.marker.exists())
        print("PASS marker:", self.pass_marker.exists())
        print("FAIL marker:", self.fail_marker.exists())
        return 0

    def preflight(self) -> int:
        if self.marker.exists():
            raise CampaignFailure(
                "Reserve A is already consumed; this epoch cannot reopen it."
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

        print("alpha4c.5f.2 readiness-locked Reserve-A preflight")
        print("Protocol ID:", self.protocol.protocol_id)
        print("Protocol SHA256:", self.protocol.protocol_sha256)
        print("Campaign:", self.protocol.campaign_id)
        print("Partition:", self.protocol.reserve_partition)
        print("Papers:", len(self.protocol.reserve_paper_ids))
        print("Pool SHA256:", self.protocol.pool_manifest_sha256)
        print("Split SHA256:", self.protocol.blind_split_sha256)
        print(
            "Readiness lock SHA256:",
            self.protocol.canonical_readiness_lock_payload_sha256,
        )
        print("5e protocol SHA256:", self.protocol.evaluation_protocol_sha256)
        print("Reserve manifest SHA256:", self.protocol.reserve_manifest_sha256)
        print("Scientific mode: evidence")
        print("Bridge required: False")
        print("New extraction LLM allowed: False")
        print("Direct consumption marker write allowed: False")
        print("Count thresholds used: False")
        print("Reserve A consumed:", False)
        print("Reserve B execution allowed:", False)
        print("LLM calls:", 0)

        if issues:
            print("Preflight: FAIL")
            for issue in issues:
                print(" -", issue)
            return 2
        print("Canonical readiness: LOCKED + CURRENT")
        print("Preflight: PASS")
        return 0

    def execute(self) -> int:
        if self.marker.exists():
            raise CampaignFailure(
                "Reserve A already consumed. alpha4c.5f.2 is single-shot."
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
            "evaluation_protocol_sha256":
                self.protocol.evaluation_protocol_sha256,
            "reserve_manifest_sha256":
                self.protocol.reserve_manifest_sha256,
            "pool_manifest_sha256":
                self.protocol.pool_manifest_sha256,
            "blind_split_sha256":
                self.protocol.blind_split_sha256,
            "reserve_partition":
                self.protocol.reserve_partition,
            "paper_ids":
                self.protocol.reserve_paper_ids,
            "started_at": now_iso(),
            "reserve_consumed": True,
            "reason": (
                "single-shot alpha4c.5f.2 scientific execution began "
                "after immediate canonical-readiness revalidation"
            ),
            "direct_marker_write_used": False,
        }

        # THIS is the new alpha4c.5f.2 consumption boundary.
        # No canonical graph has been parsed/copied by this runner, no
        # projection/Trend/Explorer/Maker has run, and no scientific output
        # has been inspected before this guarded call.
        guarded = guarded_write_consumption_marker(
            root=ROOT,
            lock_path=self.readiness_lock,
            marker_path=self.marker,
            expected_paper_ids=self.protocol.reserve_paper_ids,
            expected_domain_profile_id=self.protocol.domain_profile_id,
            marker_payload=marker_payload,
        )
        print("[alpha4c.5f.2] RESERVE A CONSUMED:", self.marker)
        print(
            "[alpha4c.5f.2] canonical readiness revalidated:",
            guarded[
                "canonical_readiness_verified_immediately_before_consumption"
            ],
        )

        # Defensive post-boundary verification. Failure here still leaves the
        # reserve consumed, as required for a single-shot blind campaign.
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
                    "protocol_sha256":
                        self.protocol.protocol_sha256,
                    "reserve_manifest_id":
                        self.protocol.reserve_manifest_id,
                    "reserve_manifest_sha256":
                        self.protocol.reserve_manifest_sha256,
                    "pool_manifest_sha256":
                        self.protocol.pool_manifest_sha256,
                    "blind_split_sha256":
                        self.protocol.blind_split_sha256,
                    "canonical_readiness_lock_sha256":
                        self.protocol.canonical_readiness_lock_payload_sha256,
                    "canonical_readiness_verified_before_consumption":
                        True,
                    "paper_ids":
                        self.protocol.reserve_paper_ids,
                    "state": "running",
                    "started_at": marker_payload["started_at"],
                    "count_thresholds_used_for_acceptance": False,
                },
            )

            # Reuse the frozen alpha4c.5f downstream scientific sequence
            # verbatim after the corrected consumption boundary.
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
                    "5e reserve evaluation did not accept campaign."
                )

            success = {
                "campaign_id": self.protocol.campaign_id,
                "protocol_id": self.protocol.protocol_id,
                "protocol_sha256": self.protocol.protocol_sha256,
                "reserve_manifest_sha256":
                    self.protocol.reserve_manifest_sha256,
                "canonical_readiness_verified_before_consumption":
                    True,
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
            }
            write_json(self.pass_marker, success)
            write_json(
                self.eval_root / "campaign_manifest.json",
                {**success, "state": "pass"},
            )
            print("\nalpha4c.5f.2 Reserve-A campaign PASS")
            print("Campaign:", self.protocol.campaign_id)
            print("Reserve consumed:", True)
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
                "failed_at": now_iso(),
                "reserve_consumed": True,
                "canonical_readiness_verified_before_consumption": True,
                "accepted": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "rerun_allowed": False,
                "automatic_scientific_output_rollback": False,
                "count_thresholds_used_for_acceptance": False,
            }
            write_json(self.fail_marker, failure)
            write_json(
                self.eval_root / "campaign_manifest.json",
                {**failure, "state": "fail"},
            )
            print(
                "\n[alpha4c.5f.2] CAMPAIGN FAIL; Reserve A remains "
                "consumed and may not be rerun under this epoch.",
                file=sys.stderr,
            )
            print(
                "[alpha4c.5f.2] Reserve B remains sealed.",
                file=sys.stderr,
            )
            print(
                "[alpha4c.5f.2] failure marker:",
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
    protocol = load_5f2_protocol(protocol_path)
    runner = ReadinessLockedRunner(protocol)

    if args.status:
        return runner.status()
    if args.preflight:
        return runner.preflight()
    if args.execute_reserve:
        if not args.confirm_consume_reserve:
            raise SystemExit(
                "--confirm-consume-reserve is required for the real "
                "single-shot Reserve-A execution."
            )
        return runner.execute()
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
