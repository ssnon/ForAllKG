from __future__ import annotations

from campaigns.sers_alpha4_epoch.alpha4.alpha4c5h1_runtime_bindings import (
    V6R2_TREND_ADAPTER,
)
import scripts.build_trend_evidence as impl


def main() -> int:
    impl.get_trend_adapter = lambda _profile: V6R2_TREND_ADAPTER
    return impl.main()


if __name__ == "__main__":
    raise SystemExit(main())
