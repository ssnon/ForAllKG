from __future__ import annotations

from dac_her.alpha4c5h1_runtime_bindings import (
    V6R2_RUNTIME_PRECISION_ADAPTER,
    V6R2_TREND_ADAPTER,
)
import scripts.build_cross_context_assessments as impl


def main() -> int:
    impl.get_trend_adapter = lambda _profile: V6R2_TREND_ADAPTER
    impl.get_trend_precision_adapter = (
        lambda _profile: V6R2_RUNTIME_PRECISION_ADAPTER
    )
    return impl.main()


if __name__ == "__main__":
    raise SystemExit(main())
