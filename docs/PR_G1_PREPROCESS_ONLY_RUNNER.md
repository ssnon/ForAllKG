# PR G.1 — Target-driven preprocess-only corpus runner

Adds `scripts.run_corpus_preprocessing`, an orchestration lane that stops before
M4.5/Strict/Bridge.

The runner accepts a single `--target-count` and performs:

1. deterministic dynamic acquisition-profile generation with quota apportionment,
2. optional D.2 access-state recovery,
3. M3.2 acquisition-aware backfill to the requested main-PDF count,
4. optional M3.1 supplementary discovery/acquisition,
5. PR G incremental M4 materialization.

The persistent run manifest remembers the latest completed M3 snapshot so a
later invocation such as `100 -> 150 -> 200` grows from the previous result
rather than restarting from the original seed.

`target_count` is an acquisition target and is also checked against final M4
extraction-ready count. A technically successful run can therefore finish with
`materialization_shortfall` when M3 reached the requested count but M4 did not.
This is deliberate; it avoids treating `100 downloaded / 70 ready` as a fully
preprocessed 100-paper corpus.

Recommended Marker concurrency starts at `--m4-workers 2`. Use
`--retry-failed-materialization` when retrying technical M4 failures. The runner
never invokes M4.5, Strict, Bridge, corpus publication, or an LLM.
