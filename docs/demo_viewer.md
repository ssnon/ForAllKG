# GraphAgentsDAC demo viewer

The demo viewer turns the existing v2.7.0 E2E JSON artifacts into one self-contained HTML file. It does not call an LLM, mutate scientific results, or require a web server.

## What it shows

For each hypothesis:

1. **Lineage** — source paper / evidence premise → hypothesis → prediction & falsifier → semantic gate → scientific scope → validation specification → physics & experimental feasibility → final decision.
2. **Verification matrix** — every physics and experimental check with status, basis, and rationale, including scope-derived N/A checks.
3. **Validation design** — controlled/varied variables, comparison requirements, candidate-concretization requirements, success/falsification patterns, and next actions.
4. **Provenance** — source paper IDs, exact premise text, inferential bridge, assumptions, semantic warnings, scope warnings, and artifact IDs.

The viewer is intentionally read-only. Its purpose is demonstration and auditability, not a second reasoning layer.

## Build from an existing E2E run

From the repository root:

```bash
python -m scripts.build_demo_viewer \
  --run-dir runs/e2e/manual_001
```

The script auto-detects `feasibility_v02/`, `feasibility/`, or another `feasibility*` child that contains the v0.2 artifacts.

Default output:

```text
runs/e2e/manual_001/demo/index.html
```

Open it directly in a browser:

```bash
xdg-open runs/e2e/manual_001/demo/index.html
```

On WSL you can also use:

```bash
explorer.exe "$(wslpath -w runs/e2e/manual_001/demo/index.html)"
```

## Explicit feasibility directory

```bash
python -m scripts.build_demo_viewer \
  --run-dir runs/e2e/manual_001 \
  --feasibility-dir runs/e2e/manual_001/feasibility_v02 \
  --output runs/e2e/manual_001/demo/graphagentsdac_demo.html
```

## Artifact expectations

Required:

```text
<feasibility-dir>/
  feasibility/intake.json
  decision/portfolio.json
```

Used when present:

```text
  manifest.json
  scope/*.json
  validation/*.json
  physics/*.json
  experimental/*.json
```

The loader is deliberately schema-tolerant and reads JSON dictionaries instead of importing the Pydantic contracts. This makes the demo resilient to small contract-version changes while preserving artifact IDs and source text exactly.

## Tests

```bash
python -m pytest -q tests/test_demo_viewer.py
```
