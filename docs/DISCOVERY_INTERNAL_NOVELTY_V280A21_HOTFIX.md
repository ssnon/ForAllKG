# GraphAgentsDAC v2.8.0-alpha2.1 discovery-selection hotfix

This hotfix responds to the 31-paper nitrogen-coordination benchmark where
alpha2 returned selected inspirations with `grounding_sem≈0.98–1.00` and
`selected_sem≈0.98–1.00`.

## Behavioral change

Discovery selection is now **quality-limited, not quota-limited**.

Default gates:

- semantic similarity to the grounding bundle must be `<= 0.95`
- exploration score must be `>= 0.05`
- pairwise semantic diversity uses the existing strict/relaxed thresholds
  (`0.88 / 0.94`)
- the bundle may contain fewer than `top_k`, including zero

The old alpha2 behavior can be reproduced only with `--force-fill` and should
be treated as a diagnostic ablation.

If zero paths survive, the intended next action is to add an
`exploratory`-mode traversal or broaden the retrieval question. The
discovery-aware Hypothesis Maker now fails closed on an empty DiscoveryBundle
unless `--allow-empty-discovery` is explicitly supplied.

## Recommended benchmark command

```bash
python -m scripts.build_discovery_bundle \
  --traversal "$RUN/traversal.json" \
  --top-k 8 \
  --output "$RUN/discovery.bundle.a21.json"
```

For the mechanism-only nitrogen-coordination benchmark, **zero or a small
under-filled bundle is a scientifically preferable result** if all candidate
paths simply replay the grounding neighborhood.
