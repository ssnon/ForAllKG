# PR-G0 — Multi-domain isolation foundation

This refactor prepares strict extraction for DAC-HER, SERS, Broad catalysis,
and future domains without intentionally changing the current scientific
behavior of the three existing profiles.

1. Compact-generation schema selection becomes adapter-owned.
2. Runtime strict relation endpoint selection becomes adapter-owned.
3. DAC-HER and Broad retain the legacy DAC endpoint matrix they already saw.
4. SERS retains only the shared measurement/claim subset it previously saw.
5. Broad compact-schema implementation hashing becomes conditional on compact
   mode so future Broad compact edits do not invalidate DAC-HER/SERS runs.

Deferred:
- Full SERS endpoint strictness.
- Domain-specific metric vocabulary views.
- Compact domain-gate recovery.
- Global vocabulary-YAML fingerprint isolation.
