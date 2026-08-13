# alpha4b.2c.1 anchor repair

The first alpha4b.2c installer stopped after successfully patching the domain
profile and both DAC-HER/SERS domain profiles. It failed before changing
`graphagents_adapter.py`.

Root cause: the installer looked for `_BACKTRACE_RELATIONS` using a string
containing literal escaped newlines rather than the actual source text.

This repair continues from that partial state, structurally locates the legacy
relation block and backtrace functions, completes projection domainization, and
runs HER/SERS regression tests.

No rollback of alpha4b.2c is required before applying this repair.
