# alpha4c.5h — Trend v6r2 Freeze & Reserve-B Confirmation Protocol

## State transition

alpha4c.5g.2r2 completed the Development-only candidate regression for:

`sers_au_ag_trend_v6r2_alpha4c5g2r2`

The accepted Development invariants are frozen here. This phase performs no
new scientific evaluation and never reads Reserve-B scientific values.

## What is frozen

The freeze binds:

- the exact v6r2 Development regression summary;
- the original alpha4c.5f.2 blind split and its SHA;
- the exact 25 Reserve-B paper IDs;
- the existing alpha4c.5e acceptance protocol ID;
- every current Python module under `dac_her`;
- existing scientific builder/hypothesis/trend/cross-context scripts selected
  by the freeze inventory.

The four historically pinned builder hashes from alpha4c.5g are also checked
before the freeze is written.

## Why execution is not performed here

Reserve B is a blind confirmation partition. Scientific semantics and
acceptance semantics must be frozen before any Reserve-B readiness/consumption
runner is allowed to operate.

Therefore alpha4c.5h emits only:

- `freeze_manifest.json`
- `reserve_b_confirmation_protocol.json`
- `freeze_status.json`

`freeze_status.json` starts with:

- reserve_b_consumed = false
- readiness_prepared = false
- consumption_marker_written = false

## Next epoch

alpha4c.5h.1 must:

1. verify the alpha4c.5h freeze byte-for-byte;
2. bind its own orchestration runner SHA;
3. prepare canonical readiness for only the 25 frozen Reserve-B IDs;
4. write a value-blind readiness lock;
5. reverify both freeze and readiness immediately before consumption;
6. only then write the guarded irreversible Reserve-B consumption marker.

No Reserve-B scientific result may be inspected before step 6.
