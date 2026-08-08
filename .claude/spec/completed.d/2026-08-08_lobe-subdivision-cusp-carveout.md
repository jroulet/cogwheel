---
date: 2026-08-08
section: Backlog
---

### Lobe-interior subdivision and the deltoid-cusp carve-out ruling

Two macro-saddle lobe items from the saddle-forensics audit
([[lensing_saddle_forensics]]) landed this build, both on
`cogwheel/lensing/surrogate_training.py`. The tracking fragments stay OPEN —
the shared-tiling-machine acceptance still wants `TubeChart` coverage, the OOP
shape, a region-scoped entry point, and the probe/trainer byte-identical test.

**`_subdivide_lobe_tile` exists and is wired.** The lobe subdivider is a thin
wrapper over the shared `_subdivide_tile` — the intended splitter/builder/gate
triple, not a third copy — with a single-sourced `_lobe_child_boxes` splitter
(midpoint halving in lobe-local `(rho_lobe, theta_local)`, four children in
row-major `(s_rho, s_theta)` order) shared between the wrapper's `child_half`
and the generic skeleton's splitter. `_subdivide_tile` now carries the
parent's `lobe` admission object onto child tiles, and `_train_band_charts`'
gated-lobe branch records `subdivided: True`, calls `_subdivide_lobe_tile`,
and sets `ladder_served_gap = subdivision['packed'] == 0` (a parent window is
now subdivided; only when ALL children fail does it remain a ladder gap).
Children re-admit through `_SaddleLobeAdmission.admits` (the single
authoritative admission predicate), retrain via `_build_lobe_chart`, and
re-gate on `config.interior_eps_max`; recursion is bounded by
`MAX_SUBDIVISION_DEPTH` inside `_subdivide_tile`. A carrier-flip
(`CarrierDiscontinuityError`) child is recorded as a ladder-served gap and
NEVER recursed — subdivision cannot fix critical-basin phase discontinuities.
INS-1-001 (child build passes the resolved `eff_w_nodes` to `_build_lobe_chart`,
matching the wedge subdivider) and INS-1-002 (the stale "lobe subdivision is
owed follow-on work" comment replaced) both RESOLVED.

**Deltoid-cusp carve-out ruling.** The separate cusp-ball carve-out is NOT
needed: `_SaddleLobeAdmission.admits`'s existing `eta_max` tube-shell
nearest-distance exclusion already rejects tiles too close to a deltoid cusp,
because the cusp vertices are points of the caustic cloud the test probes
against (Professor ruling). `_LOBE_CUSP_EXCLUSION_DISTANCE = 0.1` — sized
downward from the exterior `_CUSP_EXCLUSION_DISTANCE = 0.2` since deltoid
lobes are smaller than the full astroid — is recorded as a redundant constant
documenting the ruling; no separate admission branch was added.

Certified by `cogwheel/tests/test_lensing_lobe_subdivision.py` (19 tests):
`LobeSubdivisionTestCase` exercises the REAL `_subdivide_tile` skeleton and
`_gate_chart` against mocked `_build_lobe_chart` outcomes; `LobeCuspProximity
TestCase` pins the near-cusp admission refusal; `LobeCarrierFlipRefusalTestCase`
/ `LobeCarrierFlipSelfFalsificationTestCase` pin the catch-and-record, never-
recurse behaviour; `GhostKernelSaddleTestCase` is the first `ghost_kernel`
smoke at `gamma > 1` (structural only — see
[[lensing_saddle_forensics]] item f, still open).
