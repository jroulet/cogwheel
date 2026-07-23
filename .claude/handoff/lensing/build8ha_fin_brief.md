# Build 8h-a-fin — Complete Build 8h-a: WP4 + verification of the in-tree WP1-3

## Mission

Finish Build 8h-a, which died at the WP4 coder spawn on an
infrastructure error (argv E2BIG from an uncapped inlined agent memory
— SDK-fixed and memory-compacted since; not a content failure). WP1-3
of the APPROVED plan are implemented and sitting UNCOMMITTED in the
working tree; no Test Developer, Inspector, or Professor pass has run
on them yet. This build: (1) implement WP4 exactly as approved;
(2) author the full domain test batch; (3) verify EVERYTHING (WP1-4)
through the normal gates.

THE APPROVED PLAN IS THE SPEC: read
`.claude/handoff/lensing/build8ha_plan_approved.json` in full — its
WP4 section (What/Where/How/Verification), its Professor Inputs
(sup-over-w floor, 1e-4 bar, w_trust margin rule, morse-sign mask,
beyond-wall UNKNOWN), and its Domain Test Descriptions are binding
verbatim. Do NOT re-plan WP1-3; they are done — verify them.

## State of the tree (measured — do not re-derive)

- WP1 (cogwheel/lensing/ppgo_map.py + scripts/train_ppgo_map.py +
  registry entries), WP2 (band-split dispatch in
  cogwheel/lensing/likelihood.py, surrogate.py untouched), WP3
  (interior tiles + strata trim in surrogate_training.py) are IN THE
  WORKING TREE, uncommitted, coder-complete but UNVERIFIED. Coder
  change reports were lost with the crashed process — the diff is the
  source of truth: `git status --short` / `git diff` against HEAD
  (8a00fd9-era tree plus the memory/SDK housekeeping).
- WP4 (targeted edge-annulus subdivision in _train_band_charts) was
  NEVER STARTED. Its full spec is in the plan file.
- The plan's Domain Test Descriptions cover WP1-4; none are authored
  yet. All new tests go in cogwheel/tests/ beside the suite; the
  8g/8g-b batches (test_lensing_surrogate_training.py,
  test_lensing_farfield_envelope.py) are the style/vocabulary
  precedent and MUST STAY GREEN.
- Invariants that must hold (verify, do not assume): F005
  additive-only; envelope_definition tag dispatch (8g-b); tube
  byte-identity; per-w refusal propagation; beyond-wall UNKNOWN never
  certifies; zero-quadrature mandate (a refusal never falls through
  to numerical quadrature).
- Test tiers are LAW; tree-gate commit preflight active; in-build
  training/sweeps synthetic-scale only.

## Out of scope — hard fences

- NO re-implementation or redesign of WP1-3 (fix defects the
  Inspector finds, per the normal revision loop, but the design is
  the approved plan's).
- NO quad-double work; NO tube-chart changes; NO production sweeps or
  campaigns (driver post-build).
- NO SDK/infra changes (the E2BIG fix is already in the tree — leave
  .claude/sdk/memory.py and .serena/memories/* as found; commit them
  with the build).

## Acceptance (two-tier)

1. In-build (FAST): WP4 implemented per the plan and verified by its
   domain test (subdivision: children re-admitted, passing children
   packed, still-failing children recorded); the FULL domain test
   batch from the plan authored and green, including the sup-over-w
   synthetic-beat test, the F010 map-refusal truth table, the
   morse-sign cusp-adjacent fixture, the band-split seam test, and
   the interior telescoping identity; existing suites green (tree
   gate); tube byte-identity confirmed.
2. POST-BUILD (driver): production ppGO map sweep, campaign v4,
   ladder census (quadrature bucket 0% outside the measured
   beyond-wall tail), then the P1/P2 decision probes.
