---
section: Backlog
---

- **THE POSITIVE-PARITY GOLDEN FIXTURE RECOMPUTES GEOMETRY, so it is a
  tripwire on all of `geometry.py`, not a serve-path pin** `[housekeeping]` —
  `PositiveParityGoldenTestCase` (`cogwheel/tests/test_lensing_surrogate_lobe.py`)
  asserts the served envelope BIT-FOR-BIT against three committed
  `float.hex()` pairs plus a SHA-256 artifact digest. Its docstring claimed
  the golden constants were "literals baked into this file, so the test
  remains a valid regression forever". The CONSTANTS are literals; the FIXTURE
  they are compared against is not — `_positive_golden_arc_map()` calls
  `surrogate._caustic_arclength_map(...)`, recomputing a `(4, 2001)`
  arc-length table from live caustic geometry on every run.
  MEASURED COST (2026-08-07): the `subdivision_recursion` build cut
  `r_caustic`'s positive-parity bracket count 720 -> 48 (a real 10.6x
  speedup, 1.788 s -> 0.169 s / 200 calls). Driver cross-tree verification
  over 6080 (gamma, theta) samples: zero refusal mismatches, worst 7.59e-15
  relative — pure `brentq` convergence noise. Propagated into the golden it
  moved exactly ONE ULP in the imaginary part of one of three elements (max
  rel 8.2e-17). That one ULP turned the tree gate RED, blocked the build's
  commit, and cost ~40 min of driver forensics (worktree A/B at pre-build
  HEAD, a 6080-point sweep, a timing A/B) to establish nothing was wrong.
  `test_served_value_tracks_unchanged_physical_oracle` — the test that checks
  the VALUE — was green throughout.
  FIX: commit the arc map as a fixture artifact (10012 floats, ~80 KB `.npz`)
  and load it instead of recomputing, making the golden what its docstring
  already promises — a frozen pin on the SERVE path alone, where red means a
  real defect. Interim state already applied: constants re-frozen with the
  perturbation measured and recorded inline, both docstrings corrected to
  state the real scope and the re-freeze protocol.
  OPEN: are there other bits-exact goldens whose fixtures recompute from live
  code? Grep the lensing suites for `tobytes()` / digest comparisons and check
  each one's fixture provenance — any that recompute share this failure mode.
  Same root shape as [[lensing_brief_premises_are_unverified]].
