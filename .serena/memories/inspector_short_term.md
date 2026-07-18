# Inspector Short-Term Observations

## 2026-07-18 — Build 3e review (envelope decomposition) — VERDICT ISSUES (design/escalation)

Scope: uncommitted diff, worktree /home/tejaswi/Work/cogwheel-claude-dev,
branch claude-dev. HEAD = 26505d5 ("build3d: abort record — interpolation
layer measured-exhausted; 3e brief"). The ONLY source-relevant uncommitted
change is a NEW test file `cogwheel/tests/test_lensing_envelope_reconstruction.py`.
No change to channels.py or likelihood.py (the WP1/WP2 targets).

### Build-3d issues (INS-3-001..004) are RESOLVED by the abort/revert.
The Build-3d uncommitted global-spline diff that INS-2/INS-3 reviewed was
reverted; the abort record was committed as 26505d5. Confirmed fresh this
session:
- `_DEFAULT_KERNEL_NODES = 100` again in likelihood.py (was 32 in the
  reverted diff) — matches SPEC.md verbatim, so INS-3-004 SPEC divergence
  is gone.
- `_amplification_coefficients` returns `(delays, k0, k1, partition)`,
  splines `partition.kernels` on `_coarse_w_node_grid` — the SPEC-described
  100-node union scheme. No segmentation code, no `_MAX_SEGMENT_NODES`
  import — INS-3-001/002 collection error gone.
- Full lensing suites GREEN: test_lensing_fast_path + _likelihood + _gauge
  + _channels = 94 passed in 77 s. New envelope suite = 10 passed, 1 xfailed
  in 4.25 s.

### Build 3e: WP1/WP2 NOT delivered — legitimate escalation, objective UNMET.
The approved plan's "code-pinned efficiency finding" is FALSE against the
tree: `transition_envelopes`, `image_amplification_factor`, `_dd_image`,
`_kernel_from_image_amplification` DO NOT EXIST (zero hits outside the plan
JSON + agent memories). The engine produces only the cluster TOTAL F(w)
(`operator.F_op`/`F_op_grid`) plus the DIVERGENT geometric per-image kernel
`geometry.image_kernel` (invalid inside an unresolved cluster). A smooth
per-image wave-optics residual R_j reproducing `exact_total` across the
deep-unresolved near-cusp/near-fold band IS the unsolved envelope
decomposition the Professor still owes. Coder + Test-Dev correctly REFUSED
to fabricate it (would be an unverifiable invented oracle, and needs new
physics in forbidden files). Both BLOCKED + ESCALATED. => 10 ms warm-lnlike
objective is UNMET; this must go to the owner as a design decision, not be
shipped green.

### The delivered test file is HONEST and correct (verified).
- `_gauge.reconstructed_total(w, member_delays, kernels)` exists, signature
  matches the test's positional call; carrier = `np.exp(1j*outer(w,tau))`.
- API-boundary guard: `@expectedFailure hasattr(...,'transition_envelopes')`
  is xfail today, flips to xpass/RED when WP1 lands. Companion green pins:
  channels public API == {evaluate,evaluate_path,reset,w} (verified via
  symbol overview); `_amplification_coefficients` source still returns the
  Build-3d contract and lacks `transition_envelopes` (verified by reading
  the method body).
- Spec-7 large-phase carrier: `_gauge.reconstructed_total` vs INDEPENDENT
  pure-mpmath oracle (mpc(cos,sin), dps=50) over crown-scale w up to 2000
  (w*tau ~ 6800 rad) — genuinely uncovered (gauge suite reaches w*tau<=36).
  CARRIER_REL_TOL=1e-11 is one order above the ~(w*tau)*eps~1e-12 float64
  floor — a real bound. Self-falsification drives w*tau~8.5e12 (irrational-
  scaled factors so the PRODUCT rounds; power-of-ten would be exact and
  numpy's accurate exp-range-reduction would hide it) and proves the gate
  can go RED + mod-2pi recovers it. Anti-vacuity: CarrierTestCase.tearDown
  fails on zero comparisons; OracleIndependenceTestCase AST-guards the
  oracle against {_gauge,channels,np,exp,...}.

### Findings raised this review
- INS-4-001 (design): Build-3e objective unmet; plan premise falsified;
  owner/Professor must produce the real decomposition before WP1/WP2. Not a
  code defect — the escalation is the correct outcome, but the 10 ms gate
  and the WP1/WP2 diffs are absent, so the build is ISSUES not PASS.
- INS-4-002 (trivial): dead var `kernels_full` in
  `test_carrier_error_does_not_grow_past_the_gate_with_phase` (computed then
  only `del`'d).

### Open issues carried forward
INS-4-001 open (owner design escalation). INS-3-001..004 CLOSED (reverted).

### Pattern
- After an "abort/revert" commit, re-confirm the tree matches SPEC and is
  green rather than trusting the reverted-diff memories — here `git log`
  + a single grep (`_DEFAULT_KERNEL_NODES`) + one suite run showed the
  Build-3d issues were gone and the coder's "current shipped uses 32"
  memory was STALE (described the reverted diff, not HEAD).
- A test-only build that pins the ABSENCE of the deliverable (xfail
  boundary guard) is the honest shape of a blocked build — verify it is
  non-vacuous and independent, but report the UNMET objective as ISSUES.
