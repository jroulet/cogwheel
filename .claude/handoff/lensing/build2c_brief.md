# Build 2c — Switch-neighbourhood fix: close the crown gate

## Mission (one sentence)

Fix the one-line channel-switch neighbourhood bug that caused every crown-gate
accuracy failure, revert the dense sub-sampling that was compensating for it,
and re-base the affected tests — turning Build 2's red suite green at the
ORIGINAL tolerances.

## Context (read first)

- `.claude/handoff/lensing/META_PLAN.md` — sections "BUILD 2B SUITE RESULT",
  "PROFESSOR CONSULT VERDICT", and "INVESTIGATION CONVERGENCE" (the last one
  supersedes the middle one's re-scoping recommendation).
- The paper: `.claude/spec/lensing_paper/`, Sec. "Topology-stable
  four-component representation" — Eq. (delay-separation), Eq. (switch),
  Eq. (explicit-K). The decomposition is CORRECT near cusps; the
  implementation deviates in one line.

## Root cause — PROVED and MEASURED (do not re-derive; do not re-litigate)

`_channel_switch` (cogwheel/lensing/chang_refsdal/channels.py:313; the bug is
the line `others = real_ids[real_ids != channel]` at :342) measures each real
channel's delay separation against OTHER REAL channels only. The paper's
Eq. (delay-separation) takes the minimum over ALL cluster members —
INCLUDING labels parked at the critical point (virtual labels). On the
2-image side of a caustic, a near-critical image's actual cluster-mates ARE
the parked virtual labels (measured gap 5.5e-5 at the crown near-cusp config
vs 0.856 to the persistent image), so the switch spuriously ramps to 1 and
hands the channel to the divergent saddle kernel H (|H_0| ~ 1.8e8 there,
growth ~ gap^-2 approaching the cusp), flooding all four channels via the
residual projection (|K_a| ~ 5.2e5 cancelling coherently to |F| ~ 3).

Measured consequences of correcting the neighbourhood (two independent
agents; scratch probes only, repo untouched):

| config    | switch        | max|k0| | RB−brute lnl offset | 1.5 gate |
|-----------|---------------|---------|---------------------|----------|
| two-image | current (bug) | 40.9    | +9.768              | FAIL     |
| two-image | fixed         | 0.922   | +0.080              | PASS     |
| near-cusp | current (bug) | 5.22e5  | +6.43e8             | FAIL     |
| near-cusp | fixed         | 0.975   | +0.329              | PASS     |

- `kernel_subsamples=2` under the fixed switch: +0.069 / +0.316 (both PASS).
- p+s≤3 == p+s≤4 == p+s≤5 to <1e-4 once kernels are bounded: NO norm-moment
  change is needed; the contraction algebra is correct as shipped.
- Brute force is switch-independent (uses `exact_total`; reconstruction
  ~1e-16 under either switch) — the oracle never moved.
- Reconstruction error IMPROVES under the fix: 2.5e-10 → 5e-16.
- The fix is one-directional (can only LOWER a switch) and a no-op wherever
  all four labels are real (4-image regions; near-fold-inside unchanged).

## Work packages (code only — you write, downstream verifies)

### WP1 — the switch fix
`cogwheel/lensing/chang_refsdal/channels.py`, `_channel_switch`:
- Replace the real-only neighbour set with ALL channels ≠ self (the parked
  virtual labels are legitimate neighbours), per Eq. (delay-separation).
- Correct the docstring: it currently encodes the wrong rule ("nearest real
  neighbour").
- AUDIT the sibling `_min_delay_separation` (channels.py:352), which has the
  same real-only pattern and feeds the wave/geometric branch gate. Measured:
  `exact_total` is unaffected by the switch bug, so this sibling is NOT
  implicated in the crown failures — align it with the paper's
  cluster-separation definition if the paper requires it, otherwise document
  precisely why it is correct as written. Do not guess: cite the equation.

Acceptance: channel kernels at the crown near-cusp config are bounded
(max|K_a| = O(1)) and the reconstruction residual is at machine precision;
four-image configs bit-unchanged.

### WP2 — retire the dense-subsample compensation
`cogwheel/lensing/likelihood.py`:
- Revert the `kernel_subsamples` DEFAULT from 8 to 2. Keep the sub-sampling
  machinery itself (it is correct and harmless; it was merely compensating
  for WP1's bug at 8× the engine cost). Update the class docstring: dense
  sub-sampling is a robustness margin, not a correctness requirement.

Acceptance: engine evaluations per lnlike drop ~4× (2024 → 506 points at the
crown fixture); RB lnlike beats brute force by well over the 3× timing gate.

### WP3 — spec/FINDINGS corrections
- FINDINGS F006: the edge-secant/slope-squaring mechanism attribution is
  SUPERSEDED (it was sign-disproven: the norm term went NEGATIVE-huge, which
  slope-squaring cannot produce). Correct it — do not delete history; state
  what F006 got right (dense sampling changed nothing → mechanism must lie
  elsewhere) and record the actual cause as a new finding (switch
  neighbourhood bug, paper Eq. delay-separation, measured table above).
- New changelog fragment + spec_changelog fragment per repo convention;
  run `python scripts/render_fragments.py` after writing fragments.
- SPEC.md: integrate the Build 2 modules (waveform.py, likelihood.py) —
  the red-gate checkpoint commit 21243c7 deliberately deferred this; the
  contract is now stable (RB valid through cusp/fold; no re-scoping).

## Test amendments (Test Developer domain — coders do NOT touch test files)

domain_test_descriptions:

1. **Crown agreement (suite: test_lensing_likelihood.py)** — the existing
   BruteForceAgreement + NearCuspRegressionPin production assertions must
   pass UNCHANGED at the ORIGINAL tolerances (RB_ATOL=1.5). No tolerance may
   be widened anywhere. Expected post-fix offsets: +0.080 (two-image),
   +0.329 (near-cusp).
2. **Canary re-base (same suite)** — the current canary asserts
   kernel_subsamples=2 REPRODUCES the +6.4e8 pathology (SECANT_ALIAS_MIN);
   after WP1 that premise is void (nsub=2 passes at +0.316). Re-base the
   canary on the REAL cause: the real-only switch variant (monkeypatched or
   via direct channel evaluation) produces unbounded kernels
   (max|K| ≥ 1e3 · |F|) at the near-cusp config while the production switch
   stays O(1) — pinning the WP1 fix as load-bearing.
3. **Zero-noise floor fixture (same suite)** — repair the NaN: the fixture
   zeroes the noise then the ASD-drift estimate divides by a zero variance
   (numpy "Degrees of freedom <= 0" / "invalid value in divide" in the suite
   log). Bypass or pin the drift (asd_drift=1) for the zero-noise anchor;
   keep the ZERO_NOISE_TOL=0.01 gate itself unchanged.
4. **Macro-saddle control (test_lensing_waveform.py)** — the positive-parity
   control (y1=0.5, y2=0.25, kappa=0.5 → gamma_eff=0.5) is MIS-SPECIFIED:
   the operator's order-42 shear-series tail is 1.168e-10 > the 1e-10
   certification target, so the refusal is CORRECT engine behaviour. Replace
   the control with a config inside the certified (w, gamma') band, and keep
   a companion expectation that the band-edge config refuses cleanly
   (CancellationError) — refusal is a feature; assert it where it belongs.
5. **Timing gate (test_lensing_likelihood.py)** — unchanged SPEEDUP_MIN=3;
   with WP2 the expected margin is ~7×. Also keep the contraction-
   subdominance assertion as-is.
6. **Small-mass unlensed floor (test_lensing_waveform.py)** — the smallest
   mass (M_L=1e-12 Msun → w~1e-13) drives the shear-series prefactor
   gamma/(2w) into a genuine small-w singularity: an ENGINE gap, ticketed
   and deferred (see Out of scope). Restrict the monotone floor assertion to
   physically meaningful masses (smallest w ≳ 1e-3, expected clean
   |F|-1 ~ 1e-3 floor) and reference the ticket in the test docstring.
7. **Everything currently green stays green** — the full suites
   (test_lensing_operator.py, test_lensing_waveform.py,
   test_lensing_likelihood.py, and the pre-existing package tests minus the
   three XODE-import-gap modules) at original tolerances. The full run costs
   ~2 h; the crown suite alone is the long pole (brute-force oracle evals).

## Out of scope (do NOT do these)

- Sparse global kernel nodes (pure speed optimization; not needed for any
  gate — candidate for a later build).
- Persistent-image split alignment with Eq. (persistent-factors) (benign
  deviation; bounded under the fix).
- Small-w engine short-circuit (F→1 + leading correction) and adaptive
  strong-shear MAX_ORDER — engine tickets, deferred.
- Any change to `_gauge.py` exact channel algebra, `_norm_term`/`_data_term`
  contractions, moment orders, bin selection, or operator refusal
  thresholds. They are measured correct.

## Environment facts (pre-answered)

- Suite interpreter: /Users/tejaswi/miniconda3/envs/cogwheel_310/bin/python.
- test_waveform/test_gw_prior/test_posterior fail COLLECTION for a
  pre-existing optional-dependency gap (IMRPhenomXODE symlink absent in
  cogwheel_310) — ignore them; not this build's concern.
- The prior red suite run: 161 passed + 178 subtests, 9 failed, 2h00m
  (digest in META_PLAN). All 9 failures are dispositioned above.
