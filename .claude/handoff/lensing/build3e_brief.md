# Build 3e — Beat-free envelope decomposition; 10 ms is the gate

## Mission

Reach warm, thread-pinned `lnlike <= 10 ms` on the crown config by
fixing the problem at the layer where it actually lives. Build 3d
PROVED empirically that no interpolation scheme over the CURRENT
channel kernels gets there: the kernels carry beat-frequency
oscillation, so the 1e-3 accuracy gate binds any spline (global,
segmented, overlay) at ~50-60+ nodes. The owner's design ruling
(2026-07-18) is the mission: the oscillation is a DECOMPOSITION
artifact, not physics to be resolved by nodes — the locked design
(decisions #1 and #3: analytic per-image delay carriers; envelope
interpolation at pair delays) says oscillatory content is NEVER
interpolated. Re-express the transition-band kernels as SMOOTH
ENVELOPES x ANALYTIC CARRIERS, interpolate only the envelopes, and the
engine node count collapses to the envelope scale — small and
CONFIG-INDEPENDENT. Layer the cheap micro-levers on top. The plan's
FIRST-ORDER task is the Professor's envelope analysis; everything else
is scoped by its outcome.

## Empirical facts from Build 3d's abort (pre-answered; the evidence)

1. Segmented kink-splines: abandoned IN-BUILD by the coder — the beat
   content defeats them (kinks are not what sets the node count).
2. Global spline + uniform beat-resolving overlay: accuracy gate binds
   at 58-91 nodes across the five anchor configs (GREW vs the ~100
   baseline's intent of ~10x reduction); warm crown lnlike REGRESSED
   18.8 -> 27.4 ms like-for-like in the Inspector's harness. Aborted;
   tree reverted.
3. The shipped-then-reverted code's own docstring conceded: removing
   the in-kernel oscillation is "out of scope for this interpolation
   layer". Correct — it is THIS build's scope, one layer down.
4. Where the beats leak in: in the unresolved regime the gauge parks
   the FULL amplification oscillation F(w) inside the channels (the
   artificial single-image split `K_a = alpha_a exp(-i w tau_a) F(w)`,
   cf. FINDINGS F006 history), and the smootherstep hand-over mixes
   that with the resolved per-image decomposition across the
   transition band — the mixture oscillates at the pairwise delay
   separations. The resolved-regime kernels are already smooth; the
   transition/unresolved band is where the re-expansion must act.
5. HEAD (6a62eff) reference: lnlike 41 ms pinned in the driver's
   harness (18.8 ms in the Inspector's; the harness discrepancy is
   unresolved — gates must therefore be SELF-RELATIVE and
   arithmetic-derived, never absolute numbers imported from one
   harness into another).

## The design question (Professor-first; carries the plan)

Find the re-expansion of the transition-band channel kernels as
`K_a(w) = sum_pairs E_ab(w) * exp(-i w Delta_ab)` (or the equivalent
minimal carrier set) with `E_ab` SMOOTH (envelope-scale variation,
C-infinity within regimes), such that:
- all rapidly-oscillating content sits in the analytic carriers
  (pairwise delay separations `Delta_ab` over the FULL cluster — the
  F008 set — are known exactly from geometry);
- the envelopes are what the coarse engine grid samples and the spline
  interpolates; the carriers are applied exactly at the dense
  frequencies;
- the smootherstep switch factor (known analytically) is peeled out of
  the interpolated object, not interpolated through;
- the node count is set by the ENVELOPE smoothness — the acceptance
  criterion is that it is CONFIG-INDEPENDENT (the crown's "9 beat
  cycles" arithmetic must not appear anywhere in the node budget);
- downstream, the (k0, k1) per-bin reduction and the mode-then-image
  contraction consume the reconstructed kernels unchanged, and
  `exact_total` / brute force remain the untouched oracle.
If the Professor's analysis finds a genuine obstruction (e.g. the
residual projection cannot be carrier-expanded to envelope smoothness
in some sub-band), the step rule applies: gate at the derived floor,
name the residual, escalate the design question to the owner with the
analysis.

## Additional levers (in scope, after the envelope analysis)

- Caustic search Newton shortcut (geometry.py): 1.9 -> ~0.3 ms
  (value-preserving, branch-invariant — same obligations as 3b's WP1).
- Contraction fusion (operator.py): ~2 -> ~1 ms (weight-vector path
  internal fusion; refusal quantities and thresholds byte-unchanged).
- The 3D post-contraction surrogate (owner: wanted if it certifies and
  saves FLOPs): RE-SCOPE after the envelope analysis — beat-free
  envelopes may make the table trivial (tabulate envelopes, tiny
  domains) or unnecessary. If taken: the archived design facts hold
  (reduced (w, y', gamma') space, w = xi(M_L)*f global-domain
  constraint, LookupTable-idiom gitignored cache with engine-version
  provenance + DATA_CONTRACTS artifact + fragments, refused-cell
  masks, zero false accepts, fallback-to-exact).

Budget arithmetic: envelope nodes ~10-15 x ~0.37 ms = 4-6 ms engine +
contraction 1-2 ms + caustic 0.3 ms + non-engine ~2.5 ms => ~8-10 ms;
with the surrogate serving envelopes, ~4-6 ms.

## Scope fences

IN: `cogwheel/lensing/chang_refsdal/channels.py` and `_gauge.py`
(the transition-band kernel decomposition surface — the envelope
re-expansion lives here), `geometry.py` (Newton shortcut ONLY),
`operator.py` (contraction fusion ONLY — no threshold/refusal change),
`cogwheel/lensing/likelihood.py` (envelope interpolation wiring),
optional `_surrogate.py` per the re-scope, tests via
`domain_test_descriptions`, SPEC/DATA_CONTRACTS fragments.

OUT: `_dd.py` semantics; `_hyp1f1.py` ladder algorithm; every refusal
THRESHOLD and message (the certified-or-named-refusal contract must
survive verbatim); `F_op_grid`'s certification (it remains the exact
per-node evaluator the envelopes are built from); the F008 switch
RULE (full-cluster neighbours — the re-expansion must reproduce it,
not alter it); the stall-ringdown/template builders; priors/sampling
(Build 4); NO tolerance widening anywhere.

## Constraints

- `exact_total` and `lnlike_bruteforce` are the untouched oracles; the
  re-decomposed RB path must agree at the ORIGINAL gates (RB-vs-brute
  max(1.5, 1e-2|bf|) every config; production interp gate 1e-3
  null-safe moved verbatim onto the envelope scheme; reconstruction
  continuity across the smootherstep band — no new discontinuities).
- Label/gauge continuity: the envelope re-expansion must preserve the
  topology-stable channel tracking (crossing-scenario tests stay
  green; F002 fixture independence as always).
- Refusal symmetry on RB and brute paths, unswallowed.
- F010: py_func-chain falsification for any new njit code.
- Timing gates: structural-first (node count config-independence
  assertion across the five anchor configs; public-entry-point speedup
  floor), then the 10 ms pinned warm best-of-5 ceiling. Owner
  clarification (2026-07-18): 10 ms may honestly require the surrogate
  on top of the envelopes — BOTH outcomes are acceptable: (a) <= 10 ms
  this build (envelopes alone or envelopes + surrogate, if it fits an
  honest <= 3-WP decomposition), or (b) the envelope build gated at its
  own derived floor with CONFIG-INDEPENDENT nodes proven, and the
  surrogate named as the immediate finisher build. What remains
  unacceptable: a moved gate dressed as progress, or dropping the
  surrogate from the program.
- In-build tests FAST; full suite is the driver's post-build step.

## Environment facts

- Interpreter: /home/tejaswi/anaconda3/envs/cogwheel-newlal/bin/python
  (server nereid; SSE port 8323 via .env).
- HEAD 6a62eff green: 222 passed in 44 s (-n4, minus XODE trio);
  fast-path suite 20 tests; batched-operator suite green.
- Five anchor configs from the 3d test specs are good reusable
  fixtures (crown, near-cusp, well-separated 2-image, near-fold,
  sheared small-w).
- numba 0.58.1, mpmath 1.3.0, pytest-xdist 3.8.0.
