# Findings

Empirical discoveries, numerical-accuracy notes, and non-obvious gotchas
uncovered while working on cogwheel. Each finding gets an ID (`F001`, `F002`, ...),
a date, and a short writeup. Mark superseded findings rather than deleting them.

Use this for things like: tolerance levels at which relative binning diverges
from the exact likelihood, sampler convergence quirks, ASD-drift sensitivities,
waveform phase-convention pitfalls, numba compatibility traps.

---

## F001 — The two-channel cancellation law (2026-07-16)

The Chang–Refsdal amplification loses precision to alternating-series
cancellation through TWO channels. They are INDEPENDENT: they live at
SEPARATE code sites and do NOT compound into a single summed exponent.

- `L_1F1 = w * |y'|` — the confluent-hypergeometric (1F1) kernel series in
  `cogwheel/lensing/chang_refsdal/_hyp1f1.py`. Its partial terms reach
  `e**(w*|y'|)` while the sum is O(1). Double-double arithmetic is required
  HERE, and only here: Kahan summation does not help, because its error
  bound also carries `sum|term_i|`. dd holds the 1e-10 target out to
  `w*|y'| ~ 50` and degrades to ~1e-6 at the ceiling `w*|y'| = 60`.
- `L_op = w * gamma'/2` — the operator power series in `operator.py`. This
  channel is NOT rescued with extended precision; instead `F_op` MEASURES
  its own cancellation ratio `max_partial_term / |total|` and REFUSES
  (raising `CancellationError`) once it exceeds ~1e13. That runtime refusal
  is the operational form of the law: past ~13 lost digits the double-double
  substrate no longer protects the sum, so returning a plausible-but-wrong
  amplification is the failure mode being avoided.

Because the channels are independent, treating them as one summed exponent
`w*(|y'| + gamma'/2)` (as an earlier `_dd.py` docstring did) overstates the
precision demand and misplaces the dd requirement.

## F002 — The oracle-tautology trap in the lens-engine tests (2026-07-16)

A test fixture built by the very code it is meant to judge cannot fail, no
matter how broken that code is. Two concrete instances shaped the lens-engine
test design:

- The fold/cusp crossing-scenario builders that the label-continuity test
  judges `channels.py` against are constructed from `geometry`, `operator`,
  and `_gauge` ONLY. They must never import, call, or derive a value from
  `channels.py`; otherwise the ground truth is the tracker's own output and
  the test is vacuous. This is enforced with an AST import guard in the idiom
  the committed `test_lensing_gauge.py` already uses.
- A mass-sheet identity checked by comparing `F_op` against its own kappa-
  rescaling path is equally vacuous — the code agrees with itself by
  construction. Such identities are asserted on OBSERVABLES (the delay
  differences `Delta tau` and flux ratios `|K_a/K_c|`, which are exactly
  kappa-invariant) or gated against an INDEPENDENT mpmath computation, never
  against the code's own rescaling path.

## F003 — mpmath is an undeclared test dependency (2026-07-16)

The committed lens-engine test suites import `mpmath` as a high-precision
oracle, but `mpmath` is declared nowhere in `pyproject.toml` (no runtime
dependency, no test/dev extra). It is present only because it happens to be
installed in the `cogwheel_310` environment; a clean install would fail to
collect these tests. Recorded here per the Build 1b brief as an observation
to be resolved deliberately (e.g. a test extra), NOT fixed in this build.

## F004 — boundary-domain tests need float64-exact boundary points (2026-07-16)

`macro_matrix` rejects `(kappa, gamma)` iff `1 - kappa <= |gamma|` (strict
positive-parity `1 - kappa > |gamma|` required). A test intending to hit the
EQUALITY boundary must choose values where `1 - kappa == |gamma|` holds
bit-for-bit in float64. `(kappa=0.7, gamma=0.3)` does NOT: `1 - 0.7` evaluates
to `0.30000000000000004`, a hair above `0.3`, so that point is genuinely just
inside the domain and correctly does not raise — a test asserting it must raise
fails against correct code. Use powers-of-two endpoints (`0.5/0.5`, `0.75/0.25`)
where `1 - kappa` equals `|gamma|` exactly. Caught in the first delivered
lens-engine test suite; the code was right, the test's boundary point was not.
