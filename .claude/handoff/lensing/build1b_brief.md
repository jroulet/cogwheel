# Build 1b: COMPLETE the Chang–Refsdal lens engine (corrective build)

## Context
Build 1 ran `.claude/handoff/lensing/build1_plan_v3_approved.md` but delivered
only the foundation. COMMITTED — consume these, do not rewrite:
- `cogwheel/lensing/chang_refsdal/_dd.py` — double-double real+complex arithmetic
  (four-flat-scalar representation, numba-shaped; 37 tests green). It has no
  sqrt/exp/log/trig and does not need them. Its tests must keep passing unchanged.
- `cogwheel/lensing/chang_refsdal/_gauge.py` — exact gauge/cluster-split channel
  algebra (34 tests green). `channels.py` REUSES these primitives; do not port a
  second copy of the projection.
- `cogwheel/lensing/chang_refsdal/geometry.py` (872 lines) — quartic images,
  delays, magnifications, stationary-phase kernels. Public surface is complete and
  numpydoc'd; do NOT redesign it. **UNTESTED — treat as unreviewed**; if a test
  exposes a real defect, fix it surgically and say so loudly in the change report.

The detailed module-body spec is `build1_plan_v3_approved.md` WP2–WP5. Where it
conflicts with this brief, THIS BRIEF WINS.

## Mission — three production modules + closeout
1. `_hyp1f1.py` — dd-accumulated complex 1F1 kernel and shared-numerator k-ladder.
   Kummer reparametrization with a k-INDEPENDENT a' = iw/2 — this is a STRUCTURAL
   win (one numerator shared across all k), NOT a precision one: z is purely
   imaginary, |e^z| = 1, both forms have the same max term e^{w*Y}. NO
   k-recurrence (forward recurrence for 1F1(a+k;1+k;z) is unstable — the shared
   numerator sidesteps it). NO large-|z| branch (physically unreachable here).
   dd is MANDATORY for the Maclaurin series terms AND their summation (Kahan does
   not rescue it: the bound carries sum|term_i|, exponentially larger than the
   result). Prefactor C(w) is float64 — it is a common multiplicative factor, so
   its relative error factors out. TWO TRAPS: exp(pi*w/4) overflows at w~900 while
   Gamma(1-i*w/2) underflows — combine exponents analytically via complex loggamma,
   never evaluate exp(...)*gamma(...) as written; use the closed form
   |C(w)|^2 = pi*w/(1-e^{-pi*w}) with expm1 for small w. The (w/2)ln(w/2) phase
   cancels against Im loggamma(1-i*w/2) — cancel it analytically. Ladder length
   max_derivative = 2*max_order (84 at max_order=42; the operator's `_apply_rotated_D`
   raises the radial index by up to 2 per application). dd extends the MANTISSA,
   not the exponent: |G^(k)| ~ 1e92 at w=40,k=84 and overflows near w~700 — gate w
   with a named error above the certified ceiling. numba-shaped (plain loops/arrays,
   no closures/containers), but DEFER @njit unless the perf gate fails without it.
   Also correct `_dd.py`'s docstring WHY-section (docstring only, no code change):
   L_1F1 = w*|y'| and L_op = w*gamma'/2 are INDEPENDENT channels at SEPARATE code
   sites; dd is required only for the former; dd's ceiling (w*Y<=60 at 1e-6) exists
   to close the gap to the geometric branch's onset (w*Y>=50), which float64
   (w*Y<=22) cannot bridge; add the mantissa-not-exponent caveat.
   NOTE: the prototype calls `mp.hyp1f1` FRESH at every k and ladders only the
   Pochhammer — it is an ANTI-TEMPLATE here, not a port target.
2. `operator.py` — contour-free F_op. Rotate beta into the shear eigenframe: ONE
   real representation table, not a beta-parameterised family (and so no
   lru_cache-on-float; use a dense integer-keyed array table — the prototype's
   lru_cache hands callers MUTABLE dicts, a silent cache-corruption hazard).
   Replace `point_mass_G_derivatives`' mpmath body with WP2's kernel and DELETE the
   dps plumbing entirely (the prototype threads three different dps defaults for one
   knob). Exact mass-sheet kappa rescaling, written ONCE. Adaptive
   n_max ~ zeta + 5*sqrt(zeta) + 10. Runtime refusal when measured
   max_partial_term/|total| > ~1e13. `OperatorDiagnostics` frozen dataclass:
   order_used, converged, estimated_relative_tail, measured cancellation.
   operator.py OWNS the single implementation of the explicit branch gate
   {'wave','geometric'} — geometric when (w*delta_min >= rho1) AND (L > L_max=48);
   channels.py consumes it. The smooth switch S_j stays an error-free smoothness
   device — that distinction is load-bearing. Document the certified domain with
   its derivation.
3. `channels.py` — `ChangRefsdalChannels`, the entry point Build 2 consumes.
   Topology-stable 4-label decomposition F(w) = sum_a e^{i*w*tau_a} K_a(w): path-based
   label continuation by assignment on lens-plane markers, virtual labels at the
   nearest critical point, the smooth switch, cluster residual projection REUSING
   `_gauge.py`, exposing (tau_a, K_a(w)) plus the exact total. KEEP the brute-force
   assignment solver (bounded problem; deliberate, do not relitigate). Deterministic
   reset convention for far proposals, plus continuation from a previous point.
   The prototype's `global_tracking.py` is the worst-written module in the set —
   RESTRUCTURE it, do not transliterate; preserve the ALGORITHM exactly. Drop its
   `**_ignored` kwargs swallowing (typos pass silently). Export ONE line from
   `chang_refsdal/__init__.py`: `from .channels import ChangRefsdalChannels`
   (_dd/_gauge/_hyp1f1/operator stay private).
4. SPEC closeout: REWRITE SPEC.md's lensing row (it says "IN PROGRESS — foundation
   only") to the completed description with limitations recorded; spec_changelog.d
   fragment (bump: patch — completing a layer); FINDINGS.md F001 (the two-channel
   cancellation law) and F002 (the oracle-tautology lesson); short overview.rst
   paragraph. KEEP `.claude/spec/todo.d/2026-07-16_lensing-program.md` (Builds 2–3
   pending) — do NOT write a completed.d fragment for the program.

## Tests — put ALL of these in `domain_test_descriptions`
The Test Developer authors every test; no work package may deliver a test file.
That prohibition is about a WP's DELIVERABLE — the specs themselves should say
where the tests go, because `domain_test_descriptions` is the only channel the
Test Developer sees (it receives the WP ids/titles and these specs, nothing else).

RECOMMEND this layout in the specs — four new suites under `cogwheel/tests/`, one
per module, mirroring the two that exist:
`test_lensing_geometry.py`, `test_lensing_hyp1f1.py`, `test_lensing_operator.py`,
`test_lensing_channels.py` (the fold/cusp crossing fixture builders live as
test-local helpers in the channels suite, or a clearly-named test-support module
under `cogwheel/tests/` — never inside the package). Do not collapse them into one
file, and do not name a suite after a module that does not exist.

Give each spec a setup / operation / expected / diagnostic. Follow the house idiom
in `cogwheel/tests/test_lensing_dd.py`: stdlib unittest under `cogwheel/tests/`, a
helper base TestCase with a domain assertion, an ANTI-VACUITY tearDown that fails
if zero comparisons ran, `<Thing>TestCase` names, itertools.product + subTest,
ALL-CAPS module constants with `#:` comments, and a SELF-FALSIFICATION class
proving the suite can go red. mpmath is ORACLE-ONLY — it must never be importable
from a production path (`grep -r mpmath cogwheel/lensing/` returns nothing).
- geometry retro-tests: quartic CSV regression over all 168 rows (image count ==
  row.n_multistart — a frozen oracle from a deleted multistart solver, independent
  and non-drifting; n_quartic == n_multistart as a fixture-integrity check; fresh
  residual gate <= 1e-12). Do NOT assert `matched_position_error` (unrecomputable)
  or reproduce `max_quartic_residual` (polish-detail dependent).
- exact y=0 analytic geometry: gamma 0.05..0.4, kappa=0, source at the origin —
  |x|^2 = 1/(1-gamma) (shear axis) and 1/(1+gamma) (transverse); tau = 1/2 +
  (1/2)ln(1 -/+ gamma); Delta tau = artanh(gamma); agreement ~1e-14.
- Morse census, MEASURED across a gamma/y sweep, not asserted from prose.
- astroid cross-check: count is 4 iff y is inside the caustic. Build the caustic
  from an ANALYTIC astroid parametrization, NOT from `geometry.critical_point` — a
  winding test against the module's own critical curve is a consistency check, not
  an independent oracle.
- near-caustic behaviour on the 24 fold + 24 cusp rows: assert DELAYS and
  RESIDUALS, never positions (a fold's double root leaves positions with only
  sqrt(eps) ~ 1.5e-8; delays are quadratically insensitive because images are
  stationary points). Magnifications are genuinely ill-conditioned at critical
  points — assert their SCALING, not a tight absolute value.
- domain guard: `macro_matrix` raises `LensDomainError` (named, not bare
  ValueError) when 1 - kappa <= |gamma|, message naming the offending (kappa,
  gamma); also the Einstein-ring case in `find_images_quartic`.
- prefactor: gate the production |C(w)|^2 against an INDEPENDENT mpmath evaluation
  of the DEFINITION |e^{pi w/4 + (iw/2)ln(w/2)} Gamma(1-iw/2)|^2 at >=60 dps, w in
  [1e-3, 500], rtol 1e-14, flat in w (fit the residual, assert no trend). Assert
  the limits |C|^2 -> 1 as w->0 and -> pi*w as w->inf.
- k-ladder vs mpmath oracle (60–70 dps) over the certified (w,s) domain including
  w*Y up to the dd ceiling, rel err <= 1e-10, with a cancellation-law fit sitting
  below the eps*e^{w*Y} contour (no knee at L~13).
- ladder complexity, substantiated claims only: (a) shared-numerator evaluations
  INDEPENDENT of max_order, (b) total dd-multiply count LINEAR in max_order (fit
  and reject a quadratic). No unsubstantiated big-O claims.
- F_op vs mpmath oracle over the paper's 4 configs + stress cases (rtol <= 1e-10
  for L <= 48); named error above the w ceiling.
- geometric-optics slope test (self-oracling, no mpmath): |F_op - sum e^{iw tau} H_a|
  vs w — fitted exponent -1 without C1/C2, -3 with.
- mass-sheet identity in NON-VACUOUS form: the OBSERVABLES Delta tau_ac and the
  flux ratios |K_a/K_c| must be exactly kappa-invariant. Comparing F against its own
  rescaling path is vacuous; if a form must be checked directly, gate it against an
  INDEPENDENT mpmath computation.
- scale-aware exact reconstruction (NOT a flat 1e-12 — unachievable near folds);
  label continuity across fold and cusp crossings; assignment/reset equivalence and
  path-independence (the TOTAL is label-invariant).
- NON-CIRCULAR CROSSING FIXTURES: the fold/cusp scenario builders must be
  constructed from geometry + operator + `_gauge` ONLY and must never import, call,
  or derive a value from `channels.py`. They are the ground truth the
  label-continuity test judges channels against — a fixture built by the tracker it
  tests cannot fail. Assert it with an AST import guard in the idiom the committed
  `test_lensing_gauge.py` already uses.

## Settled facts — do NOT re-measure, do NOT relitigate
These are measured against the committed `geometry.py` and are inputs, not questions.
- Morse census: 4-image `(0,0,1,1)` — TWO minima + TWO saddles, NO maximum;
  2-image `(0,1)`. A point mass has -ln|x| -> +inf at the origin, so the Fermat
  potential has no local maximum: n_max = 0 in every regime. Any document saying
  `0,1,1,2` is WRONG. CAUTION: n_min - n_saddle + n_max = 0 holds for BOTH, so the
  invariant cannot discriminate — the test must MEASURE the census.
- Residual gate: all 168 rows clear 1e-12 (max 1.93e-13; general 1.93e-13, fold
  1.69e-13, cusp 6.66e-16). No near-caustic exception is needed. The solver's 3e-8
  `residual_tolerance` default is acceptance-filter headroom, not achieved accuracy.
- CSV fixture: 120 general + 24 fold + 24 cusp (rows 120–167; all beta=0.0, gamma
  in {0.05, 0.2, 0.5}). The claim "all 168 rows are general" is false.
- pylint is NOT installed: substitute a programmatic 79-column + `ast.parse` check
  on every file created or edited. Do not block on pylint.
- `test_lensing_gauge.py` already AST-forbids `_gauge.py` from importing modules
  named `_hyp1f1`, `operator`, `channels`, `crossings` — the filenames are pinned.

## Scope
No `pyproject.toml` changes (the committed suites already import mpmath without
one; if a clean-install collection failure is real, report it as a finding). No
`DATA_CONTRACTS.yaml` artifact — this build writes no on-disk data product; if a WP
caches a table to disk it must register the artifact plus a contracts_changelog.d
fragment.
