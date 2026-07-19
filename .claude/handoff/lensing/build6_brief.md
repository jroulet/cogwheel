# Build 6 — Negative parity, part 1: saddle geometry + the Schwinger wave branch

## Mission

Execute Build S1 of the certified negative-parity program
(design authority: `.claude/handoff/lensing/negative_parity_research.md`
— READ IT FULLY; its Sec. 11 build summary, Sec. 9 certified
domain/refusals, and Sec. 8 anchor set carry the Professor's authority;
all its numerics were actually run). Extend the Chang-Refsdal engine to
macro-saddle (negative-parity) hosts: parity-aware geometry (guard
split, centered-source saddle case, two-lobe critical utilities on the
existing `v(theta')` formula), the NEW certified saddle wave-branch
evaluator — the exact 1D Schwinger-parameter dd quadrature
(`_schwinger.py`), certified-or-refuse with ceiling `w <= 60` — and
parity dispatch in `operator.py`. POSITIVE-PARITY BEHAVIOR
BYTE-IDENTICAL: the existing operator/1F1 path and every refusal
constant are frozen; saddle configs route to the new evaluator;
`lam <= 0` (Type III) is a named refusal.

Build S2 (channels/likelihood/prior over the saddle domain) follows as
Build 7 — do NOT pull its scope forward.

TREE STATE AT RELAUNCH (attempt 5, 2026-07-18 ~22:15): prior attempts
delivered, uncommitted in the tree:
- geometry.py: WP1 COMPLETE per attempt-4 (parity-aware domain split,
  two-lobe branch-parameter critical utilities, centered-source saddle
  case; finalized by a coder that verified the four guard sites).
- _schwinger.py: 864 lines, NEAR-COMPLETE (attempt-4's WP2 coder ran
  out of turns during final verification, NOT mid-implementation).
  DRIVER-SMOKE-TESTED: imports clean; `f_schwinger(w, y_eig,
  gamma_prime)` returns finite complex values (e.g. w=3, y=(0.4,0.3),
  gp=1.3 -> 0.14470585550870085+0.4065122393352838j); the w>60 ceiling
  raises SchwingerCertificationError correctly; full dd toolchain +
  `_measure_warm_cost` present.
Fix FORWARD (attempt 6, ~23:00): ALL THREE WPs ARE DELIVERED in the
tree — geometry.py (WP1 finalized), _schwinger.py (WP2 verified and
completed by attempt 5), operator.py +284 lines (WP3 parity dispatch).
DRIVER SMOKE EVIDENCE: positive-parity F_op unchanged
(-0.35753006967142426+1.1663724461262843j at w=5, y=(0.3,0.1),
gamma=0.2); saddle F_op(w=3, y=(0.4,0.3), gamma=1.3) returns
0.14470585550870085+0.4065122393352838j — BIT-IDENTICAL to the direct
f_schwinger call (clean dispatch); the exact parity boundary
(kappa=0.5, gamma=0.5) raises the named LensDomainError. Attempt 5
died AFTER the WPs to the SDK anyio cancel-scope bug via a failed
cosmetic Tidier (non-fatal by intent). THIS ATTEMPT'S JOB: light
verification of the delivered code against the WP specs, then the
FULL test battery per domain_test_descriptions, Inspector, Professor,
commit. Code WPs should be verify-only unless a gate fails.

## Required measurement (owner sequencing input)

Record the Schwinger evaluator's warm PER-POINT COST (ms/point over a
(w, gamma', y) grid, same harness discipline as the ratio-layer
timing) in the change report and FINDINGS: this number PRICES the
envelope-surrogate decision (todo `likelihood_envelope-surrogate`).
It is a measurement, not a gate — no ceiling.

## Facts settled by the research (verify pins, do not re-derive)

1. Census (index theorem + 4000-source scan, zero anomalies): saddle
   hosts give 2 images both saddles (1,1) or 4 images (0,1,1,1); the
   critical curve splits into TWO 3-cusp deltoid lobes via the +-
   branch of the existing astroid formula.
2. The shear operator series DIVERGES for gamma' > 1 (branch point at
   the parity boundary) — measured at every w; best truncation O(1).
   The Schwinger 1D representation is EXACT at both parities, certified
   2.2e-15 against an independent rotated-contour 2D mpmath oracle;
   its single cancellation channel is L_S = pi*w/4, y-INDEPENDENT; dd
   holds 1e-10 to w ~ 64 (ceiling 60 with margin).
3. Deep band (F009-S): F -> e^{-i pi/2}/sqrt(gamma^2 - (1-kappa)^2);
   magnitude pinned with O(w) correction; Morse phase pinned 1.6e-4;
   drift model w*[tau_G + (1/2)ln(w/2) + c0].
4. Mass-sheet identity holds verbatim for lam > 0 (1e-16); lam <= 0 is
   a named refusal. geometry's Hessian/delay/Morse/`image_kernel` work
   UNMODIFIED on the indefinite matrix; geometric branch agrees at
   resolved w (2.3e-4 at w*dtau ~ 5).
5. Dead ends (documented — do not retry): Pade resummation of the
   shear series; float64 quadrature past w ~ 20 (silently fabricates —
   the saddle branch's F005 lesson; dd is mandatory).

## Scope fences

IN: `geometry.py` (parity-aware `macro_matrix` domain split, lam <= 0
named refusal, `_centered_source_images` saddle case, two-lobe
`critical_point`/`_caustic_source`/`nearest_caustic_point`), NEW
`cogwheel/lensing/chang_refsdal/_schwinger.py` (dd-integrand 1D
quadrature, certified-or-refuse via paired quadrature rules, w <= 60
ceiling), `operator.py` (parity dispatch in `F_op`/`F_op_grid` +
`select_branch` w-ceiling condition ONLY — existing positive-parity
path byte-frozen), tests via `domain_test_descriptions`.

OUT: the v-plane steepest-descent evaluator (research Sec. 6.4 —
deferred ceiling-lifting alternative); lam <= 0 / Type III support;
ANY change to positive-parity operator/1F1/refusal constants;
channels/likelihood/prior (Build 7); ratio-layer speedups; NO
tolerance widening.

## In-build gates (from the research Sec. 11, all fast)

1. Census: 200-source scan; index sum -2; census sets exactly
   {(1,1), (0,1,1,1)}.
2. Schwinger evaluator vs an mpmath dev-oracle (an INDEPENDENT
   high-dps implementation of the 1D representation — F002: never the
   production code's own path) at <= 1e-10 over a (w, gamma', y) grid
   with w <= 60 including gamma' = 1.05.
3. The 2.2e-15-class independent 2D rotated-contour oracle anchor
   reproduced at one saddle point.
4. Mass-sheet identity on observables (lam > 0), branch-wise.
5. Deep-band pins: |F| closed form AND the -pi/2 Morse-phase intercept.
6. Geometric-branch agreement at resolved w.
7. Positive-parity regression: the full existing suite untouched and
   green (byte-identical dispatch for 1-kappa > |gamma|).
Plus F010 discipline for any new njit (py_func-chain falsification)
and refusal symmetry (new refusals named, never silent).

## Acceptance (build-level)

All seven gates green; the per-point cost measurement recorded;
FINDINGS gains F001-S/F005-S/F009-S addenda per the research; SPEC row
updated for the saddle branch; fragments rendered; commit hook-clean.
Post-build (driver, detached, parallel per owner ruling): the research
Sec. 8 saddle anchor scan + full-suite regression.

## Environment facts

- Interpreter: /home/tejaswi/anaconda3/envs/cogwheel-newlal/bin/python
  (server nereid; SSE 8323 via .env). HEAD 3b3ebdb (SPEC 0.8.0);
  full-suite baseline green (fresh verification running detached).
- The research scratch scripts np_exp1..9 are in the driver session
  scratchpad; their facts are inlined above and in the report.
- mpmath 1.3.0 (test-only oracle, F003), numba 0.58.1.
