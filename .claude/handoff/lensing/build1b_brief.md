# Build 1b: COMPLETE the Chang–Refsdal lens engine (corrective build)

## Context — read this carefully before planning
Build 1 ran the approved plan at `.claude/handoff/lensing/build1_plan_v3_approved.md`
but DELIVERED ONLY THE FOUNDATION. The Professor's inference review (verdict:
CONCERN) found ~13 of 15 review specs described code not yet written. What is
COMMITTED and TESTED (do not rewrite; consume):
- `cogwheel/lensing/chang_refsdal/_dd.py` — double-double real+complex
  arithmetic (37 tests in `cogwheel/tests/test_lensing_dd.py`, all pass).
- `cogwheel/lensing/chang_refsdal/_gauge.py` — exact gauge/cluster-split
  channel algebra (34 tests in `cogwheel/tests/test_lensing_gauge.py`, pass).
- `cogwheel/lensing/chang_refsdal/geometry.py` (872 lines) — quartic images,
  delays, magnifications, stationary-phase kernels. **UNTESTED — treat as
  unreviewed; verify before building on it.**

## Mission — the GAP list (this is the whole build)
1. `_hyp1f1.py` — the dd-accumulated complex 1F1 k-ladder kernel, exactly per
   the approved plan's WP2 `how` (Kummer reparametrization with k-independent
   a' = iw/2; closed-form overflow-free prefactor |C(w)|^2 = pi*w/(1-e^{-pi*w})
   with expm1; shared-numerator ladder, NO k-recurrence, NO large-|z| branch;
   mpmath oracle in tests ONLY).
2. `operator.py` — contour-free F_op per approved WP3 `how` (beta rotated to
   shear eigenframe — ONE real representation table, no lru_cache-on-float;
   dense graded array representation; dd accumulation; exact mass-sheet kappa
   rescaling written once; adaptive n_max ~ zeta + 5*sqrt(zeta) + 10; runtime
   refusal when measured max_partial_term/|total| > ~1e13; diagnostics:
   order_used, converged, estimated_relative_tail, measured cancellation).
3. `channels.py` — topology-stable 4-label decomposition + `ChangRefsdalChannels`
   entry point per approved WP4 `how` (label continuation via assignment on
   markers, virtual labels at nearest critical point, smooth switch, cluster
   residual projection REUSING `_gauge.py`; explicit `branch` in
   {'wave','geometric'} global gate — geometric when (w*delta_min >= rho1)
   AND (L > L_max=48); reset convention for far proposals; label-permutation
   invariance documented).
4. `cogwheel/tests/test_lensing_geometry.py` — geometry was delivered untested:
   quartic CSV regression (168 rows,
   `.claude/spec/lensing_paper/data/quartic_geometry_validation.csv`;
   count == n_multistart per row; fresh residual gate <= 1e-12) + Morse-index
   census + near-caustic assertions on DELAYS/RESIDUALS not positions.
   PRE-ANSWERED — MORSE CENSUS, get this right: the 4-image census is
   **n_a = 0,0,1,1** (TWO minima + TWO saddles, NO maximum); the 2-image census
   is n_a = 0,1. Earlier plan documents said "0,1,1,2" (one min / two saddles /
   one max) — that is WRONG; ignore it wherever you meet it. A point mass has
   -ln|x| -> +inf at the origin, so the Fermat potential has no local maximum
   and n_max = 0 in every regime. This was MEASURED against the committed
   geometry.py (y=0 and general y inside the astroid, gamma 0.05..0.4): the
   census is [0,0,1,1] everywhere, [0,1] outside. CAUTION: the invariant
   n_min - n_saddle + n_max = 0 is satisfied by BOTH 0,0,1,1 and 0,1,1,2, so it
   cannot discriminate — the test must MEASURE the census, never assert it from
   prose. (Found by a coder that correctly refused to write an assertion it
   believed false; it was right.)
5. `cogwheel/tests/test_lensing_hyp1f1.py`, `test_lensing_operator.py`,
   `test_lensing_channels.py` — the approved plan's domain tests: exact
   prefactor identity (rtol 1e-14, flat in w); k-ladder vs mpmath oracle +
   cancellation-law fit; F_op vs mpmath oracle over the paper's 4 configs +
   stress cases (rtol <= 1e-10 for L <= 48); GEOMETRIC-OPTICS SLOPE TEST
   (|F_op - sum e^{iw tau} H_a| vs w: fitted exponent -1 without C1/C2,
   -3 with, self-oracling); scale-aware exact reconstruction; mass-sheet
   identity (non-vacuous forms); label continuity across fold+cusp paths;
   assignment/reset equivalence.
6. SPEC closeout: SPEC.md's lensing row currently says "IN PROGRESS —
   foundation only" — WP updating it must REWRITE that row to the completed
   description (all modules listed, limitations recorded) + spec_changelog.d
   fragment (bump: patch — completing the layer) + FINDINGS.md entries per the
   approved plan (cancellation law F001; tautology finding F002) + short
   overview.rst paragraph. The todo fragment
   `.claude/spec/todo.d/2026-07-16_lensing-program.md` stays (program has
   Builds 2-3 pending); do NOT write a completed.d fragment for the program.

## Hard process requirements (this is why 1b exists — Build 1's failure modes)
- A WP is complete ONLY when its named tests exist under `cogwheel/tests/`
  (NEVER inside the package) and `python -m unittest cogwheel.tests.<module>`
  passes — run it and include the pass count in your change report. Do not
  declare victory on partially-written modules.
- The commit gate enforces: new modules must appear in SPEC.md (item 6).
- mpmath: tests only. Production import graph must not reference it.
- Conventions: w dimensionless (= 8 pi G M_L (1+z_L) f / c^3); Einstein-radius
  units; enforce 1-kappa > |gamma| with a named error; numba-shaped hot paths
  (plain loops/arrays), @njit deferred.
- Performance asserts (generous): F_op <= 10 ms at w=20; kernels(w) over a
  10-node grid <= 100 ms.
- The full approved plan (WP-level `how` details, test definitions) is at
  `.claude/handoff/lensing/build1_plan_v3_approved.md` — treat its WP2-WP5
  bodies as the detailed spec for items 1-5. Professor memory
  `professor/microlensing_chang_refsdal` has the physics; the paper +
  prototype live in `.claude/spec/lensing_paper/` (prototype's
  `chang_refsdal_operator.py` is the reference for the operator series;
  its mpmath usage is what `_hyp1f1.py` replaces).

## HEADLESS DISCIPLINE (program-scoped — copy this block VERBATIM into every WP's `how`)
This build runs unattended: no human reads or answers questions mid-task. The
previous run failed exactly this way — coders did excellent read-only analysis,
raised good questions, ended with "let me know how to proceed", and the
pipeline moved on recording zero changes. Therefore, for THIS program's builds:
never end a work package waiting for direction. Resolve ambiguities and task-
text errors yourself with the most defensible choice, record each under a
DECISIONS line in the change report, and complete the files. A coder WP that
changes no files is a FAILED WP unless it ends with 'BLOCKED: <reason>'.

## PRE-ANSWERED QUESTIONS (from the previous run's coders — adopt these resolutions)
1. Prefactor test tautology (WP2): correct catch — do NOT assert the closed
   form against itself. Gate the production |C(w)|^2 against an independent
   mpmath evaluation of the DEFINITION |e^{pi w/4 + (iw/2)ln(w/2)} Gamma(1-iw/2)|^2
   at >= 60 dps.
2. CSV fixture errata (WP1): the plan's claim "all 168 rows kind='general'" is
   WRONG — actual distribution is 120 general / 24 fold / 24 cusp. Use the
   fold/cusp rows as genuine near-caustic fixtures (assert on residuals and
   delays there, not positions).
3. Residual gate vs solver default (WP1): find_images_quartic's
   residual_tolerance=3e-8 default vs the 1e-12 test gate is an empirical
   question — measure on all 168 rows; gate at 1e-12 where it holds and
   document (FINDINGS-worthy) any principled exceptions near caustics rather
   than silently loosening the gate.
4. Ladder complexity assertion (WP2): assert (a) shared-numerator evaluations
   independent of max_order and (b) total dd-multiply count linear (reject
   quadratic) — do not assert an unsubstantiated O() claim.
5. pylint is not installed in this env: substitute programmatic 79-col +
   ast.parse checks; do not block on pylint.

You (the Architect) own the plan — structure WPs as you see fit, but the gap
list above is the deliverable and the process requirements (including the
headless-discipline block) are non-negotiable.
