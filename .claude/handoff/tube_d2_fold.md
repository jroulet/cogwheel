# Build: fold tube-chart serving into the D2 fundamental domain

## Mission

Owner directive (2026-08-14): "When there is a symmetry of the problem, we
have to use it... it is a moral imperative." The amplification is exactly
D2-symmetric — `F(w; y1, y2) = F(w; ±y1, ±y2)`, the Fermat potential is
even in each source coordinate with external shear — and every chart kind
EXCEPT the tube already folds: exterior-polar serves in the folded quadrant
with a hard-raise domain guard (`surrogate.py:588-596` at a4ba536), wedge
charts fold (`:974-1001`), lobe charts fold (`:2989-2990`). TubeChart alone
serves in unfolded theta. That inconsistency is why F079's half-ring hole
was a live serve loss instead of being recovered by symmetry, and it makes
the training campaign pay for 4 astroid tube arcs where ONE fundamental-
domain arc suffices (~4x tube training cost for zero accuracy).

Fold the query source into the canonical first quadrant BEFORE tube-chart
lookup, reusing the EXISTING fold helper (DRY: one authoritative fold — the
same helper the other chart kinds call; a second quadrant map is the
violation this program exists to remove). Train/serve only fundamental-
domain arcs. The tiler still detects 4 cusps -> 4 arcs (geometry truth,
pinned by the F079 build's `_EXPECTED_ARCS` check — do NOT weaken it); the
FOLD selects which arcs get charts.

## Facts (verify at HEAD; the F079 wrap-fix build landed since the survey)

1. The house fold: locate the fold helper the exterior-polar/wedge/lobe
   paths share (survey pointed at `surrogate.py` ~:1001 "Fold into the
   canonical first quadrant via D2 symmetry" and the ~:2989 lobe fold).
   If the existing code has more than one folding implementation, that is
   a pre-existing DRY defect: consolidate to ONE and report it — do not
   add a third.
2. Saddle side: the deltoid lobes map to each other under the same D2.
   VERIFY (not assume) what the ~:2989 lobe fold already covers; if saddle
   tube arcs (macro-saddle deltoid wedges) are already effectively folded
   by the wedge machinery, say so and confine the change to the astroid
   tube path — the acceptance equality pin must still be exercised on BOTH
   parities either way.
3. Tube serving entry: `_tube_serves` / `_evaluate_chart` and the chart
   selection in the serve path; training entry `_train_band_charts` /
   `_build_tube_chart` (arcs from `detect_caustic_structure`).
4. The fold must commute with the cusp-window exclusion and the
   `theta_to_s` arc-length map: windows and maps are stored per-arc in
   arc-local coordinates, so a folded query must land in the charted
   arc's coordinates exactly as an unfolded first-quadrant query does.

## Scope

IN: the serve-side fold for tube lookups (both parities' tube paths);
training restricted to fundamental-domain arcs (chart count and engine-call
count reported); the census fold-consistency invariants (below); fast
decision-level tests.

OUT: any training run (campaign is 7a); wedge/lobe/exterior-polar folds
(already folded — touch only if consolidating duplicate fold helpers per
fact 1); the F079 topology check (stays exactly as landed); chart schema
changes.

## Acceptance

- MACHINE-PRECISION EQUALITY PIN: a served tube query and its three fold
  images return equal values (both parities; state the tolerance — exact
  bit-equality if the fold is applied before all arithmetic, else a stated
  near-machine bound with the reason).
- Tube training at a small synthetic config charts ONLY fundamental-domain
  arcs; chart count and engine-call count vs the unfolded incumbent
  reported (expected ~4x cut on astroid tubes).
- Serve values on the fundamental domain itself byte-identical to the
  unfolded incumbent.
- CENSUS BOOKKEEPING (owner-confirmed): route-equality across fold images
  pinned as a census invariant — a draw and its fold images must report
  the SAME serve route (a mismatch is a fold bug the census exists to
  catch). Serve fractions must be invariant to whether the population was
  sampled folded or unfolded — no quadruple-counting of coverage, no
  4x-deflating of gaps. Applies to the analytic-rung mirrors too (the c3
  admission gate is D2-equivariant by construction; the census must
  report it so).
- Full fast suite green; no second fold implementation in the tree.

## Constraints

Branch claude-dev; fragments (closes `todo.d/lensing_tube_d2_fold.md`,
`[→ spec]` — SPEC.md serving-regime text follows); values-not-paths;
in-build tests FAST; no engine sweeps in-build; if a WP believes it needs
a new measurement, escalate rather than iterate.
