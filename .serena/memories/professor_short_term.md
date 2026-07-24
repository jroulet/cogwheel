# Professor short-term (Build 8h-b3-fin, 2026-07-23)

## Round 4 — S2-3 whole-interior SACR-C near-cusp ruling (this session)

QUESTION: switching ENTIRE astroid/deltoid interior from far-field label
(E = F - sum_real H_a e^{iw tau_a}) to SACR-C tau_c-demodulated envelope. Is SACR-C
defined+bounded near astroid cusp rays where image pairs merge (Im tau_c -> 0 / delay
diffs collapse), or does it need a near-cusp interior carve-out / fallback?

RULING: CLEAN LABEL SWAP. No carve-out, no fallback. Whole-interior SACR-C is defined
and bounded at every interior point the charts admit, including near-cusp directions.

Reasoning (physics, not code):
- tau_c is the CRITICAL/VIRTUAL delay (Fermat delay at the merging-image saddle), NOT a
  delay DIFFERENCE between two live images. It stays finite+smooth as an image pair
  merges; it is the value the pair's common delay converges TO. The premise "Im tau_c->0"
  is a category error: the label has NO 1/(tau_a-tau_c) and NO Im tau_c denominator.
  Demodulation is a unimodular phase multiply e^{-i w tau_c} — never ill-conditioned.
- The gate S_a = smootherstep(w|tau_a-tau_c|,0.5,4) makes merging COLLAPSE-SAFE: as
  tau_a->tau_c the gate closes to 0, pulling the merging pair OUT of the analytic sum and
  INTO E exactly where the saddle asymptote fails. Bounded-phase theorem binds: E's
  demodulated phase <= rho_END=4 rad (switch scale IS demodulation distance). Memory
  records max|S_a H_a| <= 1.3 incl eta=+-0.002 cusp/fold crossings. Merging is the
  regime SACR-C was BUILT for — cusps are its best case, not its failure mode.
- Exterior cusp windows exist because the far-field exterior label sits on the caustic
  with NO tau_c demodulation shield. Interior SACR-C already absorbs that content into E,
  so the exterior-window analogy does NOT transfer to the interior.
- The genuinely ill-defined boundary is ELSEWHERE: det A -> 0 (parity boundary / Schwinger
  contour pinch at t=0 / mass-sheet refusal). Interior charts already admit only
  det A != 0 inside the caustic. Cusp rays are NOT that boundary.

EXCLUSION CURRENCY: none. Do NOT introduce a cusp-window half-width or a w*|tau_a-tau_c|
admission bound. (The w*|tau_a-tau_c| quantity IS used — but as the GATE argument that
routes content into E, not as an admission threshold.)

DO-NOT-CONFLATE: the crown (gamma~0.90) 1e-1 accuracy floor (quasi-symmetric
near-degenerate delay pairs, An-Evans crown quasi-symmetry) is a DIFFERENT sub-region and
DIFFERENT failure class from the cusp rays. Crown = relaxed accuracy bar; cusp = fully
1e-3-clean. A reviewer must not merge these into one "near-degenerate carve-out."

ONE GUARD TO PIN (label-consistency, NOT admission exclusion): near a cusp, WHICH critical
point is nearest can flip between adjacent cusp basins across proposals (same "tau_c can
jump between lobes/basins" class flagged on the saddle branch and for astroid
fold-to-cusp). The SUM is tau_c-permutation invariant; only E's ratio smoothness along the
greedy-node path needs a consistent tau_c. Pin: assert tau_c path-continuity within a tile
(no basin flip), else reseat via the assignment problem — the reset convention already
required for far-away proposals. This is interpolator-hygiene, not a physics exclusion.

NET: S2-3 = label swap ONLY. Any test asserting a near-cusp interior exclusion is a
FALSE-RED (would test a carve-out that must not exist).
