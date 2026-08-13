# Build Brief: Saddle deltoid interior cusp serving

## Mission

Make the cusp arm serve INTERIOR cusp sources of the SADDLE deltoid lobe
(the 4-image interior near a deltoid cusp vertex), where it currently
refuses.  Documented in `.claude/spec/todo.d/lensing_saddle_interior_cusp_serving.md`.

## Measured facts (at HEAD c24dee4)

- For gamma=1.3, sources INSIDE a deltoid lobe (rho 0.5-0.9, e.g. (-1.2, 0)
  rho=0.70, or the interior near the deltoid tip with beta=0.3,
  src=(-1.343,-0.461), rho=0.828) refuse on the cusp arm at w=50-1000.
- Root cause (driver trace): the deltoid cusp's soft axis is TANGENTIAL to
  the lobe; "interior" (toward the lobe centre) is along the HARD axis.  So
  the source offset projects onto the hard axis → the Pearcey control
  y = delta_perp·w^{3/4}/|C4|^{1/4} dominates with x ~ 0.5 (small), giving
  n_stat=1 (EXTERIOR Pearcey regime) — the interior calibration bypass
  (len==3) never fires.
- Compare ASTROID (gamma<1): the soft axis points TOWARD the caustic
  interior, so interior sources offset along it give x<0 → 3 stationary
  points → the bypass fires and they serve.
- The existing `_VERTEX_CONFIGS` saddle entries (gamma=1.3, rho=0.708) DO
  serve at w=40 via the EXTERIOR calibration path (n_stat=1), NOT the
  bypass — so the saddle interior serving at low w is pre-existing
  exterior Pearcey, and w>=80 refuses.
- The ppGO fold-band gate is parity-agnostic (nearest.distance) — NOT the
  blocker here.

## The fix direction (Professor to adjudicate)

The deltoid interior cusp region needs a control mapping that produces the
3-stationary interior Pearcey regime.  Options:
(a) Lobe-local coordinates: the surrogate's `_lobe_boundary_radius` /
    `_deltoid_cusp_axis_map` already handle the deltoid lobe geometry —
    reuse that (rho_lobe measured from the lobe centroid, u = d^{2/3}
    angular distance to the NEAREST deltoid cusp) to build the cusp-arm
    controls so interior deltoid sources map to x<0.
(b) A rotated control frame at the deltoid cusp that points the "interior"
    direction into the negative-x Pearcey half-plane.

The Professor must decide which gives a certified serve at the envelope bar.
Prefer reusing the surrogate's lobe machinery over inventing a new one.

## Acceptance
1. Saddle deltoid interior cusp sources (rho < 1 near a deltoid cusp vertex,
   both lobe branches and beta) are served by the cusp arm (fast, no exact
   engine, no live quadrature) at w >= 80 where they currently refuse, with
   the same tolerance as the astroid interior serving.
2. No regression: `test_lensing_airy_fold.py` (ServedValue*, InteriorCuspServing,
   Ppgo, the `_VERTEX_CONFIGS` saddle entries must STILL serve at w=40),
   `test_lensing_fast_path.py`, `test_lensing_operator.py` green.
3. Refusal-conservative; the exact engine is never the serving rung in the
   cusp neighbourhood (driver mandate).

## Constraints
- Fast tests only. Refusal-conservative.
- Reuse `surrogate._lobe_boundary_radius` / `_deltoid_cusp_axis_map` if
  appropriate — do NOT duplicate the lobe geometry.
- Do NOT weaken the calibration bypass for the astroid (it works); the fix
  must ADD saddle coverage, not loosen existing gates.
