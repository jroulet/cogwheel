# Professor short-term (this session)

## Inference review: InteriorWedgeChart / ffin retirement build (2026-08-06) — PASS

Reviewed test_lensing_interior_wedge_chart.py (63/63 PASS, ~43s) + ports
test_lensing_ppgo_bandsplit.py & test_lensing_exterior_windows.py
(146 passed, 4 skipped, 1 xfailed, ~3.5min). Env cogwheel-newlal.

Domain-correctness checks (not just green):
- Held-out accuracy: oracle = FRESH ChangRefsdalChannels.evaluate().envelope
  (SHIPPING engine code, not re-transcribed), eps normalized by max|E| (interior
  currency, NOT max|exact_total|), floor 5e-2. Query fan includes near-centre
  r=0.18 and pi/4 diagonal (0.30, pi/4). log-w vs linear-w gotcha handled
  correctly (served=log space, engine=exp(log_w)) — a mismatch would fake O(1)
  residual. Docstring says measured worst ~1.5e-2 < 5e-2 floor.
- D2 fold: 1e-12 atol, off-axis source (r=0.30,theta=0.60, y1&y2 both !=0), all
  4 mirrors, WITH self-falsification companion (non-mirror source must differ).
  Matches astroid dihedral-4 quotient (abs fold + atan2) in code-obs.
- Medial-axis (pi/4 + small-r) serving: consistent with my earlier ruling that
  pi/4 is the fold-arc D2 symmetry MIDPOINT (smooth carrier), not a
  discontinuity — so ffin nearest-foot degeneracy removal is genuine, not
  papering over a real seam. r_extent<1 leaves Airy edge to tube chart.
- ffin retirement verified at MODULE level: hasattr(st,'_farfield_interior_tiles')
  == False; _wedge_interior_tiles & _interior_admission present; wedge builder
  stores INTERIOR_SACR_C, farfield builder does NOT.
- Ports clean: bandsplit re-expressed vs _wedge_interior_tiles (L633); exterior
  _interior_admission tests all retained (3 refs), deleted method gone.

Operator-deferred (out of fast-test scope): production eps~1e-4 re-gate and full
posterior-sampling validation. Note code-obs red flag: near-caustic box-edge
configs can give large lnL error at production eps — the 5e-2 here is the coarse
in-build floor (5x5x5 grid), NOT a production accuracy claim. Diagnostic PNGs not
visually read (no image tool); validated the numeric asserts backing them.
