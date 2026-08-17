# Tube beat-free representation — RECOVERY PLAN (Professor re-confirm, 2026-08-17c)

Supersedes the earlier investigation notes. Physics (Q1 non-vanishing q=p,
w-bounded residual, node collapse) UNCHANGED and signed off. Code complete.

## STEP 1 — Delete orphan
Remove `cogwheel/tests/test_lensing_tube_nyquist_coordinate.py`: live tree
version imports superseded TV-axis symbols (_validate_tube_axis_schema,
_TUBE_AXIS_SCHEMA, theta_to_s_prime) -> errors on collection.

## STEP 2 — Author F083 permanent eps-sweep class (does NOT yet exist)
Add to test_lensing_tube_beat_free.py. Spec UNCHANGED:
- Build: astroid gamma=0.4 arcs[0], trimmed to servable sub-arc, n_theta=10.
- Servable sub-arc DERIVED from live boundary: scan arcs[0] for the maximal
  interior run where `_merging_fold_pair` does NOT return None; robust
  Delta_tau at both ends. NOT a pinned literal (drift -> spurious fail on a
  non-servable region OR spurious pass on padded region). Trim also dodges
  the known _tube_delay_map "not strictly increasing" full-arc defect.
- Held-out: 8 OFF-node samples at eta=0.5*eta_max, theta strictly BETWEEN
  the 10 build nodes (midpoints (theta_i+theta_{i+1})/2 across interior
  gaps). INTERPOLATION error only — no extrapolation past end nodes.
- Metric: eps = F_ref-normalized max relative error over held-out x, w in
  [40,80]. Assert eps <= 0.0237 (untightened bar; leave 10-vs-48 margin
  visible, do NOT tighten).
- GUARD: assert refused_count == 0 on the trimmed servable run (else a
  held-out w can land on a zero-filled node -> garbage measurement).

## STEP 3 — Add RAW-source serve probe pin (NEW invariant, not duplication)
Not implied by RoundTrip (nodes only, fold=identity there) or Buildability
(smoke) or D2ServeEquality (F_ref D2-invariant, so equality holds for folded
source too). Serve at a NON-fundamental octant (mirror image); spy
`_tube_f_ref`; assert its source arg == RAW (y1_eig,y2_eig), differs from the
folded coord. This is the sole pin guaranteeing serve-side None path
unreachable.

## STEP 4 — Coder audit (unrecorded Inspector 2 impl + 1 design findings)
Look hardest, in order:
1. tau_bar vs tau_c frame consistency in `_tube_f_ref`. A constant origin
   offset round-trips EXACTLY at build nodes (RoundTrip green) but leaks a
   slowly-growing OFF-node error -> exactly what F083 interp sweep catches.
   MOST LIKELY defect site.
2. isolated-node zero-fill vs serve None->RuntimeError: a served w landing
   (post-clamp) on a zero-filled node gives r=0 -> silent zero amplitude,
   NOT the loud raise. Check serve never interpolates across a zero-filled
   node unflagged. Mitigated by Step 2 refused_count==0 guard.
3. (LOWER risk) D2 gauge-image source selection: F_ref D2-invariant so a
   wrong-sign image keeps |F_ref| correct; can only mis-select residual
   coord, already pinned by d2_fold suite. De-prioritize.

## Existing 6-class suite verdict
Well-chosen, keep all. Gaps closed by Steps 2+3. No duplication introduced.
