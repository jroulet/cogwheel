# Professor short-term (session: F081/saddle-tube-fundamental review, 2026-08-15)

Reviewed uncommitted build "trim saddle tube training to D2 fundamental set"
+ F081 per-arc lobe-edge shell (worktree cogwheel-claude-dev, on top of HEAD
93f2591). Env python: /home/tejaswi/anaconda3/envs/cogwheel-newlal/bin/python.

## Verdict: PASS (fast domain tests only; heavy full-sampling operator-deferred)

## Ran (engine-free / fast)
- test_lensing_tube_d2_fold.py: 24/24 pass, 9s (orbit-partition, serve-coverage, per-arc shell).
- test_lensing_tiling_census.py: 26/26 pass, 53s (census Q1 re-derives saddle trained via production selector).
- caustic_cusps ported: InteriorAdmissionMarginRemovalTestCase + test_inflated_margin_changes_admission: 2/2 pass, 12s.
- Full 4-file run timed out (240s) only due to slow engine tests in caustic_cusps/surrogate_training, unrelated to this build.

## Physics confirmed (real band (1.1,1.15), independent recompute)
- 6 detected deltoid arcs -> 2 D2-orbit reps (orbit sizes {4,2}: four +1-branch
  lobe-edge arcs = one orbit, two -1-branch arcs at gauge 0/pi = the other).
  COUNT DERIVED via independent union-find (_circular_gap/_d2_gauge_images
  re-derived in test, NOT production helper). Mission's a-priori "3" was
  explicitly to be derived; 2 is correct for this fixture.
- arc_r_min=[0.399, 9.156], f_max=0.4 => min_eta_max=0.160, max_eta_max=3.66
  (~23x anisotropy). corridor_half=0.160=1.0*min_eta_max (NOT max) — F081 fix
  intact (matches my prior Q1 ruling: lobe eta = f_max*lobe-edge r_min).

## Why PASS (teeth verified, not just green)
- Orbit trim: anti-vacuity (reps<arcs) + self-falsification (defeat
  _circular_angular_distance -> 6->6 identity).
- Serve coverage (symmetry moral-imperative equality pin): fundamental served
  set SUPERSET of all-6 incumbent over 720-angle ring, 0 violations; teeth =
  dropping ANY rep strands a band. This end-to-end pin backs the 6->2 count
  even if arc bookkeeping were off.
- Part B shell: witness at geometric-mean distance admitted under min shell,
  flips to excluded under max shell; equal-shell + reverted-derivation legs.
- Census Q1 & caustic_cusps ported off retired max_tube_arcs to
  _tube_training_arcs(structure, parity); assert live-geometry values.

## Caveat
Could not render PNGs in this harness (no image Read tool); confirmed 3 fresh
plots generated (saddle_orbit_partition, saddle_serve_coverage,
saddle_lobe_edge_shell_witness) and verified the assertions they visualize.
Docstring says "6->2 collapse" in one self-falsif docstring — matches reality;
an earlier astroid narrative still says "6->3" nowhere binding. No concern.
