# Build: certify raw ppGO on the TRUE caustic interior

## Mission

The interior fold-ppGO rung in `cogwheel/lensing/likelihood.py` gates on three
legs. Leg 1 (`rho <= 1`) does not mean what it says, and leg 3
(`_uniform_error_estimate <= CERTIFICATION_BAR`) is inherited from the fold arm
rather than derived from what the rung actually serves (raw ppGO).

Replace leg 1 with the EXACT interior predicate, and leg 3 with a certificate
derived from raw ppGO's own asymptotics. Both changes are already measured;
this build implements and tests them.

## Measured facts (you cannot obtain these cheaply; all at SHA 87e62bb)

1. `ppgo_map.caustic_rho` normalises `|y|` by the caustic's MAXIMUM reach over
   angle. The astroid is not a circle, so `rho <= 1` is NOT the interior:
   measured over 2400 points (gamma 0.2-0.9, 4 angles), it admits caustic
   EXTERIOR sources at **58.7%** of points, and `max_reach / r_caustic(theta)`
   ranges **1.45x to 6.20x**.
2. The exact interior predicate is FREE: **four real images <=> caustic
   interior**. Measured against the closed-form caustic curve (the parametric
   form documented in `geometry.caustic_derivatives`), the two agree at
   **0 disagreements / 2400 points**.
3. On the TRUE interior, the pure `w^-3` series certificate is conservative
   with NO ghost term and NO safety factor: **1248 points, median ratio 0.587,
   p99 0.953, MAX 0.980, optimistic at 0.0%** (ratio = true_error/certificate;
   >1 is the dangerous direction).
4. On the EXTERIOR the same certificate is **77.5% optimistic, max 362,199x**.
   Every catastrophic point is a source leg 1 should have excluded.
5. **0 of 78** true-interior records in the sweep have a live ghost. Interior
   implies no ghost, as the quartic root count requires.
6. The leading omitted term is the `c3` term of the same stationary-phase
   series that produces the shipped `geometry._c1_polynomial` /
   `_c2_polynomial`. The derivation reproduces BOTH to **2.4e-15** and
   **5.8e-14** relative over 44 images spanning gamma 0.2-0.8. `c3` is purely
   imaginary and is a polynomial in the same `(prr, prt, ptt)` metric
   `geometry._saddle_metric` returns. Derived exponent **-3**; measured median
   `d log err / d log w` = **-3.010**.
7. Cost per gate call (4 images): exact `c3` **6.27 ms**. Cheap surrogates were
   tried and ALL fail — `|C2|^1.5`, `|C1||C2|`, `|C2|^2/|C1|`,
   `|C1|^3+|C2|^1.5`, `(|C1|+sqrt|C2|)^3` each under-predict `|c3|` by
   **30x-300x** somewhere over 1990 images. Ship the exact routine or derive
   the closed-form `C3(prr, prt, ptt)` symbolically.
8. `geometry.GhostAbsentError` landed in 87e62bb: it is raised ONLY when the
   real-image count proves no ghost exists, so it is the sound way to assert
   "the ghost term is exactly zero".
9. `F_op` is NOT an independent oracle above w = 60 (F069). Use
   `_schwinger.f_schwinger` directly, with the mass-sheet + eigenframe
   reconstruction copied from `operator._positive_parity_grid`. The
   demodulation origin is the `min` real-image `geometry.delay`, NOT
   `part.delays.min()`.

## Reference material (read these, do not re-derive)

- `.claude/handoff/ppgo_c3_reference.py` — WORKING `c3` series routine
  (`series_coefficients`), already validated against the shipped C1/C2.
- `.claude/handoff/ppgo_cert_sweep.json` — 434 configs x 16 w-points of raw
  measurements. Fields: `gamma, rho, theta, n_img, w, err, cert, absF, ...`
  where `cert` is the pure `w^-3` term and `err` the true ppGO error.
- `.claude/spec/todo.d/lensing_fold_ppgo_gate_is_inverted_in_xi.md` — the full
  backlog entry, including the caveats.
- FINDINGS F069 (F_op is not an oracle), F073 (`caustic_rho` is not interior).

## Scope

IN:
- A `c3`-based error estimate for raw ppGO, as a function in the
  `chang_refsdal` layer, taking the images/source/matrix and a `w`, returning
  an absolute error estimate on `|F - ppGO|`.
- Re-gate the interior fold-ppGO rung: leg 1 becomes the EXACT interior
  predicate; leg 3 becomes the new certificate.
- Tests for both, FAST (small synthetic configs, seconds).

OUT — do not do these:
- Do NOT add a ghost term. On the true interior the ghost is exactly zero
  (fact 5); if you find a gate-passing config with a live ghost, STOP and
  report it rather than adding a term.
- Do NOT change `ppgo_map.caustic_rho` itself. It is a legitimate MAP
  COORDINATE and the map is built and queried in that gauge. Only its use as a
  DOMAIN PREDICATE is wrong. Check every other consumer
  (`likelihood._ppgo_cell_coords`, `surrogate_training._train_band_charts`)
  and report — do not re-gauge them in this build.
- Do NOT retrain any surrogate. Do NOT run slow tiers
  (`COGWHEEL_BRUTE_ACCURACY`, `COGWHEEL_STRICT_TIMING` stay empty).
- Do NOT delete or rewrite the fold arm. `fold_amplification` is a separate
  consumer of `_merging_fold_pair` and is out of scope.

## Acceptance

- The rung's interior leg is exact: a source with 2 real images is refused by
  leg 1, verified by a test that derives its fixtures from the geometry rather
  than pinning literals.
- The certificate is implemented and its cost is reported (a number, not a
  claim).
- Over a sweep the build runs itself on the TRUE interior with `w <= 60`
  (where the exact engine can check), report: number of admitted points, and
  the MAX true error among admitted, against `CERTIFICATION_BAR`. This is
  REPORTED EVIDENCE, not a permanent test.
- Existing suites stay green. `test_lensing_ghost.py`,
  `test_lensing_airy_fold.py`, `test_lensing_fold_ppgo_handoff.py`,
  `test_lensing_ppgo_above_ceiling.py`, `test_lensing_likelihood.py` are the
  binding ones.
- Whatever safety factor you choose, JUSTIFY it from the measured ratio
  distribution. Fact 3 says 1.0 already suffices on the measured interior;
  a modest margin is defensible, 10x is not.

## Constraints

- Branch `claude-dev` only. Behavior change in `cogwheel/` => spec/TODO
  workflow applies: the existing `todo.d` fragment must be retired to
  `completed.d/` if you close it, and `render_fragments.py` run after.
- Assert VALUES against an oracle and a tolerance, not code paths. One
  canonical pin per routing decision, in the file that owns the predicate.
- Derive test fixtures from the live production constants; never pin a literal
  that a moved threshold would strand.
