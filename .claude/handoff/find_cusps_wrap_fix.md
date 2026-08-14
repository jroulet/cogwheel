# Build: fix `_find_cusps` wrap arithmetic; retire the dead cusp-arm coverage machinery

## Mission

F079: `_find_cusps` walks dip windows with periodic INDEX arithmetic but
computes the window span with LINEAR angle arithmetic
(`surrogate_training.py:597`: `span = abs(thetas[i] - thetas[lo]) +
abs(thetas[hi] - thetas[i])`). At the wrap point the theta = 0 cusp's span
computes as 1.5*pi (the detector's whole window) instead of the ~0.106-0.130
rad the same detector measures when 0 is window-interior — a 36-44x
overestimate at every gamma tested (0.2, 0.5, 0.7, 0.9). `_make_arc` then
gets a negative inner span for both adjacent arcs and returns None
(`:699-700`), so `_astroid_arcs` yields 2 of 4 arcs and the entire
`theta in (-pi/2, pi/2)` half of the astroid fold ring silently gets no tube
chart. The topology cross-check (`detect_caustic_structure:785-787`) counts
CUSPS (4 == 4, passes), never arcs — the exact blind spot. TubeChart serving
is not D2-folded, so the hole ships in any training run. MUST land before
`train_lens_surrogate.py`.

Same build retires the measured-dead cusp-arm coverage machinery (F079
retirement list): `_CUSP_ARM_COVERAGE = 0.07` is in the WRONG UNITS for its
consumer (image-plane polar offset vs the critical-curve parameter angle the
gate subtracts it from — 40x apart, non-monotonically related) and INERT
(0 differing serve decisions over 64 production cusp windows at 0.0 vs 0.07;
no reachable in-band interior query comes within 0.33 rad of a window).
Post-F074 there is no angular serve boundary at all — the real structure is
the w-floor 49 and a gamma-dependent serve fraction.

## Measured facts (SHA `9f331dd` for F079's numbers; surveyed again at launch HEAD)

1. Defect site is the ONE line `surrogate_training.py:597`. The serve-side
   gate (`surrogate.py:2893`) and `census_dry_run.py`'s `_is_near_cusp`
   (`:215`) already do correct mod-2pi — `abs((a - b + pi) % 2pi - pi)` is
   the house idiom; the producer is the only linear site.
2. `_saddle_arcs` uses `periodic=False` per wedge (`:662`) and its goldens
   are frozen (`_WP1_GOLDEN_STRUCTURE`, `test_lensing_surrogate_training.py
   :2360-2404`). The wrap fix MUST be gated on `periodic` so the saddle path
   stays byte-identical.
3. `_astroid_arcs:636` already handles the index wrap; only the oversized
   half-widths kill arcs 0 and 3.
4. A golden currently FREEZES THE BUG: `GOLDEN_INWARD_SIGN`
   (`test_lensing_caustic_cusps.py:277-284`) pins astroid arcs as 2-tuples
   (`assertEqual(len(structure.arcs), len(golden))` at `:790-805`). The fix
   turns it red BY DESIGN; re-freeze to 4-tuples with signs DERIVED from
   geometry (the suite's own `test_frozen_sign_is_the_geometric_two_image_side`
   at `:807` is the derivation check — signs must satisfy it, never be
   copied to make the count pass). Saddle rows (6-tuples) unchanged.
5. Interior-baseline window widths for the theta = 0 cusp: ~0.11-0.13 rad
   at the tested gammas; post-fix the theta = 0 window must agree with a
   window-interior baseline at the same gamma (rotate the grid or compare
   against the other three cusps — state the method).
6. `max_tube_arcs` default is 1 (`surrogate_training.py:302`), so
   default-config tests see no chart-count change; production sets 20
   (`scripts/train_surrogate_production.py:60`), one test sets 4
   (`test_lensing_surrogate_training.py:6594`).
7. Arc-count blast radius: `surrogate_training.py:4768-4781` (charts per
   arc, tags `chart_{label}_tube_{idx}`, `max_eta_max` over arc r_min).
   `test_lensing_surrogate_training.py:2046-2054` asserts astroid arc count
   is equal ACROSS gammas — survives 2->4 but check any literal.

## Scope

IN (production, 3 files):
- `surrogate_training.py`: mod-2pi span at `:597` (periodic-gated, house
  idiom), docstring `:563-565`; `detect_caustic_structure` gains the
  arcs-survive-the-tiler pin: expected cusps AND expected surviving arcs
  (astroid 4 -> 4) — count ARCS, not just cusps.
- `surrogate.py`: delete `_CUSP_ARM_COVERAGE` / `_SADDLE_CUSP_ARM_COVERAGE`
  (`:295-313`) and the `_tube_serves` shrink (`:2886-2892`; gate keeps
  `residual = delta_theta`).
- `surrogate_census.py`: update the `cusp-window` category note
  (`:267-272`): the category reports zero STRUCTURALLY post-F074; real
  cusp-region losses surface as eta-floor / w-cap categories. Keep or drop
  the category — decide, and say why in the note.

IN (tests):
- `test_lensing_caustic_cusps.py`: re-freeze `GOLDEN_INWARD_SIGN` astroid
  rows to 4-tuples (fact 4's derivation discipline); add the 4-cusps->4-arcs
  pin near `:497`; add the theta=0-window-vs-interior-baseline value pin
  (fact 5).
- `test_lensing_surrogate_training.py`: delete D2a/D2b classes
  (`:5995-6260`) + `_WP2_CUSP_*` fixtures; verify `:2046-2054` and the
  `_WP1_*` saddle goldens stay green (the saddle regression witness).
- `test_lensing_surrogate.py`: delete the parity-gating + self-falsification
  classes (`:5056-5297`) and their plot output.
- `test_lensing_cusp_arm_coverage.py`: fix prose `:8`; extend the RETIRED
  block (`:405-445`) with the F079 retirement.

IN (scripts):
- Delete `scripts/measure_cusp_arm_reach.py`,
  `scripts/measure_cusp_arm_actual_boundary.py`,
  `scripts/measure_saddle_cusp_arm_coverage.py`.
- `scripts/census_dry_run.py`: delete the mirrored `_CUSP_ARM_COVERAGE`
  (`:28`), the residual arithmetic (`:357`), banner (`:400`); re-express the
  `cusp_arm` route decision on F074 terms (the w-floor), not an angle.
- `scripts/calibrate_ppgo_rung.py`: its import of `_CUSP_ARM_COVERAGE`
  (`:48`) breaks at retirement — replace the probe ladder's scale with a
  script-local derivation or retire the script; decide from what the script
  still measures post-F074, state why.

OUT: any training run; tube-chart schema; the campaign's cusp-region table
targets (F079's not-a-bug residue: gamma >= ~0.45, w in [DD cap, 49) falls
to the engine correctly — that is 7a's job); saddle wedge geometry; D2
FOLDING of tube serving — a follow-on build (owner directive, fragment
`lensing_tube_d2_fold`) folds tube queries into the first-quadrant
fundamental domain so ONE arc's chart serves all four. Do NOT build any
per-arc serving machinery beyond tiler correctness here: the 4-arc pin is
about the TILER telling the truth, not about training 4 charts.

## Acceptance

- Astroid: 4 cusps -> 4 arcs at every production gamma band, pinned by the
  new arc-count check (a build that fixes the span but drops an arc for any
  other reason FAILS).
- The theta = 0 cusp window agrees with the window-interior baseline value
  (~0.11-0.13 rad at tested gammas) to the detector's own resolution —
  a VALUE pin, not a path pin.
- Saddle structures byte-identical: `_WP1_GOLDEN_STRUCTURE` and the frozen
  saddle golden rows pass unchanged.
- `GOLDEN_INWARD_SIGN` astroid rows re-frozen to 4 entries whose signs pass
  the geometric-side derivation test — not copied from the count.
- No reference to `_CUSP_ARM_COVERAGE` / `_SADDLE_CUSP_ARM_COVERAGE`
  anywhere in the tree (grep clean, imports included); full fast suite
  green.

## Constraints

Branch claude-dev; fragments per CLAUDE.md (this closes
`todo.d/lensing_find_cusps_wrap_bug.md` and
`todo.d/lensing_cusp_arm_coverage_constant_stale.md`; `[→ spec]` — census
note + FINDINGS pointer discipline: F079 gets a resolution pointer, never an
edit); values-not-paths; in-build tests FAST (synthetic/small configs); no
engine sweeps in-build. Line numbers above were surveyed at `9f331dd` —
re-locate by symbol/pattern, do not trust offsets blindly.
