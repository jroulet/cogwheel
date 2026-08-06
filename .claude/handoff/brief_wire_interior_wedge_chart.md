# Build — wire InteriorWedgeChart into training; retire the `ffin` path

## Mission

The astroid interior is the ONLY one of the four serving regions whose intended
chart class is not trained. `InteriorWedgeChart` is implemented, serve-wired
and tested but never constructed by `surrogate_training.py`; the interior is
still trained as `FarFieldChart` tiles carrying the `INTERIOR_SACR_C` label
(the "ffin" charts). Wire the wedge chart in, and DELETE the `ffin` path.

This is a COORDINATE change, not a label change. The envelope is unchanged:
`InteriorWedgeChart` already declares the `tau_c`-demodulated
`INTERIOR_SACR_C` label. Chart class and `envelope_definition` are orthogonal.

## Measured facts (current tree — do not re-derive)

`InteriorWedgeChart` (added 2026-08-03, `ff06b8a`) already has:
  - `from_wedge_values` (surrogate.py:2460) — engine-free seam for tests
  - `_assemble` (:2537) — reload path used by `_chart_from_npz`
  - `_WedgeCausticMap` (:526) — precomputed `r_caustic(gamma, theta_wedge)`
    table; "SIMPLER than `_FarFieldArcMap` — no cumulative integration"
  - `_to_wedge_fixed` (:1278) / `_from_wedge_fixed` (:1322)
  - `_wedge_serves` (:2991) and a live `select_chart` branch (:3135)
  - two passing test suites: `test_lensing_interior_wedge_chart.py`,
    `test_lensing_wedge_dd_arclength.py`

It is MISSING: an engine-backed constructor, a `_build_wedge_chart`, a wedge
tiler, and the call swap in `_train_band_charts`.

The `ffin` path to retire: `_farfield_interior_tiles` (surrogate_training.py:
1838) + `_interior_admission` (:1803) feed `_build_farfield_chart` with
`definition=INTERIOR_SACR_C`, called at ~:4239. In the 2026-08-05 production
attempt these were 106 of 165 charts.

## THE SADDLE PATH IS THE TEMPLATE — transcribe it

The macro-saddle interior is already complete end to end:

    _saddle_lobe_admissions (2276)
      -> _lobe_interior_tiles (2340, called at 4162)
      -> _build_lobe_chart (2874, called at 4364)
      -> LobeInteriorChart.from_lobe_engine

`_lobe_interior_tiles` lays uniform radial rows over a NORMALISED radius
(`rho_lobe in [0,1]`, centroid to lobe reach) and cusp-aligns the angular axis
so no tile straddles a cusp ray or the `+-pi` seam. The wedge tiler is the
same shape over `r in [0,1)` x `theta_wedge in [0, pi/2]`.

Read both `_build_lobe_chart` and `_build_farfield_chart` before writing
`_build_wedge_chart`. Do not invent a third pattern.

## Why the wedge coordinate is correct here

1. `(s, d)` needs a UNIQUE nearest-caustic foot and degenerates on the medial
   axis (astroid centre and diagonals) — hence `_FARFIELD_MEDIAL_AXIS_TOL`
   and the near-tied-foot rejection. `(r, theta_wedge)` is global inside:
   `r = 0` centre, `r = 1` caustic boundary, no ambiguity.
2. `theta_wedge = atan2(|y2|, |y1|)` in `[0, pi/2]` covers a QUARTER of the
   interior; `r_caustic` is exactly 4-fold symmetric so the fold is exact.
3. The DD product cap `w * |y| < 58` becomes `w * r * r_caustic < 58`, known
   at each grid point.

## Scope

IN — the engine-backed wedge constructor; `_build_wedge_chart`; the wedge
tiler; the swap at `_train_band_charts` (~4239); DELETION of the dead `ffin`
path (below); tests.

OUT — the macro-saddle interior (`LobeInteriorChart` owns it, already wired);
tube and exterior charts; any rename of `FarFieldChart` (deferred, see
`todo.d/lensing_farfield_name_spans_three_regimes.md`); any training run.

## Delete, do not leave reachable

`ffin` exists ONLY for the astroid interior and the wedge chart covers exactly
that region — both are positive-parity-only (`_interior_admission` raises
`ValueError` for `parity != 1`). After the swap nothing produces an
interior-tagged `FarFieldChart`, so these are dead and must go in THIS build:

  - `_farfield_interior_tiles`, `_interior_admission` (if the wedge tiler
    genuinely reuses the directional-admission geometry, MOVE it — do not
    leave a second caller);
  - the `definition=INTERIOR_SACR_C` branch of `_build_farfield_chart`;
  - the `child_definition = INTERIOR_SACR_C` branch of
    `_subdivide_farfield_tile` (~3675);
  - the "INTERIOR `FarFieldChart`" branch of `_heldout_eps` (~2988) with its
    `max|E|` normalization special case;
  - any load/serve branch reconstructing an interior-tagged `FarFieldChart`.

The `INTERIOR_SACR_C` LABEL stays — both interior chart classes carry it.
What retires is its pairing with `FarFieldChart`.

## Acceptance

1. No `FarFieldChart` carries `INTERIOR_SACR_C`. `grep INTERIOR_SACR_C
   cogwheel/lensing/surrogate_training.py` finds it only on the wedge and lobe
   paths.
2. Interior held-out eps no WORSE than the `ffin` baseline, at equal or lower
   chart count, on a small synthetic band. Report both numbers.
3. A medial-axis query (astroid centre or a diagonal) that the `ffin` path
   refused now SERVES. This is the correctness win; name the refusing case.
4. The D2 fold is exercised: a query and its three mirror images serve
   identical values to machine precision.
5. Every deletion above is done; nothing reachable constructs an
   interior-tagged `FarFieldChart`.
6. Full suite green, driver-verified post-build.

## Constraints

- Branch `claude-dev`.
- **Every domain-test description MUST name its target suite file**
  (`test_<x>.py`). A description naming no file is routed to cross-suite and,
  with several suites in play, is appended to every agent's prompt without
  being counted by the shard cap or the `60 + 20*n` budget (F057 — that is
  how a previous build died with zero tests written).
- Keep the WP count at or below 3.
- Slow tiers stay empty in-build (`COGWHEEL_BRUTE_ACCURACY`,
  `COGWHEEL_TRAIN_TIER`, `COGWHEEL_STRICT_TIMING`); fast synthetic oracles
  only. No training run.
- Assert VALUES against an oracle and a tolerance, never which branch produced
  them. No `git show HEAD` oracle (pre-commit enforced, F043/F045).
- Spec workflow: this closes
  `todo.d/lensing_interior_wedge_chart_unwired.md` — delete that fragment, add
  a `completed.d/` fragment, and run `python scripts/render_fragments.py`.
