# Build: tube trainer resolvable-sub-arc trim (promote F083)

## Mission

Full cusp-to-cusp astroid tube charts FAIL the bar at production
density: near-cusp zones where `_merging_fold_pair` refuses make build
nodes zero-fill and spread spline knots over dead regions, while F083
proved eps 4.3e-3 at n_theta=10 on the TRIMMED sub-arc of the same
gamma=0.4 band.  PROMOTE the F083 trim algorithm (derived and tested
in the test fixture) into `surrogate_training` so the tube build path
trains on the trimmed span; the fixture then imports the production
helper (DRY).  Closes
`todo.d/lensing_tube_trainer_resolvable_subarc_trim.md`; every clause
binds.  BLOCKS the training campaign's astroid tube legs and the
tiling design's per-band theta spans.

## Facts (measured at bbfcf7e unless noted)

1. Trim home: `cogwheel/tests/test_lensing_tube_beat_free.py`.
   `_f083_delta_tau` (L1058) — merging-pair delay gap at a tube node;
   `None` when the source is non-finite, images != 4, or
   `_merging_fold_pair` refuses ("the exact conditions under which
   `_tube_f_ref` ... is undefined").  Knee scan in `_f083_shared_tube`
   (L1112-1128): 80-point theta scan at the binding corner
   (gamma_hi = _BAND[1], eta_max); peak = nanargmax; low knee = first
   finite theta clearing `_F083_DTAU_FRAC = 0.6` of peak; then
   theta_lo = knee + `_F083_LO_STANDOFF = 0.20` * span, theta_hi =
   peak - `_F083_HI_STANDOFF = 0.05` * span (span = peak - knee).
   Comment rationale (L1036): "derive the robust servable sub-arc
   from the binding corner's live merging-pair profile (the full
   cusp-to-cusp arc has a non-monotone Delta_tau and must not be
   used)"; "robust at the binding corner is robust across the whole
   gamma axis".  ORDERING: r_min/eta_max/eta_floor on the FULL arc
   (L1107-1109), trim after, chart on `dataclasses.replace` (L1128).
2. Trainer path: `surrogate_training._train_band_charts` (L4926):
   `_tube_training_arcs` (L4946) -> per-arc `arc_r_min` on FULL spans
   (L4949; max/min_eta_max also size w-cap/interior-skip/wedge/
   lobe-edge shells) -> per-arc loop (L4972) sets eta_max/eta_floor =
   f*r_min (L4977-4978) -> `_build_tube_chart` (L2969), theta nodes =
   uniform-arc-length images of [arc.theta_lo, arc.theta_hi]
   (L3022-3028).  INSERTION POINT: the per-arc loop after eta sizing
   (L4977-4990) — trimmed arc feeds `_build_tube_chart`,
   `_tube_heldout_samples`, report `theta_range`.  NOT inside
   `_tube_training_arcs` (Fact 6; also circular — scan needs eta_max).
3. Evidence (HEAD 77da2e6; n_gamma=n_u=n_theta=7, w/decade=15, w<=60):
   full-arc astroid (`f_fraction_sweep_results.json`, 60 pts) eps_band
   0.125-0.840, refused 98-147/343 nodes; trimmed astroid
   (`f_fraction_sweep_trimmed_astroid.json`, 60 pts) eps_band
   0.039-0.185, refused 0-1 — excluding the ONE recorded outlier
   (gamma=0.10, f_max=0.28: 0.604, "isolated ... re-probe during
   tiling design").  Saddle full-arc: eps 0.0032-0.140.  Ruling
   (`f_constants_decision.md`): f_max=0.40, f_floor=0.08; density
   flags astroid gamma 0.10-0.40, saddle ~1.1.
4. Serve side (`surrogate.py`): `_tube_serves` (L3080) declines
   off-span queries via `_tube_theta_inframe` (L2900, `None` unless a
   D2 gauge image lands in [theta_grid[0], theta_grid[-1]]) and gates
   on F_ref buildability (L3155-3158): `if require_fref: ... if
   _tube_f_ref(np.exp(chart.log_w_grid), gamma, source_q) is None:
   return False` — "a node that refused at build refuses here too".
   Refused = 0 on trimmed builds means NO zero-filled rows; the
   silent-r=0 watch item reduces to the Acceptance boundary probe.
5. DRY: `_merging_fold_pair` (`chang_refsdal/_airy_fold.py:278`) and
   `_frame_delays` (`channels.py:928`) are already production; the
   test already imports `_tube_training_arcs`/`_build_tube_chart`/
   `_tube_source`.  `_f083_delta_tau` + the inline scan (L1058-1128)
   retire in favour of the imported helper.
6. Census counts ARCS, not spans: `tiling_census._count_tube` (L290)
   `n_arcs = len(ctx.tube_arcs)`, tiles == arcs; it calls
   `st._tube_training_arcs` (L242, `census_dry_run.py:126` too) and
   derives its OWN r_min/max_eta_max from full arcs.  A Fact-2-sited
   trim is invisible to it; one inside `_tube_training_arcs` would
   silently shift census eta/w sizing.
7. SADDLE IS NOT A NO-OP UNDER THE RAW RECIPE: saddle sweep rows
   refuse 28-180/343 nodes yet pass eps — "resolves throughout" holds
   for the served core, not per build node — and the recipe maps even
   an all-finite profile to [knee + 0.20 span, peak - 0.05 span],
   never the identity.  Byte-identical saddle charts need an explicit
   gate (parity-only, or a profile predicate) — Professor, plan time.

## Scope

IN: the trim helper in `surrogate_training` (analytic knee scan,
constants carried verbatim with F083 provenance comments); wiring at
the Fact-2 insertion point with the saddle gate per Fact 7 (saddle
byte-identical); F083 fixture re-pointed to the production helper;
fast synthetic tests — knee location on a synthetic Delta_tau profile,
the saddle byte-identity pin, drifted-core LOUD failure (refused > 0
is a hard error — the `test_trimmed_run_refused_no_build_nodes`
pattern).
OUT: campaigns and tiling design; f-constants changes (0.40/0.08
stand); serve-side changes (`_tube_serves`, `_tube_theta_inframe`,
`_tube_f_ref` untouched); low-w/Born/c3 rungs; census counting
changes; the gamma=0.10 outlier re-probe.

## Acceptance

- One production-density astroid spot-check (gamma=0.4 band,
  n_theta=7, f_max=0.40, f_floor=0.12): eps consistent with the
  trimmed sweep row eps_band = 0.108 at 77da2e6 (GENEROUS tolerance —
  configs differ slightly), refused = 0.  Cost ~200 s (one sweep row)
  — the in-build ceiling; bulk sweeps stay driver post-build.
- Saddle bands byte-identical (or, if the plan admits a benign gated
  path, measured-equal with the delta quoted).
- Boundary interval (fragment watch item): a query between the
  trimmed fence and the cusp window is DECLINED by the tube chart
  (falls through the ladder), never interpolated — one pinned probe.
- F083 suite green importing the production helper; full fast suite
  green.  Parsimony: one canonical pin per invariant; added-vs-retired
  counts reported (the retired fixture-local scan counts).

## Constraints

Branch claude-dev only.  Constants carried over, never re-tuned
in-build.  Closes the todo.d fragment with a completed.d record;
`[→ spec]` — spec_changelog.d fragment with `bump:`; render fragments.
In-build tests fast/synthetic beyond the single spot-check build.
Escalate-not-iterate: if the trimmed production build misses the
sweep-consistent eps, or a saddle chart moves without a
plan-sanctioned gate, STOP — that falsifies the promotion premise,
not the plumbing; never widen tolerances.
