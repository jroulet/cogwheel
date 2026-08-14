# Coder Short-Term Observations

## 2026-08-14 (INS-2-001 — Born band-split shape ValueError -> full-band serve + envelope zeroing)

- DEFECT: `_born_residual_analytic` (WP-B) crashed with a shape ValueError
  whenever the certified-map band-split branch fired. It sub-sliced
  `chart_w = dense_w[below_mask]` (length k) and passed chart_w as `w` while
  handing FULL-length `geom.saddle_kernels`/`geom.delays` (N rows) to
  `born_carrier_from_partition` -> `reconstruct_farfield`, which validates
  `saddle_kernels.shape[0] == w.size` and raises ValueError naming
  'saddle_kernels'. The dispatcher `_amplification_coefficients` does NOT
  swallow ValueError (only LensDomainError), so the whole likelihood eval
  crashed. Reachable on a SUPPORTED config (certified map installed + a
  Born-eligible exterior cell certifying with w_trust strictly in-band).
- FIX (mirrors the buried `_surrogate_coefficients` twin's envelope-masking):
  serve carrier / residual / ppGO over the FULL `dense_w`. partition_ns now
  `w=dense_w`; `residual = born_chart.evaluate(dense_w, ...)`; ppgo summed
  over `dense_w[:,None]*delays`; `envelope = (f_total - ppgo) * exp(1j*
  dense_w*t_min)` then `envelope[~below_mask] = 0.0`. E_ff=0 above w_trust
  telescopes to the bare ppGO image-kernel sum in the FARFIELD_KERNEL_SUM
  gauge (channels.reconstruct_farfield docstring confirms: "the ghost lives
  in the mid-band window only, never in the bare ppGO band above w_trust").
  NEVER sub-slice w against full-length geometry.
- `chart_w = dense_w[below_mask]` KEPT — but now used ONLY for the INS-1-001
  trained-band refusal `born_chart.covers(gamma, rho, chart_w)` (the served
  sub-band must lie in the trained log_w range); it is no longer passed to
  any geometry-length consumer. Null-split identity preserved: no map =>
  below_mask all-True => nothing zeroed => byte-identical whole-band Born.
- TEST: rewrote `MapBandSplitTraceTestCase` docstring (describes the fix)
  and DELETED the tripwire `test_band_split_serve_raises_shape_defect`
  (it PINNED the crash via assertRaises(ValueError)+assertIn('saddle_kernels')
  on both direct rung and dispatcher). Kept the two premise tests
  (w_trust strictly in-band; beyond-wall cell keeps whole band). Did NOT
  invert the tripwire into a positive `max|k_split-k_nomap|>0` assertion —
  that would certify my own new serve; flagged for the Test Developer (the
  docstring names it the Architect's MAP BAND-SPLIT TRACE invariant, owned
  by the test author). `dimensionless_frequency` import still used (no orphan).
- VERIFIED via the test module's own fixtures (direct execution, not the
  suite): band-split premise fires (w_trust=30.000, n_below=31/64); direct
  rung SERVES k0/k1 shape (4,4) finite=True; dispatcher SERVES (returns a
  tuple, no raise); no-map whole-band finite; beyond-wall == no-map
  byte-identical. reconstruct_farfield signature re-confirmed
  (w, envelope, delays, saddle_kernels, real_mask, definition, t_min) —
  call matches, all args full-length. ast.parse OK both files.

## 2026-08-14 (WP-A — BornResidualChart.load + writer hash/schema parity)

- Added `BornResidualChart.load(path=None)` + `_default_artifact_path`
  (staticmethod, mirrors CertifiedPpgoMap via `files('cogwheel').joinpath
  ('data', _DEFAULT_CHART_NAME)`). Loader: `np.load(allow_pickle=False)`,
  hard-refuse (ValueError naming scripts/train_born_residual.py) on
  missing/mismatched `schema` key (`born_residual_v1`), missing
  `content_hash`, or recomputed-hash mismatch; else construct via the
  frozen ctor. DESIGN CHOICE (per brief DRY-vs-coupling): DUPLICATED the
  ~5-line `_content_hash` in born_residual_chart.py rather than importing
  ppgo_map's — ppgo's variant folds certification scalars into its
  signature (different args) and importing adds an intra-lensing edge for
  a trivial helper. born hash = SHA1 over gamma/rho/log_w/real/imag
  float64 bytes (NO scalars — chart has none).
- FRAMING DECISION: content_hash + schema are SEPARATE top-level npz keys
  (per brief "How"), NOT inside provenance (ppgo puts hash in provenance).
  Writer now emits `provenance=np.array(json.dumps(...))` (was
  `str(...)`), plus `content_hash` and `schema` keys. Writer imports
  `_SCHEMA, _content_hash` from born_residual_chart (both module-public
  underscore names, intentional reuse by the shipping writer).
- RE-SAVE (no retrain, F075 CLEAN): loaded old npz arrays, parsed the OLD
  repr-string provenance via ast.literal_eval -> dict, re-dumped as JSON,
  wrote via a .tmp then os.replace after asserting real/imag_coeffs bytes
  byte-IDENTICAL to the prior artifact. Old size 7990 -> new 8714 bytes
  (grew only from json provenance + 2 new keys; numeric payload
  unchanged). content_hash = 2725474eeeda46b3cfc58bfcc6198cc7b1c9ea59.
- SMOKE-TESTED: default load() round-trips + evaluate() works; all 4
  integrity guards (missing-schema, wrong-schema, missing-hash,
  tampered-payload) raise ValueError naming the regen script. py_compile
  OK both files. Consumer note: likelihood.py L877/1808 holds a
  born_residual_chart attr (currently defaults None) — WIRING the loaded
  chart into the likelihood is a SEPARATE WP (handoff wire_serving_
  artifacts.md), NOT this one.

## 2026-08-14 (WP1 revision — INS-1-001 census diagnostic fold-consistency)

- classify_fallthrough (surrogate_census.py) HAS y1_eig/y2_eig params and
  characterize_sample passes the REAL eigenframe source, but the
  cusp-window probe called `_tube_serves(..., image_count)` with RAW theta,
  OMITTING y1_eig/y2_eig — so the cusp-window fall-through CATEGORY was not
  D2-equivariant (mirror-image draws misclassified, per-category gap counts
  deflated up to 4x). The sibling exterior-polar refusal-ball probe already
  passed them. FIX: append `y1_eig, y2_eig` to that `_tube_serves` call
  (positional, after image_count — matches the nan-default signature). Served
  flag / chart_index / aggregate serve fraction were ALWAYS fold-invariant
  (production serving folds via select_chart); this is a DIAGNOSTIC-ACCURACY
  defect only, no over-certification.
- Closed the deliberate category exclusion in test_lensing_tube_d2_fold.py
  `_assert_route_equality`: added `categories=[r.category...]` to the D2
  equality pin + corrected the stale "category deliberately NOT asserted"
  comment. Done at Inspector's explicit direction (INS-1-001 suggested fix);
  pre-existing test scaffolding for already-landed D2 physics, and the class
  docstring already claimed "same fall-through category" so the pin now
  matches its own contract. Diagnostic test figure-title ('census route is
  identical across D2 sign images') is now accurate with no change.
- LESSON: "census inherits the fold via the NaN default" holds ONLY for a
  caller WITHOUT the source in scope; any caller that DOES have y1_eig/y2_eig
  must thread them or it silently runs the unfolded (identity) diagnostic.

## 2026-08-14 (WP-D — get_init_dict JSON round-trip for born_residual_chart)

- PROBLEM: base LensedRelativeBinningLikelihood eagerly resolves the
  `_AUTO_BORN_CHART` sentinel in __init__ -> `self.born_residual_chart` is a
  chart|None, so the resolved value CANNOT distinguish auto-loaded-default
  from a caller-supplied copy of the same artifact. Fix: record intent flag
  `self._born_residual_chart_is_default = (born_residual_chart is
  _AUTO_BORN_CHART)` set BEFORE the auto-load branch (both paths covered;
  private attr, not a ctor param -> JSONMixin.get_init_dict never captures
  it, confirmed: base get_init_dict does plain getattr over ctor param names,
  no recursion/encode until to_json).
- BASE get_init_dict three-way (keyed on the flag): is_default -> pop key
  (reconstruct re-defaults to sentinel, re-auto-loads, re-serves Born);
  elif chart is None -> EMIT `born_residual_chart=None` (explicit opt-out
  must round-trip pure-engine — NOT pop; popping would re-auto-load, the
  latent bug WP-D fixes since WP-C flipped the default to the sentinel);
  else -> NotImplementedError NAMING the limitation (caller-supplied
  in-memory chart: no source path on BornResidualChart, tables not embedded;
  ctor takes only instance|None|sentinel, not a path string, so path-emit is
  genuinely unsupported). amplification_surrogate logic UNCHANGED.
- MARGINALIZED get_init_dict: this class stores the ctor value VERBATIM
  (self.born_residual_chart may still BE the sentinel), so key directly on
  `self.born_residual_chart is _AUTO_BORN_CHART` (pop) / is None (emit None)
  / else raise — no flag needed, and dropped the old engine-resolved keying
  (`self._engine.born_residual_chart`) which wrongly raised on the
  auto-loaded default. `_AUTO_BORN_CHART` already imported there (L65).
- VERIFIED: ast.parse + import both modules OK (h5py warn pre-existing);
  shipped cogwheel/data/born_residual_chart.npz exists + BornResidualChart
  .load() OK (default omit genuinely re-serves); replayed all 6 branch cases
  in isolation (default->{}, None->{'...':None}=json null, chart->raise for
  both classes). UNVERIFIED: full JSONMixin round-trip + live Born serve on
  the reconstructed object (needs heavy EventData/WaveformGenerator/par_dic_0
  fixtures — not run per write/verify split).

## 2026-08-14 (INS-2-001/002 — census mirror + dead max_tube_arcs knob)

- INS-2-001: scripts/census_dry_run.py saddle path computed arc_r_min over
  `structure.arcs[:cfg.max_tube_arcs]` — STALE MIRROR after WP1 routed
  production tube-arc selection through `_tube_training_arcs` (saddle => ALL
  arcs). Fix: `tube_arcs = _st._tube_training_arcs(structure, _SADDLE_PARITY)`
  (already importable via the `_st` = surrogate_training alias, no new logic),
  iterate arc_r_min over `tube_arcs`. Now the dry-run's reported max_eta_max
  bound == what the trainer builds (served == counted). Diagnostic/mirror
  accuracy only; production serving + aggregate serve fraction were already
  correct.
- INS-2-001 cont.: `config.max_tube_arcs` no longer consumed by
  `_train_band_charts`. Removed the dead `max_tube_arcs=20` assignment AND
  its banner print line from scripts/train_surrogate_production.py (grep
  clean, 0 residual). Added a 3-line comment on `TrainingConfig.max_tube_arcs`
  (surrogate_training.py L305) stating it no longer governs production tube
  training (superseded by `_tube_training_arcs`) and is retained only for
  tests that set it explicitly. Did NOT touch test_lensing_caustic_cusps.py
  slices (unrelated test-local uses) per Inspector direction.
- INS-2-002: NO ACTION NEEDED — the `_evaluate_chart` docstring
  (surrogate.py L3327-3333) was ALREADY reworded during the WP1 serve-fold
  edit; it now says "A tube chart also consumes them, to fold theta into the
  D2 fundamental domain via _fold_caustic_theta ... (the same fold applied at
  the _tube_serves gate)". No "ignored for a tube chart" text remains anywhere
  (grep empty). Finding was based on a pre-WP1 snapshot. All 4 files
  py_compile OK.

## 2026-08-14 (INS-1-001 — census cusp-window fold consistency)

- surrogate_census.py `classify_fallthrough`: the cusp-window probe
  `_tube_serves(relaxed, gamma, log_w_min, log_w_max, eta, theta,
  image_count)` was RAW-theta (unfolded) while its SIBLING exterior-polar
  refusal-ball probe already threaded y1_eig/y2_eig. Fix: pass
  `y1_eig, y2_eig` (positional, matching `_tube_serves` nan-default
  signature added by WP1). Now the cusp-window fall-through category is
  D2-equivariant — a mirror-image (negative-eigenframe) draw classifies
  identically to its first-quadrant counterpart; no 4x-deflated per-category
  gap counts under unfolded census sampling. Serving (served flag +
  chart_index + aggregate fraction) was ALREADY fold-invariant, so this is a
  DIAGNOSTIC-ACCURACY fix only, never over-certification.
- CORRECTION to my earlier "CENSUS NEEDS ZERO EDITS" WP1 claim: true for the
  SERVE path (fold-invariant), FALSE for the classify_fallthrough diagnostic
  category — a caller that HAS the eigenframe source in scope must thread it;
  the nan-default identity only covers callers WITHOUT a source.
- test_lensing_tube_d2_fold.py `_assert_route_equality`: closed the
  deliberate category exclusion at Inspector's explicit direction — added
  `categories = [r.category for r in records]` + a single-set equality pin,
  replaced the stale "category NOT asserted" comment. Coder editing a test is
  permitted here per inspector_knowledge precedent (PRE-EXISTING test pinning
  already-landed physics, Inspector-directed). Both files py_compile OK.

## 2026-08-14 (WP1 — D2 tube serve-fold + astroid fundamental-arc training)

- surrogate.py serve fold: new private `_fold_caustic_theta(theta, y1_eig,
  y2_eig)` before `_theta_into_frame` — exact D2 reflection (y1_eig<0 ->
  pi-theta; y2_eig<0 -> -theta), PARITY-AGNOSTIC (identical arithmetic
  astroid+saddle, no parity special-case). Applied at BOTH `_tube_serves`
  gate (threaded y1_eig,y2_eig params, NaN defaults) and `_evaluate_chart`
  tube branch. `serve()` public signature UNCHANGED; y1_eig/y2_eig already
  computed there via `_rotate_to_eigenframe`. eta (caustic_distance) is
  D2-invariant, passes UNFOLDED. NaN default => identity => surrogate_census
  (`classify_fallthrough` calls `_tube_serves` w/o source) inherits
  no-op; `characterize_sample` passes real y1_eig/y2_eig -> inherits real
  fold. CENSUS-NEEDS-ZERO-EDITS was WRONG for the DIAGNOSTIC path — see INS-1-001
  correction below. (Serving IS fold-invariant; the classify_fallthrough
  cusp-window CATEGORY was not.) Original (partly-wrong) claim: zero census
  edits, single-source the convention — single-source
  the convention, no second fold in the tree.
- FRAME SLIP CORRECTED (deviation from brief, documented in code): brief
  said train the astroid arc "bracketing pi/2" with "cusps on the DIAGONALS
  {pi/4,3pi/4,...}". That is the SOURCE-PLANE frame. In THIS code's caustic
  gauge angle the astroid cusps (caustic-speed minima from `_find_cusps`)
  sit on the AXES {0, pi/2, pi, 3pi/2} — measured deterministically via
  `detect_caustic_structure(g,1)` across gamma {0.2..0.9}; arc0=first
  quadrant [~0.14,~1.48]. pi/2 is a CUSP, not an arc interior: selecting on
  pi/2 returns ZERO arcs (a serve regression). Correct predicate brackets
  **pi/4**. `_tube_training_arcs(structure, parity)` (new helper before
  `_train_band_charts`): parity==1 -> `[arc for arc in structure.arcs if
  arc.theta_lo <= 0.25*math.pi <= arc.theta_hi]`; parity==-1 -> all arcs
  unchanged (saddle F079 closes via SAME serve fold; deltoid lobes are fold
  images handled by wedge/lobe). Deterministic ID from FoldArc fields (NOT
  a new empirical measurement) so no escalation. `_train_band_charts` uses
  `tube_arcs = _tube_training_arcs(...)` at BOTH sites (arc_r_min comp +
  enumerate loop), replacing `structure.arcs[:config.max_tube_arcs]`.
- ACCEPTANCE EVIDENCE (read off deterministic structure, no campaign):
  astroid tube charts per gamma band 4 -> 1 (arc count 4->1); engine calls
  scale linearly with tube-chart count (identical per-arc node grid,
  unchanged) => ~4x reduction. Saddle 6 -> 6 (unchanged). Verified
  `_tube_training_arcs` returns 1 for astroid g{0.2,0.5,0.9}, 6 for saddle
  g{1.1,1.5,2.0}. _EXPECTED_ARCS / detect_caustic_structure UNTOUCHED
  (topology guard still detects 4/6). Fold identity on fundamental domain
  (y1_eig>=0,y2_eig>=0) => byte-identical to unfolded incumbent there.
  py_compile OK both files; D2 fold arithmetic unit-checked incl NaN->id.

## 2026-08-14 (INS-1-001 — trained-w-band refusal on both Born rungs)

- born_residual_chart.py `covers`: extended signature to
  `covers(self, gamma, rho, w=None)`. Backward-compatible — all existing
  callers pass 2 args (verified grep: 19 test call sites, all 2-arg). When
  w given & non-empty, additionally requires
  log(w).min()>=log_w_grid[0] AND log(w).max()<=log_w_grid[-1] (in-box
  short-circuits first, so bad gamma/rho refuses before the w-check). w=None
  or empty -> box-only (original contract). Chose covers()-extension over the
  finding's inline math.exp comparison so the single-source band bound lives
  on the chart, not re-typed at two call sites.
- likelihood.py TWO guards, both on the SERVE path, both
  `if not born_chart.covers(lens['gamma'], rho, <band>): return None`:
  (1) `_born_residual_analytic` L2355 — guards `chart_w` (the band-split
  sub-band) AFTER `chart_w = dense_w[below_mask]`, BEFORE the partition_ns/
  carrier/evaluate. (2) buried `_surrogate_coefficients` Born rung L1955 —
  guards the FULL `dense_w` (no band-split there), placed AFTER the
  interior-handoff `return None`, BEFORE the duck-typed adapter.
- CRITICAL: did NOT add w to the buried rung's EARLY gate
  (`if rho <= 1.0 or not born_chart.covers(lens['gamma'], rho):`) — that gate
  routes a covers-fail into the 4-image interior handoff; adding w there would
  misroute an out-of-w-band exterior draw into the interior branch. w-band
  guard belongs only on the Born serve path after the handoff is declined.
- FP boundary note: exp∘log round-trip of an exact grid endpoint lands a hair
  BELOW the bound -> conservative REFUSE (falls to exact engine), never a
  wrong serve. Over-refusal costs coverage only; acceptable + matches the
  finding's strict </> intent. Did NOT switch evaluate() to fill_value=np.nan
  (finding marked optional; the covers() band guard is the authoritative fix
  and no test relies on evaluate extrapolation — grep of test_lensing_born*
  shows all evaluate() calls use in-band dense_w).
- VERIFIED: ast.parse both files OK; BornResidualChart.load() OK (trained band
  [5,60]); covers replayed — interior [10,50]/[6,59]->True, escapes
  low[4,50]/high[10,61]->False, bad gamma+in-band->False, None/empty->True;
  likelihood import OK. UNVERIFIED: live serve fall-through on an out-of-band
  astroid draw (needs full EventData/WaveformGenerator fixtures — reasoned
  from the guard placement, not run per write/verify split).

## 2026-08-14 (WP-E — thread born_residual_chart through marginalized engine)

- marginalized_likelihood.py: added `born_residual_chart=_AUTO_BORN_CHART`
  kwarg to LensedMarginalizedExtrinsicLikelihood.__init__ (imported the
  sentinel from cogwheel.lensing.likelihood alongside the existing
  _DEFAULT_* / _LENS_PARAMS imports). Stored VERBATIM before super().__init__
  (`self.born_residual_chart = born_residual_chart`) exactly like
  amplification_surrogate — the value may BE the sentinel; NOT resolved here.
  _set_summary forwards `born_residual_chart=self.born_residual_chart` into
  the inner LensedRelativeBinningLikelihood(...) call, so the ENGINE performs
  the single WP-C auto-load (refuse-to-None). Forwarding the sentinel (not
  resolving at the marginalized level) keeps load single-sourced in the
  engine + avoids a double load — the task's explicit directive.
- get_init_dict: added a born_residual_chart branch AFTER the existing
  amplification_surrogate branch. Since self.born_residual_chart may be the
  sentinel, it keys on the RESOLVED chart via
  `self._engine.born_residual_chart` (None-guarded on self._engine): None ->
  pop key (byte-identical round-trip, reconstruction re-defaults to
  auto-load); fitted chart -> raise NotImplementedError. Mirrors
  likelihood.py get_init_dict semantics (a default-constructed auto-loaded
  instance therefore RAISES on JSON serialization, same as the engine).
- amplification_surrogate threading UNCHANGED (byte-identical). SMOKE:
  ast.parse OK; module imports; sig default IS _AUTO_BORN_CHART (forwarded,
  not pre-resolved). Full-object construction not run (heavy EventData/
  WaveformGenerator/par_dic_0 fixtures) — forwarding is trivial dispatch
  into WP-C-verified engine load, reasoned not measured. UNVERIFIED: live
  marginalized serve via the Born path (needs a fitted chart + full engine).

## 2026-08-14 (WP-C — auto-attach born_residual_chart at construction)

- likelihood.py LensedRelativeBinningLikelihood.__init__: born_residual_chart
  kwarg default None -> module sentinel `_AUTO_BORN_CHART = object()` (defined
  after __all__ with a comment: explicit None stays distinguishable from
  "arg omitted", which is why a plain None default can't serve). Store logic:
  sentinel -> `BornResidualChart.load()` inside try/except (OSError, ValueError,
  KeyError); on failure warnings.warn(RuntimeWarning naming
  scripts/train_born_residual.py) + self.born_residual_chart=None (mirrors
  use_certified_ppgo_map refuse-to-None, ppgo_map.py L607-623). Explicit None ->
  stored None (pure-engine, byte-identical). Explicit instance -> stored as-is.
- Added `import warnings` (stdlib block) and
  `from cogwheel.lensing.born_residual_chart import BornResidualChart` (after
  ppgo_map import). NO circular import: born_residual_chart imports only stdlib+
  numpy/scipy.
- get_init_dict UNCHANGED (out of scope): a default-constructed (auto-loaded)
  instance now carries a fitted chart, so its JSON serialization hits the
  pre-existing NotImplementedError "fitted chart JSON deferred" branch — same as
  an explicitly-passed chart already did. Explicit-None JSON round-trip still
  pops the key (byte-identical). Not a regression, consistent with existing
  contract.
- ORACLE SEAM (for the domain test WP): accuracy oracles must construct with
  explicit `born_residual_chart=None` to stay engine-pure; omitting the arg now
  attaches the shipped chart.
- SMOKE: module imports; sig default IS the sentinel; BornResidualChart.load()
  returns a chart from shipped data/; load('/nonexistent') raises
  FileNotFoundError (OSError subclass, caught by the except). Full-object
  construction not run (heavy fixtures) — store branch is trivial dispatch,
  reasoned not measured.

## 2026-08-14 (WP2 — retire cusp-arm coverage constants in surrogate.py + census note)

- surrogate.py: DELETED `_SADDLE_CUSP_ARM_COVERAGE = 0.0` /
  `_CUSP_ARM_COVERAGE = 0.07` (~L295-313) and both preceding comment
  blocks; kept `_MACRO_SADDLE_EXTERIOR_IMAGE_COUNT = 2` and the
  `_DEFAULT_ARTIFACT_NAME` block. In `_tube_serves` (~L2886) dropped the
  `coverage = (_SADDLE... if parity==-1 else _CUSP...)` +
  `residual = max(0, delta_theta - coverage)` -> `residual = delta_theta`
  (full-window exclusion); rewrote the comment shrink-free (no
  `_CUSP_ARM_COVERAGE` token, notes post-F074 no angular serve boundary).
- surrogate_census.py `classify_fallthrough`: KEPT the `cusp-window`
  category (detection = relax cusp_windows to empty + re-call
  `_tube_serves`, untouched, still valid); corrected item-4 note to state
  WHY kept (tube cusp-window exclusion real+unchanged over full window)
  and per F074/F079 cusp losses now surface as eta-floor/w-cap, no angular
  arm boundary. No `_CUSP_ARM_COVERAGE` literal.
- VERIFY: grep clean (0 tokens) in both files; py_compile OK on both.
  Scope was surrogate.py + surrogate_census.py ONLY — the WP1
  surrogate_training.py wrap fix, the test-suite retirements, and the
  scripts/ deletions (census_dry_run.py, calibrate_ppgo_rung.py,
  measure_*_cusp_arm_*.py) are OTHER WPs in this build, not touched here.

## 2026-08-14 (WP-B — first-class Born intercept in _amplification_coefficients)

- likelihood.py: new private `_born_residual_analytic(self, lens, dense_w)`
  inserted AFTER `_saddle_farfield_analytic` (shaped like it). GATE (flat
  guards, no shared helper): born_residual_chart is not None; kappa==0 AND
  beta==0 (mirrors buried-path KappaBetaGuardPrecedence — chart axes are the
  kappa=0/beta=0 (gamma,rho,log_w) surface, can't represent nonzero ->
  fall through); rho=caustic_rho(gamma,|y|,kappa) > 2.0 (WP-specified, NOT
  the buried path's >1.0); born_chart.covers(gamma,rho). Cheap gates BEFORE
  geom solve; geom LensDomainError propagates unswallowed (mirrors
  _saddle_farfield_analytic).
- SERVE: geom via ChangRefsdalChannels(dense_w).geometry_partition(...)
  (copied verbatim from _saddle_farfield_analytic). partition_ns
  (SimpleNamespace) built over chart_w exactly as the buried rung.
  carrier=born_carrier_from_partition (deferred import), residual=
  born_chart.evaluate(chart_w,gamma,rho), f_total=carrier+residual, ppgo=
  sum saddle_kernels[:,real]*exp(1j*chart_w*delays), envelope=zeros(dense_w),
  envelope[below_mask]=(f_total-ppgo)*exp(1j*chart_w*t_min), reconstruct_
  farfield(FARFIELD_KERNEL_SUM). Returns (delays,k0,k1,geom) or None.
- MAP CONSULT: w_trust=_ppgo_band_split(lens); eff_ceiling=min(parity_wall,
  _ppgo_cell_ceiling); w_hi>eff_ceiling -> w_trust=None; band_split=w_trust
  not None and w_lo<w_trust<w_hi. below_mask=(dense_w<=w_trust) or all-True.
  BYTE-IDENTITY (test battery pins): no-split -> below_mask all-True ->
  chart_w=dense_w[all-True] carries IDENTICAL float values; carrier/residual/
  born_carrier are ELEMENTWISE over w (evaluate uses RegularGridInterpolator
  on log_w; born_carrier splits internally per-element on w_i*delta_tau), so
  chart_w subset == full-band restricted to below_mask. Mirrors surrogate-
  path band-split arithmetic exactly.
- WIRED: call inserted in _amplification_coefficients AFTER `if lens['gamma']
  > 1.0:` saddle block, BEFORE seed/fiducial/ratio (`w_max=...`). lens &
  dense_w already in scope there.
- NOT TOUCHED: _surrogate_coefficients buried Born rung (left flat/intact,
  no shared helper extraction); _ppgo_cell_coords saddle rho<1 refusal.
- VERIFIED: ast.parse OK; module imports (h5py warning pre-existing);
  method present, sig (self, lens, dense_w). git diff = my 2 hunks + the
  pre-existing uncommitted WP-C hunks (warnings/BornResidualChart import +
  __init__ auto-attach) which I did NOT author.

## 2026-08-14 (WP-F — census band-split mirror vs WP-B first-class Born: NO-OP + doc)

- VERDICT: FUNCTIONAL NO-OP. surrogate_census.py band-split mirror
  (~L413-444) already inlines _ppgo_cell_coords (parity + caustic_rho
  kappa=0 + saddle rho<1 skip) + _ppgo_band_split (w_trust) +
  _ppgo_cell_ceiling (w_ceiling; eff_ceiling=min(wall,ceiling)) EXACTLY.
  Verified logically equivalent to production: census "w_hi<=eff_ceiling
  AND w_lo<w_trust<w_hi" == production "NOT(w_hi>eff_ceiling) AND
  w_lo<w_trust<w_hi". WP-B's _born_residual_analytic REUSES those same 3
  likelihood methods verbatim, so the ONE census mirror already reflects
  the lifted Born band-split — no divergent arithmetic exists. Constants
  (ASTROID_WALL/SADDLE_WALL/UNKNOWN/CERTIFICATION_BAR/caustic_rho/
  _PPGO_INTERIOR_SAFETY) all bound from ppgo_map/likelihood (imports
  L59-63), not re-typed.
- WHY census can't mirror the Born SERVE: census carries NO
  born_residual_chart (grep confirms 0 refs; characterize_sample takes only
  surrogate+engine_factory) => born_chart.covers(gamma,rho) unevaluable.
  The blocker is the ABSENT chart, NOT surrogate presence — identical for
  the buried surrogate-path rung (_surrogate_coefficients, rho>1) AND WP-B's
  first-class intercept (rho>2). So WP-B's reachability lift adds NO served
  path the census can model. 'born' correctly stays a FALL-THROUGH bucket
  (rho>1, superset of both rungs' served domains) = conservative
  over-attribution to the exact engine. NOTE threshold skew: census 'born'
  rho>1 matches the BURIED rung, not WP-B's rho>2 — a chartless census
  cannot split rho>2-covered from the rest, another reason it can't model
  the serve.
- ACTION: added 2 WHY comments only (no logic): (1) band-split mirror site
  — notes it inlines the 3 methods + WP-B reuses them verbatim; (2)
  classify_fallthrough 'born' site — documents the deliberate fall-through
  (no chart => covers() unevaluable => unchanged by the lift). Preempts a
  spurious Inspector "mirror skew" finding. py_compile OK.

## 2026-08-14 (WP-G — DATA_CONTRACTS born_residual_chart false-attach fix)

- .claude/spec/DATA_CONTRACTS.yaml born_residual_chart entry: replaced the
  FALSE claim "the fact-4 slot in likelihood._surrogate_coefficients attaches
  it at construction time" + "When None (default) ... fall through" with the
  now-true story: auto-loaded via BornResidualChart.load(), attached at
  LensedRelativeBinningLikelihood construction (`_AUTO_BORN_CHART` sentinel
  default, refuse-to-None on load anomaly; opt out via explicit
  born_residual_chart=None); LensedMarginalizedExtrinsicLikelihood's internal
  engine inherits the same auto-attach default (its __init__ has NO born kwarg,
  builds engine at default). Consult site = first-class intercept
  _born_residual_analytic in _amplification_coefficients (likelihood.py L2199/
  L2469), gate kappa==0 & beta==0 & caustic-frame rho>2 (both parities) &
  born_chart.covers, band-split vs certified ppGO map. Added
  _born_residual_analytic to the consumers list (kept _surrogate_coefficients —
  buried Born rung still consults it per WP-B). Verified: yaml.safe_load OK,
  false phrase gone, load()/_born_residual_analytic present. DOC-ONLY, no code.

## 2026-08-14 (WP3 — delete dead cusp-arm measurement scripts + census re-express)

- `git rm` scripts/measure_cusp_arm_reach.py, measure_cusp_arm_actual_boundary.py,
  measure_saddle_cusp_arm_coverage.py, calibrate_ppgo_rung.py. Confirmed no
  production/test import references them (only docs: FINDINGS/COMPLETED/TODO/
  todo.d/changelog.d + one provenance comment — all Librarian/Inspector scope).
  measure_cusp_exclusion.py is a DIFFERENT script, correctly retained.
- scripts/census_dry_run.py: deleted mirrored `_CUSP_ARM_COVERAGE=0.07`;
  added `_CUSP_ARM_W_FLOOR=49.0` (no importable production constant — F074
  w-floor confirmed 49 in FINDINGS ~L4356). cusp_arm route now
  `if is_near and w >= _CUSP_ARM_W_FLOOR` (w IS in classify_draw scope) —
  angular `delta_cusp` no longer gates. Tube residual arithmetic
  (`residual = max(0, _TYPICAL_CUSP_HALF_WINDOW - _CUSP_ARM_COVERAGE)`)
  replaced by full-window exclusion `delta_cusp < _TYPICAL_CUSP_HALF_WINDOW`
  (mirrors WP2's `_tube_serves` full-window change). Banner prints w-floor
  not coverage. py_compile OK; grep clean (0 tokens) across scripts/ incl. pyc.
- FLAG -> Inspector/Librarian: cogwheel/lensing/chang_refsdal/_pearcey_cusp.py
  ~L447 has a live provenance comment "Measured: scripts/calibrate_ppgo_rung.py
  sweep..." pointing at a now-deleted script (documents _W_PPGO_FLOOR=8.0
  origin). Left untouched — out of WP3's census-only edit scope + historical
  provenance like a changelog. Adjudicate whether to reword.
</content>
