# Build 8c-cont — finish the surrogate-artifact build: registration + census + tests

## Mission

Complete Build 8c. WP1 (multi-chart surrogate) and WP3 (training
driver) are LANDED and driver-verified in the working tree — do NOT
redo them. This continuation delivers the two remaining work packages
of the approved 8c plan and the full test phase:

1. **Registration (was WP2)** — make the artifact a first-class data
   product: `DATA_CONTRACTS.yaml` entry `lens_amplification_surrogate`
   (producer `scripts/train_lens_surrogate.py`, consumers = the lensed
   likelihoods via `LensAmplificationSurrogate.load`), a
   `data_registry.yaml` path entry, and LOADERS enrollment in
   `scripts/regenerate_consumer_graph.py`. `pipeline_graph.py list`
   and the consumer-graph regeneration must see the artifact. No
   contracts_changelog/changelog fragments (post-gate doc-sync).
2. **Census tool (was WP4)** — `cogwheel/lensing/surrogate_census.py`
   (importable core) + `scripts/census_lens_surrogate.py` (thin CLI):
   served fraction with fall-through breakdown (cusp-window /
   gamma-guard / dropped-sliver / out-of-box / refusal-ball), per-chart
   held-out envelope eps, and gamma/image-count/eta-partitioned lnL
   tiers vs the exact engine (crown <= 0.01 nats; strong-shear/saddle
   <= 0.1 = a factor >= 10 below RB_ATOL = 1.5 per F016; rescued at
   RB tolerance; NEVER partitioned by gauge theta, F017). JSON report
   incl. measured artifact size. Fixture-scale runs only.
3. **Test phase** — the ten Domain Test Descriptions of the approved
   8c plan (in `/tmp/build8c_approval/plan.json`, still on disk;
   verbatim binding). New tests in `cogwheel/tests/` beside the suite;
   8a's `test_lensing_surrogate.py` must keep passing (WP1 preserved
   backward compat; tests scraping single-box internals may re-target
   `charts[0]` — flagged in coder memory).

## Landed facts (driver-verified — trust these, do not re-derive)

- WP1: `surrogate.py` rewritten multi-chart (frozen `TubeChart` /
  `FarFieldChart`, guard-stack `select_chart` keyed ONLY on certified
  `caustic_distance` + `image_count`, theta solely for cusp windows —
  F017); single-npz save/load with JSON provenance scalar; package-data
  default (`cogwheel/data/`, importlib.resources) + explicit-path
  override; 8a single-box npz loads as one-chart special case.
  `channels.py`: single additive `ChangRefsdalGeometryPartition.caustic_theta`
  field (sole constructor `geometry_partition`). `likelihood.py`:
  intercept serves via
  `surrogate.serve(eta=..., theta=..., image_count=...)`; kappa != 0
  falls through (INS-8a-001); default `amplification_surrogate=None`
  byte-identical.
- WP3: `surrogate_training.py` + `scripts/train_lens_surrogate.py`.
  Prior box read from prior classes; cusp detection via caustic-speed
  minima with topology cross-check (4 astroid / 6 deltoid, loud
  `CausticTopologyError` on mismatch); parity-agnostic arc/tube
  builders; per-chart-file resumability; JSON training report with
  measured artifact size. DRIVER ADDITIONS (post coder death, tested):
  `band_caustic_structure` (arc bounds valid across a gamma band:
  edge+center detection, intersected theta bounds, merged conservative
  cusp windows, max reach) and `stable_gamma_bands` (adaptive
  bisection into topology-stable sub-bands; metamorphosis slivers
  narrower than `TrainingConfig.min_gamma_band` are DROPPED
  refusal-conservatively and reported as `dropped_gamma_slivers`).
  Charts are built per stable sub-band, tagged
  `chart_{parity}_b{j}_{tube|farfield}_{idx}`. Rationale: the deltoid
  fold-arc partition changes at discrete gammas (measured: 6 arcs at
  gamma = 1.205 vs 4 at 1.305 within the smoke band) and the saddle
  wedge `|sin 2 theta| <= 1/gamma` narrows with gamma, so
  single-anchor arc bounds crash `critical_point` at upper-band
  gammas (the WP3 coder's death site).
- Smoke training run (driver, post-fixes): both parities end to end,
  1 stable sub-band each, ZERO dropped slivers; 4 charts, 76,666-byte
  artifact, 118 s. Held-out eps: astroid tube 0.434 / far-field
  1.6e-3; saddle tube 0.195 / far-field 7.6e-2. The tube eps values
  are SMOKE-GRID COARSENESS (4 nodes/axis over ~1.3 rad arcs), NOT
  machinery error: at an exact training node the tube spline
  reproduces the engine to 2.4e-16 (constructed coords) / 2.8e-8
  (via the query-time projection round trip). Your accuracy-bar tests
  must size their own fixtures (denser theta/u grids over a narrower
  band) rather than inherit the smoke defaults.
- THETA WRAP (driver fix, surrogate.py — know this when writing
  selection tests): `nearest_caustic_point` reports theta in
  [0, 2*pi); wedge-frame charts can span negative theta. Queries are
  unwrapped into the chart frame by `_theta_into_frame` for the range
  test and the spline coordinate; cusp windows use CIRCULAR distance.
  Near-cusp sources can legitimately project onto a NEIGHBORING arc
  (different theta*, smaller eta*, possibly different image count) —
  those fall through refusal-conservatively BY DESIGN; do not "fix"
  that behavior, test it.

## Out of scope — hard fences

- NO changes to `surrogate.py` / `surrogate_training.py` /
  `channels.py` / `likelihood.py` / engine modules beyond what a RED
  test legitimately forces; any such fix must preserve the landed
  design (band splitting, guard stack, provenance schema) and be
  called out in the change report.
- NO full-box training or full-box census (deferred post-8e per owner
  re-sequencing). NO enable-by-default flip. NO sampling/PP.
- The census draws prior samples in SAMPLED coordinates from the
  prior classes; no importance weights.

## Acceptance (two-tier)

1. In-build (FAST): registration visible to `pipeline_graph.py list` +
   consumer-graph regeneration; census core verified on the small
   fixture artifact vs hand-computed values; the ten plan test
   descriptions implemented and green, including the two design
   falsifiables (tube beats equal-budget raw chart by >= 3x at eps_95
   through the near-caustic band; fold-approach ray flat vs raw
   ~ -1/2 slope) and the F010 mutation tests (mutated chart bound
   flips a serve/fall-through decision to RED); the full
   `test_lensing_surrogate.py` suite green.
2. POST-BUILD (driver): smoke-scale training + census reports
   reviewed; full-suite parallel gate; commit.
