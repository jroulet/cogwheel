# Build 8c — The global surrogate artifact: caustic-adapted charts, contract, census

## Mission

Turn the Build-8a surrogate machinery into a GLOBAL artifact covering
the full sampled prior box, both parities, and make it a first-class
data product. Four concerns:

1. **Caustic-adapted charts (the owner's design, binding).** The
   caustic locus sweeps through raw (gamma, y1, y2) space, so
   axis-aligned tiling fights a curved surface. Instead: a NEAR-CAUSTIC
   TUBE CHART in adapted coordinates (gamma, eta, theta, log w) where
   (eta, theta) come from `nearest_caustic_point` (distance and arc
   position) — the caustic is the FIXED plane eta = 0 for every gamma —
   fitting in u = sqrt(eta) so the known fold exponent makes the
   interpolant smooth through the transition; plus plain raw-coordinate
   FAR-FIELD charts away from the tube. Cusp neighborhoods (fixed theta
   locations on the eta = 0 plane; 2/3-power scaling) are EXCLUDED from
   the tube chart in this build (served by the exact engine FOR NOW;
   the scheduled cusp fast-serving build — after homogenization —
   closes them to ms scale) — record the exclusion radii in the
   artifact. Chart selection at query time must be deterministic, cheap
   (the serving path already computes caustic_distance), and
   overlap-free or with a deterministic priority rule.
2. **The artifact as a data product.** A shipped multi-chart npz (or
   npz-set with a manifest): chart definitions, coefficient tensors,
   refused/excluded regions, provenance (training grid, engine
   version/commit, training hash). DESTINATION (owner preference,
   2026-07-20): DEFAULT to package data under `cogwheel/data/`
   (versioned with the code, no cluster-path dependence — simplicity)
   IF the trained artifact is git-comfortable in size (~tens of MB or
   less); otherwise fall back to a `data_registry.yaml` cluster path
   with the training script as the reproducible generator. The
   in-build machinery must support BOTH load paths (a package-data
   default with a path override); the census reports the measured
   artifact size so the destination call is made on numbers. Either
   way, REGISTER IT: an entry in
   `DATA_CONTRACTS.yaml` (producer: the offline training script;
   consumer: the lensed likelihoods via `LensAmplificationSurrogate`)
   + `data_registry.yaml` path + a `contracts_changelog.d/` fragment
   + ENROLL `LensAmplificationSurrogate.load` in
   `scripts/regenerate_consumer_graph.py`'s LOADERS dict (verified:
   the graph only tracks enrolled loaders — this is the mandatory
   manual step; the Librarian triage row backstops it).
3. **Training driver.** An offline training SCRIPT (scripts/ or a
   module CLI) that builds the full artifact from the prior-box
   definition (read the box from the prior classes — do not hard-code
   ranges), with per-chart engine-call budgets, resumability
   (per-chart files or completed-chart manifest), and a machine-usable
   training report.
4. **Census + production accuracy tiers (enable-by-default evidence,
   NOT the enablement itself).** A census tool measuring, on the
   trained artifact over prior-box samples: served fraction (target
   >= 95% away from exclusions), held-out envelope eps per chart, and
   the lnL tiers vs the exact engine (crown-family <= 0.01 nats;
   strong-shear/saddle <= 0.1; rescued at RB tolerance — F016: gate
   surrogate error BELOW the RB-binning floor, never past it).
   Enable-by-default itself stays an OWNER decision, additionally
   gated on PP validation which is PARKED with all sampling (ruling A).

## Measured facts (pre-answered)

- 8a machinery: `LensAmplificationSurrogate` (single-box, 4-D
  (log w, gamma, y1_eig, y2_eig), beta eliminated exactly by
  eigenframe rotation, kappa = 0 surface with the likelihood-side
  kappa != 0 fall-through guard, INS-8a-001); prefiltered
  not-a-knot B-spline coefficients, envelope query 0.37 ms at 300
  points; serving path ~6 ms (geometry_partition-dominated; the
  8b-levers build in flight reduces it).
- Envelope smoothness: smooth WITHIN an image-count region; parameter
  derivatives carry sqrt-type singularities AT caustics (fold) — the
  raison d'etre of the u = sqrt(eta) tube chart. The engine's census
  guard passes degenerate (fold/cusp-merged) censuses only with a
  near-critical witness; the serving gate must keep the existing
  image-count + caustic_distance checks per chart.
- MVP gates (8a, budget-limited fixture): held-out eps 8.4e-2 (pos) /
  1.7e-2 (saddle) at 6 nodes/axis with h^1.5 convergence; production
  target eps < 1e-3 needs the denser offline grids (hours of engine
  calls — the training run is a POST-BUILD driver step; in-build tests
  use SMALL multi-chart fixtures).
- Refusal vocabulary and the never-serve-where-wrong contract are
  unchanged; the tube chart adds NO new refusals — out-of-chart =
  fall through to exact.
- Prior box: gamma in (0, 1.6) continuous both parities; y via the
  mass-conditioned (u1, u2) box; w band from the mass range (w <= 58
  by construction; Schwinger ceiling 60).

## Out of scope — hard fences

- NO engine-module changes (geometry/operator/_schwinger/_hyp1f1/
  _gauge/_dd; channels.py only if a chart-selection helper genuinely
  needs a seam — prefer the existing geometry_partition outputs).
  NOTE: Build 8b-levers is concurrently editing geometry.py and
  operator.py internals (value-preserving certified); plan against
  their PUBLIC behavior, which is unchanged.
- NO cusp-neighborhood emulation (exclusion + exact fallback only).
- NO enable-by-default flip; NO sampling/PP runs (ruling A).
- The 8a single-box API and its tests keep working (backward compat:
  a single-box artifact is a one-chart special case).

## Acceptance (two-tier)

1. In-build (FAST): a small multi-chart fixture (one tube segment
   crossing a fold + one far-field chart per parity, coarse grids)
   demonstrating: chart selection determinism + no-overlap; tube-chart
   held-out accuracy through the near-caustic band BEATING an
   equal-budget raw-coordinate chart on the same band (the design's
   falsifiable claim); sqrt(eta) fitting verified against the engine
   across eta -> 0 approach (down to the exclusion floor); serve/
   fall-through boundaries honored with the F010 mutation idiom;
   serialization round-trip incl. the manifest; contract/registry/
   LOADERS entries present and `pipeline_graph.py list` + the
   consumer-graph regeneration see the new artifact/loader.
2. POST-BUILD (driver): SMOKE-SCALE training + census-machinery
   validation only (moderate budget: prove resumability, the manifest,
   and the census report end to end on a reduced grid). The FULL-BOX
   production training run is DEFERRED (owner ruling 2026-07-20) until
   after the homogenization build AND the cusp fast-serving build, so
   the expensive run happens exactly once, on the final engine and the
   final chart set; enable-by-default evidence (full census, price
   points) follows that run.
