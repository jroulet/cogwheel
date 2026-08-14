# Professor short-term (Leg A/B/C de-scope consult, 2026-08-14)

Consult: pin final scope for a 3-leg admission plan (transverse cone + connecting
region) after the §0 finding that the certified map keys on scalar rho and hard-refuses
saddle rho<1.

## Verified code facts (read directly, not assumed)

1. **`nearest_caustic_point` is amortized once per band-serve, NOT per w-sample.**
   `ChangRefsdalChannels.evaluate` (channels.py L1932-2036) takes `self._w` (the WHOLE
   band array) and a single `(2,)` source; it calls `nearest_caustic_point` exactly
   ONCE at L1977 and returns a `ChangRefsdalPartition` over the entire band. Same
   pattern in `geometry_partition` (L2095) and `_exact_total` (L694). So the 256-node
   seed scan + Newton polish + scipy fallback runs once per parameter point, then the
   full w-grid likelihood is evaluated from that one partition. The Simplifier's "hot,
   per-serve in likelihood eval" framing conflates per-parameter-point with per-w. It
   is per-parameter-point = per-band = amortized. Ruling: KEEP eta (the true Euclidean
   `distance`); the wedge pre-filter is NOT an acceptable substitute where the true
   directional distance is the admission criterion, and the cost concern is moot.

2. **The wedge half-width `|sin 2(theta-beta)| <= (1-kappa)/|gamma|` is the CRITICAL-
   CURVE support wedge (macro saddle), NOT a caustic-proximity predicate.** From
   `nearest_caustic_point` docstring + body: `theta_max = 0.5*arcsin(lam/|gamma|)`,
   lam=1-kappa. It bounds where the deltoid LOBES live in angle; it says nothing about
   Euclidean distance to the caustic. Membership in the wedge is necessary-not-
   sufficient for near-caustic; a wedge predicate would false-admit sources that are
   angularly inside the wedge but radially far, and cannot rank two lobes/branches.

3. **`caustic_rho` (ppgo_map.py L799) is scalar-reach: rho=|y|/caustic_reach(gamma),
   caustic_reach = MAX radius over angle.** Documented (F073): rho<=1 does NOT imply
   interior (holds at 58.7% of EXTERIOR sources over gamma 0.2-0.9); only rho>1 implies
   exterior soundly. The certified map is queried in this gauge by BOTH
   `likelihood._ppgo_cell_coords` and `surrogate_training._train_band_charts`. This
   confirms the §0 finding: no certified cell exists for saddle rho<1, so the
   connecting region has no certified serve without net-new offline map training.

## Rulings issued
- Decision 1 (connecting region -> refusal guard only): AGREE. Honest current state;
  serving it needs offline saddle rho<1 map training (out of scope).
- Decision 2: eta STAYS; wedge predicate REJECTED as a correctness substitute; cost
  justified as amortized once-per-band (not per-w).
- Decision 3: Leg C redundancy question — see plan file. Leg C lights only rho>=1
  saddle cells; needs a distinct-population check vs Leg A^B before trim/keep.
