---
section: Backlog
---

- **THE SURROGATE SERVE PATH IS 90% A CAUSTIC SCAN — derive the reach, and
  cache it** `[→ spec]` — measured 2026-07-30 (F054). A surrogate-served
  lensed `lnlike` costs 31.25 ms, of which `_surrogate_coefficients` is
  27.88 ms (89%), of which `ppgo_map.caustic_geometry` is ~27.5 ms (90% of the
  serve). The tensor-spline contraction the surrogate exists to perform —
  `_contract_tensor_spline` — is **1.7%**.

  `caustic_geometry(gamma, kappa, n_theta=720)` is a Python double loop over
  2 branches x 720 polar angles calling `geometry.critical_point` on each:
  **1440 calls per likelihood evaluation**, to find a maximum radius by
  scanning.

  ## Two independent fixes; either recovers most of the cost

  1. **DERIVE, do not scan.** The maximum of `|y(theta)|` is an extremum of a
     closed-form curve — a root of `d|y|^2/dtheta = 0` — and `y'` is exactly
     what build 1a shipped as `geometry.caustic_derivatives`. The idiom to
     copy already exists in this package and is named in
     [[lensing_analytic_derivatives]] as the model: `geometry.r_caustic`
     "samples only to BRACKET and refines every root with brentq to 4*eps".
     Same disease as F039 (`_PROBE_ETA`) and F041 (`|dot|`): a sampled
     estimator standing in for algebra nobody did.
  2. **CACHE what does not vary.** `reach` depends on `(gamma, kappa)` ONLY —
     not on the source position — yet is recomputed every evaluation.
     `_schwinger`, `_pearcey_cusp` and `prior` all use `lru_cache`; this does
     not. Cf. the standing lesson that values derived from `(source, matrix)`
     belong ON the partition rather than re-derived in a hot path.

  Do BOTH: the closed form makes each call cheap, the cache makes most calls
  disappear. Order does not matter; measure after each.

  ## Why it hid

  The surrogate's timing test asserts a SPEEDUP against the exact engine
  (9.6-20.4x measured, passing comfortably) and the exact engine is
  ~300-630 ms. A serve two orders of magnitude slower than it should be still
  reads as a triumph beside that. Nothing compared the serve against what a
  4-D spline contraction OUGHT to cost. That comparison is the acceptance
  below.

  ## Acceptance

  - `_contract_tensor_spline` becomes a MAJORITY of `_surrogate_coefficients`,
    not 1.7% of it. State the measured split before and after.
  - `critical_point` calls per served `lnlike` drop from 1440 to O(10).
  - Served values are unchanged to the F016 envelope bar — this is a cost
    change, not a physics change. A reach computed by root-finding must agree
    with the 720-point scan's answer to a stated tolerance on a gamma sweep,
    BOTH parities (the saddle's two off-origin deltoid lobes are the case a
    naive extremum search gets wrong).
  - The cache is keyed correctly: a `(gamma, kappa)` change must produce a
    different reach. Add the falsification, or the cache silently serves one
    gamma's geometry for all of them.
  - Report the served `lnlike` cost against the ~3 ms the fast path already
    targets, and the implied core-hours for a 5M-evaluation run (43 core-hours
    at today's 31 ms; ~4 at 3 ms).

  ## Scope note

  This is the SERVE path, so it needs the F016 bar and a served-value gate,
  like build 1c did — not merely a geometry tolerance. `annulus_rho` and the
  ppGO map read the same `caustic_geometry`, so check whether they inherit the
  win or need their own call sites updated.
