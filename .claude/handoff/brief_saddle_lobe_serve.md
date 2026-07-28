# Build brief — saddle lobe-frame serve wiring

## Mission

Make the macro-saddle (`gamma > 1`) per-lobe interior charts SERVABLE. The
admission and tiling geometry already exist and are certified (S2-2, frozen
WP7); what is missing is the coordinate plumbing that lets a lobe-local chart
be placed at its true physical source location at serve time. Today those
tiles are built, counted, and then thrown away with `served=False`.

Success is a saddle interior that answers queries from a chart instead of
falling through to the ladder — which is a direct contribution to the
zero-quadrature coverage goal, since the saddle interior is currently a
structural gap, not a numerical one.

## Measured facts (inline; do not re-derive)

- `surrogate_training._SaddleLobeAdmission` (~L1960) is COMPLETE and frozen:
  per-lobe winding-number interior test over every band gamma, `eta_max`
  tube-shell nearest-distance exclusion, and an inter-lobe corridor exclusion
  (`|p - centroid| + corridor_half <= |p - other_centroid|`) that guarantees no
  admitted tile straddles the lobe-equidistance line.
- `_lobe_interior_tiles` (~L2165) returns, per admitted tile,
  `((rho_lobe_center, theta_local_center), (half_rho, half_theta), i, j)`.
- Lobe-local radial coordinate:
  `rho_lobe = |y - centroid| / r_deltoid(theta_local)`, so `rho_lobe = 1`
  tracks the deltoid boundary in EVERY direction. `r_deltoid` is a periodic
  linear interpolation over `(boundary_theta, boundary_r)` from
  `_directional_lobe_boundary`. A scalar reach is NOT usable here: it
  overshoots the near-cusp directions of an elongated (sheared) lobe and
  leaves its interior untileable. `reach` on the dataclass is reporting-only.
- `_SADDLE_LOBE_CENTERS = (0.0, math.pi)` — lens-plane angular centres of the
  two lobes at `beta = 0`. Each lobe's SOURCE-plane centroid is estimated as
  the mean of its midpoint-gamma caustic points.
- THE BLOCKER, verbatim from `_train_band_charts` (~L3522): "These lobe-local
  tiles are RECORDED but NOT packed into `admitted`: the far-field serve
  mapping (`surrogate._from_caustic_fixed`) is strictly origin-centred and
  carries no lobe-centroid offset, so a lobe-local chart cannot yet be served
  at its true physical location." The report sets `interior_report['served']
  = False` with a `serve_note` (~L3634).
- `surrogate._from_caustic_fixed(gamma, rho, theta_c)` and its inverse
  `_to_caustic_fixed` are the origin-centred maps the serve path uses.
- `surrogate._FARFIELD_AXIS_SCHEMA = 'caustic_radial_offset_rho_theta_framewinv'`;
  `_validate_farfield_axis_schema` HARD-REFUSES an absent or unknown tag at
  load, so a new axis convention needs a new tag and old artifacts must refuse
  rather than reconstruct a finite-but-wrong `F`.
- Interior tiles store the `tau_c`-demodulated `INTERIOR_SACR_C` label, NOT
  the far-field kernel-sum label. Interior charts are guarded by
  `_assert_carrier_continuity` (basin-flip / medial-ridge), which measures a
  SOURCE-POSITION jump against the caustic reach.
- F022 (2026-07-28): the EXTERIOR far-field guard was re-pointed to measure a
  normalized re/im increment rather than `arg`. Do NOT harmonize the interior
  twin onto that metric — it is a different observable on a different label
  and is not implicated. Read FINDINGS F022 before touching either guard.

## In scope

- A lobe-aware coordinate map: `(lobe frame, rho_lobe, theta_local) ->`
  physical eigenframe source, and its inverse for serve-time lookup.
- Persisting the lobe frame (centroid + directional boundary) on the chart so
  it survives save/load and reconstructs exactly.
- A new axis-schema tag for lobe-local charts, with old-tag refusal preserved.
- Packing admitted lobe tiles into the served set; `interior_report['served']`
  becomes True for the saddle, and the `serve_note` is removed or restated.
- Serve-time lobe dispatch: for a `gamma > 1` query, decide lobe membership by
  the SAME admission predicate used at training (winding + corridor), and
  REFUSE — by name, to the ladder — in the inter-lobe corridor.

## Out of scope

- Any change to `_SaddleLobeAdmission`'s predicates or to the tiling geometry.
  It is frozen and certified; if it looks wrong, report it, do not edit it.
- The positive-parity (`gamma < 1`) origin-centred path. It must stay
  BYTE-IDENTICAL; prove it, do not assume it.
- Both carrier-continuity guards (interior and exterior).
- The Born rung (`_born.py`) — dormant, and the next build.
- Any coverage census or fraction measurement. That is the last step of the
  programme, deliberately, and measuring before the science is in hand is a
  standing prohibition here.

## Acceptance (build-level)

1. Round trip: for admitted lobe-interior sources, physical source ->
   `(lobe, rho_lobe, theta_local)` -> physical source agrees to <= 1e-12.
   Include at least one near-cusp direction, where a scalar reach would fail.
2. A source inside the inter-lobe corridor REFUSES by a named refusal and is
   never served from either lobe. A source inside one lobe is never served
   from the other.
3. A saddle-band training run packs lobe tiles and reports
   `interior_report['served'] is True`, with per-lobe admitted counts > 0 on a
   band where the current code already reports non-zero admitted tiles.
4. Served values at lobe-interior points match a FRESH-ENGINE oracle within
   the existing held-out eps bar for interior charts. Node-exactness at chart
   nodes where the existing interior tests assert it.
5. Positive-parity serve path proven byte-identical (same inputs -> same
   bits), not merely "tests still pass".
6. Old-tag artifacts still hard-refuse at load; the new tag round-trips.
7. Full fast suite green, driver-verified post-build.

## Constraints

- Branch `claude-dev` only. Never commit on main/master.
- Slow tests NEVER run in-build. `COGWHEEL_BRUTE_ACCURACY` /
  `COGWHEEL_STRICT_TIMING` / `COGWHEEL_TRAIN_TIER` stay unset in agent envs;
  the driver runs the tiered and slow tiers post-build.
- In-build tests must be FAST: small/synthetic bands, few-eval or analytic
  oracles. A test that needs a real multi-minute training run belongs in the
  train tier, gated, for the driver.
- Units and conventions per AGENTS.md; numba-compatible hot paths.
- Verify existing tests for backward compatibility BY READING, including
  skipped/gated ones (test_dev step 7, inspector check 5b). The pre-commit
  drift hook blocks on gated tests referencing changed APIs — if a gated test
  is fine, say why; do not silently bypass.
- If a coordinate convention has to change shape (not just value), say so
  loudly in the report: three interface migrations have already rotted
  fixtures in this area, and a silent shape change is how that happened.
