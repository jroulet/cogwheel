# Build brief — retire the training-path estimators (step 1b)

## Mission

Delete six numerical estimators in `cogwheel/lensing/surrogate_training.py`
and re-express each against the exact functions build 1a added to
`geometry`. Nothing here computes a derivative, a distance or a cusp
location by sampling or stepping when a closed form exists.

This is build **1b** of step 1 of
`.claude/spec/todo.d/lensing_caustic_relative_coordinates.md`; the inventory
and the implementation-vs-oracle rule are in
`todo.d/lensing_analytic_derivatives.md`. Read the master fragment's Part 0
principle and standing rules first. Build 1a is a prerequisite and has
shipped; 1c (the serving path, `_pearcey_cusp._cusp_vertex`) is a separate
build and is NOT in scope.

## What 1a gave you

In `cogwheel/lensing/chang_refsdal/geometry.py` (shipped `1a82046`):
`caustic_derivatives` (the primitive, `y'` and `y''`),
`caustic_curvature_radius`, `caustic_speed`, and `fold_opening_direction`
(unit vector toward the two-image side). Verified against a two-stage oracle
at `atol = 5e-13 + rtol = 1e-11` — `y'` worst 4.39e-13, `y''` worst 2.56e-14
over 110 configs; `|y'|` at the astroid cusp is 1.3e-16, i.e. cusps are exact
roots. Use these. Do not re-derive the cascade, and do not add a finite
difference anywhere.

Domain contract you inherit: positive parity IGNORES `branch` (mirroring
`critical_point`), the saddle wedge edge REFUSES by name rather than dividing
by a clamped-zero discriminant, and refusal is whole-call — never a
per-element `nan`.

## Scope

IN — `cogwheel/lensing/surrogate_training.py` only:

1. **`_min_curvature_radius`** — delete the inlined three-point circumradius
   AND its `area2 < 1e-30` collinearity guard (a stencil artifact with no
   analogue in the closed form). Return a minimum over exact
   `caustic_curvature_radius` values, endpoints INCLUDED. Keep the name,
   signature and call site.
2. **`_branch_speed_profile`** — delete the `np.gradient` and the rolled
   central difference; return `caustic_speed` directly.
3. **`_find_cusps` — LOCATION ONLY.** Cusp angles become roots of
   `caustic_speed == 0`, bracketed on the sampled profile and refined (the
   `r_caustic` idiom: sample to bracket, `brentq` to `4*eps`). Delete
   `_CUSP_SPEED_REL_FRAC` — a relative dip threshold has no meaning once the
   cusp is a root. See the WINDOW carve-out below; it is the one thing here
   you must NOT change.
4. **`_probe_arc_side` and `_PROBE_ETA`** — delete both. The fold's
   two-image side is `fold_opening_direction`; `inward_sign` follows from its
   sign against the arc normal. The image count on the served side is then a
   single well-conditioned `find_images` call at a sane offset, or derived
   from parity — your choice, but it must not depend on a step-size constant.
5. **`_caustic_inradius`** — `min |y(theta)|` over a discrete cloud becomes a
   bracketed-and-refined minimisation of the closed-form `|y(theta)|`. The
   `encloses_origin` winding test is unaffected; keep it.
6. **`_CLOUD_MARGIN_FRAC = 0.10`** — delete. It inflates an interior refusal
   threshold by a round 10% to cover a measured ~8% overshoot of the discrete
   200-point `_caustic_points` cloud. Replace the cloud-nearest distance in
   `_InteriorAdmission.admits` with `geometry.nearest_caustic_point`, which is
   exact to 9.3e-12 and already imported in this module. The margin then has
   nothing to correct.

OUT — do not touch:

- **The cusp WINDOW half-width `delta_theta`, `_CUSP_WIDTH_SAFETY`,
  `_CUSP_MIN_HALFWIDTH` and the `_SADDLE_*` variants. Leave the incumbent
  width rule byte-identical.** This is deliberate and is explained below.
- `_pearcey_cusp._cusp_vertex` (build 1c), `_DEFAULT_ETA_MAX` /
  `_DEFAULT_ETA_FLOOR` and the foot-of-normal branch (step 3),
  `ANNULUS_INNER_RADIUS` / the fences / `ppgo_map` (step 5).
- Any training run, engine sweep or chart artifact.

## Why the cusp window is carved out

`_find_cusps` returns `(theta_cusp, delta_theta)`. Only the first is a
detection artifact. The second is a load-bearing PHYSICAL exclusion: it is
persisted into `TubeChart.cusp_windows`, serialized into the artifact, and
read at SERVE time by `surrogate._tube_serves` to fall through near the
2/3-power singularity a spline cannot represent. Deleting or shrinking it
trains through a singularity.

Its width is nonetheless currently derived from a sampling artifact
(`width_safety * dip half-width`, floored at an absolute 0.05), so it IS on
the hit list — but not in this build. F040 shows the correct width is
DERIVABLE and is a FUNCTION of `w`, not a constant:

    dth_par  ~ sqrt(2/(|y''| w^{1/2})),   dth_perp ~ (6/(|y'''_perp| w^{3/4}))^{1/3}

both `w^{-1/4}`. Two reasons that is out of scope here. First, it needs
`y'''`, which build 1a did not deliver — it is scheduled with 1c. Second,
`cusp_windows` is STORED per chart as a fixed `(theta_cusp, delta_theta)`
pair, so the schema itself cannot express a frequency-dependent window;
fixing it is a schema change, not a value change, and that is its own build.

Note the incumbent is w-INDEPENDENT and 2-50x too narrow over the served
band, so charts are currently trained INTO the Pearcey region. Do not
"improve" it here by widening it — a hand-picked wider constant is the same
bug with a bigger number.

If you find yourself needing a cusp-window width to make something pass,
STOP and report — that is the signal, not a blocker to route around.

## Measured facts (driver-obtained; you cannot get these in-build)

**`_min_curvature_radius`'s consumer decision must not flip.** `r_min` is read
at exactly one place, `surrogate_training.py:3331`, as
`config.eta_max > 0.5 * r_min`. With `eta_max = 0.05` the decision is
unchanged across every band measured, incumbent vs exact:

| band | incumbent | exact | skip? |
|---|---|---|---|
| (0.25, 0.35) | 0.16136 | 0.14717 | no, both |
| (0.45, 0.55) | 0.30895 | 0.28747 | no, both |
| (0.65, 0.75) | 0.46892 | 0.44167 | no, both |
| (0.85, 0.95) | 0.78344 | 0.74692 | no, both |

and on the small-gamma bands `(0.0281,0.0462)`, `(0.0644,0.0825)`,
`(0.0825,0.1550)` (skip, both) and `(0.1550,0.3000)` (no skip, both). The
exact value is SMALLER by 4.9-9.6% — F038, the endpoint-exclusion bias. Do
NOT assert byte-identity with the incumbent and do NOT assert that margin;
both enshrine a discretization artifact.

**Deleting `_PROBE_ETA` should close the dropped topology slivers.** Measured
at fixed `n_samples = 200`, varying only the probe step:
`stable_gamma_bands((0.01, 0.30), +1)` returns 4 stable bands with 2 dropped
slivers at `_PROBE_ETA = 0.05`, and 1 stable band with 0 dropped at 0.004.
The drops are the probe overshooting a small caustic, not a detection
failure (F039).

**The probe's answer moves with its step, which is why it goes rather than
gets retuned.** `(sign, n_img)` at positive parity, `kappa = 0`:

| gamma | theta | step 0.05 | step 0.25*R_c |
|---|---|---|---|
| 0.02 | 2.3 | (1, 2) | (-1, 4) |
| 0.30 | 1.0 | (-1, 4) | (1, 2) |
| 0.70 | 1.0 | (-1, 4) | (1, 2) |

`f * R_c` is a trap: a curvature radius is not a caustic THICKNESS (at
gamma = 0.3, `R_c = 1.05` while the caustic reaches 0.72).

**Image-count conditioning.** `find_images_quartic` cannot separate the fold
pair at `eps = 6e-7` on a `gamma = 0.005` caustic; it agrees again by
`eps ~ 1e-3 * scale`. If you use an image count anywhere, keep it well above
that floor and say so.

## Acceptance

1. `eta_max > 0.5 * r_min` flips on NO band in the table above. Assert the
   decision ONCE, in the file that owns the predicate — not in every consumer
   suite (see "Assert VALUES, not code paths" in AGENTS.md).
2. `stable_gamma_bands((0.01, 0.30), +1)` returns ZERO dropped slivers. If it
   does not, report the residual cause; do not add a fence.
3. Deleting `_CLOUD_MARGIN_FRAC` changes no admission decision on the bands
   above — the distance it corrected is now exact. Show this, do not assume
   it.
4. No `np.gradient`, no finite difference, no sampled-arc estimator and no
   step-size constant remains in the six targets. `_PROBE_ETA`,
   `_CLOUD_MARGIN_FRAC` and `_CUSP_SPEED_REL_FRAC` are gone from the module.
5. Cusp ANGLES agree with the incumbent detector to within the incumbent's
   own sampling resolution, and are pinned to the analytic root at 1e-10.
   Cusp WINDOW widths are byte-identical to before.
6. `python -m pytest cogwheel/tests/test_lensing_surrogate_training.py -q`
   green, plus any suite you touched. Full-suite gate is a post-build driver
   step; do not run it in-build.

## Constraints

- Assert VALUES against an oracle and a tolerance, not code paths.
- Slow tests never run inside a build; `COGWHEEL_BRUTE_ACCURACY` and
  `COGWHEEL_STRICT_TIMING` stay empty.
- Named refusals stay named. `LensDomainError` for a genuine domain exit;
  never a silent `nan`, never a masked element.
- Prose you change must be true when you are done. `_min_curvature_radius`,
  `_find_cusps` and the cusp-window machinery are described in `SPEC.md`
  row 55, `COVERAGE_DESIGN.md`, and the `lensing_coverage_map` fragment. A
  build that changes behaviour and leaves a live document describing the old
  behaviour FAILS its acceptance, exactly like a red test.
- Deletion is the default. A test that pins a step-size constant which no
  longer exists is deleted, not re-pointed at a new number.
