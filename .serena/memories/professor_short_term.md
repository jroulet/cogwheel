# Diffractive truncation-certificate fence ruling (d609fff)

Measured at gamma'=0.41, kappa=0, beta=0 (corner direction theta_source=3pi/4+pi/32=2.454 rad):

1. **Caustic angle mismatch is order-1.** `caustic_point(gamma, theta)` takes the CRITICAL-CURVE polar angle theta; the returned caustic point's own polar angle phi(theta) differs from theta by median ~1.36 rad (78 deg), up to pi. At theta_source=2.454, y_c=(0.314,0.263) has phi=0.698 rad (40 deg) — a ~100 deg mismatch. So `|y_c(theta_source)|` is NOT the caustic radius along the source direction.

2. **rho discriminator is directionally miscalibrated.** ratio R_dir(phi)/|y_c(theta=phi)| (directional fold-crossing radius / code's reference radius) spans [0.573, 1.0] over 360 source directions. The code reference |y_c(theta_source)| is ALWAYS >= the true directional fold radius (1x..1.75x), so code rho = r/|y_c(theta_source)| systematically UNDER-estimates the true rho (source looks closer to the fold than it is) by up to 1.75x. The corner case (rho=1.34) is the ONE direction where ratio=1 (min-radius cusp region, |y_c|~0.41 = gamma'), so the brief's rho happens to be correctly calibrated only there.

3. **|y_c| range** at gamma'=0.41: 0.41 (min, near diagonal cusp) to 1.067 (max, at theta=pi/2). min |y_c| = gamma' = 0.41.

4. Operator-series small parameter = gamma' * s * w / 2 (code docstring; s=|y'|^2 reduced). Vanishes in deep interior (s->0), so the order-16 series is MOST valid there; marginal resonances (rel(w) barely > CERTIFICATION_BAR, ~1.1-1.2e-4 vs 1e-4 in ~0.1-wide w-windows) are a NEAR-FOLD phenomenon only (order-16 Taylor-in-shear can't track the coalescing-image Airy oscillation to 1e-4 relative).

5. Fold wave-optics shell width ~ Airy xi=(3 w DT/4)^(2/3) ~ w^(2/3) * Delta, so Delta ~ w^(-2/3) ~ 0.08-0.2 reduced units at the resonance w~3.5-7. Maps to delta ~ Delta/R_dir ~ 0.25-0.5 in rho. Measured corner delta = 0.14/0.41 = 0.34.
