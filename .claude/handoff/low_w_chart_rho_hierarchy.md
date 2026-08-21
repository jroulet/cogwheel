# Ruling: the low-w chart needs a rho-partitioned carrier hierarchy, not a cusp fallback

## The core correction (Professor, 2026-08-21 — after the cusp-fallback smoke bake)

The wall band (gamma' > 0.5, 57.5% of engine residual) is NOT a fold or cusp
problem. At gamma'=0.8 the caustic reaches |y| ~ 3.58; rho=2.0 is 2x the
caustic radius — two well-separated images, no fold, no cusp. The b3->0
signal meant the fold form's MERGING-PAIR assumption broke down, not "there
is a cusp here." The "fold or cusp, both finite in the right coordinates"
framing applies only to the caustic NEIGHBORHOOD (rho ~ 1); the far-exterior
wall band is neither.

## The 5800x overshoot (measured): a carrier-normalization failure

r = f_pure*sqrt(1-gamma'^2)/F_ref. The residual is O(1) only if F_ref is a
carrier of the same magnitude as f_pure. It is not, two ways:
- The Airy q=p form is safe (Wronskian Ai^2+Ai'^2 > 0) but mis-normalized
  (|F_ref| ~ w^{-1/6} -> inf as w->0, so r -> 0).
- The Pearcey uniform = cluster_sum*(P/P_asymp) has NO Wronskian guarantee:
  P has genuine zeros (cusp diffraction dark fringes), P_asymp diverges on
  fold lines, cluster_sum vanishes independently (resolved far-exterior).
- The guard min/max >= 1e-3 measures the spread of F_ref against ITSELF —
  blind to a uniformly-small carrier. The de-rate 0.0002 is the symptom: no
  scalar de-rate repairs an ill-conditioned, 3-4-orders-off-normalization
  interpolant.

## Correct hierarchy (binding) — by rho, not by gamma

1. CAUSTIC NEIGHBORHOOD (rho ~ 1): Airy fold (q=p) carrier, with the Pearcey
   fallback restricted to the GENUINE cusp window (b3->0 AND rho ~ 1, where
   the 3 images are cluster-resolved and cluster_sum is O(1)).
2. FAR EXTERIOR, RESOLVED (rho >= rho_split, w*Delta_tau >= few): the
   TWO-IMAGE GEOMETRIC-OPTICS SUM carrier — exact where the images are
   well-separated; the ratio against it is the smooth O(1) diffractive
   correction (the Born/ppGO exterior ladder's pattern, F023-F025).
3. FAR EXTERIOR, UNRESOLVED (rho >= rho_split, low w): the MACRO LEAD
   CARRIER sqrt(mu_macro) — which is ALREADY the residual's normalization
   (f_pure*sqrt(1-gamma'^2) -> 1 at low w). The diffractive correction is
   the smooth residual.

## Two requirements without which the pole returns

- GUARD CURRENCY: replace the self-referential min/max-spread guard with an
  ABSOLUTE carrier-adequacy bound: |F_ref| / |f_pure*sqrt(1-gamma'^2)| (or
  directly |r|) on the w-grid. The 5800x overshoot blew through the spread
  guard.
- MAGNITUDE: normalize the Airy F_ref to the macro limit at low w (its
  w^{-1/6} divergence forces r->0, not r->1), or fold the macro lead into
  the reference, so the fold-side residual is genuinely O(1) at the band
  bottom.

## Implication

The wall band is served by the GEOMETRIC/MACRO ladder (carriers 2 and 3),
the same representation family as the Born/ppGO exterior charts — NOT by a
fold/cusp normal form. The chart's rho-partitioned carriers cover all three
regions with a smooth O(1) residual each. The genuine-cusp window (rho~1,
b3->0) is the only place the Pearcey form applies.
