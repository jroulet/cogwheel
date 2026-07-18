---
date: 2026-07-18
---
### Microlensed likelihood reaches ~10 ms/eval: the ratio layer

The relative-binning idea is applied to the lens sector itself: each
proposal's smooth SACR-C envelope is heterodyned against a memoized
fiducial envelope (a pure function of the candidate — lens geometry
snapped to a fixed lattice, lens mass/redshift shared exactly), so only
the ultra-smooth ratio is interpolated, at ~8 engine nodes per proposal
(config-independent). Guards (image-count mismatch, envelope health,
fiducial-side refusals) fall back to the certified direct path;
candidate-side refusals propagate identically on the ratio, direct, and
brute-force paths. Measured warm single-thread lnlike: ~9.8 ms/eval,
~143x over brute force — a ~1500x cumulative speedup over the
pre-fast-path likelihood at unchanged accuracy tolerances (the
remaining ~1-nat crown-config difference vs brute force is inherited
relative-binning error, present identically in the direct path).
