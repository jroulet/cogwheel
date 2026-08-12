---
bump: patch
---
SPEC.md Born rung: `classify_fallthrough` born-attribution sentence updated —
the census now marks saddle corridor sources (`gamma > 1`, `image_count == 2`)
as 'born' in addition to `rho > 1` (exterior-to-caustic), since the deltoid
caustic does not enclose the origin so `rho < 1` does not imply interior on
the saddle. Fix for 288f37c which patched `surrogate_census.py` but left the
SPEC sentence describing only the `rho > 1` criterion.
