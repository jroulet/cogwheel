---
date: 2026-08-07
---

### Exterior charts re-charted in polar (rho, theta_c); FarFieldChart deleted

`ExteriorPolarChart` replaces `FarFieldChart` as the exterior chart class.
Exterior charts now interpolate in the tiler's own caustic-fixed polar
`(rho, theta_c)` coordinates instead of the fold-adapted `(s, d)` far-field-
smooth coordinates — self-contained, single-valued, and cusp-safe by
construction: tile edges sit on cusp rays so no tile straddles a cusp, and
no arc-length map is needed. Roughly 1500 lines of `(s, d)` machinery were
deleted (`FarFieldChart` itself, `_FarFieldArcMap`, the
`_to_farfield_smooth`/`_from_farfield_smooth` bridge, `_farfield_serves`,
and related helpers/schema constants; stale `(s, d)` artifacts hard-refuse
at load). Both positive-parity astroid and macro-saddle exteriors are now
chartable in polar coordinates. A cusp carve-out
(`_CUSP_EXCLUSION_DISTANCE = 0.2` y-units) keeps the exterior tiler off
cusp-adjacent regions that the Pearcey serving arm covers. The envelope-
definition loader set `_KNOWN_ENVELOPE_DEFINITIONS` is now the union of the
far-field and interior (SACR-C) labels, so interior-tagged charts validate.
The training path also gained an optional `m_lens_range` override
(`(m_lo, m_hi)` in Msun) so a per-region probe can train a single mass/w
stratum through the production training path.
