---
date: 2026-08-12
bump: patch
---
### Saddle corridor refusal in ppGO map; caustic_rho is origin-based, not an interior discriminator

SPEC.md FOLD-PPGO INTERIOR HANDOFF paragraph: added the PARITY-GATED saddle
corridor refusal — origin-based `caustic_rho` is NOT an interior
discriminator on the macro saddle (the deltoid lobes do not enclose the
origin), so saddle corridor sources (`gamma > 1`, 2 images, `rho < 1`) are
never routed through the ppGO map / fold-ppGO interior handoff / Born
classification; `CertifiedPpgoMap.w_cert` returns the `UNKNOWN` sentinel
for saddle `rho < 1` cells (saddle `rho >= 1` stays certified); interior is
decided by image count (`len(images) >= 4`) on both parities.

Module list: `caustic_rho` annotated as an origin-based scalar-reach gauge
that is NOT a saddle interior discriminator.
