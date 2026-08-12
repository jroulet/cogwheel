---
date: 2026-08-12
section: Backlog
---
# Saddle corridor origin-rho misclassification — RESOLVED (commit 288f37c)

The deltoid (saddle, gamma>1) caustic does not enclose the origin, so
origin-based `caustic_rho` misclassified the corridor between the two
deltoid lobes as interior (2 images exterior but rho<1).  Fixed with the
image-count discriminator (len(images)==4 interior / ==2 exterior),
parity-gated so the astroid (origin-enclosing) path is byte-identical:

- `_ppgo_cell_coords` refuses saddle rho<1 ppGO map queries.
- fold-ppGO interior handoff skips saddle non-4-image draws.
- census `classify_fallthrough` marks saddle image_count==2 as 'born'.
- `ppgo_map.w_cert` returns UNKNOWN for saddle rho<1 (defense-in-depth).
- Saddle rho>=1 (genuinely exterior) stays certified.

Exploits the D2 4-fold symmetry (image count is fold-invariant).  User
flagged; Test Dev unaware; Professor adjudicated.  Stale born-test
fixtures re-pointed to genuine saddle lobe-interior configs.
