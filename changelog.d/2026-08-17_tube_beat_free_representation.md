---
date: 2026-08-17
---

Lensing: `TubeChart` (near-caustic microlensing surrogate) now stores the
envelope as a **beat-free residual** `r(w) = E(w) / F_ref(w)` instead of
the raw envelope `E(w)`. The near-caustic envelope carries two fold-pair
carriers that interfere as `cos(w * Delta_tau)`, and no reparametrization
of the interpolation axes removes that beat — dividing out an analytic,
non-vanishing Airy-uniform two-carrier reference `F_ref(w)` does. This
collapses the node count needed to clear the accuracy bar from 48 to 10
(F083, measured on a gamma=0.4 astroid, held-out eps=4.2652e-03 vs the
0.0237 bar). Every `TubeChart` record now carries an `envelope_definition`
tag (default `'tube_beat_free_airy_v1'`); an absent or unknown tag
hard-refuses at load, so a stale pre-beat-free tube artifact cannot
silently mis-serve as a residual. `serve` reconstructs the physical
envelope transparently (`E = r * F_ref`); no downstream API changed.
**Any cached surrogate `.npz` with tube charts trained before this change
must be regenerated.**

Commit: `69c79b8`
