---
section: Backlog
---
# Standard RB zero-noise floor (8.96e-3) — fix upstream [→ docs]

Owner directive (2026-07-17): the measured zero-noise floor of the
PRE-EXISTING standard `RelativeBinningLikelihood` should be fixed in the
upstream machinery, not merely pinned by the lensing suite.

Measured (probe9, zero-noise fixture d=h0, asd_drift=1, HLV 4s
IMRPhenomXPHM, 253 bins @4 Hz): lnlike_fft(par0)=285.398401; standard
unlensed RB = 285.389439 → floor 8.962e-3; lensed RB adds only 2.676e-3
on top (that increment is separately gated in the lensing suite).

Design note from the owner: the stall-ringdown is applied to the
REFERENCE (`_h0_edges`) and not the evaluated waveform — by design. The
investigation should therefore look at the intrinsic cost of the
reference-side stall + linear-free construction at zero noise (where the
candidate/fiducial ratio should be exactly 1) rather than "symmetrizing"
the construction (measured in probe8: forcing the stall onto the
candidate explodes the floor to ~127 — the asymmetry is intentional).

Mechanism (owner, 2026-07-17): the ratio is r = h/h_0 where ONLY h_0
carries the stall-ringdown — so even at exactly fiducial parameters and
zero noise, r != 1 in the ringdown band and the per-bin LINEAR ratio
model must track real structure; its residual binning error IS the
8.96e-3 floor. If reference and candidate were constructed alike, r
would be identically 1 there and the floor would vanish — the stall buys
a smooth interpolable reference at the cost of a nontrivial fiducial
ratio.

HARD CONSTRAINT (owner, 2026-07-17): the stall-ringdown is NOT to be
removed or weakened — it is load-bearing (smooth interpolable
reference). Any plan that deletes or bypasses the stall is out of
bounds.

THE REQUIREMENT (owner, 2026-07-17): likelihood accuracy, nothing else.
r != 1 at fiducial is acceptable — the stall exists to make r SMOOTH,
not trivial — as long as the per-bin model captures it to the lnL
tolerance. The 8.96e-3 zero-noise ΔlnL says the current bins/linear
model capture it imperfectly. Candidate levers (engineering choice for
the build, not prescribed here): finer/adaptive binning through the
ringdown band, a higher-order in-bin ratio model there, or arranging
the stalled reference to cancel through the contraction. Judge any of
them purely by the measured zero-noise ΔlnL and the standard accuracy
gates.

Acceptance: standard-RB zero-noise self-floor driven to <=1e-4 (or the
mechanism documented in FINDINGS with a physical bound if irreducible);
the lensing suite's regression pin (1.164e-2 decomposed) then tightens
accordingly. Route through the normal build workflow — this touches
mature package machinery (correctness-first; brute/fft oracles exist).
