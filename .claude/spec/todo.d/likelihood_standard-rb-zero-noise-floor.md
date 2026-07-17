---
section: likelihood
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

Acceptance: standard-RB zero-noise self-floor driven to <=1e-4 (or the
mechanism documented in FINDINGS with a physical bound if irreducible);
the lensing suite's regression pin (1.164e-2 decomposed) then tightens
accordingly. Route through the normal build workflow — this touches
mature package machinery (correctness-first; brute/fft oracles exist).
