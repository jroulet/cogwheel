---
date: 2026-07-30
section: lensing
---

# Closed-form caustic reach — the serve path stops scanning (F054 / F026)

Closes `todo.d/lensing_serve_path_caustic_reach.md`. `ppgo_map.caustic_geometry`
was a Python double loop over 2 square-root branches x 720 polar angles calling
`geometry.critical_point` on each — **1440 calls per likelihood evaluation** —
to find a maximum source-plane radius by scanning. It ran on every serve, and
F054 measured it at ~27.5 ms of a 31.25 ms surrogate-served lensed `lnlike`
(90%), against 1.7% for `_contract_tensor_spline`, the contraction the
surrogate exists to perform.

It is closed form. With `lam = 1 - kappa`, `e = gamma / lam` and
`u = 1 / (lam |x|^2)`, eliminating `cos 2theta` gives

    |y|^2 = lam [ (1-u)^2 (1+2u) + e^2 (2u-1) ] / u^2

whose stationary points factor exactly:

    (u - 1)(u^2 + u + 1 - e^2) = 0

`u = 1` is the origin-crossing decoy; the interior roots
`u = (-1 +- sqrt(4e^2 - 3)) / 2` are real iff `e >= sqrt(3)/2`. The reach is
the largest admitted candidate among the axis cusps `u = 1 -+ e`, the macro
saddle's wedge turnaround `u = sqrt(e^2 - 1)`, and those interior roots — each
admitted only if `u > 0` and the implied `cos 2theta` lies in `[-1, 1]`.

## Measured

| | before | after |
|---|---|---|
| `critical_point` calls per reach | 1440 | **0** |
| `caustic_geometry` mean cost (13 configs) | 42.95 ms | **0.0054 ms** |
| speedup | — | **~7900x** (up to 16000x) |

Agreement with the retired scan is exact (0 or ~1e-16 relative) wherever the
extremum lands on a grid node, and differs by up to 1.1e-4 exactly in the
near-wall band `1 < gamma < 1.2`. **That residual is the scan's error, not the
formula's**: the scan's answer converges toward the closed form as `n_theta`
refines (gamma=1.05: 1.10e-4 -> 2.59e-6 -> 1.08e-8 on 4x refinements). The
astroid cases matched at 1e-16 only because 720 is divisible by 4, so the
`pi/2` cusps sit on grid nodes.

The cache half of the TODO is deliberately NOT implemented: at 5.4 us a
`(gamma, kappa)` `lru_cache` would add invalidation surface and a
stale-geometry failure mode to buy nothing measurable. Recorded here rather
than silently skipped.

## Notes

- No SPEC.md surface changed: SPEC.md does not describe `ppgo_map`, so the
  TODO's `[→ spec]` tag had no target. Values are unchanged (a cost fix), so
  there is nothing spec-level to state. No `spec_changelog.d` fragment.
- The build STRANDED before committing (F056: an unreadable Inspector result
  was an unbreakable revision loop). Revisions 1 and 2 were genuinely reviewed
  and their findings fixed; **revision 3's verdict was lost to that defect, so
  this work has had two inspection rounds, not three.**
- A `git show HEAD` "was 1440" witness the build added was removed before
  commit: it could only pass in the window before its own change landed, then
  would skip itself forever (F043/F045). The pre-commit guard caught it. The
  durable claim — the closed form issues ZERO `critical_point` calls — is kept.
