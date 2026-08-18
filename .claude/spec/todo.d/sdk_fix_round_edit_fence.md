---
section: Backlog
---

- **THE ESCALATION FIX ROUND SILENTLY DROPS OUT-OF-MANIFEST EDITS —
  three consumed rulings, one-line fix never applied** `[housekeeping]`
  — measured 2026-08-18 (born_farfield_completion build): the driver
  issued the SAME escalation_fix ruling THREE times for a one-line
  probe binding in test_lensing_born_analytic_reachability.py (a file
  outside the build's changed-file manifest); each round consumed the
  ruling, did other work, and left the file VERIFIABLY untouched (not
  in git diff), even after the third ruling granted explicit
  out-of-manifest authorization and demanded a
  tell-me-what-blocked-you report — which also never came. Same
  signature suspected in the low_w build (the lam*sqrt_mu
  normalization needed a literal-patch second issuance). INVESTIGATE
  in .claude/sdk/orchestrator.py's escalation-fix path: what the
  fix-coder's task prompt/permission set says about file scope; whether
  a manifest allowlist, a hook, or CHANGE_REPORT plumbing fences edits
  to build-tracked files; why the mandated
  "say exactly what stopped you" clause produced silence (is the
  driver feedback even reaching the coder's prompt verbatim, or is it
  summarized/truncated en route?). FIX: escalation-fix rounds must
  treat the DRIVER RULING as the manifest (the ruling's named files
  are in scope by definition), and any tool denial inside a fix round
  must surface in the change report, not vanish. Acceptance: a probe
  build (or the next real escalation) applies a driver-named
  out-of-manifest edit on the FIRST round.
