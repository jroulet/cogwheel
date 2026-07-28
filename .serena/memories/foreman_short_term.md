# Foreman-Lite Short-Term Observations

- INS-10-001 (this pass, 2026-07-28, repeat): ELEVENTH distinct
  finding-ID instance of the same mis-route pattern — finding text
  explicitly says "→ Librarian: extend the Born-rung paragraph..." yet
  was routed into the Foreman-Lite queue again. Declined without
  touching any files; SPEC.md editing is Librarian-owned and explicitly
  forbidden by my hard requirements. No code change made — this was pure
  doc-narrative work (rewriting the Born rung paragraph in SPEC.md to
  describe positive+saddle parity serving, Morse phase, Guard B, F026
  fence, ghost refusal above split). Now an 11x recurrence of the
  identical routing bug across sessions — strongly recommend the
  orchestrator add a pre-filter that strips "→ Librarian"-tagged findings
  from the Foreman-Lite queue before dispatch, since per-pass decline is
  not resolving the upstream bug.
