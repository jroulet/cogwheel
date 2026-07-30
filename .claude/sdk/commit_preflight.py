"""Spec/doc discipline preflight for SDK build commits.

Pure-stdlib helper (no claude_agent_sdk dependency) so it is unit
testable in the project's conda env. Runs the repo's pre-commit hook
against the staged index; if it fails because DATA_CONTRACTS.yaml or
SPEC.md is staged without its changelog fragment, generates a stub
fragment (marked for Librarian refinement), stages it, and re-runs the
hook. Residual failures raise with the hook's verbatim output.

Motivation: SDK builds that complete all work should not die opaquely
at ``git commit`` on the spec/doc discipline hook.
"""
import subprocess
import sys
from pathlib import Path



def ensure_spec_doc_fragments(project_root, message, log=print):
    root = Path(project_root)
    hook = root / ".claude" / "hooks" / "pre-commit"
    if not hook.exists():
        return

    def run_hook():
        return subprocess.run(
            [str(hook)], capture_output=True, text=True, cwd=root)

    res = run_hook()
    if res.returncode == 0:
        return

    out = (res.stdout or "") + (res.stderr or "")

    # The fragment AUTO-STUB that used to live here is gone (2026-07-30).
    # It existed only because the Librarian -- which OWNS every doc surface --
    # ran one step after the commit, so the gate was unsatisfiable by its
    # owner. Two things fixed that properly: the doc stage now runs BEFORE the
    # commit, and the pre-commit hook DEFERS spec/doc debt for an SDK build,
    # recording a receipt (.claude/doc_debt.json) the orchestrator then asserts
    # was cleared.
    #
    # The stub was worse than the gap it filled: it hardcoded `bump: patch`
    # regardless of the real change (1e-tube's schema addition was minor),
    # scraped a title from the commit subject, and rendered an
    # "(Auto-generated ... Librarian should refine)" note into the CANONICAL
    # changelog with nothing tracking it for refinement. It also ran
    # `git add -u` twice, re-introducing blanket staging. Fabricating a
    # plausible-looking wrong answer is worse than recording a debt.

    # Last remediation: the deterministic sync script auto-fixes SPEC
    # module lists (the "new module added but SPEC.md not updated" class
    # that killed builds 8g-b and 8h-b2). Run it, restage, re-check.
    sync = root / "scripts" / "sync_derived_docs.py"
    if sync.exists():
        # Stage ONLY what the sync script itself changed. A blanket
        # `git add -u` here runs INSIDE _git_commit_safe, after it has
        # deliberately staged just the build's own output, and would
        # re-sweep exactly the pre-existing dirt that staging fix excludes.
        def _dirty() -> set:
            proc = subprocess.run(["git", "diff", "--name-only"],
                                  capture_output=True, text=True, cwd=root)
            return {p for p in (proc.stdout or "").splitlines() if p}

        before = _dirty()
        subprocess.run([sys.executable, str(sync)],
                       capture_output=True, text=True, cwd=root)
        for path in sorted(_dirty() - before):
            subprocess.run(["git", "add", "--", path], cwd=root, check=True)
        log("Commit preflight: ran sync_derived_docs.py auto-fix, "
            "re-checking the hook")
        res = run_hook()
        if res.returncode == 0:
            return
        out = (res.stdout or "") + (res.stderr or "")

    raise RuntimeError(
        "Spec/doc discipline hook blocks the commit and preflight "
        "could not auto-remediate. Hook output:\n" + out)
