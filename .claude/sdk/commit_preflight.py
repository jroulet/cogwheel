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
from datetime import date
from pathlib import Path

# (marker substring emitted by .claude/hooks/pre-commit, changelog fragment dir)
FRAG_SPECS = [
    ("DATA_CONTRACTS.yaml modified but no changelog evidence",
     ".claude/spec/contracts_changelog.d"),
    ("SPEC.md modified but no changelog evidence",
     ".claude/spec/spec_changelog.d"),
]


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
    title = message.splitlines()[0][:70]
    slug = "".join(
        c if c.isalnum() else "_" for c in title.lower()
    ).strip("_")[:40] or "build_change"
    today = date.today().isoformat()
    stubbed = []

    for marker, frag_dir in FRAG_SPECS:
        if marker not in out:
            continue
        frag = root / frag_dir / f"{today}_{slug}.md"
        if frag.exists():
            continue
        frag.parent.mkdir(parents=True, exist_ok=True)
        frag.write_text(
            f"---\nbump: patch\n---\n\n### {title}\n\n"
            "(Auto-generated at commit preflight because the build "
            "staged the canonical file without a fragment; Librarian "
            "should refine this entry from the commit diff.)\n")
        rel = str(frag.relative_to(root))
        subprocess.run(["git", "add", rel], cwd=root, check=True)
        stubbed.append(rel)

    if stubbed:
        render = root / "scripts" / "render_fragments.py"
        if render.exists():
            subprocess.run([sys.executable, str(render)],
                           capture_output=True, text=True, cwd=root)
            subprocess.run(["git", "add", "-u"], cwd=root, check=True)
        log(f"Commit preflight: auto-stubbed missing fragment(s) "
            f"{stubbed} (Librarian should refine)")
        res = run_hook()
        if res.returncode == 0:
            return
        out = (res.stdout or "") + (res.stderr or "")

    # Last remediation: the deterministic sync script auto-fixes SPEC
    # module lists (the "new module added but SPEC.md not updated" class
    # that killed builds 8g-b and 8h-b2). Run it, restage, re-check.
    sync = root / "scripts" / "sync_derived_docs.py"
    if sync.exists():
        subprocess.run([sys.executable, str(sync)],
                       capture_output=True, text=True, cwd=root)
        subprocess.run(["git", "add", "-u"], cwd=root, check=True)
        log("Commit preflight: ran sync_derived_docs.py auto-fix, "
            "re-checking the hook")
        res = run_hook()
        if res.returncode == 0:
            return
        out = (res.stdout or "") + (res.stderr or "")

    raise RuntimeError(
        "Spec/doc discipline hook blocks the commit and preflight "
        "could not auto-remediate. Hook output:\n" + out)
