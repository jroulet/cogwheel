#!/usr/bin/env bash
# verify_installation.sh — post-install validation for TejaForce pipeline.
#
# Run this AFTER the skill completes Phase 6 to catch anything the
# installer missed. Checks file existence, content correctness,
# placeholder substitution, JSON validity, and hook wiring.
#
# Usage:
#   bash .claude/hooks/../../../scripts/verify_installation.sh
#   # or simply:
#   bash scripts/verify_installation.sh
#
# Exit codes:
#   0 = all checks passed
#   1 = one or more checks failed (details printed)

set -e

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPO_ROOT"

errors=""
warnings=""

err() { errors="$errors\n  FAIL: $1"; }
warn() { warnings="$warnings\n  WARN: $1"; }
ok() { echo "  OK: $1"; }

echo "TejaForce Installation Verification"
echo "===================================="
echo ""

# ── 1. Required files exist ──────────────────────────────────────────────

echo "1. Required files..."
for f in \
    .claude/sdk/orchestrator.py \
    .claude/sdk/agents.py \
    .claude/sdk/cli.py \
    .claude/sdk/build.py \
    .claude/sdk/gates.py \
    .claude/sdk/schemas.py \
    .claude/sdk/state.py \
    .claude/sdk/memory.py \
    .claude/sdk/prompts/sections.py \
    .claude/sdk/watchdog.sh \
    .claude/settings.json \
    .claude/hooks/pre-commit \
    .claude/hooks/post-commit \
    .claude/hooks/post-merge \
    .claude/hooks/install_hooks.sh \
    .claude/spec/SPEC.md \
    .claude/spec/TODO.md \
    .claude/spec/COMPLETED.md \
    ; do
    if [ -f "$f" ]; then
        ok "$f"
    else
        err "$f missing"
    fi
done

# Crew prompts (at least 9 required)
crew_count=$(find .claude/crew -name '*.md' 2>/dev/null | wc -l | tr -d ' ')
if [ "$crew_count" -ge 9 ]; then
    ok ".claude/crew/ has $crew_count agent prompts"
else
    err ".claude/crew/ has only $crew_count agent prompts (need >= 9)"
fi

# Commands
cmd_count=$(find .claude/commands -name '*.md' 2>/dev/null | wc -l | tr -d ' ')
if [ "$cmd_count" -ge 7 ]; then
    ok ".claude/commands/ has $cmd_count commands"
else
    err ".claude/commands/ has only $cmd_count commands (need >= 7)"
fi

echo ""

# ── 2. No un-substituted PLACEHOLDERs ───────────────────────────────────

echo "2. PLACEHOLDER substitution..."
stale_placeholders=$(grep -rl 'PLACEHOLDER_' \
    .claude/sdk/ .claude/hooks/ .claude/settings.json \
    .claude/build 2>/dev/null || true)
if [ -n "$stale_placeholders" ]; then
    for f in $stale_placeholders; do
        matches=$(grep -c 'PLACEHOLDER_' "$f" 2>/dev/null || echo 0)
        err "$f has $matches un-substituted PLACEHOLDER_ markers"
    done
else
    ok "No PLACEHOLDER_ markers in SDK/hooks/config files"
fi

# Check crew prompts for un-substituted {{TEMPLATE}} markers
stale_templates=$(grep -rl '{{[A-Z_]*}}' .claude/crew/ 2>/dev/null || true)
if [ -n "$stale_templates" ]; then
    for f in $stale_templates; do
        matches=$(grep -o '{{[A-Z_]*}}' "$f" 2>/dev/null | sort -u | tr '\n' ', ')
        err "$f has un-substituted template markers: $matches"
    done
else
    ok "No {{TEMPLATE}} markers in crew prompts"
fi

echo ""

# ── 3. JSON validity ────────────────────────────────────────────────────

echo "3. JSON validity..."
for f in .claude/settings.json; do
    if [ -f "$f" ]; then
        if python3 -c "import json; json.load(open('$f'))" 2>/dev/null; then
            ok "$f is valid JSON"
        else
            err "$f is invalid JSON"
        fi
    fi
done

echo ""

# ── 4. Shell scripts are executable ─────────────────────────────────────

echo "4. Executable permissions..."
for f in \
    .claude/hooks/pre-commit \
    .claude/hooks/post-commit \
    .claude/hooks/post-merge \
    .claude/hooks/install_hooks.sh \
    .claude/sdk/watchdog.sh \
    ; do
    if [ -f "$f" ]; then
        if [ -x "$f" ]; then
            ok "$f is executable"
        else
            err "$f exists but is not executable"
        fi
    fi
done

echo ""

# ── 5. Git hooks path ───────────────────────────────────────────────────

echo "5. Git hooks path..."
hooks_path=$(git config core.hooksPath 2>/dev/null || echo "")
if [ "$hooks_path" = ".claude/hooks" ]; then
    ok "core.hooksPath = .claude/hooks"
elif [ -n "$hooks_path" ]; then
    err "core.hooksPath = $hooks_path (expected .claude/hooks — run .claude/hooks/install_hooks.sh)"
else
    warn "core.hooksPath not set (run .claude/hooks/install_hooks.sh)"
fi

echo ""

# ── 6. Python syntax check on SDK files ─────────────────────────────────

echo "6. Python syntax..."
py_errors=0
for f in .claude/sdk/*.py .claude/sdk/prompts/*.py; do
    if [ -f "$f" ]; then
        if python3 -c "import ast; ast.parse(open('$f').read())" 2>/dev/null; then
            : # silent on success
        else
            err "$f has Python syntax errors"
            py_errors=$((py_errors + 1))
        fi
    fi
done
if [ "$py_errors" -eq 0 ]; then
    ok "All SDK Python files parse cleanly"
fi

echo ""

# ── 7. Settings.json hooks reference existing scripts ───────────────────

echo "7. Hook script references..."
if [ -f .claude/settings.json ]; then
    hook_scripts=$(python3 -c "
import json
s = json.load(open('.claude/settings.json'))
for phase in s.get('hooks', {}).values():
    for entry in phase:
        for h in entry.get('hooks', []):
            cmd = h.get('command', '')
            if '/' in cmd and not cmd.startswith('jq'):
                # Extract the script path (first token that looks like a path)
                for token in cmd.split():
                    if '/' in token:
                        print(token)
                        break
" 2>/dev/null || true)
    for script in $hook_scripts; do
        # Resolve relative to repo root
        resolved="$script"
        if [ -f "$resolved" ]; then
            ok "Hook references $resolved (exists)"
        else
            err "Hook references $resolved (NOT FOUND)"
        fi
    done
fi

echo ""

# ── 8. Serena or memory directory ───────────────────────────────────────

echo "8. Memory storage..."
if [ -d .serena ]; then
    ok ".serena/ exists (Serena configured)"
elif [ -d .serena/memories ]; then
    ok ".serena/memories/ exists (file-based fallback)"
else
    warn "Neither .serena/ nor .serena/memories/ exists — agent memories won't persist"
fi

echo ""

# ── 9. Data infrastructure wiring (conditional) ────────────────────────

if [ -f .claude/spec/DATA_CONTRACTS.yaml ]; then
    echo "9. Data infrastructure wiring..."

    # If contracts exist, the components should reference them
    if grep -q "DATA_CONTRACTS" .claude/crew/librarian.md 2>/dev/null; then
        ok "Librarian knows about DATA_CONTRACTS"
    else
        err "DATA_CONTRACTS.yaml installed but Librarian prompt doesn't reference it"
    fi

    if grep -qi "data.contract" .claude/crew/inspector.md 2>/dev/null; then
        ok "Inspector knows about data contracts"
    else
        err "DATA_CONTRACTS.yaml installed but Inspector prompt doesn't reference it"
    fi

    if grep -q "DATA_CONTRACTS" .claude/sdk/orchestrator.py 2>/dev/null; then
        ok "DATA_CONTRACTS in orchestrator SPEC_FILES"
    else
        err "DATA_CONTRACTS.yaml installed but not in orchestrator SPEC_FILES"
    fi

    if [ -f .claude/spec/data_registry.yaml ]; then
        if grep -q "data_registry" .claude/hooks/pre-commit 2>/dev/null; then
            ok "Pre-commit has registry version bump check"
        else
            err "data_registry.yaml installed but pre-commit has no version bump check for it"
        fi
    fi

    echo ""
fi

# ── Report ──────────────────────────────────────────────────────────────

echo "===================================="
if [ -n "$errors" ]; then
    echo "ERRORS (must fix):"
    echo -e "$errors"
    echo ""
fi
if [ -n "$warnings" ]; then
    echo "WARNINGS (review):"
    echo -e "$warnings"
    echo ""
fi
if [ -z "$errors" ] && [ -z "$warnings" ]; then
    echo "All checks passed."
fi

if [ -n "$errors" ]; then
    exit 1
fi
exit 0
