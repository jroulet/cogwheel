#!/bin/sh
# install_hooks.sh — tell git to use .claude/hooks/ as the hooks directory.
#
# Sets core.hooksPath so git runs hooks directly from .claude/hooks/.
# No symlinks, no rot. Idempotent. Safe to run repeatedly.
#
# Usage:
#   bash .claude/hooks/install_hooks.sh              # repo-level
#   bash .claude/hooks/install_hooks.sh --worktree   # worktree-scoped (for branch safety)
#   bash .claude/hooks/install_hooks.sh --check      # report status only

set -e

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "ERROR: Not inside a git repository." >&2
    exit 1
fi

CHECK_ONLY=0
WORKTREE=0
for arg in "$@"; do
    case "$arg" in
        --check) CHECK_ONLY=1 ;;
        --worktree) WORKTREE=1 ;;
    esac
done

if [ "$WORKTREE" -eq 1 ]; then
    git config extensions.worktreeConfig true 2>/dev/null || true
    CONFIG_CMD="git config --worktree"
    SCOPE="worktree"
else
    CONFIG_CMD="git config"
    SCOPE="repo"
fi

current=$($CONFIG_CMD core.hooksPath 2>/dev/null || echo "")

if [ "$current" = ".claude/hooks" ]; then
    echo "OK: core.hooksPath = .claude/hooks ($SCOPE-scoped)"
elif [ -n "$current" ]; then
    if [ "$CHECK_ONLY" -eq 1 ]; then
        echo "WRONG: core.hooksPath = $current (expected .claude/hooks)"
        exit 1
    fi
    $CONFIG_CMD core.hooksPath .claude/hooks
    echo "REPAIRED: core.hooksPath changed from $current to .claude/hooks ($SCOPE-scoped)"
else
    if [ "$CHECK_ONLY" -eq 1 ]; then
        echo "MISSING: core.hooksPath not set; run without --check to install"
        exit 1
    fi
    $CONFIG_CMD core.hooksPath .claude/hooks
    echo "INSTALLED: core.hooksPath = .claude/hooks ($SCOPE-scoped)"
fi
