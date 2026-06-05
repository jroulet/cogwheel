#!/usr/bin/env bash
# Post-merge filter: revert blacklisted paths to the local version.
#
# Reads .claude/memory-blacklist.txt for the list of paths.
# Two tiers:
#   [soft] — blacklisted by default; skipped when MIND_MELD=1 is set.
#   [hard] — always filtered, regardless of MIND_MELD.
#
# Behavior:
#   For each blacklisted path that the merge commit changed:
#     git checkout HEAD^1 -- <path>   # restore local (target-branch) version
#     git add <path>
#   Then amend the merge commit to fold the reverts in.
#
# Usage (from .claude/hooks/post-merge):
#   bash scripts/apply_memory_filter.sh
#
# Env vars:
#   MIND_MELD=1    — skip the soft blacklist (hard blacklist still enforced)
#
# Exit status:
#   0 always — never fail the merge
#
# Ported from gw_detection_ias scripts/apply_memory_filter.sh (acb7829 +
# e0f2c50 fixes). No skill-specific adaptation needed: the blacklist file
# controls which paths are filtered; the script itself is generic.

set -uo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || exit 0
BLACKLIST="$REPO_ROOT/.claude/memory-blacklist.txt"

# If blacklist file missing, nothing to do.
[ -f "$BLACKLIST" ] || exit 0

# Only run on three-way merges (merge commits with two parents).
# Fast-forward merges (e.g. `git pull` on the same branch across your own
# machines) have no HEAD^2; skipping them lets your own incoming changes
# flow without being reverted. Cross-branch merges (e.g. collaborator
# pulling from your branch into theirs) always produce a merge commit, so
# the filter applies there.
git rev-parse HEAD^2 >/dev/null 2>&1 || exit 0

# Read blacklist into soft/hard arrays.
declare -a soft_paths=()
declare -a hard_paths=()
section=""
while IFS= read -r line; do
    # Strip comments and trim whitespace.
    line="${line%%#*}"
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [ -z "$line" ] && continue

    case "$line" in
        "[soft]") section="soft"; continue ;;
        "[hard]") section="hard"; continue ;;
    esac

    case "$section" in
        soft) soft_paths+=("$line") ;;
        hard) hard_paths+=("$line") ;;
    esac
done < "$BLACKLIST"

# Which tiers are active?
declare -a active_paths=("${hard_paths[@]}")
if [ "${MIND_MELD:-0}" != "1" ]; then
    active_paths+=("${soft_paths[@]}")
fi

[ "${#active_paths[@]}" -eq 0 ] && exit 0

# Files that the merge commit changed (vs. first parent = local side pre-merge).
changed=$(git diff --name-only HEAD^1 HEAD 2>/dev/null || true)
[ -z "$changed" ] && exit 0

# For each active blacklist entry, find matching changed files and revert.
reverted=()
for path in "${active_paths[@]}"; do
    # Expand directory patterns (trailing /) to match all files under that dir.
    case "$path" in
        */)
            matches=$(echo "$changed" | grep -E "^${path}" || true)
            ;;
        *)
            matches=$(echo "$changed" | grep -Fx "$path" || true)
            ;;
    esac
    [ -z "$matches" ] && continue

    while IFS= read -r match; do
        [ -z "$match" ] && continue
        # Check if the path exists in HEAD^1 (may have been added in the merge only).
        if git cat-file -e "HEAD^1:$match" 2>/dev/null; then
            git checkout HEAD^1 -- "$match" 2>/dev/null || true
            git add "$match" 2>/dev/null || true
        else
            # Path was added by the merge (didn't exist locally) — remove it.
            git rm -f --cached "$match" 2>/dev/null || true
            rm -f "$REPO_ROOT/$match" 2>/dev/null || true
        fi
        reverted+=("$match")
    done <<< "$matches"
done

if [ "${#reverted[@]}" -eq 0 ]; then
    exit 0
fi

# Amend the merge commit to include the reverts.
# --no-edit keeps the existing merge commit message.
git commit --amend --no-edit --no-verify >/dev/null 2>&1 || true

echo "  memory filter: reverted ${#reverted[@]} blacklisted path(s) in merge commit"
if [ "${MIND_MELD:-0}" = "1" ]; then
    echo "    (MIND_MELD=1 active — only hard blacklist enforced)"
fi

exit 0
