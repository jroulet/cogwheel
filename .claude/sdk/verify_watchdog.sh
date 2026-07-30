#!/usr/bin/env bash
# .claude/sdk/verify_watchdog.sh — does the watchdog ACTUALLY guard a build?
#
# Asserts outcomes (a stalled process really dies; the fallback pattern really
# matches the launcher's argv), never code paths. Runs in ~12s, needs no live
# build, and kills nothing outside its own sandbox.
#
# Exists because the watchdog failed OPEN for three days (F055): it greps for
# an entrypoint name, launch_build.sh launches a different one, and the only
# evidence was an exit-1 line in a sidecar log nobody reads while the launcher
# printed "(watchdog 1200s)". Test 4 is the one that would have caught it.
set -u

SDK="$(cd "$(dirname "$0")" && pwd)"
WD="$SDK/watchdog.sh"
LAUNCH="$SDK/launch_build.sh"
W="$(mktemp -d)"; trap 'rm -rf "$W"' EXIT
export WATCHDOG_POLL_INTERVAL=1 WATCHDOG_STARTUP_GRACE=5
pass=0; fail=0
ok() { if [ "$2" = "$3" ]; then echo "  PASS  $1 -> $3"; pass=$((pass+1));
       else echo "  FAIL  $1 -> got '$3', want '$2'"; fail=$((fail+1)); fi; }

echo "=== 1. a stalled build MUST be killed ==="
L="$W/t1.log"; : > "$L"
bash -c 'sleep 300' & FAKE=$!
bash "$WD" "$L" 3 "$FAKE" >/dev/null 2>&1; WRC=$?
wait "$FAKE" 2>/dev/null; FRC=$?
ok "watchdog exit code (2=killed)" 2 "$WRC"
ok "orchestrator death (137=SIGKILL)" 137 "$FRC"
ok "kill marker in the build log" 1 "$(grep -ac 'KILLED BY WATCHDOG' "$L")"

echo "=== 2. a build that ends on its own MUST exit 0, unkilled ==="
L="$W/t2.log"; : > "$L"
bash -c 'sleep 3' & FAKE=$!
bash "$WD" "$L" 0 "$FAKE" >/dev/null 2>&1; WRC=$?
wait "$FAKE" 2>/dev/null; FRC=$?
ok "watchdog exit code (0=finished)" 0 "$WRC"
ok "orchestrator exited cleanly" 0 "$FRC"

echo "=== 3. a dead PID MUST be refused, not silently ignored ==="
L="$W/t3.log"; : > "$L"
bash -c 'exit 0' & DEAD=$!; wait "$DEAD" 2>/dev/null
bash "$WD" "$L" 3 "$DEAD" >/dev/null 2>&1
ok "watchdog exit code (1=setup error)" 1 "$?"

echo "=== 4. THE F055 GUARD: fallback pattern vs the launcher's real argv ==="
# The entrypoint name is knowledge held in two files. Compose the argv the
# launcher actually produces and require the watchdog's pattern to match it.
ENTRY=$(grep -oE '\$REPO_ROOT/\.claude/sdk/[a-z_]+\.py" build' "$LAUNCH" |
        head -1 | grep -oE '[a-z_]+\.py')
PATTERN=$(grep -oE "pgrep -nf '[^']+'" "$WD" | head -1 |
          sed "s/pgrep -nf '//; s/'$//")
echo "  launcher entrypoint: ${ENTRY:-NONE}"
echo "  watchdog pattern:    ${PATTERN:-NONE}"
ARGV="/opt/conda/envs/x/bin/python /repo/.claude/sdk/${ENTRY} build --log /tmp/a.log"
# Non-vacuity first: an empty PATTERN makes grep -E match everything, so the
# assertion below would pass loudest exactly when the pgrep line was deleted.
ok "both names extracted (test is not vacuous)" "yes" \
   "$([ -n "$ENTRY" ] && [ -n "$PATTERN" ] && echo yes || echo no)"
ok "pattern matches the launcher's argv" 0 \
   "$(echo "$ARGV" | grep -qE "$PATTERN"; echo $?)"
# Keep the diagnosis itself as an assertion: the shipped-until-07-30 pattern
# must NOT match, or test 4 is measuring nothing.
ok "CONTRAST: the F055 pattern would not have matched" 1 \
   "$(echo "$ARGV" | grep -qE '\.claude/sdk/cli\.py build'; echo $?)"

echo "=== 5. the launcher MUST pass the PID, not rely on the pattern ==="
ok "launch_build.sh passes \$BUILD_PID to the watchdog" 1 \
   "$(grep -c 'watchdog.sh" "\$LOG" "\$STALE" "\$BUILD_PID"' "$LAUNCH")"
ok "launch_build.sh warns when attachment fails" 1 \
   "$(grep -c 'WATCHDOG DID NOT ATTACH' "$LAUNCH")"

# A fake orchestrator whose ARGV looks like the real one, so log-path
# discovery has something true to find. `exec -a` rewrites argv in place, so
# the PID captured here is the PID pgrep will see.
#
# Backgrounded DIRECTLY, never via `x=$(helper)`: a background job inside
# command substitution keeps the substitution's stdout open (the caller hangs
# until it exits) and belongs to the subshell, so `wait` in this shell cannot
# reap it and returns 127 instead of the 137 the test is looking for. Both
# traps cost the gw pipeline a probe rewrite; neither announces itself,
# because each still produces a plausible-looking result.
_ORCH_ARGV='/env/bin/python /repo/.claude/sdk/build.py build --log'

echo "=== 6. with TWO builds running, each watchdog guards ITS OWN ==="
LA="$W/t6a.log"; LB="$W/t6b.log"; : > "$LA"; : > "$LB"
bash -c "exec -a '$_ORCH_ARGV $LA' sleep 300" & PA=$!
bash -c "exec -a '$_ORCH_ARGV $LB' sleep 300" & PB=$!
sleep 1
bash "$WD" "$LA" 3 >/dev/null 2>&1; WRC=$?          # no PID: pure discovery
wait "$PA" 2>/dev/null; ARC=$?
ok "watchdog killed a build (rc 2)" 2 "$WRC"
ok "it killed the build that owns LA" 137 "$ARC"
ok "the OTHER build is untouched" "alive" \
   "$(kill -0 "$PB" 2>/dev/null && echo alive || echo dead)"
ok "it named A's PID, not B's" "$PA" \
   "$(grep -a 'Watching orchestrator PID' "$W/t6a.watchdog.log" | tail -1 |
      sed 's/.*PID //')"
kill -9 "$PB" 2>/dev/null; wait "$PB" 2>/dev/null

echo "=== 7. NEGATIVE CONTROL: stale=0 must NOT kill a quiet build ==="
# Without this, test 1 only shows the watchdog kills — not that it kills for
# the stated REASON. A killer that always fires would pass test 1.
L="$W/t7.log"; : > "$L"
bash -c 'sleep 300' & FAKE=$!
bash "$WD" "$L" 0 "$FAKE" >/dev/null 2>&1 & WD7=$!
sleep 6
ok "quiet build survives a DISABLED staleness check" "alive" \
   "$(kill -0 "$FAKE" 2>/dev/null && echo alive || echo dead)"
kill -9 "$WD7" "$FAKE" 2>/dev/null; wait "$WD7" "$FAKE" 2>/dev/null

echo
echo "  $pass passed, $fail failed"
[ "$fail" -eq 0 ]
