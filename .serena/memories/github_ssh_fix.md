# GitHub SSH CheckHostIP Fix

On this machine (EL8, OpenSSH 8.0), git-over-SSH to github.com intermittently
fails with "Host key verification failed" in non-interactive sessions.
Root cause: `CheckHostIP` — GitHub serves from many IPs, and an unseen IP
triggers an interactive confirmation that fails without a tty
(`read_passphrase: can't open /dev/tty`).

Fixed 2026-07-17 by adding `CheckHostIP no` to the `Host github.com` block
in `~/.ssh/config`. If GitHub SSH fails again, verify that line is still
present before other debugging; `GIT_SSH_COMMAND="ssh -o CheckHostIP=no"` is
the ad-hoc workaround.
