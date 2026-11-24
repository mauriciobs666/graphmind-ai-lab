# Tier-2 runbook: install the key holder on this host and probe it (D14, S20)

For a human with `sudo` on this WSL2 host. Plan: `docs/plans/llm-gateway.md` §3.4, §5.2 (FR-8
tier 2). The coordinator and every agent run nothing privileged: you run each `sudo` line, read
what it prints, and paste back what step 16 collects. **No step puts a key or a credential on a
command line, in your shell history or in the paste-back.** Every command below was checked against
`install.sh`'s usage text, `keyholder.py <subcommand> --help` and `ccg.broker.server --help`; the
bootstrap and `install.sh` subcommands were also rehearsed in `--test-root` mode without sudo.

## What only tier 2 shows

Tier 1 (the default suite) runs the real holder, service and broker as the same uid, with an injected
"trusted uid". It cannot show these. `tests/deploy/` (below) asserts each one on the real install.

| Claim | Why tier 1 cannot show it | Test file | Requirement |
|---|---|---|---|
| Code root `/opt/llmgateway` and every ancestor root-owned, not writable by the agent; installed bytes equal the reviewed sha | trusted uid is injected; no root `tar` | `test_deploy_code_root.py`, steps 1 and 14 | FR-3, FR-8, AC-21 |
| Installed `gateway.toml` and manifest are root-owned and agent-unwritable; the installed manifest is a byte copy of the tracked one | same | `test_deploy_code_root.py` | FR-3, AC-21 |
| `check-deployment` green against the artifact table on real `lstat` (root run) and, as the agent, blocked inside `/var/lib/llmholder` | needs root and a second uid | `test_deploy_check.py`, step 7 | FR-8, AC-5 |
| `/run/llmholder` 0711 (traverse only), socket 0666, state 0700 / key 0600 holder-owned | needs `llmholder` and systemd `RuntimeDirectory` | `test_deploy_check.py` | FR-8, AC-5 |
| The agent cannot read the holder's key, files, `/proc/<pid>/{environ,mem,maps,fd}`, cannot `ptrace` it, cannot `sudo`/`runuser` into it, and has no Docker access; each refusal paired with a positive control (a planted FAKE marker in an own-uid file and process IS found; `PTRACE_SEIZE` of an own-uid process works) | needs a real second uid | `test_deploy_probe.py` | FR-8, AC-5 |
| The holder is non-dumpable, core size 0, `NoNewPrivs`, empty capability set, seccomp, `ProtectSystem=strict` etc. in effect; unit verifies | needs systemd | `test_deploy_unit.py` | FR-8, AC-5 |
| `install.sh caller` writes the credential as the caller uid in a real home, refuses a planted symlink; rotation replaces both copies and the old credential stops working with no holder restart | needs root, a real home, a real uid drop | step 4, step 11, `test_deploy_holder_wire.py` | FR-8, AC-5 |
| A direct call to the real holder with no, wrong, swapped, replayed or stale credential is refused; a valid one is authenticated; junk frames are dropped without a hang; the WSL clock vs the 30 s window | needs the real socket and credentials | `test_deploy_holder_wire.py` | FR-8, AC-5 |
| The broker starts on the real root-owned chain and exits 69 on the tracked, a user-owned or a loose-credential input, 64 on own-uid `--holder-uid`; its argv and environ carry the credential path, never the content | needs a root-owned chain | `test_deploy_broker.py` | FR-8, FR-12, AC-21 |
| Real TLS to the real provider with a FAKE key: the call fails `auth`, no error text carries the key, and a host-wide search of everything the agent can read (files, `/proc` argv/environ, git history) finds the fake key nowhere while the control finds it where planted | needs the real holder account and egress | `test_deploy_hosted.py` (opt-in, step 13) | FR-8, AC-5 |
| `sudo -u llmholder ... log` asks for a password (no sudoers rule) | needs sudo | `test_deploy_probe.py`, step 12 | FR-8 |

**Not shown, not claimed:** a hostile same-uid process that reads the broker's or service's credential
can call the holder within its ceilings (the stated limitation, FR-8); a real key's behaviour
(billing, credit and spend-limit error shapes, thinking tokens) needs the human's own live smoke
(`needs_provider_key`, step 13b; skipped by default); AC-6's ledger arithmetic stays a tier-1 result.

## 0. Preflight (no sudo)

Rules for the whole session. **No agent session (Claude Code or otherwise) runs while you do steps 1-3 and
14**: the agent shares your uid, which owns `src.git`, `$SCRATCH` and the checkout, and could swap what root
reads. Run everything from a terminal with no agent attached. After a new shell, `wsl.exe --shutdown` or a
reboot, paste the ENV block again (nothing else carries over).

```bash
# ENV block (bash). Edit SHA only.
REPO=~/prg/graphmind-ai-lab
SHA='<the reviewed commit, 40 hex characters, given by the coordinator>'
SCRATCH=~/llmgw-tier2                 # fixed name, so teardown still finds it in a new shell
G='env GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null git'
inst() { sudo /bin/bash /opt/llmgateway/deploy/install.sh "$@"; }
mkdir -p -m 0700 "$SCRATCH"
```
Checks (each must print what is expected):
```bash
[[ $SHA =~ ^[0-9a-f]{40}$ ]] && git -C "$REPO" cat-file -e "$SHA^{commit}" && echo "sha ok"
git -C "$REPO" diff --quiet "$SHA" -- llm-gateway cpg && [ -z "$(git -C "$REPO" status --porcelain -- llm-gateway cpg)" ] && echo "checkout code == sha"
ls -ld /opt/llmgateway /opt/llmgateway.new /etc/llmgateway /var/lib/llmholder /run/llmholder 2>&1 | grep -v 'No such file'; getent passwd llmholder
id -nG; getent group docker
```
Expected: `sha ok`; `checkout code == sha` (the pytest and broker in steps 9-11 and 16 run **from this checkout**, so
`llm-gateway/` and `cpg/` must be exactly the reviewed sha, clean; other directories may differ); no `ls`/`getent
passwd` lines (nothing installed; otherwise do step 17 first).

**Docker (decide now).** If `id -nG` or `getent group docker` lists you, `test_agent_account_has_no_docker_access`
FAILS by design: docker-group access is root-equivalent (`docker run -v /var/lib/llmholder:/h` reads the holder's
files), so a dedicated holder account protects nothing while the agent can reach a container runtime
(`docs/reviews/llm-gateway-security.md` B2).

- **Running on with that test red does not show AC-5.** It leaves AC-5 and D14 **open** with a recorded blocker; it
  is not a pass with a footnote. The fake-key steps (13a/13b) may run either way; **step 13c (a real key) is barred
  until this test has passed on this installation.**
- Removing the account has costs you decide first: `cypher-mcp/docker-run.sh`, `cpg/infra/ccg-falkordb.sh`,
  `falkordb-dev` and anything else that runs docker as you stop working, and `wsl.exe --shutdown` (needed for the group
  change to reach every process) stops every running container and this session. Choose the steady state (stay out of
  `docker` and run those tools another way, or accept AC-5 open) **before any real key exists**. If the account later
  rejoins `docker`, treat a real key as exposed and rotate it at the provider.
- To remove: `sudo gpasswd -d "$USER" docker`, then `wsl.exe --shutdown` from Windows, reopen, paste the ENV block, and
  verify with **both** `getent group docker` (must not list you) and a fresh `id -nG` (must not list `docker`).
  `id -nG` in the old session keeps showing the group; the suite checks the live process groups, `/etc/group` and every
  container socket (`/var/run/docker.sock`, `/run/docker.sock`, containerd, podman, Docker Desktop's WSL sockets,
  `DOCKER_HOST`).

`sudo -n true` should print "a password is required" (no passwordless sudo). Baseline:
`cd "$REPO/llm-gateway" && ../model-bench/.venv/bin/pytest -q -p no:cacheprovider` -> all passed, deploy deselected
(1113 at this commit).

## 1. Bootstrap the code root (verify, `sudo tar`, read, move)

Everything up to the `sudo tar` is checked **before** root extracts anything. One subshell, so a failed check stops
it without closing your terminal:
```bash
(
  set -euo pipefail
  [[ $SHA =~ ^[0-9a-f]{40}$ ]] || { echo "SHA must be 40 hex characters"; exit 1; }
  [ ! -e /opt/llmgateway.new ] || { echo "/opt/llmgateway.new exists: remove it first (sudo rm -rf), then redo"; exit 1; }
  [ ! -e /opt/llmgateway ]     || { echo "/opt/llmgateway exists: tear down (step 17) first"; exit 1; }
  rm -rf "$SCRATCH/src.git" "$SCRATCH/ref"
  $G -c transfer.fsckObjects=true clone -q --no-local --bare "$REPO" "$SCRATCH/src.git"  # clean object copy: no .git/config, filters or hooks
  $G -C "$SCRATCH/src.git" fsck --full --strict
  $G -C "$SCRATCH/src.git" rev-parse --verify "$SHA^{commit}" >/dev/null
  # only regular files (100644/100755): no symlink (120000), submodule (160000) or odd mode reaches root's tar
  $G -C "$SCRATCH/src.git" ls-tree -r "$SHA" -- llm-gateway \
    | awk '$1!="100644" && $1!="100755"{print "refused mode: " $0; bad=1} END{exit bad}'
  mkdir "$SCRATCH/ref"
  $G -C "$SCRATCH/src.git" archive "$SHA" llm-gateway | tar -x --strip-components=1 -C "$SCRATCH/ref"   # unprivileged copy: the reference
  $G -C "$SCRATCH/src.git" show "$SHA:cpg/ccg/system.yaml" >"$SCRATCH/system.yaml"
  sudo install -d -m 0755 -o root -g root /opt/llmgateway.new
  $G -C "$SCRATCH/src.git" archive "$SHA" llm-gateway | sudo tar -x --no-same-owner --strip-components=1 -C /opt/llmgateway.new
  echo "extracted"
)
```
Expected: ends with `extracted`, no `refused mode`, no fsck output. Then read what root will run, and check the tree:
```bash
less /opt/llmgateway.new/deploy/install.sh                    # read it all (q to quit)
find /opt/llmgateway.new ! -type f ! -type d                  # expected: prints nothing
sudo find /opt/llmgateway.new \( ! -user root -o -perm /022 \) # expected: prints nothing
diff -r "$SCRATCH/ref" /opt/llmgateway.new && echo IDENTICAL  # expected: IDENTICAL
stat -c '%U:%G %a %n' /opt /opt/llmgateway.new                # expected: root:root 755 for both
```
Only if all four expectations hold: `sudo mv -T /opt/llmgateway.new /opt/llmgateway` (`-T` refuses to nest into an
existing directory). From here `inst` runs the installed copy, which refuses to run from anywhere else.

## 2. `holder-account`

```bash
inst --dry-run holder-account                                 # prints what it would do; read it
inst holder-account
getent passwd llmholder                                       # expected: uid < 1000, home /var/lib/llmholder, shell nologin
sudo ls -ln /var/lib/llmholder                                # expected: keys  registry (owned by llmholder's uid)
ls -l /etc/systemd/system/llmholder.service                   # expected: -rw-r--r-- root root
```
Expected last line of `inst holder-account`: `next: install.sh owner-files ...`. (The key is installed in step 13
only; nothing needs one yet.)

## 3. `owner-files` (preview and confirm: twice, once per file)

```bash
HUID=$(id -u llmholder)
cat >"$SCRATCH/gateway.toml" <<TOML
[deployment]
holder_uid = $HUID
holder_socket = "/run/llmholder/holder.sock"
local_endpoints = ["http://127.0.0.1:1234"]
TOML
sha256sum "$SCRATCH/system.yaml"
inst owner-files --gateway-toml "$SCRATCH/gateway.toml" --manifest "$SCRATCH/system.yaml"
```
`$SCRATCH/system.yaml` is the manifest as committed in the reviewed sha (`git show`), not the working tree. For each
file it prints `(new file)`, the **whole** file (control bytes shown as `^[`), its sha256, and asks
`Install ... [y/N]`. Read it; the manifest's printed sha256 must equal the `sha256sum` line above; answer `y` to
each. This base config has no hosted provider (the holder starts with none; a hosted provider is step 13).
`local_endpoints` is only an example of a loopback origin.

## 4. `caller` for the broker and the service

```bash
inst caller broker  "$(id -u)" "$HOME/.config/llmgateway/callers/broker.cred"
inst caller service "$(id -u)" "$HOME/.config/llmgateway/callers/service.cred"
stat -c '%U %a %h %n' ~/.config/llmgateway/callers/*.cred     # expected: you 600 1 (both)
sudo ls -ln /var/lib/llmholder/registry                       # expected: broker.secret service.secret, owned by llmholder's uid
```
Expected output: `keyholder: caller broker added`, `keyholder: caller service added`. Never `cat` a `.cred`.

Planted-symlink refusal (a real uid drop in a real home): 
```bash
mkdir "$SCRATCH/real" && ln -s "$SCRATCH/real" "$SCRATCH/link"
inst caller probe-sym "$(id -u)" "$SCRATCH/link/probe.cred"   # expected: `keyholder: refused: link: symlink or not a directory ...`, non-zero exit
ls -A "$SCRATCH/real"; sudo ls /var/lib/llmholder/registry    # expected: nothing in real/; no probe-sym.secret
```

## 5. `service-unit`

```bash
inst service-unit "$(id -u)" "$HOME"                          # expected: `unattended runs need: loginctl enable-linger <user>`
cat ~/.config/systemd/user/llmgateway.service                 # expected: ExecStart=/usr/bin/python3 -I /opt/llmgateway/bin/service.py
```

## 6. Start the holder (`systemd` enable)

```bash
sudo systemctl enable --now llmholder
systemctl status llmholder --no-pager | head -12              # expected: active (running), User llmholder
journalctl -u llmholder -n 10 --no-pager                      # expected: `keyholder: serving on /run/llmholder/holder.sock`
ls -ld /run/llmholder; ls -l /run/llmholder/holder.sock       # expected: drwx--x--x llmholder llmholder ; srw-rw-rw- llmholder llmholder
```
If it says `refusing to start: ...`, paste the journal line (it names the cause) and stop.

## 7. Full `check` as root

```bash
inst check --caller-cred "$(id -u):$HOME/.config/llmgateway/callers/broker.cred" \
           --caller-cred "$(id -u):$HOME/.config/llmgateway/callers/service.cred" \
           --user-unit   "$(id -u):$HOME/.config/systemd/user/llmgateway.service"; echo "rc=$?"
```
Expected: no output and `rc=0` (every artifact matches the plan §3.4 table).

## 8. Start the service (user unit)

```bash
sudo loginctl enable-linger "$USER"                           # WSL: the user manager must outlive the shell
systemctl --user daemon-reload && systemctl --user enable --now llmgateway
systemctl --user status llmgateway --no-pager | head -8       # expected: active (running)
```
If `systemctl --user` says it cannot connect to the bus, open a new shell after `enable-linger`.

## 9. Start the broker (second terminal; leave it running during step 10)

This runs code from the agent-writable checkout (verified equal to the sha in step 0). Drop any cached sudo ticket
first, so that code never runs in a sudo-warm shell:
```bash
sudo -k
cd "$REPO/cpg" && .venv/bin/python -m ccg.broker.server \
  --system /etc/llmgateway/ccg/system.yaml --require-owner-protected \
  --key-holder-socket /run/llmholder/holder.sock --holder-uid "$(id -u llmholder)"
```
Expected on stderr: `ccg-broker: listening on <socket>; audit <dir>` (the holder credential defaults to
`~/.config/llmgateway/callers/broker.cred`). The suite starts its own throwaway brokers and never touches this
one; stop it with Ctrl-C after step 10.

## 10. The tier-2 suite

```bash
sudo -k                                   # the suite runs agent-writable code: no warm sudo ticket
cd "$REPO/llm-gateway"
LLMGATEWAY_DEPLOY_REF="$SCRATCH/ref" LLMGATEWAY_DEPLOY_MANIFEST="$SCRATCH/system.yaml" \
  ../model-bench/.venv/bin/pytest -p no:cacheprovider -m deploy tests/deploy --tb=short 2>&1 | tee "$SCRATCH/pytest.txt"
```
`LLMGATEWAY_DEPLOY_REF` and `LLMGATEWAY_DEPLOY_MANIFEST` are what the byte-for-byte tests compare the install to
(both come from the reviewed sha, step 1); without them those two tests fail by design. Credentials are redacted in
every traceback style (the suite checks that); do not add `-l`/`--showlocals` anyway.
Expected at this commit: **0 failed, 5 skipped** (rotation x3 until step 11, the fake-key test and
`needs_provider_key` until step 13), the rest passed (the count is printed on the last line; paste it). No test may
skip because `cpg/.venv` is missing (that fails). The one expected red is the Docker test while step 0 leaves you in
`docker` or a container socket reachable: that is a recorded blocker, **AC-5/D14 stay open**. The host-independent
control tests (`test_deploy_selfcheck.py`) always run. A failure message names the path, mode or step to redo.

## 11. Rotation

```bash
cp -p ~/.config/llmgateway/callers/broker.cred  "$SCRATCH/old-broker.cred"
cp -p ~/.config/llmgateway/callers/service.cred "$SCRATCH/old-service.cred"
inst caller broker  "$(id -u)" "$HOME/.config/llmgateway/callers/broker.cred"  --rotate
inst caller service "$(id -u)" "$HOME/.config/llmgateway/callers/service.cred" --rotate
systemctl --user restart llmgateway                            # reads its credential once, at start
# restart the step-9 broker too (Ctrl-C, run it again): it also reads its credential once.
sudo -k                                                       # `inst` re-warmed sudo; drop it before agent-writable code
cd "$REPO/llm-gateway"
LLMGATEWAY_DEPLOY_REF="$SCRATCH/ref" LLMGATEWAY_DEPLOY_MANIFEST="$SCRATCH/system.yaml" \
LLMGATEWAY_DEPLOY_OLD_CRED_BROKER="$SCRATCH/old-broker.cred" LLMGATEWAY_DEPLOY_OLD_CRED_SERVICE="$SCRATCH/old-service.cred" \
  ../model-bench/.venv/bin/pytest -p no:cacheprovider -m deploy tests/deploy --tb=short -k "rotation or credential"
```
Expected: the three rotation tests PASS (old credential refused `bad_credential`, new one authenticated, no
holder restart). The `old-*.cred` copies are credentials: step 17 deletes them.

## 12. The holder's call log (needs your password; no sudoers rule exists)

```bash
sudo -u llmholder /usr/bin/python3 -I /opt/llmgateway/bin/keyholder.py log
```
Expected: one line per authenticated probe (`caller`, `run_id` `deploy-probe-*`, endpoint
`https://deploy-probe.invalid:443`, status 499, 0 tokens), no bodies, no key; unauthenticated, wrong-credential and
stale requests leave no line.

## 13. Optional hosted path (a FAKE key; real TLS to the provider, no cost)

13a. Add a hosted provider and call site (edit `$SCRATCH/gateway.toml`: append, replacing the model id with one your
account is pinned to), then re-run step 3 (the diff shows only your addition; confirm both prompts):
```toml
[holder]
window_seconds = 3600
window_max_tokens = 20000
window_max_released = 5
unmatched = "refuse"

[providers.anthropic]
kind = "anthropic"
base_url = "https://api.anthropic.com"
key_ref = "/var/lib/llmholder/keys/anthropic.key"
model = "claude-haiku-4-5"
pinned_models = ["claude-haiku-4-5"]
max_tokens_ceiling = 64

[scopes.smoke]

[call_sites.smoke]
component = "deploy.smoke"
scopes = ["smoke"]
chain = ["anthropic"]

[[run_bounds]]
match = "deploy-smoke-*"
max_tokens = 2000
```
13b. Install a FAKE key from a file (never typed on a command line), then run the opt-in test:
```bash
( umask 077; printf 'sk-ant-FAKE-%s\n' "$(openssl rand -hex 16)" >"$SCRATCH/fake.key" )
inst holder-account --key-from-stdin anthropic <"$SCRATCH/fake.key"   # expected: `installed key for anthropic`
systemctl --user restart llmgateway
sudo -k
cd "$REPO/llm-gateway"
LLMGATEWAY_DEPLOY_FAKE_KEY_FILE="$SCRATCH/fake.key" LLMGATEWAY_DEPLOY_SMOKE_CALL_SITE=smoke LLMGATEWAY_DEPLOY_SMOKE_RUN_ID=deploy-smoke-1 \
  ../model-bench/.venv/bin/pytest -p no:cacheprovider -m deploy tests/deploy --tb=short -k "fake_key"
```
Expected: PASS (the provider answers 401, class `auth`, nothing leaks, the host-wide search finds the fake key
only where the test plants it; the file you installed it from is the one excluded path). It SKIPS with
"no usable egress" when offline, which does **not** show the hosted half of AC-5: rerun online until it PASSES. The
search is long (minutes: `/tmp` and `$HOME` are large) and FAILS rather than passes if it cannot finish. The fake key is
sent to the real provider as `x-api-key` (it is not a secret). Delete the key afterwards:
`sudo rm /var/lib/llmholder/keys/anthropic.key; shred -u "$SCRATCH/fake.key"` (`shred` is advisory on ext4/WSL).

13c. Real key (your decision; costs a few tokens). **Barred until `test_agent_account_has_no_docker_access` has PASSED
on this installation (step 10 output), and only after you decided the Docker steady state in step 0.** With the account in
`docker` the agent reads this key as root-equivalent. Type it hidden, never as an argument, then opt in the smoke:
```bash
read -rs -p 'provider key: ' K; echo
printf '%s\n' "$K" | inst holder-account --key-from-stdin anthropic; unset K
LLMGATEWAY_DEPLOY_SMOKE_CALL_SITE=smoke LLMGATEWAY_DEPLOY_SMOKE_RUN_ID=deploy-smoke-2 \
  ../model-bench/.venv/bin/pytest -p no:cacheprovider -m needs_provider_key tests/deploy --tb=short
```
Expected: PASS with `hosted` true; it SKIPS if the fake-key file variable is still exported or the two variables are unset.

## 14. Negative rehearsal: a writable code path stops the installer and the holder

No agent session may run during this step (the group-write bit lets any other uid replace files in the code root while
it is set). `g+w`, not `go+w`: the agent is not in group root, and `-perm /022` still trips.
```bash
sudo chmod g+w /opt/llmgateway/bin
inst check; echo "rc=$?"                                      # expected: `the code root is not protected: ...`, rc=1 (first, before anything restarts)
sudo systemctl restart llmholder; sleep 2; systemctl is-active llmholder   # expected: not active (the holder refuses to start)
sudo journalctl -u llmholder -n 3 --no-pager                  # expected: `refusing to start: code root is not protected`
sudo chmod 0755 /opt/llmgateway/bin
diff -r "$SCRATCH/ref" /opt/llmgateway && echo IDENTICAL      # expected: IDENTICAL (chmod does not undo a planted file; this does)
inst check; echo "rc=$?"                                      # expected: no `not protected` line (run dir / socket may show `missing` while the holder is stopped)
sudo systemctl reset-failed llmholder; sudo systemctl restart llmholder; systemctl is-active llmholder   # expected: active
```

## 15. Re-run step 7 and step 10 after the restarts

Expected: unchanged (`/run/llmholder` is recreated by systemd at each start; the test asserts its mode and the socket's).

## 16. What to paste back to the coordinator

Two blocks: the root-run collection, then (after dropping the sudo ticket) the suite from the checkout.
```bash
{
  echo "sha=$SHA  date=$(date -u +%FT%TZ)  host=$(uname -r)"; id -nG; getent group docker
  inst check --caller-cred "$(id -u):$HOME/.config/llmgateway/callers/broker.cred" --caller-cred "$(id -u):$HOME/.config/llmgateway/callers/service.cred" --user-unit "$(id -u):$HOME/.config/systemd/user/llmgateway.service"; echo "check rc=$?"
  sudo ls -ln /var/lib/llmholder /var/lib/llmholder/keys /var/lib/llmholder/registry
  systemctl show llmholder -p ActiveState,SubState,NRestarts,User,NoNewPrivileges,ProtectSystem,ProtectProc
  sudo -u llmholder /usr/bin/python3 -I /opt/llmgateway/bin/keyholder.py log
} 2>&1 | tee "$SCRATCH/report.txt"
sudo -k
cd "$REPO/llm-gateway" && LLMGATEWAY_DEPLOY_REF="$SCRATCH/ref" LLMGATEWAY_DEPLOY_MANIFEST="$SCRATCH/system.yaml" \
  ../model-bench/.venv/bin/pytest -p no:cacheprovider -m deploy tests/deploy --tb=short -rA -q 2>&1 | tail -n 150 | tee -a "$SCRATCH/report.txt"
```
Paste `report.txt`: it holds names, modes, counts and pass/fail lines, never a key or a credential (the `log` is
holder metadata only). Also paste the step-0 Docker decision, any step whose "expected" line did not hold, and the
`WSL clock vs Windows host` line the clock test prints (it prints it always). **State plainly whether
`test_agent_account_has_no_docker_access` and `test_a_fake_key_hosted_call...` passed**: AC-5 is shown only if
both did.

## 17. Teardown / rollback (reverses 1 to 16; safe to run partially, also from a new shell)

Paste the ENV block of step 0 first (it defines `SCRATCH`), then:
```bash
: "${SCRATCH:?paste the ENV block of step 0 first}"
cd ~ ; systemctl --user disable --now llmgateway 2>/dev/null
rm -f ~/.config/systemd/user/llmgateway.service ~/.config/llmgateway/callers/{broker,service}.cred
rmdir ~/.config/llmgateway/callers ~/.config/llmgateway 2>/dev/null; systemctl --user daemon-reload
rm -rf "${XDG_STATE_HOME:-$HOME/.local/state}/llmgateway"       # the service's call records (advisory, same-uid)
sudo systemctl disable --now llmholder 2>/dev/null
sudo systemctl reset-failed llmholder 2>/dev/null
sudo rm -f /etc/systemd/system/llmholder.service && sudo systemctl daemon-reload
sudo rm -rf /var/lib/llmholder /etc/llmgateway /opt/llmgateway /opt/llmgateway.new     # deletes the holder's key, ledger, log and registry
sudo userdel llmholder 2>/dev/null; getent group llmholder >/dev/null && sudo groupdel llmholder
# only if YOU enabled it in step 8 and nothing else needs it:  sudo loginctl disable-linger "$USER"
shred -u "$SCRATCH"/*.cred "$SCRATCH"/fake.key 2>/dev/null; rm -rf "$SCRATCH"
sudo -k
unset SCRATCH G K LLMGATEWAY_DEPLOY_OLD_CRED_BROKER LLMGATEWAY_DEPLOY_OLD_CRED_SERVICE
ls -ld /opt/llmgateway /etc/llmgateway /var/lib/llmholder /run/llmholder 2>&1 | grep -vc 'No such file'   # expected: 0
getent passwd llmholder; getent group llmholder                 # expected: nothing
```
Stop the step-9 broker (Ctrl-C) first. Its audit directory (`cpg/.audit` by default) and
`~/.local/state/ccg/broker/` are yours and are not removed here. **Residue you must handle:** the holder's and
service's lines stay in the systemd journals (names and run ids, no key; `sudo journalctl --vacuum-time=1s` if you
want them gone); `shred` is advisory on ext4/WSL; and **if step 13c installed a real key, rotate that key at the
provider now**, because it sat on a disk this teardown cannot overwrite, and (if the account was ever in `docker`
while it was installed) was readable by the agent.
