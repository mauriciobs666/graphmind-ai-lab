# Connected code graph — egress spec, delta after P1-29 and P1-48b

> **Status:** active · **Owner:** `architect` · **Tracks:** C-1001 (P1) · **Extends:** `docs/plans/connected-code-graph-egress-spec3.md` · **Reviews:** `docs/reviews/connected-code-graph-p1-29.md`, `docs/reviews/connected-code-graph-p1-29-security.md`, `docs/reviews/connected-code-graph-p1-48b.md`, `docs/reviews/connected-code-graph-p1-48b-security.md`

This states only what changes against `docs/plans/connected-code-graph-egress-spec3.md` (spec3), which stays
authoritative everywhere not named here. Section numbers are spec3's. Sources: the four reviews above and
the stakeholder rulings relayed by `teco`. Where a ruling is not yet in the code (it may change in a fix
round), the text below is the contract the code is brought to.

## 1. §7.4: `--project-version`, and the grammar's small corrections

**`--project-version <v>`** joins the indexer grammar: `scip-python` only, **mandatory** for it, at most once,
as two separate elements (`--project-version=v` is `forbidden-flag`). `<v>` full-matches the `--project-name`
pattern `^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$`. On `scip-typescript` it is `forbidden-flag`. A missing one is
`usage`, like `--cwd` and `--environment`. The stager (P1-15) supplies a deterministic value: the source
commit's sha or a digest of the staged tree, never a timestamp.
*Why:* `scip-python` 0.6.8 exits 1 with neither a git revision nor this flag (both reviewers reproduced it,
`--network none`, read-only `/src`); staging `.git` runs `git` in the sandbox against a target-controlled config
and widens the stager's reject-list. The flag adds no execution surface. It is mandatory, not optional,
because no `.git` is ever staged, so a tree that carries its own revision never arises. With it,
`scip-python` 0.6.8 runs no `git`, so spec3 R-E9's `git rev-parse HEAD` child is gone.

Other §7.4 changes:
- **Precedence.** A forbidden element outranks a missing mandatory one. An empty argv is `forbidden-flag`
  with hit `(missing index)`, as joern's is `(missing --script)`. Spec3's `usage` line in §7.5 applies only
  when nothing is forbidden. The symlink and containment refusal below is forbidden-class, so it too
  outranks a missing mandatory element: an escaping `--cwd` with `--project-version` missing is
  `forbidden-flag`, and the containment check runs after the grammar and before the `usage` refusal.
- **`<p>`.** `/src/` alone (zero components) is accepted. Each component of a `--cwd` and `--environment`
  path must also match `^[A-Za-z0-9._@+-]+$` (and is not `.` or `..`); anything else is `forbidden-flag`.
- **Symlink containment (runner obligation).** After the grammar check and before the `usage` refusal, the
  runner walks each `--cwd`/`--environment` value inside the staged tree, component by component, following
  every link it traverses, including links reached through the target of a relative link. It refuses
  (`forbidden-flag`, exit 64, nothing launched) when:
  - any traversed link has an absolute target, because inside the container it resolves against the image,
    not the tree;
  - the walk leaves the staged tree (`..` past its root);
  - the value cannot be resolved (a dangling link; a missing `env.json`, which is therefore `forbidden-flag`,
    not `usage`), or `--cwd` is not a directory;
  - `--environment` does not resolve to a regular file (a directory, a FIFO, a device): a non-regular file is
    `forbidden-flag`, the same class and precedence as the rest of this bullet. The test is made without
    opening the path, so a FIFO cannot hang the runner.

  Refused: `lnk -> /`, `lnk -> ../..`, `lnk -> /src/sub`, an absolute link carrying the host path of the
  staged tree, `lnk2 -> deep/abs` where `deep/abs` is an absolute link, a dangling link, a missing
  `env.json`, an `--environment` that is a directory or a FIFO (a device is not tested: it needs root). Accepted: a regular-file `--environment`, a plain subdirectory and a relative link that stays inside. The stager's own symlink
  rejection stays; this is defence in depth. Mutation of the tree after the check is the stager's and
  operator's concern.
- **Indexer wall-clock watchdog (narrows spec3 §7.4's last sentence and R-E10: wall-clock time is now limited
  for indexer entries; memory and CPU are not).** The runner holds a constant, `SBX_INDEXER_WALL_S=1800` (30
  minutes), not read from the environment, applied to indexer entries only (Joern frontends are unchanged).
  On expiry the runner takes the same steps as `sbx_on_signal`: KILL the Docker client, `sbx_remove_own` until
  none remain, `sbx_cleanup`. It is a variant, not a reuse, because that handler exits `128+n` without a
  record. The watchdog then writes the one sandbox record of the invocation, unrefused, and exits 124.
  - **Exception to spec3 §5.3 and §7.8.** A signalled invocation writes no record; the watchdog expiry is not
    a caller signal and is the one exception. INT, TERM and HUP still write none. §5.3's `exitCode` row admits
    124 for it. **First wins:** each handler (the watchdog's and the INT, TERM and HUP one) disarms the other
    on entry, so a signalled run never also writes a watchdog record.
  - A bare `timeout` is not the mechanism: a hung node may ignore the TERM that `docker run` proxies.
  - Exit 124 is a non-zero indexer exit, so the no-artifact rule below applies, and a 124 record fails
    `check_sandbox_ids` (spec3 §5.4 requires `exitCode: 0`).
  - **Outer bound.** P1-15's own timeout must exceed `SBX_INDEXER_WALL_S` plus 60 s, and P1-15 treats runner
    exit 124 as a timeout failure. The margin covers `sbx_remove_own`, whose measured worst case is about 25 s without removals and about
    40 s with them, with headroom; a hung Docker daemon is out of scope. At or below the sum the inner watchdog never fires and a TERM-ing outer path yields 143 with no
    record. The timer is cancelled on normal exit, so a stale timer never signals a runner past its trap.
  - **Owner and gates:** a follow-up to P1-48b, as the plan rows say (`docs/plans/connected-code-graph-p1.md`,
    P1-48b row; `docs/plans/connected-code-graph-p1-extraction.md`, P1-48b and P1-15 sections).
- **Image by ID, every container launch.** The runner launches the image by the `sha256:` ID it inspected and
  recorded (`sandbox.toolchain.imageId`), not by name, so the checked image is the one that runs; this covers
  every container launch and narrows §7.1's "on `ccg-toolchain:<hash>`". `launcherArgv` carries the ID. The
  `ccg-toolchain-indexer=1` label guards against a wrong image only; it is not an integrity control.
- **Counts.** "13 forbidden, 3 allowed" becomes: 13 Joern forbidden entries and 3 allowed (Joern v4.0.579),
  plus 4 `scip-typescript` forbidden entries (`--pnpm-workspaces`, `--yarn-workspaces`,
  `--yarn-berry-workspaces`, `--infer-tsconfig`), 17 forbidden in all. The files stay normative.
- **§7.8 gains: failure means no artifact.** A non-zero indexer exit means there is no artifact: the tool may
  leave a partial `<auditId>.scip` in `/out`, and P1-15 must not read it.

## 2. §10.1, test (22) and the usage list

- Launches: `scip-python` cases gain `--project-version v1` (and the reordered form); `scip-typescript`
  cases unchanged; add `--cwd /src/`.
- `usage`: `scip-python` without `--project-version` (beside the two existing cases).
- `forbidden-flag`: `--project-version` on `scip-typescript`; a repeat; the `--project-version=x` form;
  values empty, `-x`, `--x`, 65 characters; empty argv with hit `(missing index)`; a forbidden element
  together with a missing mandatory one (forbidden wins); path components with a newline, `*` or a space.
- New: every refused and accepted symlink case of §1's symlink-containment bullet, as `forbidden-flag` and
  launch respectively; an escaping `--cwd` together with a missing `--project-version` is `forbidden-flag`,
  not `usage`; `--environment` naming a directory, and naming a FIFO (the refusal returns promptly, the test
  carries a timeout), each `forbidden-flag`, with a regular-file `--environment` as the positive control that
  launches; a launched record's `launcherArgv` image element equals `toolchain.imageId`, killed by a
  mutant that launches by name.
- New: the watchdog. The runner is sourced as a library (its `BASH_SOURCE == $0` main guard) and the test
  reassigns `SBX_INDEXER_WALL_S` to a few seconds after sourcing; the hook is that reassignment, because a
  script run as `offline-run.sh` assigns the constant unconditionally and reads no environment variable, so
  production has no override. A planted indexer that sleeps and ignores TERM is KILLed, no container labelled
  with the `auditId` remains, the record has `exitCode: 124` and `refused: false`, and the runner exits 124.
  Mutants: the cap applied to a Joern frontend; the watchdog sending TERM instead of KILL (killed by the
  hang under the suite's timeout); the removal skipped.
- New: first-wins. With the watchdog armed, an INT is delivered and the watchdog fires while the INT
  handler is still removing containers: exit 130, no record; mutant: a handler that does not disarm the watchdog (it exits 124 and
  writes a record).
- New, in the test round (not the watchdog step): the environment cannot set the cap. In library mode the
  test exports `SBX_INDEXER_WALL_S=3`, sources the runner, and asserts the constant is still the literal
  `1800`; its own mutant row in `mutants.sh` reads `${SBX_INDEXER_WALL_S:-1800}`. A normal invocation cannot
  kill this mutant, because `env -i` drops the variable first; that is a second guard, not the test.
- `scip-python`'s launch is asserted (not 126/127, exit 0, non-empty `.scip`) now that the flag admits success.

## 3. §2.2: reply frame size

A reply frame carries the response body base64-encoded in its JSON. The outbound cap is
`4 * ceil(MAX_RESPONSE_BYTES / 3) + 64 KiB`, not `MAX_RESPONSE_BYTES + 64 KiB` (a 16 MiB response that §2.3
admits would otherwise be undeliverable). The inbound cap is unchanged.
Also: **one request and one reply per connection** (§2.2 said only "the connection").

## 4. §2.0 and §5.1: strictness and audit-line form

- **§2.0 ancestor rule stays strict in production.** The sticky-directory relaxation is not admitted; tests
  inject a relaxed predicate (and a stat reader, so the owner conjunct can be tested with a foreign uid).
- **Caller input never invalidates a record** (§5.1), with its cases stated:
  - a string that is not UTF-8 encodable (a lone surrogate), in a key or a value, makes the frame
    malformed (§2.2): the connection closes, no record, no decision;
  - an over-long or otherwise unfitting `payloadFields` key is recorded as `payloadFields: null` plus the
    SHA-256 of the key list in `rejected["payloadFields"]`;
  - `target.path` must also match the manifest `paths` pattern, else `null` plus digest;
  - U+2028 and similar are harmless because the line is `ensure_ascii`.
- **The audit line is `ensure_ascii` JSON** (still UTF-8, valid JSON; no raw U+2028/U+0085 reaches a
  line-splitting reader).
- **`repoId` in the audit schema** is capped at 256 characters; a loaded id (manifest cap 200) never reaches it.

## 5. §2.1, §2.5, §3.3, §3.4, §5.2: sentences spec3 lacks

- **§2.1:** `RunGrant` carries `auditId` (the `policy` record's), which an overridden refresh's report names:
  `RunGrant {runId, token, effective, systemLocalOnly, auditId}`.
- **§2.5 / §5.2:** every durable `decision` allow is followed by exactly one `close`. If the exchange (TLS
  context creation included) raises, the `close` carries `closeReason = upstream-error` and the caller gets
  that reply, never EOF.
- **§3.3:** `requestMode` for a declaration that fails sanitisation (a `codebases` that is junk, or outside the
  run) falls back to the run's full codebases. It is informational; the same request is an R0 deny.
  Relatedly, junk `codebases` in a `vulndb` run still gives `codebases-in-vulndb` (R0 check 6 precedes 7).
- **§3.4:** an IP-literal host skips the resolver but goes through the full forbidden-set check.
- **§5.2 `payloadFields`:** for `repo-names-and-identity` (R4) the list holds `identity.subject` and
  `identity.issuer` in place of `identity`; an allow with no body records `[]`, a deny `null`.

## 6. Open question (not decided here)

**Extra forbidden ranges (§3.4).** Add `fec0::/10` and `100::/64` to the always-forbidden IPv6 list, and
   the cloud-metadata addresses `fd00:ec2::254` and `100.100.100.200`, which `privateAddress: true`
   currently admits (they fall in `fc00::/7` and `100.64.0.0/10`)? Security advisory recommends yes to all
   four (never opt-in-able); the owner's call.
