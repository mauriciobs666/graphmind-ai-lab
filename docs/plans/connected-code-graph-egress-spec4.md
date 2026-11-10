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
    not `usage`), or `--cwd` is not a directory.

  Refused: `lnk -> /`, `lnk -> ../..`, `lnk -> /src/sub`, an absolute link carrying the host path of the
  staged tree, `lnk2 -> deep/abs` where `deep/abs` is an absolute link, a dangling link, a missing
  `env.json`. Accepted: a plain subdirectory and a relative link that stays inside. The stager's own symlink
  rejection stays; this is defence in depth. Mutation of the tree after the check is the stager's and
  operator's concern.
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
  not `usage`; a launched record's `launcherArgv` image element equals `toolchain.imageId`, killed by a
  mutant that launches by name.
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

## 6. Open questions (not decided here)

1. **Extra forbidden ranges (§3.4).** Add `fec0::/10` and `100::/64` to the always-forbidden IPv6 list, and
   the cloud-metadata addresses `fd00:ec2::254` and `100.100.100.200`, which `privateAddress: true`
   currently admits (they fall in `fc00::/7` and `100.64.0.0/10`)? Security advisory recommends yes to all
   four (never opt-in-able); the owner's call.
2. **Runner-side wall-clock cap for indexers (§7.4, R-E10).** Today only P1-15's timeout bounds an indexer
   container. Is a `timeout` wrapper with the existing signal cleanup worth building now, or left to the
   measured-need rule? Owner's call; `devops` input.
