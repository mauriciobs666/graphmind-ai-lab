# Review: C-1001 P1 units P1-48 (instance, indexer image, throwaway) and P1-02 (bundle-at-rest, key custody)

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (P1)

## Pass 1 (2026-11-09)

### Scope and verdict

I reviewed the uncommitted working-tree diffs of `cpg/.gitignore`, `cpg/README.md` (one row),
`cpg/infra/README.md`, `cpg/infra/ccg-falkordb.sh`, `cpg/infra/tests/test-ccg-falkordb.sh`,
`cpg/infra/toolchain/Dockerfile` and `build.sh`, and the new `cpg/ccg/atrest.py` and
`cpg/tests/test_atrest.py`. The baseline was the steps `## P1-48` and `## P1-02` in
`docs/plans/connected-code-graph-p1-extraction.md`, plus index `docs/plans/connected-code-graph-p1.md`
(§3, §6, §8, D9), `-bundle-spec3.md` §5.17 and §13 F-1, `-bundle-spec5.md` §2 and §9 (working-tree
text), `-p1-graph.md` §7, `-egress-spec3.md` §7.4, the security advisory S5-6/S5-7 and root `REVIEW.md`.
There is no `cpg/REVIEW.md`.

**Verdict: needs changes.** There is one blocker. `ensure_private` follows symlinks and, by default,
admits both areas, so a bundle-named symlink reads the HMAC key (B1). That defeats the containment
argument behind moving the key out of the repo. Six majors follow. The ACL drift rule contradicts
`-p1-graph.md` §7's user names (M1). The `CCG_CREDENTIALS_DIR` override cannot be read or written
(M2). The second published test key is not refused (M3). The indexer path has no runner support and no
regression test (M4). The guard test (c) is a bound, not proof (M5). The default bundle writer has no
owner (M6). The Dockerfile split and the throwaway refusals are sound.

CPG: considered, not relevant — no `cpg_cpg` graph is loaded for this component, and the reviewed
units are shell, Dockerfile and one new Python module, all of which I read in full.

**Runs (observed).**
- `cpg/` default suite, `.venv/bin/python -m pytest -q -p no:cacheprovider`: `4207 passed, 1 skipped, 10 deselected`
  in 150 s. The skip is the opt-in slow `test_rule_pack_leaf_mutation.py:756`, which is unrelated and
  reported unrun per D9. `test_atrest.py` was green inside that run.
- `cpg/infra/tests/test-ccg-falkordb.sh`, stubbed throughout: `checks: 510 passed, 0 failed`, 240
  coverage-probe runs.
- The real script's `--throwaway {up,status,down}` with port 6379 and with port 6390 were all refused
  before any Docker call (exit 1). `--throwaway reset-p0 --yes` was refused. A `CCG_FALKORDB_NAME=falkordb-dev`
  override was refused. No container was started.
- P1-48 (4), reproduced in the scratchpad (`CCG_AUDIT_DIR` there, not `cpg/.audit/`):
  `CCG_TOOLCHAIN_IMAGE=ccg-toolchain-indexer:73b9fb76bd2a offline-run.sh --backend container --mode offline --frontend joern --script-file scipver.sc`
  printed `SCIPVER 0.6.0`, exit 0. I found no record of the devops' own run in `cpg/.audit/sandbox.jsonl`
  (563 lines, no indexer image).
- The guard mutation on a scratch copy is described under M5.

### Findings

#### B1 (blocker) — `ensure_private` follows in-area symlinks, and by default admits the key area for a bundle path

*Verified by running it (scratchpad copy of the area layout).*
- `_inside` and `ensure_private` (`cpg/ccg/atrest.py:68-122`) work on `os.path.realpath`, so every
  component `_check_one` lstat's is already resolved. The `S_ISLNK` branch (`:98`) can never fire. A
  dangling link `bundles/shop/dang.bundle.jsonl -> bundles/shop/nowhere` is accepted, and
  `write_private` writes through it, leaving the link in place. The comment "a dangling link is refused"
  is false.
- `_area` with no roots (the production default), or with both, admits either area. A symlink
  `bundles/shop/u1.b9.bundle.jsonl -> keys/shop.hmac` passes `read_private(..., root=work, keys_root=keys)`
  and returns the key bytes (observed: `b'47cea9db'...`).
- **Why it matters.** S5-6's threat is a uid-1000 container with a `.ccg-work` ancestor mounted. Such
  a container can plant that link with any target text. The next host read of that "bundle" loads the
  key into `bundle.loads`, whose rejection paths echo values (S5-1). So "outside the tree, none of these
  reaches the key by construction" (`cpg/infra/README.md`, Bundle-at-rest) does not hold.
- **Fix (for `devops`).**
  - Pin each accessor to exactly one area: bundle functions to `WORK_ROOT`, key and credential
    functions to the key area. Drop the "either area" default.
  - Refuse any symlink on the *lexical* path from the area root to the leaf. Do this by lstat'ing each
    component of the normalized, unresolved path, and refuse a lexical `..` outright.
  - In `read_private`, open with `O_RDONLY|O_NOFOLLOW` and check owner and mode with `fstat` on that fd.
    That also closes the check-then-read window.
  - Add tests with these planted inputs: an in-area link to another in-area file, a dangling link, and
    a bundle-dir link into the key area. Each must be refused.

#### M1 (major) — the `status` ACL rule contradicts `-p1-graph.md` §7's per-system user names

*Traced: script `:115`, the test's S-P1 case, `docs/plans/connected-code-graph-p1-graph.md:295-314,325-329`.*
- `ACL_ALLOWED_RE` is `^(default|ccg_build|ccg_verdict|ccg_engine|ccg_raw_[a-z][a-z0-9_]*)$`.
  `-p1-graph.md` §7 mints `ccg_build_<s>`, `ccg_verdict_<s>`, `ccg_engine_<s>` and `ccg_raw_<s>`. It
  says a shared bare `ccg_build` would let one system overwrite another's grants.
- With P1-19's real users, `status` therefore reports drift on the P1 steady state. It also rejects valid
  system names: the whitelist `^[a-z0-9]+(?:[-_][a-z0-9]+)*$` admits a leading digit and `-`, and the
  regex rejects both.
- The plan (P1-48 (2) "the four `ccg_*` users"), `-p1-linking.md:134` and `-p1-answers.md` use bare
  names, so this is a cross-document conflict.
- **Fix.** `graph-dba`/`architect` rule on one naming (here §7, the only source that reasons about it).
  Then the regex becomes `^(default|ccg_(build|verdict|engine|raw)_<whitelist>)$` and the S-P1 case uses
  `ccg_build_shop` and friends. Add a drift case for a bare `ccg_build` if per-system naming wins.

#### M2 (major) — `CCG_CREDENTIALS_DIR` paths are refused by `read_private`/`write_private`

*Verified by running it.* With `CCG_CREDENTIALS_DIR=<dir>`, `acl_path("shop")` returns `<dir>/shop.acl`.
But `write_private(p, …)` and `read_private(p)` both raise "outside the work area / key area", because
`_area` without roots knows only `WORK_ROOT` and `KEY_ROOT`. Passing `keys_root=atrest.credentials_dir()`
works. The README documents the bare form ("Written with `write_private`, read with `read_private`"),
and P1-02 (e) says clients read "through `CCG_CREDENTIALS_DIR`". The mechanism therefore fails for the
one case the variable exists for. It fails closed, so nothing leaks.
`test_acl_path_default_and_env_override` checks only the returned path.
- **Fix.** Add `read_acl(system)`/`write_acl(system, data)`, pinned to `credentials_dir()` (this fits B1's
  one-area-per-accessor fix). Add a round-trip test with the variable set.

#### M3 (major) — only one of the two published test keys is refused; its vector ID is stale; the refusal has two homes

*Traced and recomputed.*
- The working-tree `-bundle-spec5.md` §9 publishes two keys. K1 is `00…1f`, whose `bindKeyCheck` is
  **EH-30** (`aca11439…`; EH-24 is now a path vector). K2 is `20…3f`, whose check is
  `4d6dd097…02414ac` (§9.4; I recomputed both with `hmac`).
- §2 requires refusing "either published test key". `atrest` refuses K1 only, and names it EH-24
  (`atrest.py:39-41`, `test_atrest.py:277-288`).
- §2 also puts the refusal in `ccg.valuehash.system_key`, with an `allow_test_key=True` test bypass.
  `atrest.hmac_key` states "no test-harness bypass", so two homes carry contradictory semantics.
- **Fix.**
  - `devops`: refuse a set of both checks. The tests read both expected hex values from the spec5 text,
    as P1-07 does, never from the constant.
  - `architect` (spec5 §2): state that `system_key` calls `atrest.hmac_key`, and that `allow_test_key`
    covers only conformance tests that pass a key directly. That is the README's own wording.

#### M4 (major) — the indexer has no runner support, and P1-48 (4) has no regression test

*Confirmed.*
- `SBX_FRONTENDS` (`cpg/sandbox/offline-run.sh:51`) and `-egress-spec3.md` §7.4 hold only Joern
  frontends, mapped to `$JOERN_HOME`. `--frontend scip-typescript` is `unknown-frontend`, and no step's
  file list names `offline-run.sh` (P1-15 lists only `cpg/extractors/indexers/*`).
- The README's workaround works: my run printed `SCIPVER 0.6.0`. But it works because a `joern` script
  can exec any image binary (`scala.sys.process`). So P1-15's "argv check asserts no `npm`/`pip`
  invocation" (`-p1-ml.md:151`) cannot be enforced at the runner on that route. The image also ships
  `npm`, `npx`, `pip` and `pip3` (observed with `command -v`).
- **Owner.** `architect` folds an indexer entry into `-egress-spec3.md` §7.4 in its open revision
  (P1-28, currently "needs changes"): names, executable map outside `$JOERN_HOME`, argv grammar, and
  forbidden `npm`/`pip`/`uv`. P1-15's file list then gains `cpg/sandbox/offline-run.sh`,
  `allowed-flags.txt`/`forbidden-flags.txt` and the sandbox tests. Until then, P1-48 (4) is met only by
  the workaround, and P1-15 must not adopt it.
- **Test gap.** `cpg/infra/toolchain/tests/test_toolchain.sh` is not extended. Nothing guards the
  indexer image's claims: same hash, `ccg-versions` passes offline, no `/opt/truth` and no other truth
  tool. I checked them by hand (all held).

#### M5 (major) — guard test (c): the positive controls bite, but the collector is a bound

*Verified with a mutation and a coverage probe on a scratch copy (`PYTHONPATH` on the copy).*
- **Mutation.** I made `pattern_escapes` return `False` after the NUL check. 10 of 14 guard tests went
  red, both positive controls included.
- **Probe (Appendix A).** These channels go **unflagged**: `--allowed-tools "Read(cpg/**)"` (an alias
  `claude --help` lists), `--allowedTools=Read(cpg/**)`, and `--settings`/`--mcp-config`/`--add-dir`
  (not parsed). An MCP tool `def read(target: str)` is unflagged (a name outside `PATHLIKE_PARAMS`),
  and so is any `.mcp.json` value without the substring `ccg-work` (e.g. `/…/cpg`, `cpg/`). A value
  containing `keys` is exempt.
- Root `REVIEW.md` says a filter that matches only the known sites is a bound. Today the real sources
  hold no patterns, so the real-sources test is a tripwire for future edits, and those edits arrive
  through exactly these channels.
- **Fix.**
  - Parse every flag spelling: alias, `=` form, `--settings`/`--mcp-config` files, `--add-dir` as
    `<dir>/**`.
  - Fail closed on any `str` parameter of an MCP tool outside an explicit safe-name allow-set.
  - Judge every `.mcp.json` `ccg` string value, without the `ccg-work`/`keys` filter.
  - Make Appendix A's probe the parametrized test.

#### M6 (major) — no step owns moving extractors off `ccg.bundle.write_bundle`

*Traced.* `ccg.bundle.write_bundle` (`cpg/ccg/bundle.py:93-99`) writes `<target>.tmp` with the umask's
mode. The plan preamble (`-p1-extraction.md:12`) still says "Every extractor writes bundles through
`ccg.bundle.write_bundle`". The README marks it umask-moded, but no step's done-condition names
`atrest.write_bundle`. P1-04 (the first bundle writer, `graph-dba`), P1-10, P1-11, P1-12 and P1-44 all
write bundles.
- **Fix.** `architect` rewrites line 12 to `ccg.atrest.write_bundle`. P1-04 carries the obligation with a
  test that its bundle is `0600` under `bundles/<system>/`. As defence in depth, `bundle.write_bundle`
  creates its temp file with `O_EXCL`, mode `0600` and `fchmod`, so a stray caller cannot widen it.
  That is a one-line `tdd-engineer` change, best placed in P1-07, which P1-02 blocks.

#### m1 (minor) — plan text is stale after the key move (the move itself is approved)

The move to `~/.config/ccg/keys/` is an improvement. It agrees with `-bundle-spec3.md` §5.17 and P0's
`~/.config/ccg/`, and it answers S5-6 (B1 aside). Stale homes from an unfiltered
`grep -rn 'ccg-work/keys'`:
- `-p1-extraction.md:59-60` (P1-02 (d)/(e)), and `:65` ("a path outside `cpg/.ccg-work/` is refused";
  keys are now accepted only outside it);
- `-p1-linking.md:156` (P1-19);
- `-bundle-spec5.md:585` ("Open for `devops`"; now answered), and §2's "held per P1-02(d)".

Owner: `architect`, folded in place. `docs/reviews/*` hits are history and stay. Leaving them means a
P1-19 implementer following the plan writes the ACL file where `atrest` refuses it. That fails closed.

#### m2 (minor) — a leading-zero port slips past the literal 6379/6390 refusal

*Verified for `status`; inferred for `up`.* `tw_check_port_value` compares strings, so
`CCG_FALKORDB_PORT=06379 --throwaway status` passes the check and reaches Docker. For `up` with `06390`
while `ccg-falkordb` is down, only `ss`/`docker ps` stand between it and a throwaway bound to 6390, and
`port_published` greps `06390->`, which never matches. I did not run it. **Fix:** require
`^[1-9][0-9]*$`, or normalize with `PORT=$((10#$PORT))` before the comparisons. Add `06379`/`06390`
to T1.

#### m3 (minor) — `up` on a running instance with lost config succeeds silently; a failed throwaway start leaks

- `cmd_up`'s running branch only logs `report_module_config`, so `up` exits 0 while `status` reports
  `TIMEOUT_MAX 0` drift. Make it warn and exit non-zero on a mismatch, naming the remedy
  (`down --force`, `up`).
- In `run_container`, a `die` inside `wait_ready` or `apply_module_config` after a successful
  `docker run` leaves a running throwaway and prints no name. For `--throwaway`, remove the container
  on that path, as the rc-3 path does. *Inferred from code.*

#### m4 (minor) — `hmac_key`'s first write is not atomic

`atrest.py:252-265` creates the file with `O_EXCL`, then writes. A crash in between leaves an empty
file, which is "malformed" forever. A concurrent loser that reads before the winner's write gets the
same error. Both fail closed. **Fix:** write a temp file, then `os.link` it onto the name (atomic
create-if-absent), then unlink the temp. *Inferred.*

#### m5 (minor) — the docs claim more than they deliver, and restate facts

- `cpg/infra/README.md` says the indexer image is "described in `cpg/infra/toolchain/README.md`". That
  README is untouched, and its usage block lacks `--indexer`.
- **Growth-stops (`cpg/README.md`, new scope).** The row is new scope, but it restates the modes,
  paths and retention rule whose home is `infra/README.md` § Bundle-at-rest. Cut it to "bundle-at-rest
  and key custody (P1-02): `infra/README.md` § Bundle-at-rest, which replaces `-bundle-spec3.md` §13
  F-1's default".
- `atrest.py`'s docstring likewise restates the README table. Cite it instead.
- The `infra/README.md` growth is new scope (throwaway, indexer, bundle-at-rest). The removed
  Reservations list is rework that shrank.

#### n1 (nit)

- Index §8 cites `--throwaway` as "P1-01"; it is P1-48.
- In throwaway mode, `status` prints "start with: cpg/infra/ccg-falkordb.sh up", which is the wrong verb.

### What's solid

- **The Dockerfile split does not change `toolchain`.** `ccg-toolchain:73b9fb76bd2a` (new inputs) and
  `:a82faf5219b3` (old) share image ID `sha256:a24714d0…`. `truth-fetch` is `FROM npm-fetch`, with the
  same instructions, so `/opt/truth` is unchanged. I inferred that, because `truth` was not rebuilt at
  the new hash. The `indexer` image carries only `/opt/indexer` (316 MB), no `/opt/truth`, and none of
  scip/scip-java/scip-clang/callgraph/uv. `ccg-versions` passes under `--network none`. `--print-tag`
  agrees across all three targets. The AC-8 surface widens only for runs that pick this image.
- **Throwaway ownership is layered:**
  - a pure-argument port gate before Docker;
  - the name derived from the port;
  - a second label required by every throwaway verb;
  - `reset-p0` refused;
  - a race path that removes only a `created`, labelled container.
  Every refusal has a stubbed test with a positive control (T1–T9).
- **The `status` rules have both directions:** a nonzero `TIMEOUT_MAX`, `MAX_INFO_QUERIES` 0, and drift
  cases for 0, 1000, `ccg_buildx` and `ccg_raw_`.
- **`atrest`'s mode handling is umask-independent and tested at umask 0 and 0277.** It raises on an
  exposed file rather than re-moding it. Retention parses `<unit>.<build>` unambiguously (no `.` in
  names) and spares `.tmp` and sibling units. Generation uses `secrets.token_bytes(32)`, and a
  generated test key is never written.

### Open questions

1. **ACL naming (M1).** Are the names per-system (`ccg_build_<s>`, `-p1-graph.md` §7) or bare
   (P1-48/P1-19 text)? `graph-dba` and `architect` decide. The script follows the ruling.
2. **Joern script mode as an exec channel (M4).** Script mode lets a script exec any image binary. Is
   that acceptable for `extract.sc` alone, or should `-egress-spec3.md` §7.4 state it as a residual?
   That is for `architect`, with `security-expert` advisory.

### Proposed REVIEW.md lines

Target: root `REVIEW.md`, `## Always check`. Sources: B1 and M5 of this review.

- A path-containment check that resolves symlinks is checked with planted links: one from one protected
  area into another, one between two files inside the same area, and one dangling. Containment by
  realpath alone admits the first two.
- A guard that reads a tool's configuration or launch surface is checked against every input channel the
  tool's own documentation lists (flag aliases, `--flag=value` forms, extra settings or config files),
  not only the channels in use today.

## Pass 2 (2026-11-10)

**Verdict: approve with suggestions.** B1, M1, M2, M3 and M5 are fixed, and I checked each one by
running it. Two majors stay open, and both belong to `architect`, not to `devops`: M4 (accepted as
deliberately open) and M6 (not claimed fixed). P1-04, P1-15 and P1-19 should not be dispatched until
those two and m1 are folded in. m2 is only half fixed (P2-1).

**Runs (observed).**
- `cpg/` default suite: `4251 passed, 1 skipped, 10 deselected`. The skip is the same opt-in slow
  test as in Pass 1.
- `infra/tests/test-ccg-falkordb.sh`: `533 passed, 0 failed`.
- `infra/toolchain/tests/test_toolchain.sh`: `40 passed, 0 failed`, with `truth` now built at
  `73b9fb76bd2a`. Its smoke run passes all 8, so I verified, rather than inferred, that the split left
  `truth` intact.
- A live throwaway on 16477 (labelled by the script, removed with `--throwaway down`; nothing left
  over, `falkordb-dev` untouched). `up` reported `TIMEOUT_MAX 10000, MAX_INFO_QUERIES 0`. `status`
  returned 0, and still 0 after `ACL SETUSER ccg_build_shop`. After `ACL SETUSER ccg_build` it reported
  `DRIFT: ACL users outside the P1 set: ccg_build`.

**Prior findings.**
- **B1: fixed.** I re-planted every case on a scratch area: dangling link (read and write), bundle-named
  link into the key area (`read_bundle`, and `read_private` with `area=keys`), in-area link, lexical `..`,
  system dir linked to the key area, and a key-file link. All were refused. **Mutation:** in a scratch
  copy, I changed `lstat` to `stat` in `_check_chain`. Four tests went red (into-key-area, in-area,
  dangling, key-file-replaced). The restored copy was green (99 passed).
- **M1: fixed.** `ACL_USER_RE` now accepts `ccg_<role>_<s>` with the §7 whitelist and a 1–48 length
  check. The new drift cases cover bare `ccg_build`, `shop_`, `a__b`, `-x`, upper case, unknown role and
  49 characters. I confirmed it live (above).
- **M2: fixed.** With the environment variable set, `write_acl` and `read_acl` round-trip, and a `0640`
  file is refused (I ran this).
- **M3: fixed.** `atrest` refuses both K1 and K2 (I loaded each), and the tests read the hex values from
  the spec5 text. Spec5 §2 now names `atrest.hmac_key` as the single refusal point with no bypass, which
  agrees with the module.
- **M4: open by decision.** The README states that the runner has no supported indexer path. Two
  things are still undone: `-egress-spec3.md` §7.4 has no indexer entry, and P1-15's file list does not
  name `cpg/sandbox/offline-run.sh` (`architect`). The test gap is fixed: `test_toolchain.sh` (5)
  covers the indexer image (I ran it).
- **M5: fixed for every channel in Appendix A.** I re-ran the probe against a scratch copy of the kit;
  all of Pass 1's misses are now flagged. A new bound remains (P2-2).
- **M6: not fixed.** `-p1-extraction.md:12` still reads "through `ccg.bundle.write_bundle`", and
  `bundle.py` is unchanged. The owners are as in Pass 1.
- **m1: not fixed**, with a new hit. P1-02 (d)/(e) (`-p1-extraction.md:59-60`) and `-p1-linking.md:159`
  still give `cpg/.ccg-work/keys/`. `-p1-graph.md:311`, revised in this same round for M1, also cites
  `cpg/.ccg-work/keys/<s>.acl`. That contradicts `atrest`, which refuses a key area inside the repo.
  Owners: `architect` and `graph-dba`.
- **m2: partly fixed.** See P2-1.
- **m3: fixed.** `up` on a running instance with lost config now exits non-zero and names the remedy.
  A throwaway that fails after `docker run` is removed (T10, T10b, T10c).
- **m4: fixed.** The first write is a temp file plus `os.link`.
- **m5: fixed.** The `cpg/README.md` row now only cites. The toolchain README carries the indexer
  section, and the `atrest` docstring cites rather than restates.
- **n1: fixed.**

**New findings.**
- **P2-1 (minor): a port spelling of 7 or more characters still slips past the 6379/6390 refusal.**
  *Verified for `status`.* Normalization runs only when `${#PORT_ENV} -le 6`, so `0006379` and
  `00000006390` reach `docker` as `ccg-falkordb-tw-0006379`, while `06379` and `006390` are refused.
  **Fix:** refuse anything that is not `^[1-9][0-9]{3,4}$` outright, instead of normalizing. Add a
  7-character case to T1.
- **P2-2 (minor): the guard does not judge flags that widen grants without naming a path.**
  *Verified with a kit copy in the scratchpad.* These go unflagged:
  - `--dangerously-skip-permissions`;
  - `--permission-mode bypassPermissions`, `acceptEdits`, or `--permission-mode=…`;
  - `--agents <another file>`;
  - `--plugin-dir`;
  - removing `--disallowedTools`.

  For path tools, the `CCG_GUARD` hook still contains these. That is inferred: Claude Code runs
  PreToolUse hooks in every permission mode. **Fix:** fail closed on any launcher token that is not in
  a reviewed set built from `claude --help`, and pin `--permission-mode dontAsk`. That way a new flag
  turns the test red instead of slipping past.
- **P2-3 (minor): `test_atrest.py` reads `docs/plans/connected-code-graph-bundle-spec5.md`, which is
  untracked.** On a checkout without it, collection fails (I saw this on my scratch copy). Commit
  spec5 with, or before, this change (`teco`).
- **P2-4 (nit).**
  - `ensure_private`/`read_private` infer the area kind from the directory name (`.ccg-work` means work
    area, so the root is not checked). An explicit `is_key` would not depend on a name.
  - The README says a swap after the name check "is caught". That holds for the leaf, not for a
    parent directory swapped for a link between the check and the `open`. With fixed `.bundle.jsonl`
    and `.acl` leaf names, I could not turn this into a key read. Narrow the sentence, or open the
    chain with `O_NOFOLLOW|O_DIRECTORY`.

## Appendix A — guard coverage probe (M5)

Run against the scratch copy, using `test_atrest._collect` and `_mcp_tool_patterns` unchanged:

```
missed   allowed-tools(dash)        --allowed-tools "Read(cpg/**)"
missed   allowedTools=              --allowedTools=Read(cpg/**)
FLAGGED  allowedTools(space)        --allowedTools "Read(cpg/**)"
missed   settings-extra             --settings /x/other.json
missed   mcp param 'target'         def read(target: str)
FLAGGED  mcp param 'path'           def dump(path: str)
missed   .mcp.json abs cpg root     args ["--root", "/home/u/repo/cpg"]
missed   .mcp.json rel cpg          args ["--root", "cpg/"]
missed   .mcp.json keys path        args ["--root", "cpg/.ccg-work/keys/../bundles"]
```
