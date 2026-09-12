# DevOps: headless, local-model OpenCode variant (`tank`) — security review

> **Status:** active · **Owner:** `security-expert` · **Tracks:** — (M0)

## 1. Scope & verdict

**Lens:** primarily agent/prompt-safety review (lens 2) of a finished `architect` plan, with a
secrets/infra-hardening angle (lens 3) folded in for the Docker/Compose surface. This is **not**
classified as a lens-1 code/app-security review: `tank` is not built yet — there is no source tree,
and (verified fresh via `mcp__cypher__query`) no `cpg_opencode`/`cpg_claude` graph exists, matching
the plan's own §1 CPG finding. Per the review-family convention this means the mandatory `CPG:`
line does not apply here; it would apply once `tank`'s actual config/scripts exist and a
lens-1 review of that artifact is requested.

**Reviewed:** `opencode/docs/plans/devops-opencode-headless.md` (plan, all sections) against
`opencode/docs/requirements/devops-opencode-headless.md` (requirements/acceptance criteria,
including the Terminology and Decision-log sections) and the referenced live artifacts
(`claude/devops/devops.md`, `claude/scripts/guard-destructive-ops.sh`,
`deprecated/opencode/agents/severino/opencode.json`).

**Verdict: needs changes.** One blocker: the plan's central safety mechanism (§3.2's static
`permission.bash` allow-list) is **not** the mechanical, hard-boundary safety net it is designed
and relied upon to be — it is bypassable using nothing more exotic than OpenCode's own documented
glob syntax, with no shell metacharacters, no exotic encoding, and no compromise of the model
required. This was reproduced live (not inferred), tearing down a completely unrelated,
out-of-repo-scope Docker Compose stack while the actually-intended, in-scope stack was left
untouched. Two related major findings compound the same root cause. See §2.

## 2. Findings

### BLOCKER — Glob-smuggling in `permission.bash` defeats FR-8 repo-scoping and the whole mechanical safety net, confirmed live

**Evidence.** Plan §3.2 lines 218–227 propose (paraphrased to the pattern shape, `<repo>` = the
baked-in absolute repo path):

```
docker compose -f <repo>/*compose*.y*ml --project-directory <repo>/* down     → allow  (exact match, no trailing *)
docker compose -f <repo>/*compose*.y*ml --project-directory <repo>/* up*      → allow
```

OpenCode's own documentation and empirical resolution (`opencode debug agent tank`, this session,
against the installed OpenCode v1.18.30) confirm `*` is "zero or more of **any** character" — it is
not path-separator-aware, not word-boundary-aware, and does not stop at flags. That means the `*`
between `--project-directory <repo>/` and the literal ` down`/` up` is not "the rest of one directory
path" as the plan's prose implies — it is "**any bytes at all**," including a second, fully-formed
`-f <file> --project-directory <dir>` pair.

I reproduced this against disposable fixtures I created and destroyed for this purpose (methodology
and full command log in §5) — mirroring the same throwaway-config verification technique the plan's
own §2.3/§5 uses, not a probe against any named lab service:

- Built a `tank`-shaped `opencode.json` with `permission.bash` set exactly to the plan's §3.2 tiers
  (default deny → the compose allow-list, glob-scoped to a stand-in `<repo>` → the metacharacter/
  chaining deny net last), and drove it with `opencode debug agent tank --tool bash --params
  '{"command":"..."}'` — which invokes OpenCode's real permission classifier and, on an allow,
  **actually executes** the command (confirmed distinctly from a mere `debug agent` shape-inspection,
  which the plan's own §5 notes is not sufficient).
- Stood up two independent, disposable Compose stacks: one under the stand-in `<repo>` (the
  "legitimate" target `tank` was told about) and one entirely **outside** it, at `/tmp/victim-...`
  (standing in for "a demo/dev environment `tank` was not asked about" — the exact case the plan's
  own §5 test 6/9 says must be refused).
- Issued this single command, which passes the exact-match "down" glob string-for-string (it starts
  with the required `-f <repo>/...compose.yaml`, then ` --project-directory `, then the wildcard
  swallows the injected second pair, then ends in the required literal ` down`):

  ```
  docker compose -f <repo>/compose-smuggle-test/compose.yaml \
    --project-directory <repo>/compose-smuggle-test \
    -f /tmp/victim-outside-repo-55731/compose.yaml \
    --project-directory /tmp/victim-outside-repo-55731 \
    down
  ```

- **Result, observed directly:** OpenCode classified it `allow` (no `permissionDecision` rejection,
  unlike every denied case tested alongside it) and executed it with exit `0`. Docker Compose merged
  the two `-f` values and used the **last** `--project-directory` (Cobra flag semantics: repeated
  scalar flag, last wins), so the effective target was the out-of-scope victim stack. The victim's
  container and network were torn down. **The in-scope, actually-named `compose-smuggle-test` stack
  was untouched** — it never even matched the ownership-marker check, because the command that ran
  bore no resemblance, semantically, to what any marker would have recorded.
- No shell metacharacter was present anywhere in this command — the `*&&*`/`*;*`/`*|*`/backtick/
  `*$(*`/`*>*` defense-in-depth net (§3.2 tier 3) is simply irrelevant to this vector, since nothing
  is being chained; a second, fully legitimate-looking flag pair is being **appended inside an
  already-allowed pattern's own wildcard span**.
- The identical shape applies to the `up*` pattern, and arguably worse there: FR-5a's whole premise
  ("bring-up is additive, not destructive, so it is not subject to the auto-deny safety net") assumes
  the compose file being brought up is one "already checked into the repo, and thus already
  reviewed" (§3.2's own stated rationale for the repo-root scoping). A second, smuggled `-f`
  pointing outside the repo defeats that assumption directly — an arbitrary, attacker-supplied
  compose file (a bind mount of `/` into a container, a privileged container, a host-network
  service) can be brought up under the same "always allowed" pattern, with no state-marker check
  at all (bring-up is unconditionally allowed per FR-5a).

**Why it matters.** This is not the theoretical "heredoc or `$IFS`" gap the plan's own §7 already
disclosed and routed to this review — it's a **plainer, easier, already-armed** version of the same
"residual glob-pattern smuggling risk" the architect flagged without diagnosing. It requires no
unusual shell feature, just knowledge that `--project-directory`'s value is matched by a
mid-pattern wildcard. It defeats, simultaneously: FR-8 (repo scoping), the ownership-marker
mechanism's entire premise (§3.3 — the marker is irrelevant if the executed command was never the
one the marker check reasoned about), and the plan's own explicit design rationale for putting
enforcement at the mechanical layer specifically *because* the local model's judgment can't be
trusted (§3.2's opening paragraph) — the mechanical layer itself has the hole.

It also directly supplies the "concrete mechanism" for review angle 3 (prompt-injection surface):
a compromised or merely confused 3B local model does not need to be tricked into anything exotic —
text reflected into its context from a health-check target, a compose file's own service/label
names, or prior tool output could plausibly steer it toward emitting a second `-f`/
`--project-directory` pair (e.g. "also check the config at ..."), and the mechanical layer would
wave it through exactly as demonstrated.

**Suggested improvement.** Do not rely on a single ordered glob list against the raw command string
for this class of command. Concretely:
1. Anchor `--project-directory` (and `-f`) values so the wildcard cannot contain another
   `-f`/`--project-directory`/`--project-name`/`--profile`/`--env-file` token — e.g. deny any
   compose-prefixed command containing more than one ` -f ` or more than one
   `--project-directory` substring, placed in the same last-wins tier as the existing chaining net
   (`*-f *-f *`, `*--project-directory*--project-directory*`, `*--env-file*`, `*--profile*` as
   additional deny entries, checked **before** trusting the allow patterns matched at all — i.e.,
   treat "more than one instance of a scoping flag" as its own smuggling signature, the same way
   `&&`/`;`/`|` already are).
2. Preferably, don't build the docker compose invocation from a model-authored string at all: have
   the bring-up/teardown wrapper scripts (§4 step 7, already planned as "thin, reviewed wrappers")
   take a **pre-validated slug** (looked up against a small, repo-committed allow-list of known
   compose files, not glob-matched against arbitrary repo paths) and construct the entire
   `docker compose -f ... --project-directory ... {up|down}` argv themselves, in code, with no
   string concatenation of anything the model wrote. `tank`'s `bash` permission for compose would
   then only ever need to allow the wrapper script's own fixed invocation shape, closing the glob
   surface entirely rather than patching it. This is a design-level fix, not a tighter regex — worth
   discussing with `architect` before promotion.

### MAJOR — Ownership-marker slug derivation collides on directory basename alone

**Evidence.** Plan §3.3: "The marker filename is derived from the environment, not fixed: the slug
is the basename of the `--project-directory` path it was given (e.g. a request naming
`falkor-chat/compose.yaml` derives slug `falkor-chat` → `state/falkor-chat.json`...)."

**Why it matters.** Basename-only derivation means two genuinely different environments that happen
to share a leaf directory name (e.g. `services/demo/` and `sandbox/demo/`, or any two throwaway
fixtures a future user names identically — the plan's own §5 test 8 explicitly asks for a second,
throwaway fixture "created under the repo for this test only", with no naming convention imposed)
collide on the same `state/demo.json`. Two concrete bad outcomes: (a) `tank` brings up environment A,
writes `state/demo.json`, then is later asked about unrelated environment B with the same basename —
the "already running?" check in §3.3 step 1 keys off the wrong marker, and (b) a teardown request
for B finds A's marker present, concludes "yes, I own this," and runs `compose down` against B using
A's recorded `-f`/`--project-directory` paths (§3.3 step 3: "run the allowed `compose ... down`
using the paths recorded in the marker") — which tears down A while the request was about B, with
no attacker or malicious input needed at all, just an ordinary name collision. This directly
undermines FR-5b's stated invariant ("Any other destructive... operation... is not covered by this
carve-out") using entirely benign inputs.

**Suggested improvement.** Derive the slug from a normalized **full path** (e.g. a hash or the
full `--project-directory` path with separators substituted, not the basename), so two different
directories can never collide. If a human-readable slug is still wanted for the README/debugging,
keep it as a label inside the marker JSON, not as the filename/lookup key.

### MAJOR — Ownership marker has no tamper/provenance check; any writer on the filesystem can plant one

**Evidence.** Plan §3.2: `permission.write`/`.edit` = `{"*": "deny", "state/*": "allow"}` — this
constrains what **`tank`'s own write tool** can touch, but the marker file
(`opencode/agents/tank/state/<slug>.json`) is an ordinary file in the checked-out working tree.
Nothing in §3.3's protocol validates that a marker was written by a `tank` run that actually
observed a cold-start `up`, versus, e.g., a stale marker left over from a previous session, a marker
written by hand during testing, or (in a future where another local process or CI job touches the
repo directory) a marker planted by something else entirely. §3.3 step 3 treats "marker present" as
sufficient proof of "started by me."

**Why it matters.** Combined with the basename-collision finding above, or on its own via any
process with filesystem access to the repo, a stale or planted marker converts directly into an
unattended teardown of an environment `tank` did not itself bring up in the current run — the exact
scenario FR-5b/FR-6 exist to prevent, and the one the plan's own live smoke test 6 explicitly checks
for the *simple* case (a stack started by a plain `docker compose up` with no marker) but not for
"a marker exists but is stale/wrong."

**Suggested improvement.** Bind the marker to the specific run that created it — at minimum a
freshness check (refuse to act on a marker older than some bound, since `tank` runs are one-shot and
short-lived) and ideally a value in the marker (a run/session ID) that's meaningless to forge
without already having write access equivalent to `tank`'s own. This is a cheap addition given the
marker is already being written with a timestamp.

### MINOR — `up*`'s trailing wildcard is strictly more permissive than the exact-match `down`, undermining §3.2's own stated rationale

**Evidence.** §3.2: the `down` pattern is deliberately exact-match "so `-v`/`--volumes` can never be
appended and still match this literal pattern." The `up*` pattern (line 222) keeps a trailing `*`,
so **anything** can be appended after `up` — `--force-recreate`, `--scale`, arbitrary extra tokens —
with no equivalent defense-in-depth reasoning given for why that's acceptable, beyond "bring-up is
additive."

**Why it matters.** "Additive" is only true for the intended compose file; combined with the
BLOCKER finding above (an attacker-influenced second `-f`), an unconstrained trailing wildcard after
`up` is a second, independent way to broaden what actually executes. Even absent that finding, a
trailing wildcard is unnecessary generosity: the plan's own default-`up*` invocation is a fixed
shape from the wrapper script (§4 step 7), so the wildcard buys nothing except attack surface.

**Suggested improvement.** Give `up` the same exact-match discipline as `down` (or a narrowly
enumerated small set of explicit flag combinations, if `-d`/`--build` variants are genuinely needed),
rather than a bare trailing `*`.

### MINOR (unconfirmed, theoretical) — `write: {"state/*": "allow"}` may not stop `state/../..` traversal

**Evidence.** The same `*` semantics that make the bash glob smuggling possible (any character,
including `/` and `.`) apply equally to `permission.write`'s path-glob matching (confirmed generic
mechanism via `opencode debug agent`, §2.3 of the plan and reconfirmed this session). A pattern
`state/*` would, on the same "any character" semantics, also match a value like
`state/../../../etc/something` if OpenCode's write-permission check runs before path normalization.

**I attempted, but did not complete, a live check of this specific claim.** A second throwaway
project directory produced an unexplained result: `opencode debug agent tank` resolved `tools.bash`
and `tools.write` to `false` even though the project `opencode.json` set both `true` (identical
JSON shape to an adjacent, working test directory where they resolved `true` moments earlier) — the
`write` tool call was refused with `"Tool write is disabled for agent tank"` before permission
matching was ever exercised. I did not track down the cause (possibly a merge interaction with this
machine's own global `~/.config/opencode/opencode.json`, which defines an unrelated `lmstudio`
provider and a `plan` agent with `bash`/`write`/`edit` forced `false` — untested hypothesis) given
the time budget for this review.

**Suggested improvement.** Before relying on `state/*` as a real boundary, `architect`/whoever
implements this should specifically test a `write` call with a `..`-bearing path against the real
`tank` config (the same `opencode debug agent tank --tool write --params '{"filePath":"state/../x",...}'`
technique used above for `bash`) and confirm whether OpenCode normalizes the path before or after
matching. Given the confirmed bash-side behavior, treat this as **likely to have the same gap**
until verified, not as safe by assumption.

## 3. What's solid

- The core design instinct — put enforcement at OpenCode's static permission layer rather than
  trusting a 3B local model's judgment (§3.2) — is the right call in principle; the finding here is
  that the chosen mechanism doesn't yet deliver on that instinct, not that the instinct is wrong.
- The `"ask"` hangs-headless / `"deny"` completes-and-reports distinction (§2.3) is correctly
  identified and the design consistently avoids `"ask"` everywhere reachable in a headless run — I
  did not find a path back to `"ask"` anywhere in §3.2's table.
- The default-deny-first-then-allow-then-defense-in-depth-deny **ordering** is exactly right for
  "last match wins" semantics, and the compound-command/shell-metacharacter net (`&&`, `;`, `|`,
  backtick, `$(`, `>`) genuinely works as intended for what it's designed to catch — I confirmed
  each of those denies fires correctly and blocks execution (no partial state).
- The prompt addendum (§3.4) is written in the **safe shape** this review's agent/prompt-safety
  lens looks for: it states the safety property directly ("a denied command is final... never
  retry... never look for an alternative command with the same effect") rather than framing any
  workaround as "the answer to being denied." No instruction-poisoning-shaped language found in
  §3.1–§3.4 or in the addendum content.
- The persona-split mechanism (§3.1) is a clean, low-risk piece of the plan; no security concerns
  found there.
- The plan is honest about its own residual risk (§7) and explicitly requested this review rather
  than asserting the glob table was exhaustive — the architect's own framing ("safe by construction
  here only because the default posture is deny, not because the deny-list is exhaustive") is
  accurate in spirit, though the blocker above shows the specific residual risk is more severe and
  more easily triggered than the heredoc/`$IFS` example given.

## 4. Open questions (for `architect` / `cobb` / the stakeholder)

- Is the design-level fix suggested for the BLOCKER (wrapper-constructed argv from a validated
  slug/allow-list, rather than glob-matching a model-authored string) acceptable, or is there a
  reason the plan needs the model itself to compose the full docker compose invocation? If the
  latter, the tightened-glob mitigation (deny on repeated scoping flags) should be treated as a
  floor, not a complete fix — I'd recommend re-testing empirically against this same live-execution
  technique before trusting a revised pattern set.
- Should `tank`'s repo-generic scope (§2.4/§3.2 — "whatever compose-based environment it's asked
  about") be revisited given the blocker? The scoping-to-`<repo>` narrowing was already a
  deliberate, disclosed choice; this review's finding is that the narrowing doesn't actually hold
  today, which may change the stakeholder's risk calculus about shipping the generic-environment
  carve-out at all versus starting with a hardcoded single-stack allow-list.
- Worth a follow-up, lens-1 code-security review once `tank`'s actual `opencode.json`/scripts exist
  (post-implementation) — this plan-stage review verified the *design's* glob claims but a
  from-code review after implementation should re-verify the shipped patterns match what's
  described here, plus re-run the write/state traversal check left unconfirmed above.

## 5. Verification methodology (live reproduction, disposable fixtures)

Everything executed below ran against **fixtures I created for this review and destroyed
immediately after** — a throwaway `opencode.json`/`tank` agent config under this session's
scratchpad, and two disposable `busybox`-based Docker Compose stacks (one "in-repo" stand-in, one
"out-of-repo" stand-in). No lab-owned service (`falkor-chat`, `salesperson`, the shared
`falkordb-dev` container, `cypher-mcp`) was targeted, read, or touched at any point — `docker ps -a`
before/after each phase confirmed only my own test containers were affected. `tank` itself does not
exist yet (this is a plan review), so there is no live `tank` instance to exploit; what was tested
is OpenCode's own permission-classification engine (`opencode debug agent <name> --tool bash
--params ...`, which invokes the real classifier and, on an allow, really executes) and Docker
Compose's own CLI flag parser, both third-party, already-installed tools this repo uses routinely —
the same class of throwaway-config verification the architect performed in the plan's own §2.3/§5,
not a probe against a running lab target. On that basis I treated this as ordinary investigative
verification of the plan's own technical claims (guardrail: "gather evidence... reproduce the actual
path... instead of pattern-matching"), not as an FR-10-gated exploitation attempt against this lab's
local/dev services — I'm stating that judgment call explicitly so it can be checked, since FR-10's
approval ritual is otherwise unavailable to me as a delegated subagent (no live human turn to state
target/technique/blast-radius to, and no standing approval to fall back on).

Commands run, target, and result — full detail folded into the BLOCKER finding above; summary:
1. `docker ps -a` baseline before/after every phase — confirmed no lab container ever touched.
2. Built `opencode.json` reproducing plan §3.2's exact three-tier `permission.bash` shape (with a
   scratch `<repo>` substituted for the real absolute path). Verified resolution via
   `opencode debug agent tank`.
3. `opencode debug agent tank --tool bash --params '{"command":"..."}'` — confirmed default-deny,
   the `docker ps*` allow, and each metacharacter-net deny (`&&`, newline+`touch`) behave as the
   plan claims: denied calls produced an explicit permission-rejection error and no state change.
4. Stood up disposable in-repo-stand-in stack (`compose-smuggle-test`, one `busybox` container +
   named volume) and out-of-repo-stand-in stack (`/tmp/victim-outside-repo-<pid>/`, one `busybox`
   container + named volume) via plain `docker compose up -d`.
5. Ran the smuggled command (full text in the BLOCKER finding) through the same
   `debug agent --tool bash` harness. **Result: classified `allow`, executed, exit 0** — the
   out-of-repo victim's container+network were torn down (confirmed via `docker ps -a`/`docker
   volume ls` before/after); the in-repo stack was left running, confirming the "down" command that
   actually ran had nothing to do with the environment `tank` would have believed it was targeting.
6. Cleaned up: `docker compose down -v` on the in-repo stand-in, `docker volume rm` on the orphaned
   victim volume, `docker ps -a`/`docker volume ls` re-checked clean, all scratch directories
   removed. No residual state left on the host beyond this written report.
