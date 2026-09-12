# DevOps: headless, local-model OpenCode variant (`tank`) — implementation plan

> **Status:** active · **Owner:** `architect` · **Tracks:** — (M0) · **Version:** 3

**Revision notes** (one dated line each):
- 2026-09-12 — Generalized FR-5a/5b from a `falkor-chat`-only design to any compose-based
  environment under the repo, per stakeholder clarification (via `teco`).
- 2026-09-12 — Replaced §3.2's model-authored, glob-matched compose-invocation patterns with a
  wrapper-script + repo-committed environments allow-list, per `security-expert`'s review
  (`opencode/docs/reviews/devops-opencode-headless.md`) finding a live-reproduced glob-smuggling
  blocker; also fixed the marker-slug-collision and marker-tamper/freshness findings the same
  review raised, and removed `tank`'s `write`/`edit` tool access entirely (state I/O moved into
  the wrapper scripts' own code, closing the review's unconfirmed write-path-traversal question by
  eliminating the surface rather than testing it — see §3.3 and §7).

## 1. Goal & scope

Build `tank`, a new **headless, local-model OpenCode agent** that gives the `devops` role a way
to run without a live Claude Code session or a human present: a read-only health/hygiene check by
default, plus the ability to bring up **a** demo/dev environment headlessly and tear back down
(only) what it itself brought up. Everything else destructive stays auto-denied and reported, never
attempted. **The environment is generic, not tied to any one project** — the mechanism (§3.2/§3.3)
works for any Docker-Compose-based environment checked into this repo; `falkor-chat`'s stack is
used throughout this plan as the concrete worked example because it's the only such stack that
exists today, not because the design is scoped to it (stakeholder-confirmed, see §7).

**In scope:** `opencode/agents/tank/` (OpenCode project-scoped config + prompt-sharing + scripts),
the shared-persona split at `claude/devops/` (`devops-persona.md` + `devops.md` + a sync script),
and the small doc updates the new component needs (root `AGENTS.md`, a new `opencode/AGENTS.md`).

**Out of scope** (per the requirements doc, unchanged here): the actual retirement/merge of
`claude devops` into `tank`; a wall-clock scheduler; kaizen-graph writes from a headless run
(blocked on C-310); global (every-project) local-model config; any headless task beyond the
read-only check and bring-up/teardown of a demo/dev environment; teardown of anything `tank`
didn't itself bring up, or any other destructive/shared-state op.

**CPG:** considered, not relevant — queried `mcp__cypher__query` for loaded graphs; only
`cpg_falkorchat` and `cpg_deprecated_salesperson` exist, no `cpg_opencode`/`cpg_claude`. This is a
greenfield config/prompt-authoring task with no existing call graph to analyze for impact.

## 2. Context & findings

### 2.1 Terminology (binding — from the requirements doc, restated only where it drives design)

**devops** = the role. **`claude devops`** = today's interactive/subagent Claude-Code identity at
`claude/devops/devops.md`. **`tank`** = this version's headless, local-model OpenCode identity.
**interactive** = human present, turn by turn. **subagent** = live Claude Code session, no human,
dispatched by another agent (`claude devops`'s existing non-interactive mode). **headless** = no
live Claude Code session at all (`opencode run --agent tank`). The long-term direction — `tank`
eventually absorbing `claude devops` entirely (interactive + subagent + headless) — is **out of
scope to build**, but every design choice below was checked against "does this assume the two
identities stay separate forever?" and rejected if so; see §3.6.

### 2.2 `claude devops` today

`claude/devops/devops.md` — YAML frontmatter (`name`, `description`, `permissionMode:
acceptEdits`, one `PreToolUse`/`Bash` hook) + a Markdown body that is the literal system prompt
(confirmed against `code.claude.com/docs/en/sub-agents`: a subagent's body has **no `@import`
expansion** — whatever bytes are in the file are the prompt, verbatim). The hook
(`claude/devops/hooks/guard-destructive-ops.sh`) thin-wraps `claude/scripts/guard-destructive-ops.sh
devops`: a regex-based `Bash`-matcher that escalates (`permissionDecision: "ask"`) on
`docker volume rm|prune`, `docker system prune`, `docker rm -f`/`--force`, `compose down -v/
--volumes`, `FLUSHALL`/`FLUSHDB`/`GRAPH.DELETE`, and the ad-hoc `pipeline.sh --reset` wrapper
match — **ask-only, allow-by-default on everything else**, because a human is assumed reachable to
answer the prompt.

The body's sections (verified by reading the file top to bottom): identity paragraph → "Orient
yourself in the project first" → "Core expertise" (a pointer to `claude/devops/ops-quirks.md`, not
inlined) → seven domain subsections (Containerization, Reproducible dev environments,
Dependencies/config/secrets, Automation & scripting, CI/CD, Observability, Security) → "Operating
principles" (nine bullets, including the destructive-ops-confirmation bullet with a
hook-mechanism parenthetical, and an "Interactive-mode commit" bullet) → "How you work" (5 steps)
→ "Boundaries & handoffs" (routes to `graph-dba`/`coder`/`qa-engineer`/`cobb`/`teco`, a
`tico`-mid-conversation demo-environment handoff bullet, and a closing "you are a subagent... stop
and return to the caller" line) → "Learning capture" (writes `:KaizenEntry` nodes to `kaizen_team`
via `mcp__cypher__query`).

### 2.3 OpenCode mechanics — verified live against the installed binary (v1.18.30), not guessed

I ran throwaway `opencode.json` configs through `opencode debug agent <name>` (prints the fully
resolved agent config with no model call needed) and `opencode run --agent <name> "<msg>"` against
this box's live LM Studio (reachable at `http://localhost:1234/v1`, confirmed via
`curl .../v1/models`) to settle every load-bearing mechanism below:

- **`{file:./path}` in a `prompt` string can be mixed with literal text in the same field.**
  Confirmed: `"prompt": "{file:./persona.md}\n\nADDENDUM..."` resolves (via `opencode debug agent`)
  to the file's contents followed by the literal addendum, concatenated — this is the FR-3
  sharing mechanism (§3.1).
- **`permission.bash` (and `.write`/`.edit`) accept an object of `{glob-pattern: allow|ask|deny}`,
  evaluated in declaration order with the last matching pattern winning** (matches the public docs
  at `opencode.ai/docs/permissions/`, confirmed empirically: a config with `"*":"deny"` followed by
  specific `"allow"` entries followed by specific `"deny"` overrides resolves to exactly that
  ordered list via `opencode debug agent`, in the order declared). `write`/`edit` use file-path
  globs the same way (confirmed: `"write": {"*":"deny","state/*":"allow"}` resolves to two ordered
  rules) — a general OpenCode capability, noted here for completeness; §3.2 explains why the
  shipped design ends up not using it at all.
- **OpenCode's own baseline (nothing set) is permissive**, not deny-by-default: the resolved
  config for an agent with no `permission` block shows a bare `"*": "allow"` catch-all, with only
  `question`/`plan_enter`/`plan_exit` denied and `external_directory`/`.env` reads asking. **`bash`
  has no rule at all by default** — it falls through to the `"*": "allow"` catch-all. This means
  FR-6/7's deny-by-default posture does **not** come for free from OpenCode; `tank`'s
  `opencode.json` must set it explicitly (§3.2).
- **`"ask"` hangs forever in a headless run with no human present.** Confirmed live: a
  `permission.bash: {"*":"ask"}` config, run via `opencode run` with no `--auto` flag, produced no
  output and had to be killed at a 45s timeout — there is no one to answer the prompt, and the
  run does not time out or fall back on its own. This is why `tank`'s config below uses only
  `"allow"`/`"deny"`, never `"ask"`, anywhere reachable in a headless run.
- **`"deny"` lets the run complete.** Confirmed live: a `permission.bash: {"*":"deny"}` config,
  asked to run `docker system prune -f`, had the tool call rejected, continued its reasoning loop,
  and produced a final text answer ("REFUSED: No permission to execute `docker` commands...") with
  exit code `0`. This is the FR-6/7 mechanism: deny, don't ask, and the agent narrates the refusal
  in its own final report.
- **Project-scoped config requires running from that directory** — confirmed against this repo's
  own prior verified note (`deprecated/opencode/agents/severino/README.md`'s troubleshooting
  table: "you didn't `cd` into `severino/`... project-scoped config is only loaded from the
  current working directory") and consistent with the `run --help` output exposing a separate
  `--dir` flag for pointing elsewhere without `cd`.
- **`docker compose` supports `-f <file> --project-directory <dir>` from any cwd** — confirmed
  against the installed `docker compose version` (v5.4.0): `--project-directory` is a real,
  present flag ("Specify an alternate working directory"). This lets `tank` invoke compose with
  one flat command (no `cd ... &&`), which matters for the compound-command defense in §3.2.
- One benign, unrelated finding: OpenCode's own internal "generate a session title" sub-call
  against `mistralai/ministral-3-3b` logged a Jinja template error (`After the optional system
  message, conversation roles must alternate...`) — cosmetic, logged only, did not affect the main
  turn or the final answer. Recorded as a kaizen entry (§6); not a blocker.

### 2.4 The demo/dev environment mechanism is generic; `falkor-chat` is today's worked example

FR-5a/5b say "a demo/dev environment" generically, and the stakeholder confirmed (via `teco`,
mid-plan) that this is deliberate: **`tank` must be able to bring up/tear down whatever
Docker-Compose-based environment it's asked about, not one hard-coded stack** — the ownership-
marker logic in §3.3 is the general mechanism, and §3.2's permission patterns are written against
*any* compose file under this repo, not against `falkor-chat/compose.yaml` specifically.

`falkor-chat/compose.yaml` (FalkorDB + the M1 server) is used throughout this plan as the concrete
example because it's the **only** live (non-deprecated) Docker Compose stack in the repo today —
it's what gets exercised in the Test strategy (§5) and what a first real invocation will target,
but nothing in §3.2/§3.3 names it specifically. Its own header comment states the safe lifecycle
worth knowing as the running example: `docker compose up --build` / `docker compose down` —
**`down -v` is explicitly forbidden** ("the volume is shared live data") — and the file warns that
its `falkordb` service binds the same port and volume as the separately-shared `falkordb-dev`
container (`falkor-chat/scripts/start_falkordb.sh`); running both is documented as **corrupting**,
not merely conflicting. Because Compose refuses to bind an already-held port, a `docker compose up`
attempted while `falkordb-dev` is running fails cleanly at the port bind (no partial state, no data
touched) — this is part of why treating bring-up as safe-by-construction in §3.3 holds for this
example, and is a property of Compose generally, not of `falkor-chat` specifically.

### 2.5 Verifying the redesign — live-reproducing the review's finding, and re-testing the fix

Before writing §3.2/§3.3's revised design into this plan, I reproduced the reviewer's finding and
then verified the fix, both live, rather than trusting either the original design or the proposed
fix by inspection alone:

- **Reproduced the classification gap the fix relies on.** Built a throwaway `opencode.json` with
  a `permission.bash` table shaped like the *new* design (default deny → `docker ps*` → the three
  absolute-path-anchored wrapper-script patterns → the metacharacter net) and confirmed via
  `opencode debug agent <name>` that it resolves in the intended order, with no raw `docker
  compose` pattern present anywhere in the table.
- **Verified the wrapper-script argument-validation logic directly** (plain bash, no OpenCode
  involved) against a mock of `compose-down.sh`'s intended logic (argc check + `case`-statement
  exact-match lookup): a known slug resolves; a quoted single-argument smuggle
  (`"falkor-chat -f /tmp/victim/compose.yaml --project-directory /tmp/victim"`) is refused by the
  lookup (no such key); the same content passed as separate, unquoted arguments is refused by the
  argc check (`got 5`, not `1`); a `../../../etc`-shaped value is refused by the lookup. All four
  cases fail before constructing any `docker` command — full transcript kept in this session's
  scratchpad, not committed (throwaway verification, same as §2.3's original mechanism checks).
- Did **not** re-run the reviewer's full live tear-down-of-a-disposable-victim-stack reproduction
  against the new config in this pass (time-bounded) — §5's smuggling-regression test specifies
  exactly that as a required step before this revision is considered verified, not merely
  designed.

## 3. Design & rationale

### 3.1 FR-3 — one canonical persona, shared mechanically

**Split `claude/devops/devops.md`'s body into two files:**

- **`claude/devops/devops-persona.md`** (new, no frontmatter) — the canonical, runtime-agnostic
  persona: identity paragraph, "Orient yourself", "Core expertise" (keep the `ops-quirks.md`
  pointer — sharing it is a small down payment on the eventual merge, §3.6), all seven domain
  subsections, "How you work", and "Boundaries & handoffs" **minus** its final two runtime-specific
  pieces (see below). Two edits versus today's `devops.md` text when extracting:
  - In "Orient yourself", drop the parenthetical hierarchy detail `(enterprise→user→project→local)`
    from the "Auto-loaded" bullet — that's Claude Code's specific memory-file resolution order;
    OpenCode also auto-loads `AGENTS.md` but resolves it differently, and the persona shouldn't
    assert a resolution order that's only true for one runtime.
  - In "Operating principles", the "Guarded ops" bullet keeps its behavioral core ("stop and get
    explicit confirmation before anything destructive... state the exact command and its blast
    radius") but drops the parenthetical describing the Claude Code `PreToolUse` hook mechanism
    (`guard-destructive-ops.sh`) — that's `claude devops`-specific machinery, not part of the
    persona, and factually wrong for `tank` (which has no hook, and can't "wait").
  - Drop the "Interactive-mode commit" bullet entirely from the shared file — it's Claude-Code
    git-commit-authority policy (`claude/AGENTS.md`'s commit rules), not applicable to `tank` in
    this version (no git-commit capability is requested or built for `tank`).
  - Drop "Boundaries & handoffs"'s closing `tico`-mid-conversation-handoff bullet and the final
    "You are a subagent... stop and return to the caller" line — both assume a live conversational
    caller, which headless `tank` never has.
- **`claude/devops/devops.md`** keeps its frontmatter untouched, then a body of: an HTML-comment
  marker pair `<!-- SHARED-PERSONA:BEGIN -->` / `<!-- SHARED-PERSONA:END -->` wrapping a byte-for-
  byte copy of `devops-persona.md`'s content, followed by the four Claude-only pieces dropped
  above (the hook parenthetical folded back into the Guarded-ops bullet, the Interactive-mode-
  commit bullet, the `tico`-handoff bullet + closing subagent line, and the unchanged "Learning
  capture" section).
- **`claude/devops/scripts/sync-persona.sh`** (new) — a small, idempotent script: read
  `devops-persona.md`, replace everything between the two markers in `devops.md` with it verbatim,
  fail loudly if the markers are missing or duplicated. Run it whenever `devops-persona.md`
  changes, in the same commit (documented in a one-line comment at the top of both files). This is
  the mechanical generation step FR-3 asks for on the Claude side — Claude Code subagent files have
  no live-include mechanism (§2.2), so regeneration is the only option there.
- **`opencode/agents/tank/opencode.json`**'s `agent.tank.prompt` field is
  `"{file:../../../claude/devops/devops-persona.md}\n\n<tank-specific addendum>"` (path relative to
  the config file's own location, `opencode/agents/tank/`, three levels up to repo root then down
  into `claude/devops/` — verified path-relative resolution semantics in §2.3). **This side needs
  no sync script and cannot drift**: OpenCode reads the live file content on every run.

This gives one canonical source (`devops-persona.md`) for the substance FR-3 protects, and the
acceptance criterion ("the claude devops prompt is later edited... all forms reflect it") is
satisfied differently on each side: `tank` reads it live; `devops.md` is kept in lockstep by a
one-command regeneration that any future editor must remember to run (documented at the point of
edit, not hidden).

### 3.2 FR-5/6/7 — the safety net is a static permission table, not the model's judgment

The requirements doc's own risk framing ("zero tolerance for slip-through... deny-by-default on
any command the guard can't confidently classify as safe") plus the live-verified fact that a
3–4B local model is a **less reliable judge of "is this safe?" than the frontier model behind
interactive `claude devops`** (§2.3's benign-but-real template glitch; `deprecated/opencode/
agents/severino/README.md`'s own note that "a 4B local model can mangle files or run shell
commands unreliably") together argue for putting the enforcement at the **mechanical** layer
(OpenCode's static `permission.bash`/`write`/`edit` rules), not the prompt layer. The prompt
addendum (§3.4) only shapes how `tank` *narrates* a refusal — it is never the thing that prevents
one.

**Rejected alternative:** mirror `guard-destructive-ops.sh`'s design (regex deny-list, allow-by-
default, ask on match). Rejected because (a) `"ask"` hangs forever headless (§2.3 — a hard
blocker, not a style preference), and (b) allow-by-default is the opposite of FR-7's explicit
"deny-by-default on ambiguity" instruction. The two mechanisms are deliberately asymmetric, and
that asymmetry is the correct reading of the requirements doc, not an oversight.

**Revised 2026-09-12 in response to `security-expert`'s review** (`opencode/docs/reviews/
devops-opencode-headless.md`), which live-reproduced a bypass of the compose allow-list this
section originally proposed (`docker compose -f <repo>/*compose*.y*ml --project-directory
<repo>/* down`): `*` in OpenCode's glob matcher is "any bytes at all," not "the rest of one
directory path," so a second, fully-formed `-f <file> --project-directory <dir>` pair fits
inside the wildcard span — Docker Compose then applies its own last-flag-wins semantics and
executes against whichever `--project-directory` was named *last*, not the one the allow pattern's
literal prefix appeared to scope it to. The reviewer tore down a disposable out-of-repo stack this
way with no shell metacharacter involved, so the tier-3 metacharacter net (still below) is simply
the wrong tool for this vector — it doesn't fire because nothing is being chained.

**The fix is architectural, not a tighter regex** (the stakeholder chose this over narrowing to a
single hardcoded environment, keeping `tank` generic): **no `docker compose` invocation is ever
built from a model-authored string.** `tank`'s bash tool never constructs a `-f`/`--project-directory`
pair itself; it only ever invokes one of three fixed, repo-committed wrapper scripts by an exact
absolute path, passing a single **slug** token. The scripts — not the model, not the permission
glob — look the slug up in a small, repo-committed allow-list and construct the entire compose
argv themselves, in code, with no string concatenation of anything the model wrote:

- **`opencode/agents/tank/environments.json`** (new, repo-committed, human-edited) — the trusted
  registry of every environment `tank` may manage, e.g.:
  ```json
  {
    "falkor-chat": {
      "composeFile": "falkor-chat/compose.yaml",
      "projectDirectory": "falkor-chat",
      "label": "falkor-chat dev stack (FalkorDB + M1 server)"
    }
  }
  ```
  Paths are **repo-root-relative**; the scripts resolve `REPO_ROOT` from their own location
  (`opencode/agents/tank/scripts/<name>.sh` is four directories below the repo root — `opencode`
  → `agents` → `tank` → `scripts` — so `REPO_ROOT="$(cd "$(dirname
  "${BASH_SOURCE[0]}")/../../../.." && pwd)"`, four `../` hops, not three: don't copy §3.1's
  `opencode.json`-relative `{file:...}` path depth here, that file sits one level shallower, in
  `tank/` itself, not `tank/scripts/`), so no absolute, machine-specific path needs baking into
  this file (only `opencode.json`'s bash-permission entries below still need one, to anchor the
  trusted script paths). Adding a second environment is a one-line, reviewed JSON edit
  — this is what keeps `tank` generic without reopening the glob surface: the enumeration lives in
  a committed file a human reviews, not in a pattern the model's own words can influence.
- **`opencode/agents/tank/scripts/compose-status.sh <slug>`, `compose-up.sh <slug>`,
  `compose-down.sh <slug>`** (new) — each: (1) refuses unless invoked with **exactly one**
  argument; (2) looks that argument up in `environments.json` via an exact-string match (a `case`
  statement or equivalent — never a substring/prefix/glob match), refusing with a clear message on
  any miss (unknown slug, extra text appended to a slug, a `../`-shaped value — all fail the exact
  lookup identically, verified against a mock of this logic: quoted smuggled flag-pairs, unquoted
  multi-argument smuggling, and path-traversal-shaped slugs were all refused, none reached a
  `docker` invocation); (3) only then constructs the fixed `docker compose -f <composeFile>
  --project-directory <projectDirectory> {ps|up -d --build|down}` command from the looked-up
  values (never from the argument text itself) and runs it. `compose-up.sh`/`compose-down.sh` also
  own the state-marker read/write (§3.3) — **directly, as plain file I/O in the script's own code,
  not via any OpenCode tool call** — which is what lets `tank`'s `write`/`edit` tools be disabled
  entirely (below).

`tank`'s `permission.bash` is one ordered list (OpenCode: last matching pattern wins), built in
three tiers:

1. **Default deny:** `"*": "deny"` first (lowest precedence — anything not explicitly allowed
   below stays denied). This alone satisfies FR-7.
2. **Explicit allow-list**, safe/needed commands only:
   - Read-only hygiene, host/repo-wide, unaffected by the redesign above (none of these take a
     repeatable scoping flag whose last value silently redirects a mutating action, so the
     smuggling class the blocker describes doesn't apply to them): `docker ps*`, `docker images*`,
     `docker volume ls*`, `docker network ls*`, `docker system df*`, `df*`, `du -s*`.
   - **The three wrapper scripts above, and only the wrapper scripts** — anchored to their exact,
     absolute, trusted path (`<repo>` = this machine's absolute repo path, baked in the same way
     `severino`'s `opencode.json` baked in absolute LM Studio URLs), never a leading wildcard
     (a leading `*` here would itself be a smuggling vector — it would just as happily match an
     attacker-planted `/tmp/evil/compose-up.sh`):
     - `<repo>/opencode/agents/tank/scripts/compose-status.sh *` — allow
     - `<repo>/opencode/agents/tank/scripts/compose-up.sh *` — allow
     - `<repo>/opencode/agents/tank/scripts/compose-down.sh *` — allow
   `tank` has **no allow pattern anywhere for a raw `docker compose` invocation** — every path from
   the model's own words to an actual `docker compose` call runs through a script that validates
   the slug by exact lookup before touching a compose flag. This is what "closing the glob surface
   entirely rather than patching it" (the reviewer's phrase) means concretely: the vulnerable
   pattern class doesn't exist to be bypassed, rather than existing in a tightened form.
3. **Defense-in-depth denies, placed last (highest precedence)** so they win even if a broader
   allow pattern above accidentally overlaps:
   - Mirroring `guard-destructive-ops.sh`'s own catalog: `docker volume rm*`, `docker volume
     prune*`, `docker system prune*`, `docker rm*`, `docker container rm*`, `*compose*down*-v*`,
     `*compose*down*--volumes*`, `*FLUSHALL*`, `*FLUSHDB*`, `*GRAPH.DELETE*`, `*pipeline.sh*
     --reset*`.
   - **A repeated-scoping-flag net**, added per the review's floor-level fallback recommendation
     and kept as a permanent second layer even though tier 2 no longer has a raw compose pattern
     for it to protect: `*compose*-f *-f *`, `*compose*--project-directory*--project-directory*`,
     `*--env-file*`, `*--profile*` all denied. This is insurance against a *future* regression —
     someone adding a raw compose allow-pattern back later without re-deriving this finding — not
     load-bearing for the design as shipped.
   - A **compound-command / shell-metacharacter net**: `*&&*`, `*;*`, `*|*`, `` *`* ``, `*$(*`,
     `*>*` all denied. Rationale: OpenCode's patterns are glob, not regex — an allow pattern like
     `docker ps*` glob-matches `docker ps && rm -rf /` just as happily as `docker ps -a`. Since
     these deny rules are declared *after* every allow rule, "last match wins" means any command
     that smuggles a second command behind an allowed prefix is caught here regardless of which
     allow pattern it also matched. Confirmed by the reviewer to work exactly as intended for what
     it's designed to catch (chaining) — it was never the right tool for the blocker's smuggling
     vector (§ above), which involves no chaining at all; the two nets address two different attack
     shapes and both stay.

`permission.write`/`permission.edit`: **removed** — see `tools` below. There is no longer any file
`tank` writes via an OpenCode tool call at all, so there is nothing for a path-glob permission rule
to scope (the review's unconfirmed "does `state/*` stop `state/../..` traversal?" question is moot
for the shipped design, not answered — see §7 for why eliminating the surface, rather than testing
the old one, is the intended resolution).

`tools`: `bash: true, read: true, glob: true, grep: true, write: false, edit: false, webfetch:
false, websearch: false, task: false, skill: false, todowrite: false, question: false` — `write`
flipped from the original design's `true` to `false`: the state marker is now read/written by the
wrapper scripts' own code (§3.3), never by a `tank`-invoked OpenCode tool call, so `tank` itself
has no write surface of any kind. Minimum surface for the job; no subagent spawning, no web
access, no in-place file editing, no file writing.

### 3.3 FR-5a/5b — symmetric bring-up/teardown via a state marker

**Revised 2026-09-12** — `security-expert`'s review found two problems with the original design
here, both fixed by the same change: moving marker read/write out of `tank`'s own tool calls and
into the wrapper scripts' code (§3.2), keyed off `environments.json`'s trusted slugs instead of a
path-derived one.

- **MAJOR (slug collision, fixed):** the original design derived the marker filename from the
  `--project-directory` path's *basename* — two different environments sharing a leaf directory
  name (e.g. `services/demo/` and `sandbox/demo/`) would collide on `state/demo.json` with no
  attacker input needed, letting a teardown request for one tear down the other. **Fixed by
  construction, not by a smarter hash:** the slug is no longer *derived* from anything — it is the
  literal, human-assigned key in `environments.json` (§3.2), checked into git and reviewed like
  any other config change. Two different environments can only collide if a human gives them the
  same key in that file, which is an authoring mistake caught at review time, not a runtime
  collision triggerable by an ordinary or malicious request.
- **MAJOR (no tamper/freshness check, fixed):** the original design treated "marker file present"
  as sufficient proof of "I started this," with nothing to distinguish a fresh, legitimate marker
  from a stale leftover or a hand-planted one. Fixed with a freshness bound and provenance fields
  (below) — disclosed honestly as a *correctness* safeguard against staleness/collision, not a
  cryptographic guarantee against a adversary who already has the same filesystem write access
  `tank`'s own scripts have (no local secret exists to make forgery meaningful against that threat
  model, and that threat model is already outside what any of this plan's mechanisms defend
  against — the repo's own git-commit trust boundary is the relevant one, same as for
  `environments.json` itself).

**Marker location and ownership:** one file per environment, `opencode/agents/tank/state/
<slug>.json`, where `<slug>` is exactly the `environments.json` key — written and read **only** by
`compose-up.sh`/`compose-down.sh`'s own code (plain shell file I/O inside the trusted script, not
an OpenCode `write` tool call — §3.2), so `tank` itself never has a write path to reach or corrupt
it. Contents: `{"slug": "<the key>", "broughtUpAt": "<ISO-8601 timestamp>", "sessionId": "<random
nonce, e.g. PID + high-resolution timestamp>", "pid": <script's own PID>, "host": "<hostname>"}`.
**The marker never stores compose paths** — `compose-down.sh` always re-resolves `composeFile`/
`projectDirectory` fresh from `environments.json` at teardown time, never trusting a copy in the
marker, so a corrupted or hand-edited marker can at most affect the ownership *decision* (whether
to act at all), never *what* gets torn down.

Protocol (encoded in the prompt addendum, §3.4 — `tank` never touches the marker directly; it
only ever asks the wrapper scripts to act and reads their output):

1. **Before bring-up:** `compose-up.sh <slug>` runs `compose-status.sh <slug>`'s check internally.
   If already up, it refuses to run `up` and refuses to (re)write the marker, printing "already
   running, not started by me" — this stops a false claim of ownership that would let a later
   teardown request tear down an environment a human or another agent started.
2. If cold, `compose-up.sh <slug>` runs `docker compose -f <composeFile> --project-directory
   <projectDirectory> up -d --build` (fixed flags, never anything from the caller beyond the slug
   selecting which paths to use). On success, it writes/overwrites that environment's marker with
   a fresh timestamp and a new session nonce. On failure (e.g. `falkordb-dev` held the port in the
   `falkor-chat` example, §2.4), it prints the error verbatim and does **not** write the marker;
   the prompt addendum (§3.4) instructs `tank` to report that verbatim and never attempt to free
   the port/resource itself.
3. **Before teardown:** `compose-down.sh <slug>` looks up the slug in `environments.json` (unknown
   slug → refuse immediately), then checks for `state/<slug>.json`. Absent → refuse, "not something
   I brought up." Present but **older than a configurable freshness bound** (default 6 hours,
   `TANK_MARKER_MAX_AGE_SECONDS` env override — `tank` runs are one-shot and short-lived, so a
   marker this old is presumed stale) → refuse, "marker present but stale, treating as not mine; a
   human should verify." Present and fresh → run `docker compose -f <composeFile>
   --project-directory <projectDirectory> down` (fixed, no flags beyond that, so `-v`/`--volumes`
   is structurally impossible here regardless of anything in the permission table) and delete the
   marker on success.

**Disclosed trade-off:** an environment `tank` genuinely brought up and that is still running, but
past the freshness bound, will be refused at teardown (a false "not mine" rather than a wrong
teardown) — deliberately erring toward FR-7's "unsure → refuse" posture rather than trusting an
old marker. A human can always tear it down by hand in that case; `tank` will not.

`opencode/agents/tank/state/` is gitignored (bookkeeping, not a deliverable) — added to a new
`opencode/.gitignore`.

### 3.4 The tank-specific prompt addendum (not shared with `claude devops`)

Appended after the shared persona in `tank`'s `prompt` field (§3.1), covering only what's
genuinely different about running headless:

- Context: invoked via `opencode run --agent tank`, no live Claude Code session, no human present
  mid-run, running on a small local model — be conservative, don't improvise around a denial.
- Destructive-ops posture override: where the shared persona says "stop and get explicit
  confirmation, then wait" (fine for `claude devops`, which always has someone to wait for), `tank`
  cannot wait — a denied command is final. Never retry with different flags, never look for an
  alternative command with the same effect, never treat a permission denial as something to work
  around. Say plainly in the final report what would have been done and why it was refused.
- FR-7 ambiguity clause: if unsure whether something is safe, don't try it — treat it as unsafe,
  note it as skipped/deferred to a human, and move on.
- How to act on a bring-up/teardown request (revised for the wrapper-script design, §3.2/§3.3):
  identify which known environment is meant (consult `environments.json`'s labels — read-only,
  `tools.read` stays enabled — or the README) and invoke the matching wrapper script with exactly
  that slug (`compose-up.sh <slug>` / `compose-down.sh <slug>`); if no listed environment matches
  what was asked for, say so and stop rather than guessing at a compose file path — `tank` never
  constructs a `-f`/`--project-directory` value itself, by design, so there is no "close enough"
  path to fall back to. Report exactly what the script printed (success, "already running", "not
  mine", "stale marker", or an error) — the script's own refusal messages already say why.
- What "the health/hygiene check" means by default (container/service status, disk usage, dangling
  images/volumes, docker build-cache usage) when no more specific request is given — ties FR-1's
  default behavior to the allow-listed read-only commands in §3.2 so the model isn't guessing at
  what it's permitted to try.

### 3.5 FR-8 — repo scope

`opencode/agents/tank/opencode.json` is a project-scoped config (no `~/.config/opencode/
config.json` changes). Per §2.3's verified project-scoping behavior, it is only picked up when
`opencode` is run from `opencode/agents/tank/` (or with `--dir opencode/agents/tank`) — the
convenience scripts in §5 always `cd` there first, mirroring `deprecated/opencode/agents/severino/
tests/run.sh`'s own `( cd "$PROJECT_DIR" && opencode run ... )` pattern.

### 3.6 Not foreclosing the eventual merge

Checked explicitly, since the Intent section records the long-term direction as `tank` eventually
absorbing `claude devops` entirely:

- The persona split (§3.1) makes `devops-persona.md` the actual center of gravity — the day
  `claude devops` is retired, `devops.md` and its sync script disappear, `devops-persona.md` and
  its `{file:...}` consumer stay. Nothing here assumes two permanently-separate personas; it
  assumes one persona with (today) two consumption mechanisms.
- The tank-specific addendum (§3.4) is deliberately scoped to *headlessness*, not to "being
  tank" — if `tank` later gains an interactive mode, that addendum's destructive-ops override
  section would simply stop applying (interactive `tank` could "wait" the same way `claude devops`
  does today), not need a rewrite of the shared core.
- The permission-table mechanism (§3.2) is OpenCode-native and doesn't reference `claude devops`
  or Claude Code hooks anywhere except in the defense-in-depth list's parity comment — nothing
  there needs `claude devops` to keep existing.
- The `environments.json` + wrapper-script mechanism (§3.2/§3.3, revised post-review) is *more*
  future-proof than the design it replaced, not less: a second environment is a one-line,
  reviewed JSON edit with **no permission-table change at all** (the three wrapper-script allow
  patterns are already generic across every `environments.json` entry). A second *mode* of `tank`
  (e.g. interactive) would extend the addendum split in §3.4, not this mechanism or the shared
  core.

## 4. Step-by-step implementation

1. **`claude/devops/devops-persona.md`** — extract from today's `claude/devops/devops.md` per the
   exact section list and two prose edits in §3.1. No frontmatter.
2. **`claude/devops/devops.md`** — keep frontmatter; replace the body with the marker-wrapped copy
   of `devops-persona.md` plus the four Claude-only pieces, per §3.1. Add a one-line HTML comment
   near the top of the marked span: `<!-- generated from devops-persona.md; edit that file, then
   run claude/devops/scripts/sync-persona.sh -->`.
3. **`claude/devops/scripts/sync-persona.sh`** — read `devops-persona.md`, splice it between the
   two markers in `devops.md`, fail loudly (non-zero exit, clear message) if either marker is
   missing/duplicated. Run it once now to prove idempotency (second run = no diff).
4. **`opencode/agents/tank/environments.json`** (new) — seed with the one `falkor-chat` entry per
   §3.2's shape. This is the trusted, human-reviewed allow-list every other compose-facing piece
   below reads from — build it before the scripts and config that depend on it.
5. **`opencode/agents/tank/scripts/compose-status.sh`, `compose-up.sh`, `compose-down.sh`** (new,
   the *inner* scripts `tank` itself invokes via bash) — implement the slug-lookup-then-fixed-argv
   logic in §3.2/§3.3: exactly one argument required, exact-match lookup against
   `environments.json` (refuse on any miss — extra text, unknown key, a `../`-shaped value — same
   handling for all of them, verified against a mock of this logic before wiring it into the real
   scripts, per §5), `REPO_ROOT` resolved from the script's own location (no baked-in absolute
   path needed here). `compose-up.sh`/`compose-down.sh` own the state-marker read/write in their
   own code per §3.3 — plain shell file I/O, not an OpenCode tool call.
6. **`opencode/agents/tank/opencode.json`** — provider block (`lmstudio`, same shape as
   `deprecated/opencode/agents/severino/opencode.json`, default model `mistralai/ministral-3-3b`
   per the live-verified deny+report flow in §2.3; list `nvidia/nemotron-3-nano-4b` as a second
   entry in `provider.lmstudio.models` for an easy swap, per severino's own convention). One agent,
   `tank`: `mode: "primary"`, `model: "lmstudio/mistralai/ministral-3-3b"`, `prompt` per §3.1,
   `tools` per §3.2 (`write`/`edit` both `false`), `permission.bash` per §3.2's three tiers, with
   this machine's actual absolute repo path anchoring only the three wrapper-script allow patterns
   (never a raw `docker compose` pattern).
7. **`opencode/agents/tank/state/`** — empty dir (`.gitkeep`); **`opencode/.gitignore`** — ignore
   `agents/tank/state/*` except `.gitkeep`.
8. **`opencode/agents/tank/README.md`** — setup/run instructions mirroring `deprecated/opencode/
   agents/severino/README.md`'s shape: prerequisites (OpenCode + LM Studio, link to
   `opencode/docs/manuals/local-llm.md`), how to run the three invocation shapes (health check,
   bring-up, teardown — via the outer scripts in step 9), how to add a second environment (edit
   `environments.json`, nothing else), what the state marker is and where it lives, its freshness
   bound and how to override it, and an explicit note that editing `claude/devops/devops-persona.md`
   is how you change `tank`'s persona (never edit the `prompt` field's addendum text expecting it
   to affect `claude devops`, and never edit `devops.md` directly expecting it to survive the next
   `sync-persona.sh` run).
9. **`opencode/agents/tank/scripts/health-check.sh` (no args), `bring-up.sh <slug>`,
   `tear-down.sh <slug>`** (the *outer* scripts a human/scheduler invokes — distinct from the
   *inner* scripts in step 5 that `tank` itself invokes) — each a thin, reviewed wrapper: `cd
   opencode/agents/tank && opencode run --agent tank "<fixed message template, with <slug>
   substituted in>"`. The bring-up/tear-down scripts take an `environments.json` slug as their one
   argument (e.g. `./bring-up.sh falkor-chat`) so the fixed message names the environment
   unambiguously. These are the reviewed, scriptable invocation surface (and the natural hook point
   for the explicitly-out-of-scope future scheduler — nothing here builds one, but nothing blocks
   dropping a cron/systemd-timer entry that calls `health-check.sh` or `bring-up.sh <slug>` later).
10. **`opencode/AGENTS.md`** + **`opencode/CLAUDE.md`** (new, `@AGENTS.md` stub) — a short context
   file per root `AGENTS.md`'s convention: what `opencode/` is now (greenfield, `agents/tank/` is
   the only live agent, `deprecated/opencode/` is read-only history), the persona-sharing rule
   from §3.1 stated as a "don't" (don't hand-edit `tank`'s persona text in `opencode.json`;
   edit `claude/devops/devops-persona.md`), and a pointer to `opencode/docs/manuals/local-llm.md`.
11. **Root `AGENTS.md`** — two small edits to fix now-stale text (predates this plan, but this is
   the natural moment): the `opencode/` bullet's parenthetical (drop the retired
   `rpg`/`coding-senior`/`severino` list, describe `agents/tank/` instead) and the Component docs
   table row (`opencode/AGENTS.md` · `opencode/docs/manuals/local-llm.md` · `opencode/agents/tank/
   README.md`, dropping the stale `opencode/agents/severino/README.md` / `opencode/local-llm.md` /
   `opencode/skills/README.md` entries that no longer resolve).
12. **Verify** — run through the Test strategy below; fix forward. Given the design change is a
   direct response to a live-reproduced bypass, re-running the reviewer's own reproduction against
   the shipped config (not just this plan's prose) before calling this done is not optional — see
   §5's new smuggling-regression test.

## 5. Test strategy

**Wrapper-script logic checks, no OpenCode/LM Studio needed** (plain bash, run against
`compose-status.sh`/`compose-up.sh`/`compose-down.sh` directly, or a mock of their argument-
validation logic as done during this revision — see §2.5):
- Exactly-one-argument enforcement: 0 args and 2+ args (e.g. a slug followed by unquoted
  `-f x --project-directory y`) both refuse before any `docker` call.
- Exact-match lookup: a known slug (`falkor-chat`) resolves; an unknown slug, a slug with trailing
  garbage as a single quoted argument, and a `../`-shaped value are all refused identically — none
  reach a `docker compose` invocation. (This is the same battery §2.5 ran against a mock of this
  logic before writing it into this plan; re-run it against the real scripts once they exist.)
- `compose-down.sh` re-resolves `composeFile`/`projectDirectory` from `environments.json` at
  teardown time and never reads them from the marker (grep the implementation for any read of
  those two keys from the marker JSON — there should be none).

**Config-shape checks** (no LM Studio needed, run from `opencode/agents/tank/`):
- `opencode debug agent tank` — assert the resolved `prompt` contains both a known
  `devops-persona.md` phrase and the addendum's headless-context sentence; assert `model.providerID
  == "lmstudio"`; assert `tools.write == false` and `tools.edit == false`; assert the `permission`
  list, filtered to `bash`, is exactly: `*` deny first, then the read-only allow-list and the three
  wrapper-script allow patterns, then the defense-in-depth denies (destructive-catalog +
  repeated-scoping-flag net + metacharacter net) last, in that order (this is the property that
  makes "last match wins" resolve correctly — an ordering bug here silently reopens FR-7); assert
  **no `bash` allow pattern's literal text contains the substring `docker compose`** — the absence
  of a raw compose pattern is itself the property under test, not just its ordering.
- `claude/devops/scripts/sync-persona.sh` run twice in a row on an unchanged
  `devops-persona.md` → second run is a no-op diff (idempotency).

**Smuggling-regression test — reproduce the reviewer's exact finding against the shipped config,
required before this plan is considered addressed, not optional:**
- Using the same `opencode debug agent tank --tool bash --params '{"command":"..."}'` technique
  the review used (confirmed on this box to actually invoke the real permission classifier and, on
  an allow, actually execute — not just print the resolved shape), submit the review's exact
  smuggled string (a legitimate-looking `docker compose -f <in-repo-file> --project-directory
  <in-repo-dir> -f <out-of-repo-file> --project-directory <out-of-repo-dir> down`, disposable
  fixtures only, same cleanup discipline as §2.5/the review's own §5) and confirm it is now
  classified `deny` (falls through the default catch-all, since no allow pattern's literal prefix
  can ever match a bare `docker compose` invocation any more).
- Additionally attempt the same smuggling *shape* against the wrapper-script call itself —
  `<repo>/.../compose-down.sh falkor-chat -f <out-of-repo-file> --project-directory
  <out-of-repo-dir>` as a single, unquoted bash command — and confirm the permission layer allows
  the invocation (the pattern matches, as expected) but the **script itself** then refuses on the
  argument-count check before touching `docker`, per the wrapper-script logic checks above. Both
  layers should be independently verified: the permission table's job is only "can this script be
  reached," the script's job is "is this exactly one known slug."

**Live smoke tests** (LM Studio up, run from `opencode/agents/tank/`, `falkordb-dev` confirmed
stopped first so the `falkor-chat` example can bind :6379 cleanly):
1. `health-check.sh` — completes (exit 0), report mentions container/volume/disk signals, and
   `docker ps`/`docker volume ls` before/after the run are byte-identical (nothing was touched).
2. Ask `tank` directly to run something destructive (e.g. "delete the falkordb-data volume") —
   confirm refusal + a report naming what it would have done, exit 0, `docker volume ls`
   unchanged.
3. Ask `tank` to run an arbitrary non-allow-listed command (e.g. `pip install requests`) — confirm
   default-deny fires (not a specific pattern) and it's reported as refused.
4. `bring-up.sh falkor-chat` from cold — `falkor-chat` containers come up healthy,
   `state/falkor-chat.json` exists with a fresh `broughtUpAt`/`sessionId`, and it contains no
   compose-path fields.
5. `tear-down.sh falkor-chat` immediately after (4) — containers stop, marker is removed,
   `falkordb-data` volume still exists (`docker volume ls`) — proves `down` never became `down -v`.
6. Start the `falkor-chat` stack **outside** `tank` (plain `docker compose up`, no marker written),
   then ask `tank` to tear it down — confirm refusal ("not something I brought up") and the stack
   is left running.
7. Confirm repo scoping: run `opencode run --agent tank ...` from the repo root (no `cd`, no
   `--dir`) and confirm it fails to find the `tank` agent / doesn't pick up the lmstudio provider —
   proves the config is genuinely project-scoped, not accidentally global.
8. **Genericity, not just the one example:** add a *second*, throwaway entry to a scratch copy of
   `environments.json` pointing at a trivial fixture compose file under the repo, and confirm
   `bring-up.sh`/`tear-down.sh` work against it with **no permission-table change** — proves the
   allow-list, not a glob, is what makes this generic.
9. **Unknown/unlisted environment:** ask `tank` (or `bring-up.sh`) about a slug that isn't in
   `environments.json` — confirm refusal at the script layer ("not a known environment"), never a
   fabricated `-f`/`--project-directory` guess.
10. **Marker freshness:** hand-edit a fresh marker's `broughtUpAt` to be older than
    `TANK_MARKER_MAX_AGE_SECONDS`, then ask for teardown — confirm refusal ("stale, treating as not
    mine") and the environment is left running.

**Edge cases already covered above, called out explicitly:** ambiguous/unknown command (3),
unlisted environment (9), already-running environment at bring-up time (§3.3 step 1 — worth a
dedicated live case: run `bring-up.sh` twice in a row for the same slug and confirm the second call
reports "already running" without rewriting the marker), teardown-of-not-self-started (6), and
stale-marker teardown (10).

## 6. Kaizen capture

Recorded in `kaizen_team` (below) rather than restated here: OpenCode's `{file:...}` prompt-
templating mix-with-literal-text behavior, the `"ask"`-hangs-headless / `"deny"`-completes-and-
reports distinction, the permissive-by-default baseline permission table, and the benign Ministral
title-generation Jinja error — all verified live in §2.3 and not previously documented anywhere in
this repo. Added this revision: the glob-smuggling fact `security-expert`'s review surfaced (§3.2)
— OpenCode's `permission.bash` `*` matches across a mutating command's own flag boundaries, so a
command with repeatable, last-wins scoping flags (like `docker compose -f`/`--project-directory`)
can have a second, fully-formed instance of those flags smuggled inside an allow pattern's wildcard
span with no shell metacharacter involved — worth a team-wide flag since it generalizes past this
one agent to any future `permission.bash` allow-list gating a command with similar flag semantics.

## 7. Risks & open questions

- **Resolved:** which environment(s) "a demo/dev environment" covers. Stakeholder confirmed (via
  `teco`) that it's generic — `tank` must handle whatever compose-based environment it's asked
  about, not one hard-coded stack. Delivered via `environments.json` (§3.2), not via glob-matching
  arbitrary repo paths (that approach was tried, found exploitable, and replaced — see next
  bullet). Adding a new environment is a one-line, reviewed JSON edit.
- **Resolved (was the BLOCKER): glob-smuggling in the compose allow-list.**
  `security-expert`'s review (`opencode/docs/reviews/devops-opencode-headless.md`) live-reproduced
  tearing down an out-of-repo-scope disposable stack by smuggling a second `-f`/
  `--project-directory` pair inside the wildcard span of this plan's original `down`/`up*` compose
  patterns — no shell metacharacter needed, since Docker Compose's own last-flag-wins semantics did
  the redirecting, not a shell feature the metacharacter net could catch. Fixed architecturally
  (§3.2/§3.3): `tank` never builds a compose invocation from a model-authored string at all; three
  fixed wrapper scripts do the construction from a slug looked up in a repo-committed allow-list.
  §2.5/§5 specify live verification of the fix (both the permission-classifier layer and the
  script's own argument validation) as a required, not optional, step before this is closed out.
- **Resolved (was MAJOR): marker-slug basename collision.** Fixed by construction, not a smarter
  hash — the slug is now `environments.json`'s human-assigned key, not derived from a path, so two
  environments can only collide if a human gives them the same key (an authoring mistake, caught
  at review, not a runtime hazard) — §3.3.
- **Resolved (was MAJOR): marker has no tamper/freshness check.** Added a freshness bound
  (default 6h, overridable) and provenance fields (session nonce, PID, host) — disclosed honestly
  as a staleness/collision safeguard, not a defense against an adversary with `tank`'s own
  filesystem write access (no local secret could make that meaningful; that threat model is
  already outside every mechanism in this plan, same as `environments.json`'s own git-commit trust
  boundary) — §3.3. Trade-off disclosed there too: a genuinely-`tank`-started environment past the
  freshness bound is refused at teardown (false "not mine"), never torn down on a stale marker.
- **Resolved (was MINOR): `up*`'s wildcard more permissive than `down`'s exact match.** Both are
  now wrapper-scripted with the identical fixed-argv discipline (§3.2) — there is no longer a
  bare `up*`/exact-`down` asymmetry to reason about, since neither is a raw glob pattern anymore.
- **Not independently re-verified (was MINOR, unconfirmed): `write: {"state/*": "allow"}`
  path-traversal.** The review could not complete its own live check of this (an unrelated
  OpenCode tool-resolution snag, documented in its report) and asked that it be explicitly
  verified rather than left unconfirmed. **This revision does not verify it — it removes the
  surface instead:** `tank`'s `tools.write`/`tools.edit` are now both `false` (§3.2), because the
  state marker is written by the wrapper scripts' own code, never by an OpenCode `write` tool call
  `tank` invokes. There is no `write` permission rule left to test. This is a stronger resolution
  than confirming the old pattern was safe, but it is a different action than the one requested —
  flagged explicitly so a re-reviewer can judge whether "eliminated, not verified" is an acceptable
  answer to the original ask.
- **New, disclosed residual: `environments.json`'s own trust boundary.** Anything in that file is
  reachable by any bring-up/teardown request naming its slug — this is by design (it's the
  allow-list), but it means the file's contents are exactly as trustworthy as anything else checked
  into this repo (ordinary git-commit review), the same boundary §3.2 already relies on for "a
  compose file has to already be checked into the repo to be reachable at all." Not a new hazard
  relative to the rest of the plan, but worth naming since it's now the single point where
  "which environments can `tank` touch" is decided.
- **Open, per the review:** whether the repeated-scoping-flag deny net (§3.2 tier 3, `*-f *-f *`
  etc.) is worth keeping now that it protects nothing load-bearing (no raw compose allow-pattern
  exists for it to guard). Kept as cheap insurance against a future regression; a re-reviewer may
  reasonably judge it dead weight instead — not a blocking disagreement either way.
- **Model choice was validated shallowly.** `mistralai/ministral-3-3b` completed the deny+report
  flow correctly and quickly in live testing (§2.3); `nvidia/nemotron-3-nano-4b` (severino's
  preferred model for persona-following) was not conclusively exercised in the same test (a 60s
  probe timed out with no output, inconclusive rather than a confirmed regression). Recommend the
  implementer keep `ministral-3-3b` as the shipped default and treat a `nemotron` swap as a later,
  separate A/B, not a blocker.
- **Who implements this.** The bulk of the work is agent/prompt authoring across two tools
  (Claude Code + OpenCode), which is `cobb`'s stated specialty (its description explicitly covers
  "OpenCode (agents, skills)") more than general application code — `tank` is not, however, added
  to `claude/AGENTS.md`'s agent roster (it's an OpenCode-side artifact, not a `claude/` team
  member), so the roster-update maintenance rule there does not apply. `coder` is an acceptable
  alternative for the mechanical script/JSON pieces if the work is split. This plan does not
  mandate one over the other.
- **Recommended next step:** a second `security-expert` pass over this revision specifically —
  re-run the smuggling reproduction against the new design (§5), confirm the wrapper-script
  argument-validation logic as actually implemented (not just the mock verified in §2.5), and
  weigh in on the two still-open items above.
