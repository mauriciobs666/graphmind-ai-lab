# The one salesperson UI — Plan Review

> **Status:** active · **Owner:** `analyst` · **Tracks:** — (M<n> TBD) · **Reviews:** `docs/plans/salesperson-ui.md`

## 1. Scope & verdict

**Reviewed:** `docs/plans/salesperson-ui.md` (owner `architect`, `Status: active`, 726 lines, 17
steps S0–S16), statically, against its source requirement `docs/requirements/salesperson-ui.md`
(FR-1…FR-11, AC-1…AC-11, and its binding "Out of scope" section). No implementation exists yet;
this is the pre-implementation plan gate (U2 of `docs/plans/salesperson-ui-coordination.md`).

**Baseline for verification:** `falkor-chat` at `4bb96e1` (working tree carries only `teco`'s
uncommitted `falkor-chat/AGENTS.md` v4→v5 refresh, excluded from review), the live FalkorDB
instance at `localhost:6379` (v4.18.11), and the pinned interpreter at
`falkor-chat/server/.venv` (anyio 4.14.1, uvicorn 0.49.0). I re-verified every load-bearing
source citation in §2.2/§2.3/§4 rather than accepting the plan's summary; the audit trail is
in §5 (Appendix A).

**Mid-run premise change.** `teco` delivered the stakeholder's §8 answers while this review was in
progress. Two change premises and are reviewed as corrected: **OQ-5** — the
`FALKORCHAT_PRESENTER_KEY` design is rejected in favour of **localhost-bound presenter routes**
(reviewed at B1; the plan's "is a single operator key a new authentication system?" argument is
moot and is not adjudicated here); **OQ-3** — there is no `salesperson-ui/`; the retired Streamlit
app moves to `deprecated/salesperson/` and the new client takes the freed `salesperson/` name
(consequences at M9, m10, m14). OQ-1/2/4/6 were accepted as the plan proposed.

**Verdict: needs changes.** Four blockers. The plan's *grounding* is the best I have reviewed in
this repo — I could not find a single false source claim in §2 — and its core tenancy decision
(§4.3, one workspace, one actor anchor per participant) is correct and correctly evidenced. What
it is missing is at the edges of that decision: the isolation argument covers only the surface the
plan builds and not the two unauthenticated surfaces already mounted in the same process (B2);
the bring-up sequence provisions the wrong workspace (B3); two repository primitives the design
requires do not exist and are in no step's scope (B4); and the newly-chosen presenter control is
weaker than the one it replaces under the deployment the plan's own diagram promises (B1).

**CPG: used `cpg_salesperson` — queried only to establish that it is a live, populated CPG of the
*retired* Streamlit app (359 `METHOD` nodes), which is what makes the `cpg_<component>` naming
collision at m14 real; `cpg_falkorchat` was excluded by the dispatching brief as mid-rebuild, so
every `falkor-chat` claim below was verified against source and the live instance directly.**

---

## 2. Findings

### Blockers

#### B1 — Localhost-bound presenter routes are defeated by the deployment §3 promises, and by one environment variable

*(Against the corrected OQ-5 premise. Evidence executed on the pinned venv — Appendix A.1.)*

`request.client.host` derives from the ASGI `scope["client"]`, which is not settable by a client
header — so a bare check is sound. But uvicorn 0.49.0 wraps the app in `ProxyHeadersMiddleware`
with **`proxy_headers=True` by default**, trusting `os.environ.get("FORWARDED_ALLOW_IPS",
"127.0.0.1")`. Two plausible configurations invert the check completely:

1. **Any TLS-terminating reverse proxy.** §3's diagram says the phones reach the server over
   **HTTPS**; plain uvicorn does not provide that, so a proxy is the realistic path. Behind one,
   every request's peer *is* `127.0.0.1` — which is trusted by default — so a LAN client sending
   `X-Forwarded-For: 127.0.0.1` has `scope["client"]` rewritten to `('127.0.0.1', 0)` and gets the
   presenter surface. Without XFF at all, every request simply *is* loopback: reset-everyone open
   to the whole audience, unconditionally.
2. **`FORWARDED_ALLOW_IPS=*`** inherited from the operator's shell (it is read from the
   environment, not only a CLI flag) makes a `192.168.x.x` peer trusted, same outcome.

This is **weaker than the key it replaces**: the key required network reach *and* a secret; this
requires only that the server be wrong about one field. Concretely, the design must: derive the
peer from `scope["client"]` only and **fail closed when it is `None`**; accept `127.0.0.0/8`,
`::1` and `::ffff:127.0.0.1` (uvicorn's own default trust list does **not** include `::1` — a
presenter opening `http://localhost:PORT` on a dual-stack host can land on it); run uvicorn with
`--no-proxy-headers` in `start_demo.sh` and assert it at startup; **refuse the presenter routes
outright, loudly, if any `X-Forwarded-*` header is present**; and carry a §6.2 contract test that
`client=("192.168.1.50", …)` **plus** `X-Forwarded-For: 127.0.0.1` is refused. Two further
consequences for the architect's revision: §5.2's `POST /shop/api/presenter/session` and S10's key
exchange + rate-limiting disappear, so §5.2's blanket "all presenter routes require `Bearer
presenter.<presenterToken>`" and S10's done-condition ("a wrong key is refused and counted") are
both void; and **§6.3 #8's AC-5 pass can no longer be driven from a phone** — the qa-engineer must
run it on the host, which needs saying in the test plan.

I also note for the record that the plan's own §4.3 rejected this option because it "fails when
the presenter is on their phone." That trade-off is the stakeholder's to make; the security
inversion above is not, and should go back with this finding.

#### B2 — The unauthenticated legacy REST + web + MCP surface is an unaddressed AC-2 read path

§4.3's four-part isolation argument covers `/shop/api` only. `api.build_router` is mounted on the
same app with **no authentication of any kind**, resolving every call through the process-constant
`config.get_context()` → `CallContext(ws=FALKORCHAT_WS_ID, actor=FALKORCHAT_USER_ID)`
(`falkor-chat/server/falkorchat/config.py`, `get_context`). It exposes `GET /channels`,
`GET /channels/{cid}/threads`, `GET`+`POST /threads/{tid}/messages`, `GET /search?q=`,
`GET /messages/{id}`, `GET /threads/{tid}/participants`. `falkor-chat/web/app.js` (`:66`, `:107`,
`:139`, `:225`, `:726`) drives exactly that surface, and `app.py` mounts it at `/` — the URL a
participant reaches by trimming `/shop` off the demo link. `/mcp` (`mcp.py:_get_context`) is the
same seam.

If `FALKORCHAT_WS_ID` resolves to the demo workspace, any phone on the LAN that opens the demo
host **root** gets a working roster of every participant's `Channel`/`Thread` with full
transcripts, a workspace-wide full-text search over every message, and a post box that writes into
anyone's thread as `u1`. That is an AC-2 failure by inspection, and §4.3, §5.1 S8/S11, §6.2 and
§10 are all silent on it — S8's only mount assertion is that `/shop` *does not shadow* `/`, i.e.
the plan deliberately keeps the surface reachable.

**Fix:** (a) make the storefront refuse to start when `FALKORCHAT_WS_ID == FALKORCHAT_DEMO_WS`,
and (b) gate the legacy router *and* the `/` mount off when `FALKORCHAT_STOREFRONT_ENABLED=1`
(a `create_app(..., dev_surface=False)` parameter — the same shape as the existing `web_dir`
switch). Add to §6.2: with the storefront enabled, `GET /channels` and `GET /search?q=` either 404
or resolve to a workspace containing no participant data. See B3 — these two are the same trap
from opposite sides.

#### B3 — S11's bring-up sequence never provisions the `Agent` into the demo workspace; every participant's first message fails

`scripts/seed_demo.sh` takes the workspace as `WS_ID="${1:-${FALKORCHAT_WS_ID:-acme}}"`
(`seed_demo.sh:41`). S11 lists `bootstrap_schema.sh <demoWs>` followed by a **bare**
`seed_demo.sh` (and a bare `seed_salesperson.sh`). With B2's fix in place — `FALKORCHAT_WS_ID` ≠
the demo workspace — those seed the *wrong* graph, and `ws:{DEMO_WS}` ends up with no
`Agent {agentId: config.AGENT_ID}` node and no materialized def snapshot.

Every storefront post carries `mentions=[AGENT_ID]`. `services.post_message` calls
`_validate_and_derive_role` (`services.py:813-836`) first, which resolves author **and** mentions
in one `resolve_member_kinds` lookup and raises `UnknownMemberError` on an unresolvable mention
**before any write**. So every participant's first message 500s, the agent never fires, and S11's
own done-condition ("a reachable `/shop` with a working join") is met while the demo is dead.

The trap is symmetrical and worth stating explicitly in §4.3: the natural thing for the S11
`devops` delegate is `FALKORCHAT_WS_ID=$DEMO_WS`, because that is how every other seed script
derives its target — and that is exactly what opens B2. **Fix:** S11 passes the demo workspace
explicitly to every seed script (`seed_demo.sh "$DEMO_WS"`, `seed_salesperson.sh "$DEMO_WS"`,
`seed_workflows.sh "$DEMO_WS"` — see m4), while leaving `FALKORCHAT_WS_ID` pointed elsewhere; and
`Storefront` startup fails loudly if the demo `Agent` node is absent.

#### B4 — Two repository primitives the design requires do not exist, are in no step's scope, and S7 cannot be built without breaching the layering rule

§4.6 requires (i) the participant's **current order** — `orderId`, `status`, frozen lines, total —
inside `GET /shop/api/state`, and (ii) a `(:Customer {customerId})-[:PLACED]->(:Order {orderId})`
ownership check before the CAS. Neither exists. `services` exposes only
`get_order_status(ctx, order_id=…)` (`services.py:2795`) and `repository` only
`get_order(ws, order_id=…)` (`repository.py:2972`). The complete cart/order/profile repository
surface is `ensure_customer`, `ensure_cart`, `add_to_cart`, `adjust_cart_item`, `read_cart`,
`clear_cart`, `place_order`, `get_order`, `fulfill_order`, `deliver_order`, `cancel_order`,
`upsert_profile`, `get_profile` — **there is no method that resolves a customer's orders at all.**

S4's six named methods are provisioning/reset only; S0's scope is provisioning Cypher, the two
reset deletes, a `GRAPH.PROFILE` check and an index yes/no; and §5.0 pins `repository.py` to "S4
only" and `services.py` to "S2 only" while S7's Files column is `storefront.py` alone. The S7
delegate therefore either stalls or writes Cypher into `storefront.py`, against
`falkor-chat/AGENTS.md` rule 1 and DESIGN §14.2's "all Cypher lives in `repository.py`" (quoted in
`db.py`'s own module docstring).

**Fix:** add `get_customer_current_order` (or `list_customer_orders`) and
`order_belongs_to_customer` to **S0's scope and S4's method list**, with their `QUERIES.md` §18
entries (§17 is the current highest — S4's "new §18" is correct); update §5.0's `repository.py`
and `services.py` rows; and add to S7's done-condition that the order block in `get_state` is
populated from a repository read, not composed in `storefront.py`.

---

### Majors

#### M1 — §4.8 and §5.2 contradict each other on what "reset mine" does to the participant's identity

§4.8's survivor column: *"their `User` + `Channel` + token; a **fresh** `Thread` is minted and
`User.threadId` repointed."* §5.2's row for the same route: *"`204`; token invalidated, client
returns to join."* Both cannot hold. The choice is load-bearing across S0 (does the delete touch
`User`?), S4 (`reset_participant`'s contract), S7, S10 (`reset_all` "invalidates every participant
token" — same mechanism or not?) and S12 (the prefilled-rejoin flow). If the token *is*
invalidated and the participant re-joins, they get a fresh `participantId`, and the old
`User`/`Channel` become orphans that still appear in `GET /shop/api/presenter/participants`.

**Recommend** keeping the identity: token survives, a fresh `Thread` is minted, `204` returns the
new thread id, and the client returns to a "pick your language" step rather than the full join
screen (which also preserves §4.8's own stated rationale for re-offering the language).

#### M2 — The turn queue has no per-participant serialization; two rapid messages start two concurrent runs on one thread

`trigger.maybe_trigger` step 2 resumes only a run in status **`waiting`** (`trigger.py`, via
`find_waiting_run_for_thread`). A run still in `running` is invisible to it, so step 3 fires and
`start_workflow_run` creates a **second** run against the same thread. Both drive the same
`assistant` step, both call `post_message`, and both consume the resource §4.4 identifies as the
ceiling. Nothing in the platform guards start-while-running — only the `waiting→running` resume
CAS is single-flight.

§5.1 S9's done-condition asserts only *global* FIFO ("positions 0/1/2, complete in order"), and
§4.4 measure 1 relies on the client disabling send while `turn.state == thinking` — a client-side
control, which AC-2/AC-3's load harness (§6.4, scripted 5-turn conversations) will not honour.

**Fix:** key the queue by `participantId` — at most one in-flight **and** one pending turn per
participant, further posts refused with `409` — and add the done-condition: two posts issued
100 ms apart by one participant produce exactly **one** `WorkflowRun` for that thread.

#### M3 — `build_image_manifest()` points at the source directory, not the served one; AC-11 can silently degrade to "every product text-only"

§4.7 places assets at `salesperson-ui/public/products/<productId>.webp` and says the server "lists
that directory" at startup, while `app.py` serves the bundle from `FALKORCHAT_STOREFRONT_DIR`
(the build output). Vite copies `public/**` into `dist/`, so the two agree only when the source
tree happens to sit next to the build output. Ship `dist/` alone (the shape OQ-6 contemplated), or
point `FALKORCHAT_STOREFRONT_DIR` anywhere else, and the manifest is empty, every `imageUrl` is
`null` — and **§6.3 #9 and S14's DOM assertion still pass**, because AC-11's negative branch
("no placeholder element") masks the total failure of its positive branch.

**Fix:** the manifest lists `<FALKORCHAT_STOREFRONT_DIR>/products/`; `imageUrl` is
`/shop/products/<id>.<ext>` consistent with Vite `base: "/shop/"`; the accepted extension set is
named; and S7's done-condition asserts a **non-empty** manifest against a fixture asset directory
while §6.3 #9 gains a positive case (at least one product renders an `<img>`).

#### M4 — AC-8 / FR-10 is not actually covered: the join display name never reaches the profile, and v5's prompt asks early, not at order time

AC-8 is *"…**when they place an order**, then the UI's conversation prompts for and then displays
their name/delivery address."* §10 satisfies it with "S1 carries v5's `get_profile`/`save_profile`
prompt guidance forward" — but that guidance (`proof_defs.py`, `SALESPERSON_DEF`'s third
`systemPrompt` paragraph) reads *"Call `get_profile` once, early in the conversation… Ask for
whichever of name or delivery address `get_profile` shows as missing — only once per
conversation."* There is no order-time address confirmation, and `services.place_order`
(`services.py:2748`) does not require an address.

Separately: `join(display_name, language)` writes only `User.displayName`, while
`services.get_profile` keys on `ctx.actor`'s `Customer` — so the profile panel shows an em-dash
for `name` until the model asks for a name the participant **already typed thirty seconds
earlier**. That is a visible parity regression against the old app's upfront-name sidebar, which
§2.4's own parity table sets as the FR-10 bar.

**Fix:** `join()` calls `services.save_profile(ctx, name=display_name)`; S1's one-sentence prompt
change grows a clause covering order-time delivery-address confirmation. And because that second
half is a *prompt-adherence* claim on a 3 B model, it belongs in §6.3's measured-run list beside
AC-9, not in the code-review column of §10.

#### M5 — "Roughly halves load on the LLM endpoint" (§4.4 measure 3) is off by 3–9×

Per storefront turn the dropped work is **one** embedding call (`background._safe_embed` →
`text-embedding-qwen3-embedding-0.6b`, 0.6 B, `embedding` timeout 30 s in
`falkor-chat/config/models.json`) against **up to eight** chat completions on
`mistralai/ministral-3-3b` (`SALESPERSON_DEF`, `maxIterations: 8`, `agent`/`step` timeout 180 s).
By request count that is 11–33 %; by GPU-seconds, far less.

The measure itself is right and should stay — with no `graphrag_retrieve` in the def, the
embedding is pure contention for zero benefit. But §4.4 is the section that feeds OQ-1's capacity
conversation with the stakeholder, and it is the one place in this plan where a wrong number costs
hardware money. **Fix:** restate as "removes one embedding call per posted message (~1 of 2–9
endpoint calls per turn, and a small fraction of GPU time)", and let §6.4 Run B measure the actual
delta rather than the plan asserting one.

#### M6 — §5.0's shared-file map is incomplete in three places, and §5.0 is what dispatch is gated on

Cross-checking every Files cell in §5.1 against the map:

- `storefront.py` is mapped "S6, S7, S10 — Serialize S6 → S7 → S10", but **S9 also owns it**. Read
  literally, the map permits S9 ‖ S10 on the same file.
- `storefront_api.py` is mapped "S8 only (created), S10 (extends)" — **S9 touches it too**.
- `falkor-chat/AGENTS.md` is touched by **S11** (script-table row) *and* **S16** (routes, scripts,
  env vars) and appears in **no row** — only root `AGENTS.md` does.
- Unmapped but currently conflict-free, worth listing for completeness: `schemas.py` (S8),
  `trigger.py` (S2), `falkor-chat/docs/QUERIES.md` (S4).

**Fix:** regenerate the map mechanically from §5.1's Files column rather than by hand; the
`storefront.py` row becomes `S6 → S7 → S9 → S10`.

#### M7 — Neither reset has a contract for turns already in flight

Both resets delete `Thread`, `WorkflowRun`, `StepRun`, `TraceEvent` and `Message` nodes that a
turn executing on S9's thread pool may be mid-write against. `services.post_message` raises
`ThreadNotFoundError` on a vanished thread; `_record`/`suspend_run` writes against a deleted run
silently no-op. "Reset everyone" mid-demo, with up to `turn_workers` turns in flight and a queue
behind them, is the realistic case, and it is precisely R4's "wrong and it either bricks the demo
or wipes a bystander" scenario arriving through the *timing* dimension R4 does not consider.

Neither §4.8, S0's scope, nor S7/S10's done-conditions mention quiescing. **Fix:** name the
quiesce contract in §4.8 and add it to S0 — a reset drains/cancels that participant's queued turns
and refuses (or waits, bounded) while one is in flight; `reset_all` stops intake first. S7/S10
gain the done-condition: a reset issued while a stub-LLM turn is in flight leaves no orphan
`StepRun`/`TraceEvent`/`Message`.

#### M8 — S0's reset scope and S4's done-condition are under-specified against the real label inventory

`scripts/bootstrap_schema.sh` creates indexes/constraints for `Agent, Cart, CartItem, Channel,
Chunk, Customer, Document, Entity, Message, Order, Product, ReadCursor, Step, StepRun, Thread,
TraceEvent, User, WorkflowDef, WorkflowDefSnapshot, WorkflowRun, WorkspaceConfig`, plus unindexed
`OrderLine`. §4.8's two columns never mention three of them:

- **`ReadCursor`** (per-member/thread, `HAS_CURSOR`) — orphaned by a thread delete, in neither column.
- **`WorkspaceConfig`** — the K-042 workspace model-override singleton. A broad "reset everyone"
  sweep that catches it silently changes model resolution mid-demo, undoing K-056's Ministral
  re-point for every subsequent turn. This must be an explicit survivor.
- `Document`/`Chunk`/`Entity` appear as survivors of `reset_all` but not in the reset-mine row.

S4's done-condition asserts only disjointness, `Agent` + `WorkflowDefSnapshot`/`Step` surviving,
and idempotency. It does not assert the rest of §4.8's survivor list, that `reference` is
untouched, or — the one most likely to be got wrong — that the delete is scoped to **the thread,
not the author**: the agent's own replies live inside the participant's thread, and an
author-scoped delete orphans them against a deleted `Thread`. **Fix:** hand S0 the label inventory
above as an explicit keep/delete checklist, and expand S4's done-condition to assert every
survivor by label plus a post-`reset_all` `verify_salesperson.sh` + `verify_catalog.sh` exit 0.

#### M9 — S16's acceptance command cannot pass as written, and S16's file list misses a live agent prompt that this work invalidates

S16's done-condition is *"`rg -n 'salesperson/' --glob '!docs/**'` is clean apart from intended
history."* Run verbatim at `4bb96e1` it returns **36 matches**. `--glob '!docs/**'` contains a
slash, so it is root-anchored and excludes only the root `docs/` tree — 21 of those matches are in
component `docs/` trees. The remaining **15 are outside every `docs/` tree**, in files S16's own
Files column does not touch: `cypher-mcp/README.md:433`, `claude/frontend-engineer/kaizen/plan.md`,
three `claude/*/kaizen/history.md` files, and — the one that matters —
`claude/frontend-engineer/frontend-engineer.md:18,20`. Under OQ-3's rename the check becomes
permanently red anyway, since both the new component and `deprecated/salesperson/` match the
pattern. (Also environment-specific: `rg` here is a **shell function**, not on `PATH`.)

`claude/frontend-engineer/frontend-engineer.md` is the prompt of the agent S12–S14 are dispatched
to. It states *"In this repo the running UIs are **Streamlit** apps (`salesperson/chatbot.py`)"*
and names `cpg_salesperson` as the CPG for `salesperson/chatbot.py`. After this work both
sentences are false, and per `claude/AGENTS.md` an agent edit must land with its `kaizen/{plan,
history}.md` and `claude/README.md` in the same change — i.e. this routes to **`cobb`**, not to
S16's `coder`. **Fix:** add a `cobb`-owned step (or unit) for the agent-prompt update; replace the
done-condition with a check that actually runs. This one is clean today apart from the two
frontend-engineer hits it is meant to catch (verified — Appendix A.2):

```
rg -n --no-heading -g '!deprecated/**' -g '!**/docs/**' -g '!**/kaizen/**' \
  'salesperson/(chatbot|cart|customer_profile|session_manager|diagnostics|agent|graph|cypher|prompts|utils_common)\.py'
```

---

### Minors

- **m1 — S1's blast radius is complete for code, not for docs.** The four listed files are exactly
  right: `proof_defs.py:301`, `seed_salesperson.sh:124`, `verify_salesperson.sh:55`,
  `test_salesperson_scaffold.py:325` are the only `v5` pins in code/scripts/tests. Two prose
  surfaces also pin it and are reachable only via S16: `falkor-chat/AGENTS.md` rows 82–83 (the
  script table narrating the whole `v1…v5` chain and `verify_salesperson.sh`'s expected version)
  and `falkor-chat/docs/BACKLOG.md`'s K-060/K-062 headings. Add the AGENTS.md rows to S1 so the
  doc is not stale for the whole S1→S16 window; leave BACKLOG to those defects' own tracks, and
  say so.
- **m2 — The anyio limiter must be raised *inside* the async lifespan.** Verified on
  `falkor-chat/server/.venv` (anyio 4.14.1): `to_thread.current_default_thread_limiter()` returns
  `total_tokens == 40`, setting it to `100` works, and calling it **outside a running event loop
  raises `anyio.NoEventLoopError`** — it is event-loop-scoped. §4.4 measure 2 and S9 both say only
  "at startup". Specify "inside `create_app`'s `_lifespan`, before `yield`". Also worth stating
  that with measure 1 in place the raise is close to cosmetic (50 clients × 2 polls / 2 s ≈ 50
  req/s of millisecond-scale reads against 40 threads) — keep it, but don't let it read as
  load-bearing.
- **m3 — §2.3 and R10/S12 disagree on the poll interval** (3 s → "~33 req/s" vs 2 s → "~50
  req/s"). Immaterial to the conclusion; pick one so the two capacity paragraphs agree.
- **m4 — `GET /workspaces/{ws}/readiness` will not be green in the demo workspace**, so R9's
  mitigation does not hold as stated. `DEMO_EXPECTED_DEFS` (`services.py:650`) is **two** pairs —
  `(TRIGGER_DEF_KEY, TRIGGER_DEF_VERSION)` *and* `access-request@v1` — and S11's sequence never
  runs `seed_workflows.sh`. Either add it to S11 or drop the readiness route from R9 and rely on
  `verify_salesperson.sh`, which is scoped correctly.
- **m5 — S12 is too large to checkpoint.** Routing + API client + bearer/401-rejoin + TanStack
  polling + i18n scaffold + three locale bundles + mobile-first layout + sticky header + bottom
  sheets + safe-area insets + a 360 px Playwright assertion is three deliverables under one
  done-condition. Split: S12a (session/API client/routing/401-rejoin — done when join→chat
  round-trips), S12b (layout shell + bottom sheets + Playwright 360×740), S12c (i18n + three
  bundles). S8 and S11 are also large but each has one coherent artifact; S12 does not.
- **m6 — S12's prerequisite on S5 is not in the step table.** §9's dispatch order sequences S5
  first, but S12's Parallel cell says only "after S8" — and S12 writes into the scaffold S5
  creates. Make it "after S5 **and** S8".
- **m7 — `create_app`'s signature change is implied but never named.** The `/shop` mount must be
  registered *inside* `create_app`: `/` is a catch-all `Mount` registered last (`app.py:316-320`)
  and Starlette matches in registration order, so anything added after `create_app` returns is
  unreachable. S8's "Interface / key symbols" names only HTTP status codes. Name the new
  parameters so S8 and S9 (lifespan executor shutdown) do not diverge.
- **m8 — The participant registry's durability is unstated, and it collides with R7.** §4.8 treats
  "the participant registry" as something `reset_all` clears, and S6 holds `ParticipantRecord`s in
  `storefront.py`. If that map is authoritative, a process restart invalidates every token and
  every participant is bounced to join with a fresh `participantId` — losing their **cart and
  order**, not just their session. R7 already notes that one file write under `falkor-chat/`
  restarts uvicorn under `--reload`; the consequence is much larger than R7 states. Say explicitly
  whether `resolve_token` re-reads `User.tokenHash` from the graph, and give S6 a done-condition
  for restart survival.
- **m9 — OQ-4's now-confirmed answer needs a UI consequence.** Participant self-serve advance is
  accepted, but a customer tapping "Fulfil" then "Deliver" on their own purchase reads as a broken
  product to a business audience — exactly R2's amplification concern. Have S14 label the control
  as a demo simulation (e.g. a distinct "demo controls" affordance), not as a normal storefront
  action, and keep `cancel` presented as the ordinary customer action it is.
- **m10 — §4.1's stated obstacle to reusing the `salesperson/` name does not exist.** Alternative
  (c) is rejected partly "on top of a checked-in `.venv`". `git ls-files salesperson/` returns 25
  files, **zero** of them under `.venv`, and there is no `.venv` directory in the tree at all. The
  rejection still stands on its other grounds, but with OQ-3 now choosing that name the false
  premise should be struck rather than carried into the revision.
- **m11 — `list_catalog()` needs an explicit bound.** `services.filter_products` defaults
  `limit=20` (`services.py:2583`); an all-`None` call lists the whole catalog *bounded by that
  limit*. Correct for today's 15 products, silently wrong at 21. Pass an explicit bound and assert
  the row count in S7's done-condition.
- **m12 — §4.2's HTMX fallback stops being a fallback once S12 starts.** R5 offers it "if the Node
  toolchain proves genuinely blocked", but S12–S14 *are* the client. Give the fallback a decision
  deadline — S5's done-condition — rather than leaving it as a standing option.
- **m13 — §3's HTTPS promise vs. plain uvicorn.** The diagram says the phones reach the server over
  HTTPS; `start_demo.sh` runs plain uvicorn. Over plain HTTP on a shared LAN every participant
  bearer token is on the wire, which R6 should name as the residual. Under B1 this stops being
  cosmetic: the obvious way to get HTTPS is the one thing that breaks the presenter control.
- **m14 — OQ-3's rename invalidates §2.4/§4.2's evidence paths and collides with a CPG name.**
  §2.4's parity table and §4.2's Streamlit rejection cite `salesperson/chatbot.py`,
  `salesperson/cart.py`, `salesperson/customer_profile.py` and `salesperson/session_manager.py` —
  the plan's single most load-bearing "why not Streamlit" argument, all of which move to
  `deprecated/salesperson/`. Re-path them in the revision. Separately, `cpg_salesperson` is a live,
  populated CPG (359 `METHOD` nodes) of the *retired* app, carrying the name the `cpg_<component>`
  convention (`skills/cpg-analysis/SKILL.md` §1) assigns to the *new* component. Decide now:
  rename it to `cpg_deprecated_salesperson`, or drop it. R11 is correspondingly weaker as a risk —
  the app survives on disk, not only in history — and can be downgraded.
- **m15 — FR-4 names the presenter as a participant; the plan never says how they hold both roles.**
  *"Every participant — presenter included — has their own independent conversation, cart, and
  order state."* Under the corrected OQ-5 the presenter has no separate token, so this is now purely
  a client question: does `/shop/presenter` coexist with their own participant session in one
  browser, and does `reset_all` bounce the presenter's own conversation too? Both are cheap to
  settle; neither is settled.

### Nits

- §2.4 cites `salesperson/session_manager.py:11` for `_active_session_id`; it is line 13 (the
  quoted early-return block is verbatim and the finding is correct).
- §4.7 builds the image manifest at startup only — dropping an asset in later needs a restart.
  Worth one line in the demo brief.

---

## 3. What's solid

**The grounding is exceptional.** I re-verified roughly twenty source claims across §2.2, §2.3 and
§4 and found no false one (Appendix A.3 lists them). That includes several that are easy to get
wrong and load-bearing: `ctx.actor == customerId` across all nine service methods; `advance_order`
implementing the guarded CAS with no REST route; no catalog route; `ensure_user` existing with no
`MEMBER_OF` primitive; posting requiring only the actor's node; the `/` catch-all mount ordering;
`trigger`'s four-step ordered rule; `salesperson@v5` with `maxIterations: 8` and create-only
`config.model`; `_assemble_messages`' `CONTEXT` carrier; the chat path's hardcoded
`{"threadId": …}`; `DEMO_EXPECTED_DEFS`' first entry; the anyio ~40-thread quote at `mcp.py:57-63`;
every `api.py` handler being a sync `def`; the 180 s `agent`/`step` timeouts; `querygen`'s
two-dataset registry; the responder's workspace-wide retrieval under `channel_id=None`; the
614 msg/s capacity figure; the 6 GB VRAM line; and the Streamlit process-global. Two live claims
(the `reference` `Product` property set; two independent `Customer` anchors in `ws:qa-cart-totals`)
reproduce exactly.

**§4.5's language carrier is correct and, importantly, durable** — which the plan asserts but does
not prove. I traced it: the chat-path initial ctx is written once at `start_run`, and `_drive_loop`
reloads it from the run node on every resume (`executor.py:604`) without ever rewriting it on the
chat path. So `language` survives every turn of a long conversation, and forwarding `run_ctx`
through `maybe_trigger` also covers a *re-start* after a failed run. This is the right design.

**§4.3's one-workspace decision is right**, its rule-7 argument is correct, and the live
`ws:qa-cart-totals` two-anchor proof is exactly the right evidence for it. §4.6's
decision not to drive `order-fulfillment@v1`, §4.7's file-presence-over-graph-property decision,
§4.4's honest framing of R1, and the def-bump discipline (republish `config.model` cumulatively)
are all correct and well-argued, each with a stated reversal trigger.

One strengthening observation the plan should claim, because it changes the residual-risk picture:
§4.4 measure 3 (do not embed storefront messages) is a **second, independent** structural barrier
under §4.3 part 4. With no participant message ever embedded, the responder's `Message` ANN pool
contains no participant data at all, even with S3's flag on. Say so — so that a future reversal of
measure 3 is known to reopen part 4 rather than being treated as a pure performance knob.

---

## 4. Open questions

1. **(For the stakeholder, via `teco` — the seventh, and the one the plan missed.)** Given B1, is
   the presenter expected to drive the demo **from the machine running uvicorn**? Localhost binding
   makes reset-everyone reachable only from that host, and only over plain HTTP with no reverse
   proxy in front. If the answer is "the presenter is on their phone" or "we want HTTPS", the
   decision needs revisiting — the honest third option is a presenter token bound at
   `start_demo.sh` startup and *printed to the operator's terminal* (not an env-configured shared
   secret), which keeps FR-7's "presenter-only" without either a standing secret or the loopback
   fragility.
2. **(For the `architect`.)** M1's contradiction is a genuine design fork, not an editing slip —
   which of the two reset-mine semantics is intended?
3. **(For `teco`.)** M9's agent-prompt update routes to `cobb`, not to S16's `coder`. Is that a new
   unit in the coordination, or an S16 sub-step with a different specialist?

Of the plan's own six §8 questions, I agree with the defaults on OQ-1, OQ-2 and OQ-6 as the
stakeholder confirmed them, and note only that OQ-1's chosen basis (stub run + published live
curve + staggered script) makes AC-3's literal wording — "no noticeable degradation **for any
participant**" — unmeetable for agent turns; the test report should say so plainly rather than
recording a pass against wording it does not satisfy.

---

## 5. Appendix

### A.1 — uvicorn proxy-header behaviour (executed, `falkor-chat/server/.venv`)

```
uvicorn 0.49.0
  proxy_headers: bool = True,                                  # Config.__init__ default
  self.forwarded_allow_ips = os.environ.get("FORWARDED_ALLOW_IPS", "127.0.0.1")

_TrustedHosts('*')        trusts '192.168.1.50'        -> True
_TrustedHosts('*')        .get_trusted_client_address('127.0.0.1') -> ('127.0.0.1', 0)
_TrustedHosts('127.0.0.1') trusts '192.168.1.50'       -> False
_TrustedHosts('127.0.0.1') trusts '127.0.0.1'          -> True
_TrustedHosts('127.0.0.1') trusts '::1'                -> False
behind-proxy case: peer=127.0.0.1 (trusted by default), XFF='127.0.0.1'
                                     -> scope["client"] rewritten to ('127.0.0.1', 0)
```

`ProxyHeadersMiddleware.__call__` rewrites `scope["client"]` from `X-Forwarded-For` whenever the
immediate peer is in `trusted_hosts` — which, behind any reverse proxy, it always is.

### A.2 — S16 acceptance-command evidence (executed at `4bb96e1`)

```
rg -n 'salesperson/' --glob '!docs/**'          -> 36 matches   (21 in component docs/ trees)
rg -n 'salesperson/' --glob '!**/docs/**'       -> 15 matches   (none in S16's Files column)
```

The 15: `cypher-mcp/README.md:433`; `AGENTS.md:10,11,66,78` (S16 covers these);
`claude/cobb/kaizen/history.md:2612`; `claude/frontend-engineer/kaizen/plan.md:30,45`;
`claude/frontend-engineer/kaizen/history.md:16,134,135`;
`claude/frontend-engineer/frontend-engineer.md:18,20`; `claude/devops/kaizen/history.md:12,140`.
The replacement command in M9 was run and returns exactly the two
`claude/frontend-engineer/frontend-engineer.md` lines it is designed to catch.

### A.3 — Source claims re-verified for §3 ("What's solid")

`services.py` §16 comment `:2610`, `add_cart_item` `:2653`, `place_order` `:2748`,
`get_order_status` `:2795`, `advance_order` `:2812`, `get_profile`/`save_profile` `:2840`/`:2851`,
`filter_products` `:2583`, `_validate_and_derive_role` `:813`, `start_workflow_run` `:2023-2033`,
`DEMO_EXPECTED_DEFS` `:650`; `repository.py` `get_order` `:2972`, full cart/order method list;
`trigger.py` (whole module); `api.py` route list + `post_message` `:92`; `app.py` `:290-340`
(mount order) and `_build_default_app` (`WorkflowTrigger(..., responder=responder)`);
`executor.py` `_read_thread_context` `:1224`, `_assemble_messages` `:1243`, `_drive_loop` `:604`;
`proof_defs.py` module docstring + `SALESPERSON_DEF` `:301`; `querygen.py` `DATASET_REGISTRY`
`:222`; `responder.py` + `services.hybrid_search` `:1003` (`channel_id=None` ⇒ workspace-wide);
`transport.py` (blocking `urllib.request`); `config/models.json` (timeouts);
`mcp.py:57-63` (the anyio quote); `config.py` `get_context`; `db.py` `LazyFalkorDB`;
`scripts/bootstrap_schema.sh` (label inventory); `scripts/seed_demo.sh:41`;
`scripts/seed_salesperson.sh:124`; `scripts/verify_salesperson.sh:55`;
`server/tests/test_salesperson_scaffold.py:325`; `docs/QUERIES.md` (§17 is the highest section);
`docs/DESIGN.md` §1.3 + M2 line; `docs/test-reports/capacity-report.md:61`;
`salesperson/session_manager.py:13`. Live: `GRAPH.LIST`; `reference` `Product` keys;
`ws:qa-cart-totals` `Customer` anchors.

---

## Pass 2 — 2026-09-02, against plan v1.1

**Scope:** `docs/plans/salesperson-ui.md` v1.1 (1092 lines, 19 steps, S12 split into S12a/b/c),
re-gated against Pass 1's 4 blockers / 9 majors / 15 minors / 2 nits and the stakeholder's
OQ-1…OQ-6 answers. Baseline unchanged (`4bb96e1`); `teco`'s U5 `deprecated/` move has landed in
the working tree, so §2.4's parity citations are checkable at their new paths, and U6's
`frontend-engineer.md` refresh has landed too. Re-verification this pass was targeted, not
exhaustive: §4.9 (new, unreviewed) probed in full against source; §5.0 re-derived independently
from §5.1 rather than trusted; S16's replacement acceptance command executed verbatim; everything
else spot-checked.

**Verdict: approve with suggestions.** All four blockers are closed, and §4.9 in particular is a
better answer than the one I proposed. One new Major (N1) and two new Minors. **The one thing I
would gate on** is that N1 lands in §4.8/S0 *before* S0 is dispatched — S0 is the first step out
of the door and it is the step that would otherwise bake the defect into the reset Cypher; that is
a cheap ordering constraint, not a re-review.

**CPG: considered, not relevant — this pass re-read the revised plan against `falkor-chat` source
and the live instance directly; `cpg_falkorchat` is fresh (`sourceCommit 4bb96e1`) but every
question here was about *absent* code (routes not yet written, a `create_app` parameter not yet
added), which a CPG of the current tree cannot answer.**

### Disposition of Pass 1 findings

**Blockers.**

- **B1 — fixed.** §4.3 restores `FALKORCHAT_PRESENTER_KEY` with `POST /shop/api/presenter/session`
  (§5.2), S10's rate-limited exchange, and R6 carrying the standing-secret residual. I re-read the
  "do not simplify this to a localhost check" paragraph against my own executed evidence: the
  uvicorn version, the `proxy_headers=True` default, the `FORWARDED_ALLOW_IPS` default, both
  inversion paths and the "weaker than the key" conclusion are all reproduced accurately, with no
  overclaim. m15 rides along, settled properly (two credentials in one browser; `reset_all` clears
  the presenter's conversation but not their presenter token).
- **B2 + B3 — fixed, and better than my fix.** §4.9's move 1 (`dev_surface`, a function parameter
  with no env var behind it) makes the dangerous configuration inexpressible rather than merely
  detected. Verified against source: `create_app` already carries `mount_mcp: bool = True` so the
  shape is precedented; `GET /health` lives *inside* `api.build_router` (`api.py:55`) and would
  have vanished with it — S3 correctly adds a bare liveness route and asserts it, so §4.9's route
  table claim is backed by an actual work item; `_lifespan` calls
  `services.ensure_actor(provider())` and `_sweep_loop(services, provider, …)` both resolve through
  `config.get_context`, so move 2's claim that one variable fixes the sweep and the actor
  projection is exactly right. All eight surviving `FALKORCHAT_DEMO_WS` mentions are rejections or
  narrative, none a usage.
- **I accept the reasoning for declining my B2 fix (a), without reservation.** It is correct and I
  had it backwards: "these two variables must disagree" is only a fix while the legacy surface
  stays mounted, and adopting it *alongside* the un-mounting would have mandated the two-variable
  split and therefore mandated B3's trap. Collapsing to one variable is the stronger move. The one
  consequence it carries is N1 below — which is a gap in §4.8, not a reason to reopen §4.9.
- **B4 — fixed.** `get_customer_current_order` and `order_belongs_to_customer` are named in S0's
  interface column, S4's method list and §10's AC-7 row; S4's Files now carries `services.py` and
  `repository.py`; §5.0's `services.py` row is `S2 → S4`; S7 explicitly routes through
  `services.get_current_order` "a repository read, not composed here". The layering breach is
  closed.

**Majors.** M1 fixed (survival, with three reasons and §5.2's row corrected to `200 {threadId,
language}`). **M2 partially fixed; I accept the decline.** The correctness half — server-side
single-flight, `409 TurnInProgress` *before* the write, with the reasoning about `maybe_trigger`
step 2 reproduced accurately — is adopted, and the "one pending slot" I proposed was ergonomics
carrying a second queue-position concept; the retained-composer path is the better trade. M3 fixed
(served directory, non-empty-manifest assertion, both AC-11 branches in §6.3 #9). M4 fixed (§4.10;
`join()` writes the profile name, v6 gains the order-time sentence, §10's AC-8 row and §6.3 #5 are
both measured runs). M5 fixed, with the 3–9× correction stated and the measure re-justified on its
real value — the second AC-2 barrier — rather than on performance. M6 fixed (see N2 for the one
residual). M7 fixed (quiesce contract, `FALKORCHAT_STOREFRONT_QUIESCE_S`, `503` on timeout,
stop-intake-then-drain for `reset_all`, done-conditions on S7 and S10). M8 fixed (`ReadCursor`,
`WorkspaceConfig` and `Document`/`Chunk`/`Entity` all adjudicated; thread-scoped-not-author-scoped
stated and testable). **M9 fixed — verified by execution:** the replacement `grep` runs verbatim
from the repo root and now returns zero matches; at `HEAD` it returned exactly the two
`frontend-engineer.md` lines the plan claims, so the parenthetical was true when written and U6
has since closed them. The `cobb` routing note in S16 is correct.

**Minors.** All fifteen fixed. Spot-checked: m1 (S1's Files carries `falkor-chat/AGENTS.md`, with
BACKLOG explicitly and correctly excluded), m2 (`_lifespan` before `yield`, with the
`NoEventLoopError` reason stated), m3 (2 s throughout, with the old 3 s figure named as the
correction), m5 (S12a/b/c with distinct done-conditions), m6 (S12a "after S5 **and** S8"), m7
(`create_app(..., storefront, storefront_dir, dev_surface)` in S8's interface column), m8 (graph
is authoritative, in-process map is a read-through cache, restart-survival done-condition on S6,
and R7's blast radius correctly re-scoped), m10 (the `.venv` claim struck by name — I re-confirmed
`git ls-files` returns 25 files, zero under `.venv`), m11, m12 (S5's done-condition *is* the HTMX
deadline), m13, m14 (all four `deprecated/salesperson/*.py` citations resolve on disk; the
`cpg_salesperson` collision correctly routed out of the plan to `teco`/`graph-dba`), m15. **Both
nits fixed** — `session_manager.py:13` corrected, and the startup-only manifest is now an
operational note for the demo brief.

### New findings

#### N1 (Major) — collapsing to one workspace variable lands the storefront on `ws:acme` by default, where §4.8's label-based survivor list cannot protect the pre-existing subgraph

§4.9 move 2 is right, but it changes what "the demo workspace" *is*: `config.WS_ID` defaults to
`acme`, and S11's `start_demo.sh` uses `"$FALKORCHAT_WS_ID"` everywhere without ever **pinning** it
to a dedicated value. Run `start_demo.sh` without setting it — the natural first run — and the
storefront serves `ws:acme`, the repo's primary dev/demo workspace. Live inventory of `ws:acme`
today: **2 `Channel`, 2 `Thread`, 52 `Message`, 1 `User`, 1 `ReadCursor`**, alongside 544 `Entity`
/ 87 `Chunk` / 29 `Document` and ten snapshots.

`POST /shop/api/presenter/reset-all` then deletes "every participant `User` and `Channel`" plus
each participant's thread subtree — on the *shared* graph that holds `seed_demo.sh`'s
`demo-general`/`demo-welcome` and the M2/M5 hand-verification transcript those 52 messages are.
The design intent is safe (scope the channel delete via participant `User`s, which the
`User.tokenHash IS NOT NULL` roster filter already implies), but **§4.8's survivor column cannot
express it**: victims and survivors share the labels `Channel`, `Thread` and `Message`, and S4's
done-condition asserts "every §4.8 survivor **by label**", which is structurally incapable of
catching an over-broad channel delete. This is the one blind spot left in an otherwise excellent
checklist, and move 2 raises its stakes from "a dedicated `ws:demo` that has nothing in it" to
"the workspace everything else in this repo uses".

**Fix, three lines and one test:** (i) `start_demo.sh` sets `FALKORCHAT_WS_ID` to a dedicated
default (`demo`) with a comment that this is still a *single* variable, not the rejected
two-variable split — §4.9's whole argument survives intact, since there is no second value to
disagree with; (ii) §4.8 gains a non-label survivor clause: *every `Channel`/`Thread`/`Message`
not reachable from a participant `User` survives both resets, `seed_demo.sh`'s
`demo-general`/`demo-welcome` included*; (iii) S4's done-condition adds a positive assertion —
seed a non-participant channel + thread + message into the probe graph, run `reset_all`, assert
all three survive. Without (iii) the label-based assertions pass on a delete that wipes them.

#### N2 (Minor) — the rebuilt §5.0 map is complete except for the SPA's shared entry files, and that is exactly where S12b ‖ S12c collide

I re-derived the map independently from §5.1's Files column rather than trusting the rebuild:
every one of the 28 rows matches, the three v1.0 gaps are genuinely closed, and the five
`salesperson/src/**` subtrees are disjoint as claimed (`components/sheets/` vs
`components/message/`; `views/Chat*` vs `views/{Cart,Order,Profile,Catalog}*`).

The gap is what falls *outside* all five: `src/main.tsx`, `src/App.tsx` and the Tailwind entry
(`src/index.css`). S5 scaffolds them (`salesperson/**`), then S12c wires the `react-i18next`
provider and S12b wires the layout shell — both of which conventionally live in `App.tsx`/
`main.tsx`, and the two run **in parallel**. That is precisely the collision M6 existed to
prevent, reintroduced by the S12 split. **Fix:** add a `salesperson/src/{main.tsx,App.tsx,
index.css}` row owned by **S12a**, and give S12a's done-condition the provider/layout **slots**
that S12b and S12c mount into, so neither needs to edit a shared entry file.

#### N3 (Minor) — §4.9's route-table assertion keys on an import-time module constant, not on the parameter it guards

The assertion is specified as *"if `config.STOREFRONT_ENABLED` is set and any legacy route or the
`/` or `/mcp` mount is present in `app.routes`, `create_app` raises."* But `config.py` resolves
every flag at **import time** (`ENABLE_AGENT: bool = _env_flag(...)`, and so on), so an S8
`TestClient` test that builds a storefront app through `create_app(storefront=True,
dev_surface=False)` — the way every other test in this suite wires the app — leaves
`config.STOREFRONT_ENABLED` `False` and **skips the guard entirely**. The guard is then only live
in the one configuration nobody tests. Fix: key it on the `storefront` parameter (`if storefront
and not dev_surface: assert …`, or raise when `storefront and dev_surface`), which is both
testable and closer to the invariant — the two surfaces are mutually exclusive regardless of how
the app was constructed. S3's done-condition already asserts on `app.routes` with `dev_surface=
False`, so this is a one-word change to §4.9's wording, not new work.

#### Nits

- **S16's acceptance command exits 1 when clean.** `grep` returns 1 on "no matches", so a delegate
  wiring the done-condition into a `set -e` script gets a false failure on success. Say
  `! grep -rEn … .` or check for empty output. It also does not exclude `.venv` or `dist/`; neither
  can match this pattern, so it is cosmetic, but a delegate copying the idiom elsewhere will
  inherit the omission.
- **§4.3's reason for rejecting the startup-minted presenter token cites a hazard S11 removes.**
  "…cannot survive the `--reload` restart R7 describes" — but S11 sets a non-empty `UVICORN_ARGS`
  precisely so `--reload` is off, and R7's own mitigation says so. The rejection still stands on
  its first reason (the presenter must be at the server console at start time), and the stakeholder
  has chosen; only the second reason is self-undercutting and should be dropped or re-grounded on
  crash-restart rather than reload.

### What v1.1 does notably well

The revision does not merely apply the findings — it re-derives them. §4.9 solves B2/B3 by making
the bad state inexpressible rather than by adding a check, and then explains why my own proposed
fix was the wrong shape, with reasoning I had to concede. §4.4 measure 3 is kept while its stated
justification is inverted (its value is AC-2, not throughput) rather than quietly retained on the
corrected number. §4.8 now carries a per-label adjudication, a scoping rule, *and* a timing
contract, and names `WorkspaceConfig` as "the single most expensive mistake available in the reset
design" — which is right. §10's AC-3 row says plainly that the criterion is not met as literally
worded; §10's AC-8 and AC-9 rows route to measured runs rather than to code review. Declining M2's
pending slot and pushing the `frontend-engineer.md` refresh out to `cobb` are both the correct
calls. Two findings I raised are answered by being proven wrong, which is the outcome I want from
a re-gate.

---

## Pass 3 — 2026-09-02, against plan v1.9 (the v1.2 → v1.9 delta)

**Reviewed:** the **delta only** — `docs/plans/salesperson-ui.md` v1.9 diffed against its committed
v1.2 state (`git show 55d4e57:docs/plans/salesperson-ui.md`; +89/−45 by `diff -u`). Pass 1 and
Pass 2 covered the plan whole and their findings stay closed. Upstreams read in full:
`docs/plans/salesperson-ui-graph.md` v1.3 (§7, §12), `docs/reviews/salesperson-ui-impl.md`
(Passes 1–3), `docs/requirements/salesperson-ui.md`. Delivered S1/S2/S4 code and
`falkor-chat/docs/QUERIES.md` §18 read as ground truth for "was this absorbed".

**CPG: used `cpg_falkorchat` — queried for the delivered storefront repository methods and found
none of them (`ensure_participant`, `list_participants`, `reset_participant`,
`reset_all_participants` return zero `METHOD` rows), which is correct for uncommitted S4 work and
is itself the evidence behind M-3 below; source reading, not the graph, carried the verification.**

**Verdict: needs changes** — 1 blocker, 3 majors, 8 minors, 2 nits.

The delta is mostly good work: v1.7's quiesce rewrite is faithful and the new condition can fail;
v1.9's S12d closes the half of AC-5 it set out to close; the S1/S4 probe-workspace pins are
accurate against the scripts. What did not survive the compression is the part the architect
predicted it might — the graph note's §12 five-row anomaly table lost one signal to an inversion,
and the F8 mandate landed on the wrong side of the wire.

### What I checked by execution vs. judged statically

**Executed:** `diff -u` of v1.2 vs. v1.9; `redis-cli GRAPH.CONFIG GET MAX_QUEUED_QUERIES` → `25`
(confirms §6.4's new figure) and `GRAPH.INFO` → exposes a `Waiting queries` section (bears on
M-8); four `cpg_falkorchat` queries; `git log`/`git show` for the baseline. **Read as source of
truth (static):** `falkorchat/config.py:29`, `falkorchat/db.py:44`,
`falkorchat/repository.py:3564-3598`, `QUERIES.md` §18.0–§18.9, `scripts/seed_salesperson.sh:133`,
`scripts/verify_salesperson.sh:60`, `tests/test_repository.py` test-name inventory. **I did not run
the pytest suite** — a `coder` holds the database and `repository.py`/`services.py`, and nothing
below needed it.

### M-1 · **Blocker** · §4.8's F8 bullet re-attributes a server-side failure to the browser, and the server half of the mandate is now absent

v1.8's revision note records "F8's client-side reset-timeout rule (**S12a**)" as landed. It did not
land — it moved layers. `FALKORDB_SOCKET_TIMEOUT` is the falkor-chat **server's** Redis socket
timeout to FalkorDB (`falkorchat/config.py:29` → `falkorchat/db.py:44`, `socket_timeout=`), and the
note is explicit: "**S7/S10 must** treat a client-side timeout on a reset as *unknown*"
(`docs/plans/salesperson-ui-graph.md` §7; the delivered `QUERIES.md` §18.7 carries it correctly).
§4.8's new bullet instead reads it as the *browser's* socket timeout and assigns re-reading to S12a.

Consequence: no server step carries the rule (`grep -in timeout` over the plan returns lines 719,
723–725, 933 — §4.8, S12a, and unrelated LLM timeouts; nothing in S7, S8 or S10). §5.2's reset row
still says only "`503` on quiesce timeout, nothing changed", so the natural implementation catches
the reset call's exception and returns exactly that — the participant is told nothing changed while
their state was committed as deleted. S12a's rule never fires, because the browser got a clean 503.

**Suggested fix:** keep the S12a clause, and add the server half: a `redis` `TimeoutError` raised by
either reset must **not** map to the quiesce `503`; give it its own code (e.g. `504
reset_state_unknown`), state it in §5.2 on both reset rows, and put it in S7's and S10's
done-conditions. Correct §4.8's attribution to name the server→FalkorDB socket, not the browser's.

### M-2 · **Major** · the `Thread` UNIQUE-violation signal landed on the one route that cannot raise it

§5.2 attaches "a `Thread` UNIQUE violation propagates as `5xx` and is **never retried**" to
`POST /shop/api/presenter/reset-all`, and omits it from `POST /shop/api/reset`. It is the wrong way
round. The raise comes from the duplicate-marker fail-safe, which fires because
`reset_participant` **re-mints** a thread with one `$newThreadId` under a `FOREACH` that runs twice
(`QUERIES.md` §18.4, `CREATE (c)-[:HAS_THREAD]->(:Thread {threadId: $newThreadId…})`).
`reset_all_participants` contains **no** `CREATE (:Thread` at all (§18.5 read end to end), and the
note says so itself: "`reset_all` still collects the participant, because it never re-mints"
(`salesperson-ui-graph.md` §4). The note's §12 row says "**Either** raises", which is over-broad;
the plan narrowed it, and picked the wrong one. S10's row asserts the contract "on both reset
routes", so the behaviour is recoverable — but the reset-mine route's own contract is silent about
its only structural failure mode, and S8 (which builds that route) is not told about it either.

**Suggested fix:** move the clause to `POST /shop/api/reset` in §5.2 (keeping it on reset-all is
harmless but should say "if it ever arises"), and add it to S8's error map alongside `409
unscoped_participant`.

### M-3 · **Major** · §5.2's presenter roster promises six fields; three of them have no producer, and S12d's done-condition therefore cannot be met

§5.2 contracts `GET /shop/api/presenter/participants` →
`[{participantId,displayName,language,messageCount,cartTotal,orderStatus}]`. The delivered
primitive S10's row names — `repository.list_participants` — projects
`participantId, displayName, channelId, threadId, language, joinedAt`
(`falkorchat/repository.py:3584-3589`, matching `QUERIES.md` §18.3 verbatim). **`messageCount`,
`cartTotal` and `orderStatus` are produced by nothing**: no repository method, no service wrapper,
no clause in S4's, S7's or S10's row. S10's done-condition asserts only that the roster excludes
non-participant `User`s — it never asserts the field set.

v1.9's S12d then makes it load-bearing: "three rows carrying each participant's display name **and
order status**, asserted on rendered text". As the plan stands that assertion is unsatisfiable, so
the step whose whole purpose is to stop AC-5 passing vacuously has a done-condition that cannot
pass at all. The gap is not free to close either: the obvious composition is per-participant
`get_cart` + `get_current_order` + a message count, i.e. ~150 extra graph queries per presenter
poll at 50 participants — outside R10's 50 req/s budget and never profiled by S0's §8.

**Suggested fix:** decide the shape in the plan, not at S10 time. Either (a) trim §5.2's roster to
what §18.3 gives and let S12d assert on display name + language only, or (b) give S10 an explicit
sub-task for a single aggregate roster query (an append to `QUERIES.md` §18.3, `graph-dba`-designed
since it is a new label-scan-plus-aggregation on the demo's hot path), with a done-condition
asserting the six-field shape. Either way S10 must gain a field-set assertion.

### M-4 · **Major** · AC-5's participant half is still assigned to a step that does not build it

This is the defect v1.9 exists to fix, surviving on the other side. AC-5's row now reads "…**S12d**
(the presenter view itself…), S12a (presenter session + token storage + response typing; **the
participant's own reset call and its post-reset language step**), S12b (**the participant-facing
reset control in the sheets**)". Neither cited clause appears in the step it is assigned to: S12b's
scope is "sticky header with cart/order/profile icon buttons, bottom-sheet overlays, safe-area
insets…" and its done-condition asserts only viewport behaviour and sheet open/close — no reset
control, no confirm step. S12a's done-condition covers `401 → rejoin`, `409`, the F8 re-read and the
mount slots — not the post-reset language step that §5.2 promises the client performs.

So "reset mine" has server steps, a route, and an acceptance item (§6.3 #8), but no build step whose
done-condition renders or exercises the control. It first goes red at S15.

**Suggested fix:** add the control to S12b's scope text ("the participant's own reset in the profile
sheet, behind a confirm step") and to its done-condition ("the reset control is present; confirming
it calls `POST /shop/api/reset` and the client returns to the language step, asserted on rendered
state"). One clause each, and AC-5's row then matches its steps.

### Minors

- **m-5 · R4's risk column still asserts what §4.8 now disproves.** v1.7 swept R4's *mitigation*
  ("S7/S10 assert the note's §7 (a)–(d)") but left the risk statement "a reset racing an in-flight
  turn leaves orphan rows" — the exact framing §4.8 lines 710–712 and the note's F3 supersede for
  `Message`/`StepRun`/`TraceEvent`. It is not flatly false (`advance_cursor` mints a real orphan
  `ReadCursor`), which is why it reads as fine. Rewrite to "…and a reset racing an in-flight turn
  burns a turn and can mint an orphan `ReadCursor` (`…-graph.md` §7, F3)". A partial sweep is how
  the vacuous condition survived the first time.
- **m-6 · §2's deliberate-baseline status should be stated at the §2 heading — ruling on flagged
  item 2, and it needs to cover the CPG paragraph too.** Today the statement is buried mid-§2.2,
  inside the `salesperson@v5` bullet; a reader of §2.1 or §2.3 never meets it. Worse, the paragraph
  immediately under the §2 heading is the CPG block, which reads as a live claim ("rebuilt
  mid-flight and **now fresh**, `sourceCommit 4bb96e1` = `HEAD`") and is the one part of §2 a reader
  will act on. `HEAD` is now `4f88fd0`, and I confirmed by query that `cpg_falkorchat` contains
  none of S4's delivered repository methods. Add one italic line under `## 2. Context & findings`:
  *"§2 is the verified baseline the design was derived from, as of v1.0 — including the CPG
  freshness stamp. Line numbers, figures and the CPG's contents have moved with S1/S2/S4 and are
  deliberately not retro-fitted; §4 and §5 are the current surface."* Then the §2.2 clause can go.
- **m-7 · §5.0's SPA row carries two stale counts from the v1.9 sweep.** It now names six subtrees
  and sequences "S13 ‖ S14 ‖ S12d", but still says that is what "makes the two parallel **pairs**
  safe" and that the shared entry files "fall *outside* all **five**". Both should read *three
  parallel groups* / *all six*.
- **m-8 · S15's queue-depth check needs its observation mechanism named.** §6.4 says "assert the
  observed depth against the cap"; nothing says how depth is observed, and a ~240 ms window is not
  sampleable by guesswork. `GRAPH.INFO` exposes a `Waiting queries` section on this instance (I ran
  it); name it, and say the harness polls it for the duration of the `reset_all` call. Without that
  the condition degrades to "no query was rejected", which passes whenever the run never approached
  the cap — the headroom number, which is the point, goes unmeasured.
- **m-9 · S7's "§7's four conditions (a)–(d), asserted as written" is not literally executable for
  reset-mine.** (b) parenthesises "(intake stopped)" and (d) says "after `reset_all`" — both are
  worded for the presenter path. S7 has no global intake stop. Add "…(a)–(d), read participant-
  scoped for reset-mine: (b)/(c) over that participant's own posts, (d) over cursors owned by the
  reset participant". The citation is right; the "as written" is what over-reaches.
- **m-10 · S10's parenthetical is a third copy of the anomaly contract and drops two of its five
  signals** — it lists `scoped=false`, `unscopedCount > 0` and the UNIQUE violation, omitting
  zero-rows → `404`/`401` and the clean path. It also cites §5.2, so nothing is lost in practice;
  but §5.1's own new **cite, don't re-list** rule says a row "cites the section and names only what
  the row itself adds". Under that rule the parenthetical should go. Ruling on the rule's
  consistency: it is applied correctly to S7 and S10's *quiesce* conditions, and violated by the
  same v1.8 edit that introduced the anomaly contract.
- **m-11 · The fifth anomaly signal is only implicit.** The note requires `unscopedCount == 0` →
  `200` **with no `incomplete` flag**. §5.2 says what happens when the count is positive and leaves
  the clean path to inference; nothing asserts the flag is absent when clean. One clause on
  reset-all's row, and one in S10's done-condition, closes it — otherwise an implementation that
  always emits `incomplete: false` (or, worse, `true`) satisfies every stated condition.
- **m-12 · S4's row says "implementing S0's Cypher verbatim" of nine methods; S0 specifies five.**
  The note's §12 mandates "§3, §4, §5, §10.1 and §10.2 verbatim" — `ensure_participant`, both
  resets, and the two order reads. `add_channel_member`, `get_participant_record`,
  `set_participant_record` and `list_participants` have **no** Cypher anywhere in the note (grep
  returns only §8's `GRAPH.PROFILE` rows for the roster's predicate); they were designed by S4 into
  `QUERIES.md` §18.2/§18.3. Harmless in retrospect — S4 shipped — but a re-execution reads the row
  and hunts for four blocks that do not exist. Split the row: "the five §12-verbatim queries, plus
  four this plan specifies (…)".

### Nits

- **n-13 ·** S1's done-condition names the probe workspace but not `bootstrap_schema.sh
  <probe-ws>`, which `seed_salesperson.sh`'s own prerequisites list as step 1 (script header,
  line 109). A fresh probe graph has no indexes or constraints until it runs.
- **n-14 ·** §5.0 assigns `salesperson/playwright.config.ts` to S5 · S12b; S12d owns
  `tests/e2e/presenter.spec.ts` but no config entry. Fine if the config globs `tests/e2e/**`; if
  the presenter spec needs its own project/viewport, the file has no S12d owner and the collision
  the map exists to prevent reappears. One clause in the §5.0 row settles it.

### Rulings on the six flagged uncertainties

1. **The §5.2 anomaly compression — diffed signal by signal against the note's §12 table.**
   (i) `scoped=false` → `409 unscoped_participant`, never `200`: **survived**, on the reset row and
   in S8's error map. (ii) zero rows → `404`/`401`: **survived**, on the reset row. (iii)
   `unscopedCount > 0` → `200` + `incomplete` + `unresolved`: **survived**, and S12d now renders it.
   (iv) `unscopedCount == 0` → `200`, **no flag**: **implicit only** — m-11. (v) `Thread` UNIQUE →
   `5xx`, no retry: **survived but inverted onto the wrong route** — M-2. So four and a half of
   five, with the half and the inversion both worth fixing. The architect was right to ask.
2. **§2's staleness.** Correct as a policy and I did not file the stale figures. But **yes, state
   it in-document** — and at the §2 heading, not where it now sits, and worded to cover the CPG
   paragraph, which is the one part of §2 that reads as live and is now demonstrably not (m-6).
3. **S12a is not over-loaded enough to split, and splitting would cost more than it saves.** The
   only extractable piece that does not touch `main.tsx`/`App.tsx`/`index.css` is the `reset-all`
   response typing — and that lives in `src/api/**`, S12a's own subtree, so moving it to S12d
   re-creates a shared-directory collision one level down. Pass 2's N2 stands. What S12a *does*
   need is an explicit statement that the three entry files and their mount slots are its **first**
   deliverable: S12b, S12c, S13, S14 and now S12d all block on it, and a half-finished S12a stalls
   five steps. Its done-condition already names the slots; make the ordering explicit in the row.
4. **S12d's roster dependency is worse than "executed by nothing" — it is currently unsatisfiable**
   (M-3). The "asserted on rendered text rather than on the fetch" clause is itself **strong**, and
   is the right instinct: it is the one phrasing that cannot go green on a mocked fetch. Keep it
   verbatim; fix what it asserts against.
5. **S12d's numbering is fine.** Renumbering to S15 would invalidate live citations for zero
   benefit, and the family reading ("S12* is the SPA") is intact. §9's dispatch line, §5.0's row,
   §10's AC-5 row and S15's dependency all name it consistently — I checked all four. The only
   residue is m-7's two stale count words.
6. **AC-10 is accurate as written and needs no plan edit — but it needs a stakeholder question.**
   The requirement (`docs/requirements/salesperson-ui.md:124`) gates the first live demo on **K-056
   alone**, and K-056 is resolved; the plan says exactly that. What has moved is around it: K-060 is
   still `🟡 in-progress` in `falkor-chat/docs/BACKLOG.md:43`, R2 rates it **High** for precisely
   AC-10's stated rationale ("a business audience reads a wrong answer as a broken product"), and
   the burned `v6` was a third wording attempt at it — which also makes the backlog item's own "two
   attempts… before a third" framing stale. Whether the gate should widen is `tico`'s and the
   stakeholder's call, not the architect's. See Open questions.

### Verification of the architect's §12 audit

I walked all eleven §12 bullets plus the three open items against the plan text **and** the
delivered code. The audit holds on count but not on kind: what it reports as absorbed includes one
mandate that changed meaning in transit (M-1) and one that changed route (M-2). Full item-by-item
table in the Appendix (Pass 3 · A). Summary: **7 genuinely absorbed** (quiesce (a)–(d) for both
steps; profile-name re-write → S7; `MAX_QUEUED_QUERIES` → §6.4/S15; `DESIGN.md` arrow → §5.0;
`.query`/`.ro_query` routing; bare `tokenHash IS NOT NULL` roster predicate; open item 2 correctly
out of scope) · **2 absorbed in delivered code, not plan text, and verified there** (`QUERIES.md`
§18's five documentation riders — all five present, §18.0/§18.3/§18.4/§18.5/§18.7; S4's fixture
mandate including the three v1.1 shapes and the cross-member strengthening — all present as
`test_reset_participant_with_a_mismatched_marker_is_a_total_no_op`,
`…_without_a_member_edge_is_a_total_no_op`, `test_off_chain_message_survives_both_resets`,
`test_reset_participant_of_a_cross_member_leaves_the_demo_channel_whole`) · **1 misabsorbed**
(F8 → M-1) · **1 partially absorbed with an inversion** (the anomaly table → M-2, m-11) · **2
correctly out of plan scope** (the extraction-assert tripwire, "do not relax either guard" — both
S4-implementation guidance, and the tripwire was honoured per impl-review Appendix F).

### What's solid in the delta

The **v1.7 quiesce rewrite is faithful and the new condition can fail** — I confirmed (a) requires
asserting the in-flight `WorkflowRun` reached terminal status *before* the delete, and (d) is a
concrete cursor query with a non-trivial answer; neither is satisfiable by doing nothing, which was
the whole complaint. §4.8's supersession note ("not the v1.0 'no orphan rows' wording that note
supersedes as vacuous") is the right way to retire a defect: it names what it replaces. The
**S1/S4 probe-workspace pins are accurate** — `${1:-${FALKORCHAT_WS_ID:-acme}}` is verbatim what
both scripts do (`seed_salesperson.sh:133`, `verify_salesperson.sh:60`), and the "`reference` is
written either way, only the materialize target is in question" clause is confirmed by the seed
script's own header. The **`v6` burned-version explanation** is now stated once, in §4.5, and cited
from S1 rather than restated — the cite-don't-re-list rule working as intended. **S12d's "asserted
on rendered text rather than on the fetch"** and **§6.3 #8's "the roster lists every joined
participant by display name before the reset and is empty after it"** are both written by someone
who has internalised the vacuous-evidence pattern. And the **§5.0 additions** (`test_process_input.py`,
`QUERIES.md` dual ownership, `falkor-chat/docs/HISTORY.md`) each carry their own collision
assessment rather than just appearing.

### Open questions (Pass 3)

1. **Does AC-10's readiness gate stay pinned to K-056 alone?** K-056 is resolved; K-060 and K-062
   are open, and R2 rates the first High on AC-10's own stated grounds. This is a scope decision
   for the stakeholder via `tico`, not something the architect should widen unilaterally. The plan
   is correct either way — it faithfully restates the requirement as written.
2. **M-3's roster: trim the contract or build the aggregate?** (a) is free and weakens the presenter
   view; (b) needs a `graph-dba` query on the demo's hot path and a `QUERIES.md` §18.3 append after
   S4 has closed. `teco`'s call, and it changes what S10 and S12d are.

## Appendix

### Pass 3 · A — `docs/plans/salesperson-ui-graph.md` §12, item by item

| §12 mandate | Where it should land | Status | Evidence |
|---|---|---|---|
| Implement §3/§4/§5/§10.1/§10.2 verbatim | S4 | absorbed (over-stated to nine methods) | plan S4 row; note has no Cypher for 4 of the 9 — m-12 |
| Assert the extraction itself (block count / length / `markdown-it`) | S4 implementation | out of plan scope, honoured in delivery | `salesperson-ui-impl.md` Appendix F |
| Routing: writes `.query()`, reads `.ro_query()` | S4 | absorbed | plan S4 Interface column |
| `QUERIES.md` §18 + five documentation riders | S4 | in delivered code, not plan text | `QUERIES.md` §18.0 (guards/provenance), §18.5 (`dev_surface=False` dependency), §18.4/§18.5 (`HEAD`/`NEXT` invariant), §18.5/§18.7 (`ReadCursor` orphan + `Agent` residual) |
| `DESIGN.md` §5.1 arrow gains `Channel {…participantId…}` | S4 | absorbed (v1.8 §5.0 row) | `falkor-chat/docs/DESIGN.md:189-192` |
| No `u.userId > ''`; bare `tokenHash IS NOT NULL` in all three places | S4/S10 | absorbed (positive form) | plan S10 row; `repository.py:3584` |
| The five-row anomaly response contract | §5.2 / S8 / S10 | 4.5 of 5; one inverted | **M-2**, m-11 |
| Client-side timeout on a reset means *unknown* | **S7/S10** | **misabsorbed to S12a** | **M-1** |
| S4's positive fixture (§2.1 + the three v1.1 shapes + cross-member) | S4 | in delivered code, not plan text | four test names in `tests/test_repository.py`, listed above |
| S7/S10 quiesce = §7's four-part replacement | S7/S10 | absorbed (v1.7) | plan S7/S10 done-conditions |
| Do not relax either guard | S4 implementation | out of plan scope | — |
| Open item 1 — post-reset profile-name re-write | S7 | absorbed (v1.8) | plan S7 row + §6.1 bullet |
| Open item 2 — no cursor advancement (resolved) | — | correctly out of scope | note §12 resolves it itself |
| Open item 3 — `MAX_QUEUED_QUERIES 25` under `reset_all` | S15 | absorbed (v1.8); mechanism unnamed | plan §6.4 + S15; **m-8**. Cap confirmed live: `GRAPH.CONFIG GET MAX_QUEUED_QUERIES` → `25` |

---

## Pass 4 — 2026-09-02, against plan v1.10 (the Pass 3 fix pass)

**Reviewed:** `docs/plans/salesperson-ui.md` v1.10, re-gating every Pass 3 finding plus the four
items `teco` flagged. Baseline for the delta remains `git show 55d4e57:` (v1.2). Upstreams
re-consulted where a disposition needed them: `docs/plans/salesperson-ui-graph.md` §7/§12,
`falkor-chat/docs/QUERIES.md` §18.3/§18.4/§18.5/§18.7, `falkorchat/db.py`, `falkorchat/app.py`.

**CPG: considered, not relevant — the delta is document text plus two dependency-behaviour
questions (redis-py's retry policy, falkor-chat's status-code inventory) that a Joern CPG of
falkor-chat cannot answer; `grep` over `falkorchat/*.py` and live introspection of the pinned venv
did.**

**Verdict: approve with suggestions** — 0 blockers, 1 major, 3 minors. The major is a one-clause
deletion: M-3's trim swept §5.2, S10 and S12d's *done-condition* but not S12d's *scope* column,
which still commissions the fields the trim removed. Everything else in v1.10 lands, and two of the
fixes are better than what Pass 3 proposed.

### What I checked by execution vs. judged statically

**Executed (pinned venv `falkor-chat/server/.venv`, redis-py 8.0.1):** the exception hierarchy
(`redis.exceptions.TimeoutError` and `.ConnectionError` are **siblings** under `RedisError`, not
parent/child); the retry policy on a connection built by falkor-chat's own `db.connect()`
(`retry._retries == 0`, `NoBackoff`, `socket_timeout == 10.0`) versus a bare `redis.Redis()`
(`_retries == 10`, supported errors include `TimeoutError`); `redis.client.Redis._execute_command`
source. Plus a `grep` inventory of every HTTP status code in `falkorchat/*.py`. **Static:** the plan
diff, `db.py` (connect-only exception wrapping), `app.py:125-139`'s error table, `api.py:63`,
`QUERIES.md` §18.3/§18.4/§18.5/§18.7, and the committed v1.2 for M-3's provenance. **Suite not
run** — the `coder` holds the DB and nothing below needed it.

### New findings

#### P4-1 · **Major** · S12d's scope column still commissions the roster fields M-3 removed

§5.1's S12d row, first sentence: "The roster table over `GET /shop/api/presenter/participants`
(**one row per participant: display name, language, message count, cart total, order status**)".
Its own done-condition, four lines later, says the opposite and cites the reason — "display name
**and language** (§5.2's roster keys — the roster carries no activity data, see S10)". §5.2 now
contracts four keys, S10 asserts "exactly §5.2's four keys", and `repository.list_participants`
projects no activity data at all. The scope column is the build instruction; leaving it in place
re-creates exactly the producer-less contract M-3 existed to remove, one column away from the
correction. **Fix:** delete the parenthetical or replace it with "(one row per participant, §5.2's
four keys)". Nothing else in the row changes.

#### P4-2 · **Minor** · the `504` path's own re-read is the thing most likely to time out, and its failure is unspecified

S7's and S10's new condition is "returns `504 reset_state_unknown` **after re-reading state**". But
the state re-read is another query against the same graph, and FalkorDB serialises writes per graph
— the condition that produced the first `TimeoutError` (a reset still executing) is precisely the
condition that will stall the re-read for another `FALKORDB_SOCKET_TIMEOUT`. The plan says nothing
about the double fault, so the natural implementation lets the second `TimeoutError` escape as a
500, losing the `504` the participant needs. **Fix:** one clause — "if the re-read also raises,
still return `504 reset_state_unknown`, with no state body" — and name the second test case in the
done-condition, because a fake repo that times out on the reset and *succeeds* on the read tests the
easier half.

#### P4-3 · **Minor** · S12a should key the `504` branch on the status code, not on `reset_state_unknown`

§3 states TLS is a reverse proxy's job, and a proxy generates its **own** `504 Gateway Timeout`
with an HTML body when an upstream is slow — which on a reset means exactly what falkor-chat's
`504` means: *unknown, re-read state*. The semantics coincide; the handling should too. As written,
S12a's branch reads as keyed on the named code, so a proxy `504` would fall through to the
"fetch that itself times out" branch at best, or be unhandled at worst. **Fix:** say the branch is
"any `504`, whether or not the body carries `reset_state_unknown`". This is a one-line clarification,
not a design change, and it makes the branch strictly harder to get wrong.

#### P4-4 · **Minor** · "never retried" is currently guaranteed by a falkordb-py default, not by anything falkor-chat sets

The plan's one absolute prohibition on the reset path ("never retried") is honourable at the
application layer only if nothing below it retries. On this build nothing does — I introspected a
connection built by falkor-chat's own `db.connect()` and got `retry._retries == 0` with `NoBackoff`.
But that is falkordb-py's choice, not falkor-chat's: a bare `redis.Redis(socket_timeout=…)` on the
same redis-py 8.0.1 yields `_retries == 10` with `redis.exceptions.TimeoutError` in its supported
set, and `Redis._execute_command` routes every command through `conn.retry.call_with_retry`. Had the
default applied, a timed-out reset-mine would be re-issued with the same `$newThreadId` and the
observable outcome would be a `Thread` UNIQUE violation — the plan's own "the graph needs repair"
signal, raised by a benign already-committed reset. **Fix:** record the verified premise in §4.8
("client-layer retry is off: `retry._retries == 0` on falkordb-py's connection, redis-py 8.0.1")
so a dependency bump has something to contradict, and have S8's contract test assert the
*application* layer by call count (a stubbed repository raising `TimeoutError` is called exactly
once). The call-count test cannot see the library layer — which is why the premise needs writing down.

### Disposition of Pass 3 findings

- **M-1 (blocker) — fixed.** §4.8's F8 bullet now names `FALKORDB_SOCKET_TIMEOUT` as falkor-chat's
  own Redis socket timeout (`config.py:29` → `db.py:44`) and calls out the "client" ambiguity
  explicitly; `504 reset_state_unknown` appears on both §5.2 reset rows, in S7's and S10's
  done-conditions and in S8's error map; v1.8's revision note is corrected in place. Rechecked all
  six sites. The split is clean — see the ruling below.
- **M-2 — fixed, and better than proposed.** The clause is on reset-mine with the mechanism cited
  (`QUERIES.md` §18.4's `CREATE (c)-[:HAS_THREAD]->(:Thread {threadId: $newThreadId…})`), and
  reset-all states it re-mints nothing and structurally cannot raise it. Ruling below.
- **M-3 — fixed in §5.2/S10, not in S12d's scope column.** §5.2 is now a strict subset of §18.3's
  delivered projection (rechecked field by field), S10 carries the field-set assertion and the
  reversal trigger. → **P4-1.**
- **M-4 — fixed.** S12b owns the control with the subtree boundary spelled out against S14's
  `views/Profile*`, and its done-condition asserts rendered state and the return to the language
  step. §10's AC-5 row re-attributed across S12b/S12a/S12d — the wider sweep was right; leaving AC-5
  pointing at S12a would have preserved the mis-attribution.
- **m-5 — fixed.** R4's risk column now reads "burns a turn … and can mint an orphan `ReadCursor`
  (§7, F3 — the `Message`/`StepRun`/`TraceEvent` orphans v1.0 feared do not occur)".
- **m-6 — fixed**, at the §2 heading and covering the CPG paragraph. Wording ruling below.
- **m-7 — fixed.** "three parallel groups" / "outside all six".
- **m-8 — fixed.** `GRAPH.INFO`'s `Waiting queries` section named, polled for the duration of the
  call, with the reason a named mechanism is required.
- **m-9 — fixed.** S7 now reads (a)–(d) participant-scoped and says why (b)'s "(intake stopped)"
  and (d)'s "after `reset_all`" are worded for the presenter path.
- **m-10 — fixed.** S10's parenthetical re-list is gone; it now cites "§5.2's anomaly contract holds
  in full". The cite-don't-re-list rule is now applied consistently across S4, S7, S10 and S12d.
- **m-11 — fixed.** "`unscopedCount == 0` returns no `incomplete` field at all, not
  `incomplete: false`" appears in both §5.2 and S10's done-condition.
- **m-12 — fixed.** S4's row splits the five S0-verbatim queries (each with its note section) from
  the four the plan specifies. Ruling below.
- **n-13 — fixed.** `bootstrap_schema.sh <probe-ws>` is first in S1's done-condition, with the
  reason.
- **n-14 — fixed, and better than proposed.** §5.0 makes `playwright.config.ts` single-owner after
  S12b: if the presenter spec needs its own project, that edit is S12b's, and S12d never touches
  the config. That removes the ambiguity rather than documenting it.

### Rulings on the four flagged items

1. **m-6's wording — the architect's is right and mine was wrong.** "As of v1.0" would have put a
   date error inside a staleness disclaimer. The CPG paragraph is present verbatim in the committed
   v1.2 (`git show 55d4e57:`) and references `analyst`'s re-verification of ~20 source claims, which
   happened at the Pass 1 gate — so it was written in the v1.1 era, not v1.0. "The design passes that
   produced it (v1.0–v1.2)" is the accurate envelope. Adopt as written.
2. **m-12's S4 edit is descriptive and does not reopen delivered scope — confirmed.** The row's
   Files column, Interface column ("the nine repository methods + two service wrappers"), method
   count and done-condition are all unchanged; only the *attribution* of where each query's design
   came from is new, and it is checkable: `docs/plans/salesperson-ui-graph.md` contains Cypher for
   the five and none for `add_channel_member` / `get_participant_record` / `set_participant_record`
   / `list_participants`, which live in `QUERIES.md` §18.2/§18.3. All nine exist in the delivered
   `repository.py`. Nothing is owed back to S4.
3. **`504` collides with nothing inside falkor-chat.** The complete status-code inventory across
   `falkorchat/*.py` is 201, 400, 404, 409, 503 — no 502, no 504 anywhere in the package or in
   `docs/SERVER.md`, and `app.py:125-139`'s exception→code table has no gateway-class entry. There
   is no convention to break. The one external overlap is a reverse proxy's own `504`, which is
   **semantically identical here** (unknown, re-read) — so it is a handling clarification, not a
   collision: **P4-3.** Uvicorn itself generates no 504.
4. **M-3's origin — confirmed, nothing owed back to S4.** In the committed v1.2 the string
   `messageCount` appears **exactly once in the whole plan**: in the §5.2 route row. No step
   commissioned it, no repository method was ever asked for it, and S0's §12 mandated only the bare
   `tokenHash IS NOT NULL` roster, which S4 delivered exactly (`QUERIES.md` §18.3 ≡
   `repository.py:3584-3589`). The six-field contract was the plan's own v1.0 invention written
   against no capability. S4 shipped what it was asked for.

### The three "also worth attention" questions

- **Is the `503`/`504` split clean? Yes, and structurally so.** The two codes have different
  *sources*, not just different meanings: `503` comes from the application-level quiesce wait
  (S7/S10's own timer, no graph call made), `504` from a `redis.exceptions.TimeoutError` raised by
  the reset query itself. They cannot be confused by exception handling either — I verified the MRO:
  `redis.exceptions.TimeoutError` and `redis.exceptions.ConnectionError` are **siblings** under
  `RedisError`, so no `except ConnectionError` can swallow the timeout. And a *connect*-time timeout
  is already translated to `FalkorDBUnreachableError` by `db.py:46` — correctly a `503` case, since
  nothing was sent. The one gap is the double fault: **P4-2.**
- **Can S7/S10's `504` condition genuinely fail? Yes.** A fake repository raising
  `redis.exceptions.TimeoutError` from `reset_participant` and an assertion on the status code goes
  red the moment the implementation returns the quiesce `503` — this is a real red/green, not a
  restatement. Add P4-2's second case so both orderings are covered.
- **S12a's narrowed half is not vestigial.** Two events remain browser-side and the server cannot
  handle either: a `504` still has to be *rendered* as unknown rather than as failure, and a fetch
  that times out in the browser is an event the server never sees at all. The narrowing is the right
  boundary; P4-3 only sharpens its trigger.

### What's solid in v1.10

Two fixes improved on what Pass 3 asked for, both by removing an ambiguity instead of documenting
one: reset-all's "structurally cannot raise it" is stronger than my "if it ever arises" — that
phrasing would have invited a defensive branch on the one route where a retry re-raises forever, and
the claim is checkable (`QUERIES.md` §18.5 contains no `CREATE (:Thread`, and the note's §4 says so
independently). §5.0's single-owner rule for `playwright.config.ts` does the same for n-14. S10's
"exactly §5.2's four keys — asserted, not assumed" is the assertion that would have caught M-3 in
the first place, and pairing the trim with a named reversal trigger (one aggregate query designed
by `graph-dba`, never a per-participant fan-out) preserves the option without leaving a
producer-less contract behind. S12a leading with the shared entry files as its *first* deliverable
closes the sequencing exposure raised under Pass 3's flagged item 3. And correcting v1.8's revision
note in place, rather than only fixing the text it described, keeps the document's own history
honest about the misattribution.

---

## Pass 5 — 2026-09-02, against plan v1.13 (the Pass 4 fix pass + the new §5.3)

**Reviewed:** the v1.10 → v1.13 delta (`diff -u` against my own Pass 4 working copy; `55d4e57` is
still the last commit to touch the file, so a `git diff` spans v1.2→v1.13 and is not the delta
here). Subject: the four Pass 4 dispositions, the two architect-found defects, and the new **§5.3**
in full — its faithfulness to §4.3/§4.8/§5.2, and whether C1–C8 are individually failable.

**CPG: considered, not relevant — the delta is one new plan section plus step-row edits; the two
factual questions it raised (does a disabled storefront 503 or 404, does `/state` 401 after a
committed sweep) were answered from §4.9's mount derivation and §4.8's delete inventory, which a
CPG of falkor-chat does not model.**

**Verdict: approve with suggestions** — 0 blockers, 1 major, 3 minors, 2 nits. §5.3 is the right
call and the right size: it is a genuine consolidation, §4 is byte-unchanged, and both architect-found
defects are real and correctly fixed. The major is that §5.3 stops one status code short of §5.2 —
and the code it omits is the one the quiesce contract exists to produce.

### What I checked by execution vs. judged statically

**Executed:** the v1.10→v1.13 `diff -u`; `grep` sweeps for `unscoped`, `503`, `health`, `Bearer` and
`STOREFRONT_ENABLED` across the plan; a `grep` of the v1.2 and v1.10 copies to date the bearer-header
formats. **Static:** §5.3 read whole against §4.3:464-472, §4.8's delete inventory, §5.2's route
table and §4.9's mount derivation; S7/S8/S10/S12a/S12b/S12d rows; `docs/plans/salesperson-ui-graph.md`
§5 for the unscoped branch's reachability. **Suite not run**; nothing below needed it, and I left
`reference` and the four probe graphs untouched.

### Disposition of Pass 4 findings

- **P4-1 — fixed.** S12d's scope column now reads "one row per participant, **§5.2's four keys** — no
  activity data, see S10". The three-fields string is gone from the row.
- **P4-2 — fixed, and better placed than proposed.** The double-timeout contract is stated **once**,
  in §4.8, and S7's and S10's done-conditions each carry "**both orderings**… a stub whose re-read
  *also* raises `TimeoutError` still returns `504`, with no state body, never a `500`". Rechecked
  both rows.
- **P4-3 — fixed.** §5.3 C4 keys the branch on the status code and names the proxy's body-less `504`
  as the reason; S12a asserts both a `reset_state_unknown` body and a bare proxy-style `504`.
- **P4-4 — fixed in full.** §4.8 carries the premise, the redis-py contrast and a reversal trigger
  naming `db.connect()` as the pin point; S8's done-condition carries the call-count assertion with
  an explicit note that it cannot see the library layer.

### New findings

#### P5-1 · **Major** · §5.3 covers five of §5.2's reset responses and omits two — and C6's blanket `409` mis-handles one of them

§5.3 is "the client half of §5.2". §5.2's reset routes can return `200`, `401`/`404`, `409
unscoped_participant`, `503` (quiesce timeout), `504`, and reset-all's `incomplete`/`unresolved`
`200`. §5.3 gives rules for `200` (C7), `401` (C3), `504` (C4/C5) and `409 TurnInProgress` (C6).
It gives **no rule for `503`, and none for `409 unscoped_participant`** — `grep -n unscoped` returns
only server rows (S4, S8, S10, §5.2); `503` appears nowhere in §5.3.

The `503` gap is the live one: a quiesce timeout on reset-everyone mid-demo is the exact scenario
§4.8's quiesce contract was built for, and the client's behaviour on it is unspecified in the
section created to specify client behaviour on reset responses. The `409` gap is sharper in kind:
C6 says "`409 TurnInProgress` retains the composer text **and re-enables send**", so an
undifferentiated `409` handler renders `unscoped_participant` — the graph note's "never let it read
as success" alarm — as an ordinary busy signal. That is the same defect shape as the
undifferentiated `401` this section exists to fix, one status code over. (Severity is bounded: the
note establishes the unscoped branch is "unreachable on a healthy graph", so this is
defence-in-depth — but it is defence-in-depth for a corrupt graph, which is when a lying client
costs most.)

**Fix:** split C6 into C6a (`409 TurnInProgress` — current text) and C6b (`409
unscoped_participant` — surface as an error that does not read as success or as busy; no retry, no
composer re-enable), and add one rule for `503`: *nothing changed, a retry is safe, say so*. Both
are one line each and complete the matrix.

#### P5-2 · **Minor** · S8's second `503` source has no path in the plan that produces it

S8's error map reads "`503` quiesce timeout **or** storefront disabled". §4.9 move 1 derives
`dev_surface = not config.STOREFRONT_ENABLED` and registers the `/shop/api` router **only** in the
storefront deployment — so a disabled deployment has no `/shop/api` routes to answer, and a request
gets `404`, not `503`. No path described anywhere in the plan produces the second variant. This is
the answer to the `503`-ambiguity question, and it is better than either proposed option: **strike
the phrase** rather than give the quiesce `503` a discriminating body it does not need. If some
path does produce it (a `storefront=True` app whose `Storefront` failed to construct, say), name it
— at which point it needs a body, and P5-1's new `503` rule needs to know about it.

#### P5-3 · **Minor** · the "each of C1–C8" claim outruns its own enumeration for C8

S12a's done-condition opens with "**each of §5.3's C1–C8 has a test that goes red when the rule is
broken**", then enumerates C2, C3, C4, C5, C6, C7. **C1** is fine — it is an emergent property of
C2 and C3 both passing, and no client can satisfy both with a global handler. **C8** has no named
test and is not implied by any other rule, yet it is the cheapest of the eight to assert: both
polling hooks read one shared exported constant (not two literals), and the catalog query is
fetched once. Leaving it inside a blanket claim is precisely the "reads as covered, isn't" pattern
the rest of the row is written against. **Fix:** name C8's test, or narrow the blanket claim to the
rules enumerated.

#### P5-4 · **Minor** · §6 now has a client bullet, but its unit tier still excludes the client by title

The new §6.2 bullet is the right one to have added, and citing C5 as the reason for
assert-on-requests-not-renders is well judged. Two residues remain: §6.1 is still headed "Unit /
offline (**every server step**)", so the SPA has no unit tier in §6 by construction; and
S12b/S12c/S12d/S13/S14 appear nowhere in §6 at all — their tests exist only inside their own
done-conditions and the Playwright project. The server gets per-step bullets across §6.1 and §6.2;
the client gets one. **Fix:** one sentence in §6.2 pointing at where the other five SPA steps' tests
live (own done-conditions; the Playwright mobile project owned by S12b; `presenter.spec.ts` by
S12d), and drop "every server step" from §6.1's heading. One bullet is enough for **S12a**; it is
not enough for the seven-step track.

### Nits

- **P5-5 ·** §5.3's credentials table, "Sent on" row: the presenter cell says "the three
  `/shop/api/presenter/*` routes" — but `POST /shop/api/presenter/session` is where the token is
  *minted* and cannot carry it. The parallel exclusion was applied to the participant column ("every
  route except `GET /shop/api/health` and the two `session` routes") and not to the presenter one.
  Read "the two authenticated presenter routes".
- **P5-6 ·** S12a's scope column says C1–C8 are "stated there and deliberately not restated here" and
  then, in the same sentence, restates C8 ("TanStack Query polling (2 s state+messages; catalog
  fetched once)"). A leftover from v1.10; drop it and let C8 be the source.

### Rulings on the three questions

1. **`localStorage` is sound, correctly labelled as a decision, and does not need a stakeholder
   round-trip — but re-ground one sentence.** The stakeholder has already accepted R6 (no
   authentication, a standing shared secret, and every bearer token on the wire in clear over a LAN);
   a storage-medium choice sits strictly inside that residual and is smaller than what they signed
   off. Keep it in the plan. The one weak link is the reasoning, not the conclusion: **the XSS delta
   between `localStorage` and `sessionStorage` is essentially nil** — any script executing in the
   page reads either one, so §4.2's `textContent`-only rendering is a real mitigation but not the
   discriminator between the two options. The genuine discriminators are the ones §5.3 already
   names first: R7 restart-survival and not losing a cart to a closed tab, plus cross-tab sharing,
   which C3 actually needs. That argument stands alone; lead with it and demote the XSS sentence to
   "and the standing XSS objection applies equally to either medium, so it does not decide between
   them". Worth stating as a strength, because it is the security-relevant part and it is right:
   **the presenter *key* is never stored** — only the exchanged token, per the table's `{token}`.
2. **The `503` question — the consistency edit is not the one to make.** See **P5-2**: the second
   source appears unreachable, so the discriminating body would discriminate against nothing. Strike
   the phrase. The `503` work that *is* worth doing is on the client side, where the code has no rule
   at all (**P5-1**) — which is the same lesson as `504`, arrived at from the other end.
3. **One bullet is enough for S12a and not enough for the track.** See **P5-4**. The asymmetry that
   matters is not bullet count but tier coverage: the client now has an integration/contract tier and
   still has no unit tier, in a §6 whose unit heading names server steps explicitly.

### The three "also worth checking" items

- **The consolidation is faithful; nothing moved semantically.** I diffed §4 (byte-unchanged, as
  reported) and traced each C-rule to its source: C1–C3 to §4.3:464-472 (verbatim promise —
  "invalidates their *participant* token, but **not** their presenter token, so they can keep driving
  the demo through the reset"), C4/C5 to §4.8's F8 and delete inventory, C6 to §4.4 measure 1a, C7 to
  §4.8's language-step paragraph, C8 to §2.3/R10's 2 s basis. The credentials table's header formats
  (`Bearer <participantId>.<token>`, `Bearer presenter.<presenterToken>`) are **not** new — they are
  in §5.2 in the committed v1.2, so the table restates rather than invents. The one thing §5.3
  genuinely *decides* is the storage medium, and it says so in bold. C5's honesty is the standout:
  documenting that a `/state`-for-both-paths client "appears to work and is standing on luck" is a
  disclosure most plans would have quietly resolved.
- **C1–C8 failability: seven hold, C8 does not** — see P5-3. C2, C3, C4, C6, C7 have named tests that
  go red on the stated defect; C5 is a reading rule that C3+C4's tests jointly pin; C1 is emergent
  from C2+C3. The "assert the intercepted request and the stored credentials, never the rendered
  outcome" clause is correct and load-bearing for exactly C3 and C4/C5, which are the two rules whose
  wrong implementation produces the right render — the row says so, and that is the sharpest
  statement of this pattern in the document.
- **The C6/C8 dead-composer reading is right, and cheap to shrink.** Up to one poll interval of
  disabled composer after a reply lands is acceptable for a demo, and the alternative — re-enabling
  optimistically on the client — would race the server's `409` and re-open the double-post that §4.4
  measure 1a exists to close, so the conservative direction is the correct one. S13's `turn`-driven
  thinking indicator covers the perception. One cheap improvement, not a finding: C8 polls `/state`
  and `/messages` as two queries, so the reply and the re-enable land on independent timers; aligning
  them (one query key, or one shared interval) collapses most of the window at no design cost. Worth
  a clause in C8 if the architect agrees. Note the interaction is currently documented **only in the
  architect's report, not in the plan** — a future reader meets it as a bug.

### What's solid in v1.13

Both architect-found defects are real, and the second is the kind a plan review is for: a
*successful* reset-all bouncing the presenter off `/shop/presenter` fires on the success path, every
demo, on AC-5's most visible action, and it contradicts a §4.3 promise I verified word for word.
Escalating from a clause to §5.3 was the right call rather than an over-reaction — the two defects
share one cause (a five-word client contract), and the enforced-vs-upheld table at the section's
close is the part I'd keep above all: stating that every isolation guarantee is a left-hand row, and
that C1–C8 place no security obligation on the browser, is exactly the boundary a reviewer needs to
not re-litigate client trust on every pass. C4's per-path split is derived rather than asserted (the
`/state` `401` follows from §4.8's delete inventory, and the roster is what the surviving credential
can reach), and C5 then names the luck it removes.

---

## Pass 6 — 2026-09-02, against plan v1.14 (Pass 5 fixes + the completeness table) — convergence check

**Reviewed:** the six Pass 5 dispositions, the new **C6a/C6b/C9**, the **§5.2-response → C-rule
completeness table**, the in-pass **C2** split, and — as the pass's main question — whether the
defect class that has now surfaced four times is structurally closed. Baseline: my own v1.10 and
v1.2 copies for row-level churn; delivered code now committed at `5a5a257`/`b4cbdc7`.

**CPG: considered, not relevant — this pass is entirely about the plan's own response matrix and
its internal consistency; the two code-touching claims (S10's rate-limiter wording, the delivered
`list_participants`) were settled from the plan text and `QUERIES.md`, which a CPG does not model.**

**Verdict: approve with suggestions** — 0 blockers, 1 major, 4 minors. All six Pass 5 findings are
fixed, three of them better than proposed. The major is a **fourth instance of the class**, and half
of it was *introduced by the completeness table itself* — which is the most useful thing this pass
found, because it answers the convergence question with evidence rather than opinion.

### What I checked by execution vs. judged statically

**Executed:** per-step row hashes across v1.2 / v1.10 / v1.14 (the stability evidence below);
`grep` sweeps for `C1–C8` residue, `unscoped`, `503`, and every status code in §5.2. **Static:**
§5.3 read whole against §5.2's route table row by row; §6.1/§6.2; S10's rate-limiter wording;
S12a's done-condition per C-rule. **Suite not run**; `reference` and the four probe graphs untouched.

### Disposition of Pass 5 findings

- **P5-1 — fixed, and the completeness table is more than I asked for.** C6a/C6b split with C6b
  keyed explicitly on the error body, C9 added for `503` with the *why* (retry is safe **because**
  nothing changed), and the "not shipped until it has a row here" rule makes the table live rather
  than decorative. Rechecked all three against §5.2.
- **P5-2 — fixed.** The phrase is struck; §4.9's `dev_surface = not config.STOREFRONT_ENABLED`
  derivation is what makes the disabled deployment `404`, and I re-verified that line.
- **P5-3 — fixed, with the strongest test in the row.** C8's "change the constant in the test and
  observe **both** intervals move, so two literals fail" is a genuine red/green on a rule that is
  usually only asserted by inspection.
- **P5-4 — fixed, and the architect's objection to my heading-only fix is correct** (see below).
- **P5-5 — fixed.** The presenter "Sent on" cell now names the two authenticated routes and excludes
  `presenter/session` with the reason.
- **P5-6 — fixed.** S12a's scope says "TanStack Query as the polling layer" and leaves the cadence
  to C8. The one surviving `C1–C8` is inside v1.13's revision note, where it is historically correct.

### New findings

#### P6-1 · **Major** · the fourth instance — and the completeness table created half of it

The table's row **"`401` / `404` on any participant route | C3"** over-generalizes on exactly the
axis the table exists to police. `404` carries two meanings across participant routes: on
`POST /shop/api/reset` it is "not a participant / already deleted" (C3's case, correct), and on
**`POST /shop/api/order/advance` it is "no order of theirs"** (§5.2) — an *ordinary* outcome of a
stale order button. Applying C3 to it clears the participant credential and returns the view to
join: **pressing a stale `cancel`/`deliver` logs the participant out of the demo.** And
`/order/advance`'s **`409` stale CAS** — a third meaning of `409`, alongside C6a's and C6b's — has no
row and no rule at all; its correct behaviour is neither C6a's composer re-enable nor C6b's alarm,
but "re-read state and show the current status".

**Fix:** re-key the table on **(route, response)** rather than on (response, rule) — one row per
§5.2 route × non-trivial response. Over-generalization then becomes unexpressible, because a row
that spans two routes cannot be written. Concretely this pass: split the `401`/`404` row, and add a
C-rule for `/order/advance`'s `409`/`404` ("re-read `/state`, render the current order, clear nothing,
navigate nowhere").

#### P6-2 · **Minor** · the table certifies completeness against a §5.2 that is itself incomplete

§5.2 bounds three request bodies (`displayName ≤ 60`, `language ∈ locales`, `text ≤ 2000`,
`limit 1..200`) and S8 implements them as "size-bounded Pydantic models" — which means FastAPI
returns **`422`** on violation. That response exists on the wire and appears in neither §5.2 nor the
completeness table, whose only relevant row is "ordinary `200` on … the two `session` routes; none
needed". A `422` with an unhandled shape on join renders as a dead button. **Fix:** add the
validation-failure response to §5.2's three bounded routes, which then obliges a table row; the rule
itself is trivial ("show the field error in place, clear nothing").

#### P6-3 · **Minor** · C1 is keyed on the credential, C2 and C3 on the route

C1: "**which credential the failed request carried** decides what is cleared." C2/C3: "a
**presenter-route** `401`/`403`…", "a **participant-route** `401`…". These coincide only because the
credentials table's "Sent on" rows make route ≡ credential a bijection — a fact stated elsewhere and
never joined to C1. Read literally, the section's governing rule is stated on a different key than
its two implementing rules, in a section whose entire premise is that dispatch keys matter. **Fix:**
one clause on C1 — "on this client route implies credential (see the *Sent on* rows), which is why
C2 and C3 may be stated by route".

#### P6-4 · **Minor** · C2's new half misdescribes S10's rate-limiter

C2 tells the client to "expect S10's rate-limiting to make repeated attempts **progressively
slower**". S10 specifies "**fixed delay** + attempt counter". A fixed delay is not progressive, so
either C2 is wrong or S10 is under-specified — and a client that assumes growing backoff may add its
own, which is the "retry around it" C2 forbids. **Fix:** align the wording to S10 ("a fixed
per-attempt delay and an attempt counter"), or decide S10 should back off progressively and change
S10. This is the one slip in the in-pass C2 fix, and it is exactly the kind an independent pass
catches.

#### P6-5 · **Minor** · the `localStorage` decision attributes to C3 a requirement C3 does not impose

The re-grounding is otherwise right, and the XSS demotion is now stated correctly. But
"**Cross-tab sharing:** C3 requires the presenter's two credentials to have independent lifetimes
across whatever tab each view is open in, which `sessionStorage` scopes away per tab" conflates two
things. C3's requirement is that clearing one credential does not clear the other — satisfied by
separate keys in **either** medium, within one tab. Cross-tab sharing is a *usability* property
(open `/shop/presenter` in a second tab without re-entering the key), not a C3 requirement. **Fix:**
re-label that discriminator "usability: one browser, two tabs, no re-authentication", and let R7
restart/tab survival carry the decision — it does, on its own.

### Ruling on the in-pass C2 fix

**Fixing it in-pass was right, and the fix is substantively correct.** Sitting on a *known* instance
of a class that had already recurred twice under review, in order to report it, would have cost a
round trip to buy nothing — the fix is additive (it splits a rule, changes no decision, and
cross-references C6b), and it came with a clean revert. I checked its three claims: on
`presenter/session` there is genuinely no credential to clear (C2's first half is what puts the user
at key entry, having already cleared it); the user is already on key entry; and "report in place" is
the only behaviour that does not double-clear or double-navigate. All hold.

The cost is visible too, and it is small and exactly of the expected kind: **P6-4**, a paraphrase of
S10 that S10 does not say. The discipline that would have caught it is the one the architect applies
elsewhere — when a rule cross-references another row, quote that row rather than characterise it. My
recommendation for the pattern, not a rule change: in-pass fixes of a *known* class are the right
default; the check that makes them safe is that every claim about another section is re-read from
that section, not recalled.

### The convergence question

**The table is a genuine structural improvement and not yet a structural fix.** The evidence is in
this pass: it caught the `403` instance immediately (which nothing else had, across four passes),
and it simultaneously *created* the `404` instance in **P6-1** — because a table keyed on
*(response → rule)* can group two routes' responses under one row, which is the same
over-generalization at one level up. It also inherits §5.2's own gaps (**P6-2**): it certifies
completeness against a list that is itself missing the `422`s.

So the honest scoring: the table converts "find the gap by luck" into "find the gap by enumeration",
which is worth having and is why the `403` instance took minutes rather than a pass. Re-keyed on
**(route, response)** it becomes something stronger — a shape in which the class is *unexpressible*,
because every row names exactly one route and one response and there is nowhere for a second meaning
to hide. That is the edit I would make before calling the class closed.

**Instance count, for the record:** `401` (v1.13, architect) · `409` (Pass 5, me) · `403` (v1.14,
architect via the table) · `404` + `409`-on-advance (Pass 6, me, half of it table-introduced). Four
instances, and the search that found the fourth was mechanical — I read §5.2's eleven rows against
the table's nine. That search is now cheap and repeatable, which is the table's real contribution.

### The four "also worth checking" items

- **Rejecting "one query key" was right.** Two queries share a schedule; one query shares a
  *result and an error surface*. Folding `/state` and `/messages` behind one key means one fetcher,
  which either implies a combined endpoint §5.2 does not have, or hides two independent failure
  modes behind one `isError` — and `/state` carries `turn`, whose whole job is to be authoritative
  separately from the transcript. "Aligns, does not merge", with R10's two-routes-per-tick budget
  restated, is the correct formulation and I would not change a word of it.
- **The §6.1 pairing is right and my heading-only fix was not.** A section titled "Unit / offline"
  containing only server tests is worse than one titled "every server step", because it advertises
  coverage it does not have — the same failure mode as a green test asserting nothing. Heading plus
  the client paragraph plus §6.2's five-step bullet is the correct shape. **The fetch-boundary detail
  is right and load-bearing:** stubbing at the fetch boundary is precisely what makes §6.2's "assert
  the intercepted request" possible — a stub at the module boundary would let a wrong URL pass, which
  is the C4/C5 defect.
- **The cross-tab claim about `sessionStorage` is factually true** — it is scoped per tab, so two
  tabs of the same origin get separate stores, while `localStorage` is shared origin-wide. The
  *attribution* is what slips: see **P6-5**.
- **The maintenance risk: the table is on the right side of the line today.** Every cell in its
  right-hand column is a rule *number*, never a restatement of what the rule does — the one cell
  that carries prose (`incomplete`/`unresolved`) explicitly says there is no client rule and points
  at S12d. That is a pointer index. The failure mode to watch is precisely the one the architect
  named: the moment a cell explains a behaviour, §5.2 / the C-rule / the table can disagree three
  ways. P6-1's fix does not endanger this — re-keying on (route, response) makes rows *more*
  numerous and *less* prosy, which is the safe direction.

### Is the plan stable enough to dispatch S3 and S6? — **Yes, and the evidence is quantitative**

I hashed each step's row across v1.2, v1.10 and v1.14:

| Step | v1.2 | v1.10 | v1.14 |
|---|---|---|---|
| **S3** | `bdd89374` | `bdd89374` | `bdd89374` |
| **S6** | `8c62ed54` | `8c62ed54` | `8c62ed54` |
| S7 | `940acc6d` | `10b7e829` | `b1fb5e6d` |
| S8 | `fd462c21` | `3e53f238` | `cf19797a` |
| S10 | `08a5c4a0` | `f897f6fc` | `fcf9cdc8` |
| S12a | `788728ac` | `dfc9f910` | `87a04766` |

**S3 and S6 are byte-identical across twelve revisions and four review passes.** Every finding from
Pass 3 onward has landed in the error-contract surface — §4.8's F8, §5.2's response column, §5.3, and
the S7/S8/S10/S12* rows that implement them — and neither S3 nor S6 touches any of it: S3 is two
wiring switches over `config.py`/`app.py`, S6 is the registry, join, token verify and turn-state map,
whose contract (`resolve_token` re-reads the graph, join writes the profile name, restart survival)
has not been questioned since Pass 1. Their upstream, S4, is delivered and committed at `5a5a257`.
**Dispatch both now.**

**What I would not dispatch yet: S8, S10 and S12a**, which own the response map, the presenter
surface and the client contract respectively — all three rows changed in this pass, and P6-1 changes
two of them again. S7 is a judgement call: its row changed in v1.14 only through the F8/quiesce
clauses that are now settled, and P6-1 does not reach it, so S7 is dispatchable once S6 lands.
The convergence signal I would watch is not "no more findings" — it is *where* they land: four passes
running, they have landed in one surface, and that surface is now enumerated.

---

## Pass 7 — 2026-09-02, against plan v1.15 (the Pass 6 fix pass) — the standing class question

**Reviewed:** the v1.14 → v1.15 delta (P6-1…P6-5, C10/C11/C12, the re-keyed 36-row table) and, as
this pass's main question, whether the defect class — *one client rule spanning server responses
that share a status code but not a meaning* — is structurally closed. Fresh reviewer: Passes 3–6
were one reviewer, and Pass 6 prescribed the re-keying that v1.15 implements. Baseline: the v1.2
committed copy (`git show 55d4e57:docs/plans/salesperson-ui.md`), the previous reviewer's v1.14
working copy (authenticated — see Appendix P7-A), the delivered server at `5a5a257`, and the
**installed** `@tanstack/query-core@5.102.8` under `salesperson/node_modules`.

**CPG: used `cpg_falkorchat` — queried for registered exception handlers/middleware behind P7-3;
the call-name projection returned nothing usable, so that finding rests on a direct read of
`falkor-chat/server/falkorchat/app.py` and `db.py` instead, and the CPG contributed only the
negative.**

**Verdict: approve with suggestions** — 0 blockers, 3 majors, 3 minors, 1 nit. All five Pass 6
findings are fixed and the re-keying is a real improvement. **The class is not closed.** Instance
six exists, twice, on two axes the table cannot express, and the architect's own caveat about §5.2
proved concrete in a worse place than a `422`.

### What I checked by execution vs. judged statically

**Executed:** per-step row hashes across v1.2 / v1.14 / v1.15 — I re-derived Pass 6's undocumented
method (`md5sum` of the whole row line, first 8 hex) and reproduced its entire v1.2 and v1.14
columns, which both validates the method and authenticates the v1.14 copy I diffed against
(Appendix P7-A); `diff -u` v1.14→v1.15; reads of the pinned installed library source
(`retryer.js:89-92`, `mutation.js:81`, `query.js:240`, `mutationCache.js:103`) and of
`salesperson/package-lock.json`; `grep` sweeps of `falkor-chat/server/falkorchat/*.py` for
exception handlers; one `cpg_falkorchat` count + call query. **Static:** §5.2 and §5.3 read whole,
the 36 rows against §5.2's Returns column one by one, C10/C11/C12 against §4.4/§4.6/§4.8/F8, and
the S6–S10 / S12a rows. **Suite not run** — S3 is in flight and holds the database; `reference`,
`ws:test` and the four probe graphs untouched, no tree-mutating git command, no `GRAPH.DELETE`.

### Disposition of Pass 6 findings

- **P6-1 — fixed as prescribed.** Table re-keyed on (route, response), 9 → 36 rows; C10 added for
  `/order/advance`'s `404`/`409` with the log-out-for-a-stale-button reason stated. See P7-6 for the
  seven rows that still name more than one response.
- **P6-2 — fixed, larger than the finding.** `422` added to five routes, not three; C11 splits the
  two kinds. The split's boundary is wrong on one route — **P7-1**.
- **P6-3 — fixed.** C1 now states the route→credential function and names the condition under which
  it breaks.
- **P6-4 — fixed.** S10's "fixed delay + attempt counter" is quoted, and the no-backoff-assumption
  clause is explicit. What the counter *does* is still unstated — **P7-5**.
- **P6-5 — fixed.** Cross-tab sharing is re-labelled usability; R7 carries the decision.

### New findings

#### P7-1 · **Major** · instance six, *inside* a single (route, response) cell — C11's `422` boundary is wrong on `presenter/session` and ambiguous on `session`

The table's key is (route, status). Two of C11's five routes carry **two meanings in one cell**:

- **`POST /shop/api/presenter/session` `422` is classified client-bug-only.** §5.2 says the `422`
  fires on a *missing/blank* `key` — which is what a human produces by pressing Enter on an empty
  key box, the most ordinary mistake in the presenter flow. C11's client-bug branch says "there is
  no field to blame: surface it as an error and do not retry"; there *is* a field, and the right
  behaviour is the user-reachable one.
- **`POST /shop/api/session` `422` is classified user-reachable, and covers two kinds.**
  `displayName ≤ 60` is user-fixable; `language ∈ locales` is not — the chooser is S12c's three
  bundles and nothing reads `/health`'s `locales`, so a server `FALKORCHAT_STOREFRONT_LOCALES`
  narrower than the bundles makes this reachable **by demo bring-up config drift**, showing a field
  error next to a picker the user cannot fix.

The discriminator here is the error body's field (`loc`), not the (route, status) pair — a key the
table cannot express. **The plan already has a rule finer than its table's key: C6b dispatches on
the error body, not the code.** **Fix:** classify by *field*, not by route (blank `key` and
`displayName` → user-reachable; `language`, `limit`, `transition` → client-bug-only), and have S8's
row commit to a stable `422` body shape C11 may key on.

#### P7-2 · **Major** · C12's premise is correct (now verified from source) but "queries retry harmlessly" is not — TanStack's retry is status-blind, and it contradicts C11 and delays C3

Read from the pinned installed source, not the docs: `mutation.js:81` `retry: this.options.retry ?? 0`
(mutations: no retry ✓) and `query.js:240` + `retryer.js:89` `config.retry ?? (isServer() ? 0 : 3)`
(queries: 3 ✓), `retryer.js:6-8` backoff `min(1000·2^n, 30000)` → 1 s + 2 s + 4 s. The premise
holds. What does not is the inference. `retryer.js:92`'s `shouldRetry` is **blind to status** — it
retries any rejected fetch, and a query's fetcher must reject on `401` for C3 to fire at all. So:

- `GET /shop/api/messages` `422` (`limit`) — C11 says **do not retry**; the blessed default retries
  it three times. Two rules of §5.3 contradict each other on one table cell.
- `GET /shop/api/state` `401` — C3's headline scenario (every successful `reset-all`) surfaces
  **~7 s late**, with the presenter's participant view stale meanwhile.
- R10's budget of two requests per 2 s tick becomes up to **eight**, per participant, precisely
  during an outage — at 50 participants, amplification when the server is already failing.

**Fix:** re-state C12 on the transport axis rather than the status axis — polling queries pin
`retry: 0` (`refetchInterval` *is* the retry; a retry adds latency and load, not resilience), the
one-shot catalog fetch may keep a bounded 5xx-only predicate, mutations keep the default. Re-aim the
reversal trigger: it currently watches the mutation default, which is the case that did not bite.

#### P7-3 · **Major** · §5.2's other gap today — F8's timeout contract is scoped to the two resets, and the same FalkorDB stall answers an unmapped `500` on the other nine routes

§4.8 F8 is worded "a `redis` `TimeoutError` **from either reset**", and S7/S10 carry it. Every other
route reads the graph under the same 10 s `FALKORDB_SOCKET_TIMEOUT`. Verified in the delivered code:
`_register_error_handlers` (`app.py:80-150`) maps `ServiceError` and the workflow errors only;
`db.py:39-52` converts a **connect-time** timeout to `FalkorDBUnreachableError`, which has **no
handler** (`grep`: raised at `db.py:47`, referenced only in lifespan comments). A query-time
`redis.exceptions.TimeoutError` therefore escapes as a bare `500` — on `/state` and `/messages`, for
every polling participant at once, in the exact scenario F8 exists for; and by P7-2 each one is
retried three times. Neither §5.2 nor the table has a row. If S8 instead maps *any* `TimeoutError` to
`504 reset_state_unknown`, the other horn: a poll answers a reset code and C4 fires reset-re-read
logic on a read. **Fix:** register a typed handler for `redis` `TimeoutError`/`ConnectionError` and
`FalkorDBUnreachableError` → one documented code (`503 "FalkorDB unreachable"` already exists as
precedent at `api.py:63` / `app.py:345`), following this codebase's stated convention of typed
handlers "without a blanket handler masking real bugs" (`app.py:136-137`); then give §5.2 and the
table their rows. Routes to **S8**.

#### P7-4 · **Minor** · `networkMode` pauses a reset mutation offline and auto-resumes it — an automatic dispatch C12 does not cover

`retryer.js:9-10, 51-52` and `mutationCache.js:103`: with the default `networkMode: 'online'`, a
mutation fired while the browser is offline is **paused, not failed**, and `resumePausedMutations()`
fires it when connectivity returns; continuation also requires window focus. On a LAN demo with a
phone, the presenter presses "reset everyone", sees nothing (no error to report, so C9's "nothing
changed" never renders), and the sweep executes later — possibly after they have moved on. **Fix:**
one clause on C12: the reset mutations pin `networkMode: 'always'` so an offline attempt fails fast
into C9's path instead of queueing; assert it in S12a alongside the no-retry request count.

#### P7-5 · **Minor** · S10's rate-limiter attempt counter has no specified effect and no response code

S10: "rate-limited: fixed delay + attempt counter"; done-condition: "a wrong key is refused and
**counted**". Nothing says what the count changes. Either it is inert — in which case the only
defence is a fixed delay and the wording oversells it — or it eventually locks out, which is an
unlisted response (`429`/`423`, or a `403` that now means *throttled* rather than *wrong key*) with
no §5.2 row and no client rule. That second branch is instance seven pre-loaded: C2's second half
would tell the presenter "your key is wrong" when the key may be right. **Fix:** state the counter's
effect and its response in S10 and §5.2, or strike it and let the fixed delay stand alone.

#### P7-6 · **Minor** · the table bends its own key in 7 of 36 rows, and 8 rows have no §5.2 counterpart, so "certifies against §5.2" runs in the wrong direction

Seven rows name more than one response: `POST /reset` `401 / 404`; `POST /reset` and `reset-all`
`504 … and any bare 504`; `presenter/participants` and `reset-all` `401 / 403`; and the two `5xx`
rows (a class, not a response). Each grouping is exactly the shape the key exists to forbid — the
`presenter/participants` `401 / 403` pair, for instance, spans *session gone* and *wrong credential
type* (§6.2's auth matrix), which happen to share an action. Conversely, eight rows — every
participant-route `401`, both presenter routes' `401`/`403`, and `reset-all`'s `503` — have no entry
in §5.2's Returns column at all; they come from S8's error map and §4.8. The table is now the more
complete document, while the plan says it certifies against §5.2. **Fix:** name the table as the
source of truth for the response set and make §5.2's Returns column its prose view; where a row must
group, say why the two responses share one meaning.

### Nit

- **P7-N1 ·** `GET /shop/api/health` publishes `locales` and nothing consumes it. Wiring S12c's
  chooser to it would delete P7-1's config-drift trigger outright.

### Is the class closed? — **No, and the table cannot close it**

The re-key does what Pass 6 predicted on the axis it keys: across (route × status) a second meaning
has nowhere to hide, and finding instances is now mechanical. **Instance six is not on that axis.**
It appears twice:

1. **Below the key** — inside one cell, discriminated by the error body's field (**P7-1**). The
   plan's own C6b already dispatches on the body rather than the code, which is a proof from inside
   the document that (route, status) is not a sufficient key.
2. **Beside the key** — on the transport axis (**P7-2**). The table has no column for
   query-vs-mutation, so C12's rule stated over `5xx` silently governs `401` and `422` as well and
   contradicts C11 on a specific cell.

And the architect's caveat — *the class is unexpressible in the table, not in the plan; §5.2 is the
unverified upstream* — is correct and lands harder than a `422`: **P7-3**, where a `TimeoutError`
means "unknown, re-read" on two routes and an unmapped `500` on nine.

**Ruling on the proposed closure (S8's contract tests).** Necessary, not sufficient, and mis-aimed
as stated. An enumeration test proves every *listed* pair is producible; it cannot prove no
*unlisted* pair is producible, and the unlisted half is what has failed six times. Two cheap totality
guards do close it, and I would gate S8 on the first:

- **Server — make the error map total by type.** With every escaping exception class mapped by a
  typed handler (P7-3's fix, in this codebase's existing idiom), the set of producible responses is
  bounded by construction. The check at S8's implementation gate is then mechanical and, unlike
  prose-reading, decidable: **{registered handlers} × {routes} ⊆ table**, and `TestClient` asserts
  each. Check the enumeration against the *handler set*, not only against the table.
- **Client — one loud default branch.** Any (route, response) with no matching rule renders an
  explicit "unhandled response" failure rather than falling into the nearest handler. "Unexpressible
  in the table" is a document property; "unhandled ⇒ loud" is a runtime property, and only the second
  survives §5.2 being wrong. It is also the only guard that would have caught instances 1, 3 and 4 as
  they happened, in the demo, rather than in a review pass.

### Rulings on the four flagged items

1. **C12's verification debt — the right call in form, and now dischargeable without Node.** Premise
   + reversal trigger + a named verifier is the correct discipline for an unexecutable claim. But the
   claim is checkable **today**: `node` is genuinely absent (`node: command not found`; only a Windows
   `npm` shim on PATH), yet `salesperson/node_modules` is populated at the lock's exact
   `@tanstack/query-core@5.102.8`, so the defaults are readable as source. Both halves are **correct**
   — see P7-2 for the citations. Promote C12 from "stated premise" to "verified against the lock",
   keep the reversal trigger but re-aim it at the query default.
2. **C12's route-scoping — flagging rather than deciding was right; the reading it proposes should be
   rejected.** "Retrying is what makes a poll resilient" is false for these two polls: `refetchInterval`
   re-issues every 2 s regardless, so `retry: 3` buys nothing and costs ~7 s of dispatch latency plus
   4× the R10 tick budget (P7-2). Decide it as: no automatic retry anywhere except the one-shot
   catalog fetch.
3. **C11's split — correct on three routes, wrong on `presenter/session`, ambiguous on `session`**
   (P7-1). "Do not retry a client-bug `422`" is the right rule and currently unenforceable on the one
   client-bug cell that is a query (P7-2); it should also say where the error goes — a dev-visible
   surface, since the user can do nothing about it.
4. **36 rows is still on the right side of the line, with two caveats.** Every right-hand cell is a
   rule number or an explicit pointer; the two "**C10**, *not* C3" cells are annotation, not
   restatement, and earn their words. The one cell that has crossed into mechanism is `reset-all`
   `200 clean` → "C3 fires on the presenter's own participant poll; S12d re-renders", which can now
   drift from C3. Row count is not the cost driver; the missing *generation rule* is (P7-6). At 36
   hand-written rows with 8 lacking an upstream, name the source of truth before the next pass.

### Dispatch judgment for S7 / S8 / S10 / S12a

Pass 6's method reproduced and extended (Appendix P7-A). Its v1.2 and v1.14 columns reproduce
exactly, so these numbers are comparable to that table.

| Step | v1.2 | v1.14 | v1.15 | Call |
|---|---|---|---|---|
| S3 | `bdd89374` | `bdd89374` | `bdd89374` | in flight; untouched by this pass |
| S6 | `8c62ed54` | `8c62ed54` | `8c62ed54` | dispatch (unchanged from Pass 6) |
| **S7** | `940acc6d` | `b1fb5e6d` | `b1fb5e6d` | **dispatchable once S6 lands** |
| S8 | `fd462c21` | `cf19797a` | `aae76b9e` | **hold** |
| S10 | `08a5c4a0` | `fcf9cdc8` | `fcf9cdc8` | **hold**, pending P7-5 only |
| S12a | `788728ac` | `dfc9f910`→`87a04766` | `806410ee` | **hold** |

- **S7 — yes.** Its row is byte-identical for the first time across two versions, and no Pass 7
  finding routes to it: P7-3's fix lands on the error map (S8) and §5.2, not on `get_state`'s
  composition or `reset_participant`'s quiesce.
- **S8 — no.** The row moved again, and two majors land on it: P7-3's typed timeout handler plus the
  §5.2/table rows it obliges, and P7-1's requirement that the `422` body carry a discriminator C11
  can key on. Both change the error map S8 builds.
- **S10 — no, but cheaply.** First version with no movement, and only P7-5 (a one-answer question)
  routes to it. It is sequenced after S9 → S8 anyway, so the hold costs nothing.
- **S12a — no; its row has not stopped moving.** It absorbed the whole v1.15 delta and takes three
  more done-conditions from this pass (P7-1's field-level `422`, P7-2's retry pinning, P7-4's
  `networkMode`). It is gated on S8 regardless.

**Worth saying plainly:** the chain S7 → S8 → S9 → S10 → S12a means holding S8 already holds S10 and
S12a. The only dispatch this gate actually unblocks is **S7**, and it is a yes.

### What's solid in v1.15

The re-key is the right edit and was executed beyond the finding — 36 rows, `422` on five routes
rather than the three I would have found, and C10 written with the concrete harm ("logs a participant
out for pressing a stale `cancel`") rather than as a rule. C12 is the strongest new rule in the
section: it found a hard absolute in §5.2 with no client rule at all, and it states its own
unverified premise, its verifier and its reversal trigger rather than asserting a library default —
which is exactly why it was cheap for me to discharge. C1's new clause (route → credential is a
function, and here is the condition under which the bijection breaks) is the kind of statement that
makes a future reviewer's job mechanical. And the architect's caveat about §5.2 being an unverified
upstream is the sharpest sentence in the delta: it is right, and P7-3 is what it was pointing at.

### Open questions (Pass 7)

1. **What shape is the `422` body, and may C11 key on it?** FastAPI's default carries `loc`; P7-1's
   fix needs S8 to commit to that (or to a mapped shape) as contract, not as an accident of the
   framework. Architect's call.
2. **What does S10's attempt counter do** (P7-5) — inert, or a lockout with an unlisted response?
3. **Which document is the source of truth for the response set** — §5.2 or §5.3's table (P7-6)?
   They currently certify in a direction that eight rows contradict.

## Appendix

### Pass 7 · A — step-row hash method, reproduced and authenticated

Pass 6 published hashes without naming the method. Recovered by trial against its v1.2 column:
`md5sum` of the entire step row (the full `| **Sn** | … |` line, trailing newline included), first 8
hex characters. All six of Pass 6's v1.2 values reproduce exactly, and all six of its v1.14 values
reproduce from the previous reviewer's working copy — which is how that copy is authenticated as
genuinely v1.14 rather than trusted (its header also reads `Version: 1.14`). The v1.14→v1.15 delta I
reviewed is therefore the real delta, not a reconstruction.

```
for s in S3 S6 S7 S8 S10 S12a; do
  grep -m1 "^| \*\*$s\*\* |" "$copy" | md5sum | cut -c1-8
done
```

### Pass 7 · B — TanStack Query v5 defaults, read from the pinned installed source

`salesperson/package-lock.json` resolves `@tanstack/query-core` to **5.102.8**; that exact version is
present under `salesperson/node_modules/@tanstack/query-core`. Node is not runnable on this box
(`node: command not found`), so this is a source read, not an execution — but of the pinned artifact
rather than of documentation.

| Claim | Evidence |
|---|---|
| Mutations default to no retry | `build/modern/mutation.js:81` — `retry: this.options.retry ?? 0` |
| Queries default to 3 retries | `build/modern/query.js:240` passes `context.options.retry` (undefined) → `build/modern/retryer.js:89` `config.retry ?? (isServer() ? 0 : 3)` |
| Backoff 1 s / 2 s / 4 s | `retryer.js:6-8` — `Math.min(1e3 * 2 ** failureCount, 3e4)` |
| Retry is status-blind | `retryer.js:92` — `shouldRetry` tests only the count/predicate, never the error |
| Offline mutations pause and auto-resume | `retryer.js:9-10, 51-52` (`networkMode ?? "online"`, focus-gated continue); `mutationCache.js:103` `resumePausedMutations()` |

---

## Pass 8 — 2026-09-02, against plan v1.16 (the Pass 7 fix pass) — the two totality guards

**Reviewed:** the v1.15 → v1.16 delta — S8's total-by-type error map and its decidable gate, the
read-vs-write timeout split, C11's field re-key, C12's transport re-key, new **C13**, the
cross-cutting sub-table and the generation rule, S10's counter decision, and the `SERVER.md` map
gap — plus the four judgments `teco` asked for and the S8 dispatch call. Baseline: my own v1.15
copy (authenticated in Pass 7 against the committed v1.2 and the previous reviewer's v1.14), the
delivered server at `5a5a257`, and the pinned `@tanstack/query-core@5.102.8` source.

**CPG: considered, not relevant — the delta is plan text plus code claims that are line-level
citations of `app.py`/`db.py`/`api.py`, which I verified by direct read in Pass 7 and re-checked
here; `cpg_falkorchat` models none of the `/shop/api` code, which does not exist yet.**

**Verdict: approve with suggestions** — 0 blockers, 4 majors, 3 minors, 1 nit. The two guards are
built correctly and the read-vs-write split is the right generalisation. **The majors are all one
thing:** generalising F8 from "either reset" to "every write" extended C4's *domain* faster than
its *content*, and the cross-cutting `504` row hid the two cells that opened. This is the
mis-ruled residual, live, in the delta that was supposed to close the class — which is the
strongest possible confirmation of the architect's own scoring, and the reason my answer to
"ship?" is yes with a one-touch edit and **no Pass 9**.

### What I checked by execution vs. judged statically

**Executed:** `diff -u` v1.15→v1.16; step-row hashes at v1.16 by the Pass 7 method (independently
reproducing the architect's post-edit check of S6/S7); `grep` sweeps for the scoring language, the
health route, and presenter-token storage. **Static:** §5.3 read whole, the cross-cutting table
against S8's row and §4.8's F8, C4's re-read list against the set of writing routes, C9/C11/C12/C13
against their sources, S8/S9/S10/S12a rows. **Suite not run** — a `coder` holds
`storefront.py`/`config.py`/`test_storefront.py` and the database; `reference`, `ws:test` and the
four probe graphs untouched, no tree-mutating git command, no `GRAPH.DELETE`.

### Disposition of Pass 7 findings

- **P7-1 — fixed, and the boundary is now right.** C11 keyed on `field`; `key` moved to
  user-supplied, `language` to UI-supplied with the config-drift reasoning intact; S8 owns a stable
  `{error, field}` body. See P8-N1 for the multi-error case.
- **P7-2 — fixed beyond the finding.** Transport axis, `retry: 0` pinned explicitly on mutations
  rather than inherited, `5xx`-only predicate confined to the one-shot catalog fetch, and the
  premise promoted to verified with the source citations correct (I re-read them).
- **P7-3 — fixed; the generalisation went further than I asked and was right to.** See P8-1/P8-2 for
  the cost of the extra distance.
- **P7-4 — fixed.** `networkMode: 'always'` on the reset mutations, with the "sees nothing at all"
  failure named.
- **P7-5 — fixed by decision** (observational only; see judgment 4).
- **P7-6 + nit — fixed.** Source of truth flipped, generation rule added, groupings split or
  justified, `/health`'s `locales` given a consumer in S12c.

### New findings

#### P8-1 · **Major** · C4's new `POST /messages` re-read confirms the write but not the *turn* — and the likely post-timeout state is the one §4.4 measure 1a exists to prevent

The storefront post path is `services.post_message` **then** enqueue (S9). A query-time
`TimeoutError` therefore fires *during the write*, before the enqueue — so the overwhelmingly likely
committed state is **message written, turn never scheduled**. C4's new row sends the client to
`GET /shop/api/messages`, which shows the message present and says nothing about the turn; the
participant then waits forever on a reply that was never queued. §4.4 1a refuses *before* the write
precisely because "a written message with no reply would sit in the transcript forever" — the new
C4 row reintroduces that state through the back door. **Fix:** re-read `/state` as well (it carries
`turn`), and state the reconciliation: message present **and** `turn.state === 'idle'` ⇒ the turn was
lost — say so and re-enable send. Also say what re-sending costs (a duplicate line in the
transcript), because that is the only recovery the API offers.

#### P8-2 · **Major** · C4 covers four of the six writing routes; `POST /shop/api/session`'s `504` has no re-read and cannot have one

The cross-cutting table routes **every** writing route's `504` to C4. C4 enumerates four re-read
endpoints. The two credential-minting routes are missing, and join is not an oversight that a fifth
bullet fixes: `POST /shop/api/session` **writes** (S6 provisions `User`+`Channel`+`Thread`+profile),
and if it commits but the response is lost, **the token was never delivered** — there is no
credential with which to re-read anything. The graph keeps a `User` with a `tokenHash` that nobody
holds: a ghost row in the presenter roster, owning a `Channel` and `Thread`, while the person joins
again as a second identity. **Fix:** give C4 an explicit fifth case — *no surviving credential, so
report "your join may not have completed, join again" and warn the presenter that a stale roster row
may appear* — or make join idempotent on a client-supplied nonce (permitted: §5.2's invariant bans
`ws`/`threadId`/`customerId`/`orderId`, not an idempotency key). `POST /presenter/session` needs the
row too, or the classification in P8-4 that excludes it.

#### P8-3 · **Major** · the `504` cross-cutting row is the one grouping whose rule is route-*dependent*, and its by-construction licence is literally false

The stated licence is "produced by one typed handler rather than by route logic". For the two `503`
rows that holds. For `504 <op>_state_unknown` it does not, twice over: (a) the new **precedence
rule** says the two reset routes catch the timeout *themselves* and answer their own named
`504 reset_state_unknown` — so the row already spans two producers, one of which is route logic; and
(b) its rule is C4, whose entire content is a **per-route** re-read endpoint. A row may span routes
that share one meaning — these share a meaning ("the write may have committed") but not an action,
and the action is the rule. This is not theoretical: the grouping is exactly what hid P8-1 and P8-2.
**Fix:** split the `504` row per writing route (five or six rows), or keep one row with an explicit
per-route re-read column; and tighten the licence to *"a row may span routes that share one meaning
**and one action**"*.

#### P8-4 · **Major** · the route classes the gate computes over are never enumerated — and two routes plausibly belong to a third class the split has no name for

The cross-cutting table says "every route", "every route that **writes**", "every route that only
**reads**", and the plan never says which route is which. That is the input to
`{registered handlers} × {routes} ⊆ table`, so as written the gate is not computable. Two routes
resist the binary: `GET /shop/api/health` returns `{status, storefrontEnabled, locales}` with no
stated graph access, and `POST /shop/api/presenter/session` mints an in-process token from a key
(§4.3/S10: the presenter is not a `User` and survives `reset-all`) — neither may touch the graph at
all, so none of the three cross-cutting responses can arise on them. Under "every route" they get
`503 graph_unavailable` rows with **no producer**, and S8's own gate is symmetric: "a handler with no
row, **or a row with no producer**, fails the step". A coder following the plan literally hits a
failing gate and the tempting repair is to loosen the gate. **Fix:** enumerate the eleven routes into
`writes` / `reads-only` / `no graph access` in §5.3, and say that the third class takes none of the
three cross-cutting rows. This is the one Pass 8 finding that lands on **S8**.

### Minors

- **P8-5 ·** **C9's three sources share a meaning but not an action.** "Nothing changed" is true of
  all three, but the quiesce `503` follows a *user-initiated* reset (a retry control is right) while
  `graph_read_timeout` arrives on a **2 s background poll**, where there is nothing to offer a retry
  button for — the next tick *is* the retry, and what the presenter needs is a staleness indicator,
  which C9's own closing sentence gestures at without making it the rule. Split the action:
  user-initiated ⇒ retry control; poll ⇒ staleness indicator, no control.
- **P8-6 ·** **C13's headline overclaims, and the residual is not in the plan at all.** "The guard the
  other twelve are checked by" is not what C13 does: it detects the *absence* of a matching rule, and
  is silent when a rule matches and is wrong — which is the state P8-1, P8-2 and P8-3 are in. The
  unruled/mis-ruled scoring exists in the architect's report to `teco` and appears nowhere in the
  document that ships (`grep`: no occurrence). One sentence in C13: *a rule that matches and is wrong
  is invisible here; that residual is carried by each rule stating its own discriminator and by
  S12a's per-rule tests.*
- **P8-7 ·** **S8's "the only way it can be, by execution" is one way, not the only one.** A handler
  that `return`s a `JSONResponse` is invisible to the handler set — true — but FastAPI's per-route
  `responses={…}` declarations make a route's own return set machine-readable, which turns the second
  half of the gate static as well: `{declared} ∪ {handler-produced} == table`, with the contract tests
  then proving each declared entry is producible rather than being the only evidence any of it exists.

### Nit

- **P8-N1 ·** `{error: "validation_failed", field: "<name>"}` says "one field name per response", but
  Pydantic reports **all** violations at once. State the selection rule (first error in declaration
  order) and that `field` is the client-facing name (`displayName`, not `body.displayName`), so C11's
  highlighting is deterministic.

### Is the class closed enough to ship? — **Yes, with one touch, and stop the plan gates here**

**Plainly: the unruled half is closed, the mis-ruled half is open, and shipping that residual is the
right call — but not for the reason the plan currently gives.**

- **Unruled — closed, and I can say why structurally rather than hopefully.** Five of the seven
  instances (1, 3, 4, 5, 7) were responses the plan did not know it could receive. The server map is
  now total by type, so the producible set is bounded by construction, and C13 makes any survivor
  visible **in the demo, to the person who can act**. That pair does not depend on anyone having
  enumerated correctly, which is the property the four table re-keys never had.
- **Mis-ruled — open, and demonstrably still being generated.** P8-1, P8-2 and P8-3 are three
  mis-ruled instances **created by the v1.16 delta itself**: C13 stays quiet (a rule matches), and the
  server map is innocent (it produces a documented `504`). The architect's scoring is right, and one
  degree understated — this residual is not a static leftover, it is *fed* by every generalisation,
  because extending a rule's domain is one edit and extending its per-route content is six.
- **The mitigation is honest but it is a review aid, and the plan should say so.** Each rule now
  states its discriminator ("on the body, not the code" / "on the `field`, not the route" / "on the
  transport, not the status"), which converts an implicit assumption into a visible claim — real
  value, and it is what let me find P8-3 in minutes. It is **not** a guard: C4's discriminator is
  stated too ("keys on the status code, not the error string"), and that is precisely the rule that
  broke here. Label it as a review aid in the plan (P8-6), not as the third leg of the guard.
- **The mis-ruled half does need a mechanism, and it is already in the plan's vocabulary — it just
  needs one word.** S12a mandates "each rule has a test that goes red when the rule is broken".
  Require that **each rule's test enumerates the routes the rule spans**: C4's test then has to name
  all six writing routes, and P8-2 fails at implementation time, mechanically, without a reviewer.
  That is the residual's guard, it costs a clause, and it lands where evidence is executable.

**Recommendation on further passes: stop.** Fold P8-1…P8-4 (plus P8-5…P8-7) into **one architect
touch** on C4, the cross-cutting table and S8's classification; `teco` verifies by `diff` and by the
step-row hashes rather than by commissioning Pass 9. A ninth full pass has negative expected value:
Passes 5–8 each returned roughly one major plus a short tail, all in one surface, and the marginal
instance is now being produced *by the fixes* rather than found *in the original* — which converges
slowly under review and quickly under execution. Resume review at the two **implementation** gates,
where the evidence is runnable: S8's `{handlers} × {routes}` assertion (check that adding a handler
with no row actually fails it) and S12a's per-rule tests (check the route enumeration above). That is
a stopping rule, not a preference.

### The four judgments

1. **The architect's scoring — correct, and I will argue against my own prescription where it
   deserves it.** My guards do less than the framing of them in §5.3 implies. "Total by type" bounds
   only what is *raised*; C13 fires only where *nothing* matches. Between them sits every rule that
   matches and is wrong, and that band contains two of the seven instances plus three new ones this
   pass. So: right residual to ship (a wrong rule is a bounded, visible-in-testing defect, while an
   unruled response is a silent wrong branch in front of a stakeholder), wrong to present the pair as
   closure of the class. §5.3 should say what it closes and what it does not, in C13, in one
   sentence — the same honesty the section applies everywhere else.
2. **The bounded hole — the clause is honest but is not at its true strength, and the cover is
   partial.** "Total over raised exceptions" is exactly right and rare to admit. But the residue is
   covered by *two* things the plan names (per-route contract tests, C13) and one it does not
   (P8-7's `responses={…}` declaration), and "the only way it can be" should go. Contract tests
   assert the returns someone thought of; that is the same enumeration weakness one level down, and
   it is why C13 rather than the tests is the real backstop. Rule: **the hole is acceptable and
   correctly located; the sentence overclaims and the static half is cheap to add.**
3. **The permitted grouping — holds for two of the three rows and fails for the third.** See P8-3.
   `graph_unavailable` and `graph_read_timeout` are genuinely route-independent by construction
   (nothing sent; a read changes nothing) and I would keep them grouped. `504 <op>_state_unknown` is
   not: its rule is per-route by design and its producer already varies by route under the new
   precedence rule. **Would it survive a handler that varies by route?** No — and the plan already
   contains one, which is the cleanest possible answer to the question the architect asked. The
   licence also needs "and one action" (P8-5 is the same defect inside C9).
4. **The three self-answered questions — all three right, two with a caveat.** (a) `{error, field}`
   over FastAPI's `loc` is correct: a body a version bump can change is not a contract, and C6b's
   precedent makes body-dispatch a pattern rather than an exception — caveat P8-N1. (b) Observational
   counter with lockout rejected is **right and well-argued**: one shared key plus an open LAN makes a
   lockout a self-DoS anyone present can trigger mid-demo, and naming K-016 as the real answer if
   exposure widens is the correct place to stop. Caveat: with lockout gone, the fixed delay is the
   only brute-force defence and neither its value nor the key's entropy is stated — cheapest fix is
   S11 generating a random key by default; this sits inside R6's accepted residual either way.
   (c) Table as source of truth with §5.2 as its prose view is right, and the **generation rule** is
   what makes it operable rather than aspirational — it is the best single addition in v1.16.

### Dispatch

Hashes at v1.16, same method (Appendix P7-A), extended with S9:

| Step | v1.14 | v1.15 | v1.16 | Call |
|---|---|---|---|---|
| S3 | `bdd89374` | `bdd89374` | `bdd89374` | committed (`673342b`) |
| S6 | `8c62ed54` | `8c62ed54` | `8c62ed54` | in flight — **independently confirmed unchanged** |
| S7 | `b1fb5e6d` | `b1fb5e6d` | `b1fb5e6d` | cleared, three versions stable |
| S8 | `cf19797a` | `aae76b9e` | `cbccecb4` | **unblocks on one clause** — P8-4 |
| S9 | `f8e278cd` | `f8e278cd` | `f8e278cd` | not previously gated; stable across three versions |
| S10 | `fcf9cdc8` | `fcf9cdc8` | `9c5898d9` | **unblocks** — moved only to decide P7-5 |
| S12a | `87a04766` | `806410ee` | `49f58b4f` | **hold** (as expected; gated on S8 regardless) |

- **S8 — yes, conditional on P8-4 and nothing else.** Its row moved *by construction* (it implements
  P7-1 and P7-3), so churn is the wrong test here; the test is whether an open finding routes to it,
  and exactly one does. Without the route classification the step's own gate is not computable and
  its symmetric half ("a row with no producer fails the step") fails on `/health` and
  `presenter/session`. That is a one-clause edit, verifiable by `diff`. Land it with the C4 fix and
  dispatch S8 without another pass.
- **S10 — yes.** The only finding holding it (P7-5) is decided, the row moved for exactly that, and
  no Pass 8 finding reaches it. It is sequenced after S9 anyway.
- **S12a — hold**, and it will absorb P8-1, P8-2, P8-3 and P8-5/P8-6. Its row is still the plan's
  moving surface, and that is now expected rather than alarming: it is the terminal consumer of every
  rule, and it is gated on S8 regardless.

### What's solid in v1.16

The read-vs-write split is the right axis and the reasoning is airtight in the direction it is
argued: *a read that times out changes nothing by definition* is the kind of sentence that makes a
whole class of rules derivable instead of remembered. Finding instance seven by **building** the
guard rather than by reading is the delta's best evidence for the guards themselves. C12's verified
block, with file-and-line citations to the pinned artifact and a reversal trigger aimed at the
default that actually bit, is the most rigorous paragraph in the plan. And the S8 row now does
something plans rarely do: it names the gap in its own guarantee, in the same breath as the
guarantee — P8-7 sharpens that sentence, it does not dispute its honesty.

### Open questions (Pass 8)

1. **Is join idempotent, or does a lost `POST /session` response cost a ghost participant?** (P8-2) A
   nonce is permitted by §5.2's invariant; the alternative is accepting the roster artifact and
   telling the presenter. Architect's call, and it is the only Pass 8 finding with a design fork.
2. **Does `GET /shop/api/health` touch the graph?** It decides the route's class (P8-4) and whether
   the storefront's liveness answers `503 graph_unavailable` like the platform's does (`api.py:63`).
