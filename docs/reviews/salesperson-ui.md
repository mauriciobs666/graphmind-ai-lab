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
