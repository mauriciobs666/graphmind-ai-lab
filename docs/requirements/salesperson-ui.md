# The one salesperson UI — Feature Requirements
> **Status:** Ready for design · **Owner:** `tico` · **Tracks:** — (M<n> TBD) · **Last updated:** 2026-08-29

## Intent
Give business people (not developers/sysadmins) a modern, pleasant chat interface for talking to
the salesperson agent — one that shows the same basic shopping information the original
standalone Streamlit app showed (cart summary, order state, customer profile), but built against
the new, workflow-engine-backed salesperson agent (`salesperson` `WorkflowDef`, falkor-chat M6)
rather than the old standalone backend. This becomes **the one salesperson UI** going forward —
it fully replaces the existing `salesperson/` Streamlit app, which is retired.

## Problem & current state
Two different things exist today, and neither is the right shape for a business-facing demo of
the *new* agent:

- **`salesperson/`** (standalone Streamlit app, "Pastel do Mau" pastel shop) — already shows the
  business-relevant info (cart, running total, customer name/delivery address) in Brazilian
  Portuguese customer-facing copy, but it talks to an older, separate backend (`kg_pastel` graph
  + a plain LangChain/LangGraph agent) — not falkor-chat's workflow engine, and not the same
  agent being built under M6.
- **falkor-chat's own web UI** (`web/index.html`/`web/app.js`) is the only way to reach the new,
  workflow-engine-backed `salesperson` agent today (via `@mention` in a thread) — but it is
  deliberately minimalist by design (`docs/requirements/web-api-coverage.md` FR-9: "coverage
  grows, visual weight does not"), aimed at developers/sysadmins doing hand-verification, not at
  a business audience. It also doesn't yet expose several things a shopping/ordering story
  needs (e.g., workflow-run/step progress, and note that `order-fulfillment`'s lifecycle
  advancement isn't wired to a REST route at all yet).

Nothing today lets a business audience see the *new* agent presented as a finished product. Also
worth noting for the architect: the product catalog itself (K-052, `reference` graph) carries no
image data at all today — name/category/price only — so showing product pictures (FR-11) starts
from zero, not from an existing-but-unsurfaced field.

## User stories
- As a demo driver, I want to show the new salesperson agent to business people in a modern,
  pleasant chat interface, so the audience judges a real product experience, not an engineering
  console.
- As an audience member, I want to open the UI on my own phone and have my own conversation with
  the agent — seeing my own cart, running total, and order status, the same basic information the
  original salesperson UI showed — so I can try the product myself during the demo, not just
  watch the presenter use it.
- As an audience member, I want to see a picture of a product the agent mentions, not just its
  name and price, so browsing the catalog feels like a real shopping experience.
- As the presenter, I want to clear every participant's state at once between demo runs, and each
  participant wants to be able to reset just their own conversation, so demos can restart cleanly
  without one person's mess affecting anyone else's.
- As the stakeholder, I want this new UI to be **the one salesperson UI** going forward, so there
  is a single business-facing surface to maintain and show, not two.

## Functional requirements
- **FR-1** — The UI is for controlled, live-demo settings only: a presenter plus a live audience
  the presenter has invited into the room/session (e.g., via a shared link/QR code) — never
  opened to the general public, and never given real customer data. Joining requires only
  entering a display name — no account, password, or other real login. No new
  authentication/access-control system is required; general good data-handling practice (no
  secrets/PII leakage, sane input handling) is sufficient. *(Revisit if the usage model ever
  changes to unsupervised/public access — that would reopen the currently-deferred real-auth
  question, K-016.)*
- **FR-2** — The UI presents whatever catalog the new `salesperson` agent actually has (the
  ~15-product consumer-electronics catalog, K-052) — no re-theming as a pastel/pastry shop.
- **FR-3** — Each participant chooses the customer-facing language for their own conversation
  when they start it — independent of every other participant's choice — not a single setting
  fixed for the whole demo session. The choice is fixed for that conversation once made (no
  mid-chat switching); default is English if a participant doesn't choose.
- **FR-4** — Every participant — presenter included — has their own independent conversation,
  cart, and order state, distinguished by the display name entered at join. One participant's
  messages, cart, or order must never be visible in, or affect, another participant's.
- **FR-5** — The UI supports at least ~50 participants each holding their own simultaneous,
  independent conversation in a single live session without degrading the experience.
- **FR-6** — The UI is usable on phone-sized screens, since participants join and converse from
  their own phones — not only a desktop/projector layout.
- **FR-7** — Two reset controls exist: a presenter-only control that clears every participant's
  conversation/cart/order state at once (for starting a fresh demo run), and a per-participant
  control that resets only that participant's own state, independent of everyone else's.
- **FR-8** — Cart contents and an accurate running total are visible in the UI as items are
  added, removed, or the quantity changes (parity with the old Streamlit app).
- **FR-9** — Placing an order and its lifecycle status (placed → fulfilled/shipped → delivered,
  or cancelled) are visible in the UI (parity with the old app's order-state display).
- **FR-10** — The customer's profile info (name, delivery address) is captured through the
  conversation and shown in the UI, mirroring the old app's flow (name collected upfront, address
  confirmed once there's an order) — parity with the old app's sidebar profile panel.
- **FR-11** — Each product shown in the UI displays a picture. A generic/stock photo (not
  necessarily the exact real product) is acceptable. If no picture is available for a given
  product, it falls back to the existing text-only presentation — no placeholder image.

## Out of scope
- falkor-chat's own minimalist developer/sysadmin web UI (`web/index.html`) — untouched, FR-9
  (`web-api-coverage.md`) stands as-is; this is a separate, additional surface, not a change to
  that page.
- Real login/access control and any other M2.5-hardening-track capability (K-016/K-017/K-018) —
  not needed for the controlled-live-demo usage model this document scopes (FR-1); would need its
  own requirements pass if the usage model changes.
- Re-theming the new agent's catalog as a pastel/pastry shop — the electronics catalog stands.
- Live, in-conversation (mid-chat) language switching — a participant's choice (FR-3) is fixed
  once their conversation starts, not changeable partway through it.
- The old app's session/diagnostics support tooling (`diagnostics.py`'s snapshot helper) — not
  requested; not carried over unless raised later.

## Acceptance criteria
- **AC-1** (FR-1) — Given a live demo session's shared link, when a new participant opens it and
  enters a display name, then they get their own session with no password/account/login step.
- **AC-2** (FR-4) — Given two participants in the same live session, when one adds items to their
  cart or exchanges messages with the agent, then the other participant's cart and conversation
  show no trace of it.
- **AC-3** (FR-5) — Given a single live session, when at least ~50 participants join and each
  hold an independent conversation, then the UI keeps responding without noticeable degradation
  for any participant.
- **AC-4** (FR-6) — Given a common phone-sized screen, when a participant uses the UI, then chat,
  cart, and order status are all usable without horizontal scrolling or unreadably small text.
- **AC-5** (FR-7) — Given a live session with several participants' state, when the presenter
  uses the "reset everyone" control, then every participant's conversation/cart/order state
  clears; when a single participant uses their own reset, only their state clears.
- **AC-6** (FR-8) — Given a participant's cart, when they add/remove an item or change its
  quantity, then the displayed cart and running total update correctly.
- **AC-7** (FR-9) — Given a participant has placed an order, when its lifecycle status changes
  (fulfilled/shipped/delivered, or cancelled), then the UI reflects the current status.
- **AC-8** (FR-10) — Given a participant hasn't yet provided delivery info, when they place an
  order, then the UI's conversation prompts for and then displays their name/delivery address,
  mirroring the old app's upfront-name / confirm-address-once-ordering flow.
- **AC-9** (FR-2, FR-3) — Given the UI is running, when it launches, then it shows the agent's
  actual electronics catalog; when a participant starts their own conversation, then the
  customer-facing copy appears in the language they chose (English if they didn't choose one),
  independent of any other participant's choice.
- **AC-10** (readiness gate) — This UI is not put in front of a live audience while K-056 (the
  agent skipping tool calls and fabricating catalog facts) remains open — build/test work may
  proceed, but the first real demo is gated on K-056 resolving first.
- **AC-11** (FR-11) — Given a product with an available picture, when it's shown in the UI, then
  its picture is displayed alongside its text info; given a product with no picture available,
  then it's shown text-only, with no placeholder image in its place.

## Open questions
None outstanding — pending stakeholder confirmation at readback.

## Decision log
2026-08-29 — Opening ask: falkor-chat's minimalist web UI (FR-9) is a deliberate,
stakeholder-approved decision for developers/sysadmins doing hand-verification — not the
audience for this request. Not being reopened; this is a separate, additional surface.
2026-08-29 — Reframed: the business-facing audience is served by the *old* standalone
`salesperson/` Streamlit app today, but that app talks to the old (`kg_pastel`) backend, not the
new workflow-engine-backed `salesperson` agent (M6). The need is a new UI for the new agent,
carrying over the same basic info the old UI showed (cart summary, order state).
2026-08-29 — Decided: full replacement, not a second surface — the new UI becomes **the one
salesperson UI**; the old `salesperson/` Streamlit app is retired once it ships.
2026-08-29 — Stakeholder preference (not a requirement): build it "from scratch with the best
available frameworks and security" — noted as a stated preference/context for the architect.
2026-08-29 — Security need pinned down: demo-only, always presenter-driven, no unsupervised
access, no real customer data — so no new authentication is required (FR-1); this does **not**
reopen falkor-chat's deferred real-auth track (K-016). Revisit if the usage model ever changes.
2026-08-29 — Domain: retiring the pastel-shop theme entirely is fine — the new UI presents the
new agent's actual (electronics) catalog, no re-theming effort (FR-2).
2026-08-29 — Language: not fixed to Portuguese or English at build time — configurable at setup
before a given demo, not a live switcher, and a real built/tested capability rather than a
manual hand-edit (FR-3). Default is English.
2026-08-29 — Parity confirmed as must-haves for launch: cart + running total, order placement +
status, customer profile capture, and a reset-for-demos control (FR-8, FR-9, FR-10, FR-7)
— plus a mobile requirement raised alongside them (see below).
2026-08-29 — Mobile need pinned down: driven by audience members following along on their own
phones during a live demo, and (see next entry) each holding their own independent conversation,
not a mirrored view of the presenter's (FR-6). Stakeholder suggested a specific interaction
pattern (minimal icons with pop-ups for detailed info) — noted as a preference for the
architect/frontend-engineer, not a mandated design.
2026-08-29 — **Scope correction to FR-1**: audience members don't just watch — each drives their
own independent conversation from their own phone, simultaneously, joining by entering a name
(no real login). This is genuine concurrent multi-participant use, not a single presenter-driven
session as first scoped — FR-1 rewritten in place to reflect a controlled live-audience session
rather than "always presenter-driven"; added FR-4 (per-participant state isolation) and FR-5
(concurrency target). Security conclusion is unchanged: name-only join still isn't a real login,
so K-016 stays out of scope.
2026-08-29 — Concurrency scale: up to ~50 simultaneous participants in one live session (FR-5).
2026-08-29 — Reset scope: both a presenter-level "reset everyone" and a per-participant "reset
just mine" are needed (FR-7), now that state is per-participant rather than singular.
2026-08-29 — Timing: build/test work may proceed in parallel with K-056, but going live with an
audience is gated on K-056 (the agent skipping tool calls and fabricating catalog facts)
resolving first (AC-10) — the first real demo must not be undermined by the agent making things
up.
2026-08-29 — **Correction, raised at readback**: language is not a single choice made at setup
for the whole demo session after all — each participant picks the language for their own
conversation when they start it, independent of everyone else's choice, fixed for that
conversation once made (no mid-chat switching). FR-3, the corresponding out-of-scope line, and
AC-9 rewritten in place to match; the earlier setup-time framing above is superseded by this
entry.
2026-08-29 — New ask at readback: show product pictures (FR-11). Checked the catalog data
directly — no image field exists today (K-052's `Product` nodes carry only name/category/price),
so this isn't wiring up an unsurfaced field, it starts from nothing. Resolved: generic/stock
photos are acceptable, not necessarily the real product; a product with no picture available
falls back to text-only, no placeholder image.
2026-08-29 — Stakeholder confirmed the full readback (including the language and product-picture
additions) with no further changes; flipped to Ready for design.
