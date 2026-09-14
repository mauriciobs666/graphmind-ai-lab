# Review: salesperson-ui S14 (Cart / Order / Profile / Catalog panels)

> **Status:** active · **Owner:** `analyst` · **Tracks:** S14 (`docs/plans/salesperson-ui.md` §5.1)

## Scope & verdict

Reviewed the delivered S14 unit — `salesperson/src/views/{Cart,Order,Profile,Catalog}Panel.tsx`,
their four `*.test.tsx` files, `salesperson/public/products/*.jpg`, and the `salesperson/README.md`
"Product images" section — against `docs/plans/salesperson-ui.md` §5.1's S14 row, §2.4's parity
table, §4.6 (order lifecycle / "demo controls"), §4.7 (image manifest), OQ-6 (image licensing), and
`salesperson/AGENTS.md`'s file-ownership table. Baseline for "in scope" is the working tree as of
this review (uncommitted); out of scope: S13's/S12d's own concurrent `routes.tsx`/`views/Chat*`
work, and the pre-existing `Shell.test.tsx`/`ResetControl.test.tsx` failures (S12b's ownership,
already dispatched elsewhere as a separate fix).

Independently re-ran `npx vitest run` (165 passed / 9 failed, exact same 9 test names — both
`Shell.test.tsx` and `ResetControl.test.tsx`, zero S14 files) and `npx tsc -b` (clean); read
`git diff -- salesperson/src/routes.tsx` directly and confirmed it is entirely S13's
`ChatScreen`→`ChatView` swap with zero touch to any Cart/Order/Profile/Catalog code; read the
`fulfill_order`/`deliver_order`/`cancel_order` CAS guards in
`falkor-chat/server/falkorchat/repository.py:3736-3779` to verify the client's gating logic against
the actual state machine (`placed→fulfilled→delivered`, `placed→cancelled` only); verified the
README's Unsplash-License claim for Lorem Picsum against a web search; and confirmed all 14
delivered `.jpg` files are genuine, distinctly-sized 640×480 JPEGs (`file`/`ls -la`), matching the
15-product catalog literal in `falkor-chat/scripts/seed_catalog.sh` with `smart-home-hub` the one
deliberate omission.

**Verdict: approve with suggestions.**

CPG: considered, not relevant — `salesperson/` has no loaded CPG (confirmed by the implementer and
independently by `teco`; not worth building one at this component's size), and this review's
grounding was done by direct file/diff reading instead.

## Findings

### Minor — `isOrderLine` guard's filtering behaviour is untested (confirmed independently)

`OrderPanel.tsx:37-47`'s type guard silently drops any `lines` entry that doesn't match the
expected shape. `OrderPanel.test.tsx`'s only fixture builder, `orderWith()` (lines 23-32), always
emits a fully-typed line — no test ever exercises the guard's negative branch. I confirmed this by
reading the file, matching `teco`'s own mutation-tested finding (gutting the guard to
`return true` left all 9 `OrderPanel.test.tsx` tests green).

Severity is genuinely low, for a reason worth recording rather than just asserting: I traced the
guard's actual input contract at `falkor-chat/server/falkorchat/repository.py:4067-4071` (`_CURRENT_ORDER_CYPHER`)
and `:4459` — the repository already filters `lines` down to rows with a non-null `productId`
before the API ever serialises them, and every field in the `collect({...})` projection comes from
one `OrderLine` node's own properties (never a partial/optional read). So there is no reachable
server response today that would exercise the guard's reject path — it defends against a future
shape drift, not a live one. **Suggested improvement:** add one `OrderPanel.test.tsx` case using a
`lines` array with one well-formed entry and one malformed entry (e.g. missing `unitPrice`,
matching the "malformed line" shape the docstring already describes) and assert the malformed one
is absent from the rendered list while `order.total` still renders from the fixture's own total —
this is a same-file, low-cost addition, not a design change.

### Minor — order-status axis coverage stops one state short of the state machine

The plan's status enum (`falkor-chat/server/falkorchat/repository.py:3736-3779`) has four terminal
shapes: `placed`, `fulfilled`, `delivered`, `cancelled`. `OrderPanel.test.tsx` drives `placed`,
`fulfilled` and `delivered` explicitly (each asserting the right `canCancel`/`canFulfill`/
`canDeliver` combination) but never `cancelled`. The button-gating logic (`order.status === 'placed'`
etc., `OrderPanel.tsx:141-143`) happens to produce the same disabled/hidden shape for `cancelled` as
for `delivered` (both: no Cancel button, both simulation buttons disabled) so this is not a
suspected live bug — but it is an unexercised cell on the axis the panel actually varies over.
**Suggested improvement:** add a fourth status case (`orderWith('cancelled')`) asserting the status
chip reads "Cancelled" and both demo-control buttons are disabled with Cancel hidden — mirrors the
existing `delivered` test almost verbatim, so the cost is a few lines, not a new fixture shape.

### Nit — README's "fetched at build time" wording could be misread as a live build dependency

`salesperson/README.md:166-167` (the "Source and licence" paragraph) describes each image as
"fetched at build time from `https://picsum.photos/seed/<productId>/640/480.jpg` … and committed
verbatim." Read in isolation, "fetched at build time" suggests `./build.sh` reaches the network on
every build — it does not (`grep -n "picsum\|curl\|wget\|fetch" salesperson/build.sh` returns
nothing; the images are static, already-committed assets under `public/products/`, which Vite just
copies). This is consistent with `salesperson/AGENTS.md`'s "Node is a build-time dependency only"
constraint and does not contradict it, but a reader skimming just this sentence could draw the
wrong conclusion about `build.sh`'s network posture. **Suggested improvement:** reword to something
like "fetched once, during this step's authoring, from `https://picsum.photos/seed/<productId>/…`
… and committed verbatim; `build.sh` performs no network access" — one clause closes the ambiguity.

## What's solid

- **Status-gating logic matches the plan exactly, not just approximately.** I traced the actual
  server-side CAS guards (`fulfill_order`: `placed→fulfilled` only; `deliver_order`:
  `fulfilled→delivered` only; `cancel_order`: `placed→cancelled` only, explicitly enforcing "cannot
  cancel once fulfilled") against `OrderPanel.tsx:141-143`'s `canCancel`/`canFulfill`/`canDeliver`
  and found them identical, cell for cell — no daylight between the client's gating and the
  server's guarded transitions.
- **§4.6's "demo controls" framing is genuinely structural, not cosmetic.** `fulfill`/`deliver`
  live inside a visually distinct, dashed-border, explicitly-labelled box
  (`data-testid="order-demo-controls"`) that `cancel` sits outside of — and the test suite asserts
  the DOM containment relationship (`demoControls).not.toContainElement(cancelButton)`), not just
  that both render somewhere on the page.
- **AC-11's no-placeholder rule is honoured structurally.** `CatalogPanel.tsx:47-61` renders an
  `<img>` only when `imageUrl` is truthy, with a genuinely separate text-only branch — no `onError`
  swap anywhere in the file (the plan explicitly forbids this pattern since it flashes a broken
  image first). `CatalogPanel.test.tsx` asserts zero `<img>` elements in the DOM for the no-image
  case, not just that the fallback text is present — the stronger of the two possible assertions.
- **The four test files are substantive, not smoke tests.** Every one drives real response
  fixtures through `useShopState()`/`useCatalog()` and asserts rendered text/attributes/DOM
  structure (line quantities, running totals, em-dash fallback per-field, image `src`, button
  enabled/disabled state, POST body content for both `cancel` and `fulfill`) — I read all four in
  full and found no test that merely checks "the component rendered without crashing."
- **The `smart-home-hub` deliberate omission is a good call, documented as such in both the
  component and the README** — it means AC-11's negative branch is exercised by the real running
  app, not only by a mock, and the README explains exactly how to add the asset back later.
- **File-ownership discipline held.** `git status`/`git diff` confirm S14 touched only its owned
  subtree; the one file outside it that shows as modified (`routes.tsx`) is entirely S13's
  concurrent, disjoint swap, verified by reading the diff rather than taking the characterization
  on faith.
- **The Unsplash-License claim in the README checks out** — Lorem Picsum's own images are sourced
  from Unsplash and distributed under the Unsplash License (free commercial/non-commercial use, no
  attribution required), confirmed via web search against Picsum's own documentation.

## Open questions

None — the two Minor findings and the one Nit are self-contained, low-cost additions the owning
agent (`frontend-engineer`) can pick up without further input; nothing here blocks acceptance.
