# Review: salesperson-ui welcome-turn follow-up — implementation

> **Status:** active · **Owner:** `analyst` · **Tracks:** S13 welcome-turn follow-up (`docs/plans/salesperson-ui.md` §4.12, v1.37)

## Scope & verdict

Reviewed the already-committed diff `7ab7a97` ("feat(salesperson): welcome turn — render join
response's greeting once") against its design spec: `docs/plans/salesperson-ui.md` §4.12 (v1.37,
the grant), §5.2 *The join greeting*, and §5.3's credentials table (`welcome` excluded from
`ParticipantSession`). This is a **post-commit** gate — `teco` committed after its own
verification (diff read, `tsc -b`, `vitest run`, one independent mutation) before dispatching this
review, a process slip relative to this coordination's default pre-commit `analyst` gate, per the
brief. Ten files, 310/34 lines. Read every touched file in full (`SessionContext.tsx`,
`hooks.ts`, `ChatView.tsx`, `welcome.ts`, `MessageBubble.tsx`, `Transcript.tsx`, and all four test
files' diffs); traced the server side of the contract (`storefront_api.py`'s `_welcome()`,
`services.py`'s `msg_id`/`createdAt` minting) to check the synthetic row can't collide with a real
one; ran `npx tsc -b` (clean) and `npx vitest run` (190/190, 22 files — matches the commit
message) myself; performed one further independent mutation beyond the implementer's 3 and
`teco`'s 1 (below).

**Verdict: approve.**

CPG: considered, not relevant — `cpg_salesperson` does not exist in this FalkorDB instance;
grounded by direct source reading instead, same posture as this coordination's S13/S17 reviews.

## Findings

None at blocker, major, or minor severity. One nit.

**Nit — the "renders once" done-condition test doesn't independently pressure the `useMemo` deps
list; only the "does not reappear" test does.** I mutated `SessionContext.tsx`'s `useMemo` deps
array to drop `welcomeMessage` (a mutation neither the implementer's 3-item table nor `teco`'s own
check covers) and reran the suite: `ChatView.test.tsx`'s "does not reappear on a second ChatView
mount within the same join" reddened correctly (the stale memo kept showing "Welcome to the store,
Ada." after the toggle), but "a fresh join's welcome line renders once" stayed green. The reason is
structural, not a real gap: `setParticipant()` and `setWelcomeMessage()` both fire inside the same
`onSuccess` batch, and `participant` is (correctly) still in the deps array, so the memo recomputes
anyway on the very first join regardless of whether `welcomeMessage` is listed. Net effect: the
deps-list mutation is caught, just by the second test rather than the first. No action needed —
noted so a future edit that separates those two `onSuccess` calls in time doesn't quietly lose this
coverage without anyone noticing which test was doing the work. Mutation restored byte-identical
(`diff -q` against the pre-mutation backup; `git status --short` on both touched files: clean).

## What's solid

- **Design conformance is exact.** §4.12's three load-bearing decisions — mirror
  `pendingLanguageStep`'s shape precisely, set once in `useJoin()`'s `onSuccess`, clear once in
  `ChatView.tsx` via a capture-then-clear pattern — are all implemented exactly as specified, down
  to the field ordering in `SessionContextValue` and the `useMemo` deps list. No edit strayed
  outside the two named grant files (`SessionContext.tsx`, `hooks.ts`) plus S13's own unrestricted
  `views/Chat*`/`components/message/**` subtree (§5.0's shared-file map, confirmed by direct
  reading — `routes.tsx`, `session/types.ts`, `session/storage.ts`, `api/endpoints.ts` are all
  untouched, matching §4.12's explicit out-of-scope list).
- **The three done-conditions are genuinely proven, not just exercised.** `ChatView.test.tsx`'s
  three new tests drive a real `useJoin()` mutation through a purpose-built harness
  (`JoinThenChat`/`JoinThenToggleChat`) rather than injecting `welcomeMessage` directly — so "a
  fresh join renders once," "a reload-shaped mount (persisted `ParticipantSession`, no `useJoin()`
  call) starts `null`," and "does not reappear on a second mount within the same join" are each
  checked against the actual wiring, including a genuine component unmount/remount for the third.
  This is a materially stronger test than a shallow prop-injection test would have been.
- **No collision or leak risk from the synthetic row, verified against the server, not assumed.**
  `msgId: '__welcome__'` can never collide with a server-assigned id (`services.py`'s
  `_default_id()` is `uuid.uuid4().hex`, never a literal); `createdAt: 0` can never collide with a
  real timestamp (`_default_clock()` is wall-clock ms since epoch); the greeting is never persisted
  as a `Message` node server-side (`storefront_api.py`'s `_welcome()` is a pure string computation
  returned only in the join response) — so there is no path for it to reappear via the polled
  transcript. `mentions: []` and `threadId: ''` are inert: grepped every `src/components`/`src/views`
  consumer and neither field is read anywhere outside test fixtures. No sort-by-`createdAt` logic
  exists anywhere in the tree (grepped) that the `createdAt: 0` placeholder could disturb; row order
  is append/prepend order only.
- **`withWelcomeRow`/`withOptimisticRow` call order in `ChatView.tsx` doesn't matter, and the code
  doesn't lean on it mattering.** `withWelcomeRow` unconditionally prepends regardless of what it's
  given; `withOptimisticRow`'s own dedup is keyed on the *real* server `msgId`, which is disjoint
  from `WELCOME_ROW_ID` by construction (previous bullet). Composing either order produces the same
  result.
- **`showTimestamp`'s default (`true`) preserves every existing `MessageBubble` caller exactly.**
  Traced both call sites (`Transcript.tsx`'s ordinary-row map and its `pendingText` bubble) — the
  render condition `(pending || showTimestamp)` reduces to the old unconditional render in both
  pre-existing cases (`pending=false, default true` → `true`; `pending=true` → `true` regardless).
  Confirmed by running the pre-existing `MessageBubble.test.tsx` cases (markup-escaping, alignment,
  "Sending…" caption), none of which pass `showTimestamp` explicitly, and all three still assert
  what they asserted before.
- **The `eslint-disable-next-line react-hooks/exhaustive-deps` on the mount-only clearing effect is
  safe, not a paper-over.** `setWelcomeMessage` is a raw `useState` setter (not wrapped in
  `useCallback`, unlike `setParticipant`/`clearParticipant`/etc.) — React guarantees a `useState`
  setter's identity is stable for the lifetime of the component, so omitting it from the deps array
  introduces no staleness; the suppression is the same shape as any deliberate "run once on mount"
  effect. No mutation of `setWelcomeMessage`'s identity is possible to construct that would expose
  this as a bug.
- **Mutation rigor exceeds the norm for a follow-up this size.** Implementer's 3 + `teco`'s 1 +
  this review's 1 (the `useMemo` deps drop, above) span four independent files/mechanisms
  (`ChatView.tsx`'s clearing call, `ChatView.tsx`'s capture site, `welcome.ts`'s prepend order,
  `Transcript.tsx`'s `showTimestamp` wiring, `SessionContext.tsx`'s deps list) with zero survivors.

## Open questions

None. This unit is small, narrowly scoped, and fully conforms to its grant; no follow-up decision
is needed before `teco` moves to the next queued unit.
