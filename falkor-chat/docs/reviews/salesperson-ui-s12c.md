# S12c — i18n (`salesperson/src/i18n/**`, `src/locales/**`)

> **Status:** active · **Owner:** `analyst` · **Tracks:** S12c (salesperson-ui)

## Scope & verdict

Reviewed: `salesperson/src/i18n/{config,Provider,useLocale,LanguageChooser,format}.tsx?`,
`salesperson/src/locales/{en,pt-BR,es}.json`, and their tests
(`LanguageChooser.test.tsx`, `format.test.ts`, `locales.test.ts`) — S12c's owned subtree
(`docs/plans/salesperson-ui.md` §5.0/§5.1's S12c row) — plus the two out-of-subtree edits
(`salesperson/src/routes.tsx`, `salesperson/tsconfig.app.json`), diffed against
`bbd9eb7` (the S12a commit) rather than taken on report. Read `docs/plans/salesperson-ui.md`
§4.5, §5.0, §5.1's S12c row, §5.2 (`/health`'s `locales`, `/session`'s `language`, the route-class
table), §5.3's C11 (the config-drift argument S12c's row is built to close), and
`falkor-chat/docs/reviews/salesperson-ui-s12a.md` for house style and the routes.tsx-ownership
history this unit inherits. `salesperson/AGENTS.md` read for file ownership.

Not re-checked (per brief, teco's pass stands unless a specific reason surfaced to doubt it):
`routes.tsx`/`tsconfig.app.json` diff scope, the full-suite/`tsc`/`build` green claims, the
`join.languageHint` deletion mutation, and the `/health`/`storefront_api.py` `STOREFRONT_LOCALES`
read. Independently run instead: `npx vitest run src/i18n` (9/9 green) as a baseline, then three
of my own reverted mutations (below) against the live source, each restored to a byte-identical
`diff`-clean state and the suite re-confirmed green.

**Verdict: approve with suggestions.** No blocker. One Major is a plan-conformance gap
(`routes.tsx` ownership) that v1.34's own fix for the identical problem, one file-row up, missed
for S12c — the code change itself is safe and exactly the sanctioned pattern, but the plan text
is now factually wrong about who may touch this file, and the next writer to touch it (S13/S12d's
serialized swaps) needs to know a third edit already landed in between. The rest are Minor/Nit
quality points: a test-suite gap that lets the `<label>`/`<select>` association be deleted
undetected, a live-preview sync edge case, and a small DRY issue in `config.ts`.

**CPG:** considered, not relevant — `salesperson/` has no loaded CPG (`cpg_salesperson` does not
exist among this FalkorDB instance's graphs), and the reviewed surface is six small,
freshly-authored files plus three two-key JSON bundles, well within direct-reading range.

## Findings

### Major — the plan's `routes.tsx` ownership map (v1.34) still doesn't authorize S12c's edit, and now needs a third entry

`docs/plans/salesperson-ui.md`'s `routes.tsx` row (line 1063) reads: "S12a (builds it), then
**S13**, **S12d**" — v1.34's whole point was closing `falkor-chat/docs/reviews/salesperson-ui-s12a.md`'s
Major finding (S12d/S13/S14's views had no authorized way to reach `routes.tsx`) by granting S13
and S12d one narrow additive swap each. But S12c's own row (line 1104) commits to exactly the
same shape of thing — "the join-screen language chooser feeding `POST /shop/api/session`" — and
the join screen lives in `routes.tsx`, not in S12c's subtree. The delivered diff
(`git diff bbd9eb7 -- salesperson/src/routes.tsx`) is precisely the same pattern already
sanctioned for S13/S12d — one import added, one inline `<label>/<select>` block swapped for
`<LanguageChooser .../>`, nothing else touched — but no row anywhere authorizes *this* step to
make it. v1.34 fixed the identical problem for two of the three rows that needed it and missed
the third, which is exactly the "unabsorbed" pattern this plan's own version history calls out
repeatedly (e.g. v1.8, v1.18) when a sweep reaches some rows and not others.

This isn't just paperwork: the ordering note on line 1063 ("no step touches this file after
S13's and S12d's swaps land," and `teco` "serializes only that one swap apiece" between S13/S12d)
was written assuming exactly two more edits land on top of S12a's original file. A third,
undocumented edit (S12c's) has now landed *before* that pair, so the file S13/S12d will each
receive at their own dispatch is no longer the one the plan's ordering note describes — S13's and
S12d's diffs need to be taken against the delivered (S12c-inclusive) `routes.tsx`, not against
`bbd9eb7`. That's a small but real correction someone needs to make before dispatching S13/S12d,
or a future reviewer re-checking "did S13/S12d touch only their placeholder" will diff against
the wrong baseline.

**Fix:** `architect` adds a v1.35 line-1063 amendment mirroring v1.34's own mechanism: `routes.tsx`'s
row gains S12c between "builds it" and "S13, S12d," stating that S12c's swap already landed and is
included in the file S13/S12d receive. No code change needed — the delivered edit is exactly the
right shape and needs no rework, only the plan text needs to catch up to what was actually (and
correctly) done.

### Minor — the test suite proves the picker's behavior but not its accessible-name wiring; a mutation that deletes the `<label>` entirely still passes all three tests

`LanguageChooser.tsx` associates its `<select>` with `{t('join.languageLabel')}` via label
wrapping (no explicit `htmlFor`/`id` — the sole caller in `routes.tsx:87` passes no `id`, so that
prop is currently dead). I mutated the component to replace the wrapping `<label>` with a plain
`<div>` and an unassociated `<span>` (verified: `npx vitest run src/i18n/LanguageChooser.test.tsx`
→ **3/3 still pass**), then restored it (`diff` clean against the pre-mutation copy, suite green
again). All three tests query by `getByRole('combobox')` or by the visible text nodes directly,
neither of which requires the label/control association a screen-reader user depends on — so a
future edit that silently drops the `<label>` (e.g. during a styling pass) ships undetected.

**Fix:** add `screen.getByLabelText(/language/i)` (or the bundle's exact `t()` string) as an
additional assertion in the first test — Testing Library's `getByLabelText` only resolves through
a genuine wrapping-or-`htmlFor` association, so it fails exactly when the association breaks.
Separately, the hint span (`join.languageHint`) sits inside the `<label>`, which folds its full
sentence into the select's computed accessible name on every focus; consider `aria-describedby`
pointing at the hint's own `id` instead, so the accessible name stays "Language" and the hint is
exposed as a description, not part of the name.

### Minor — `LanguageChooser`'s live preview isn't synced on initial mount, only on selection

`useLocale().setLocale()` fires from `handleChange` (`LanguageChooser.tsx:48`) only, never on
mount. `routes.tsx:34`'s `useState(pendingLanguageStep ?? locales[0] ?? 'en')` seeds the select's
initial value from whichever `locales` array is present at first render. In the common case
(fresh page load, `useHealth()` not yet resolved) that's the `['en']` fallback, matching
`config.ts`'s `DEFAULT_LOCALE`, so there's no visible gap. But on a re-mount with an
already-cached `/health` response (e.g. returning to the language step after `reset-mine`, §4.8's
C7) *and* a deployment whose `FALKORCHAT_STOREFRONT_LOCALES` puts a non-English locale first, the
select can pre-select e.g. `pt-BR` while the surrounding chrome (heading, hint, button) is still
rendering in whatever locale was last active — until the participant touches the control. This
never affects what's actually POSTed (the `language` state and the select stay bound correctly),
only the chrome/selection visual consistency the live-preview mechanism is meant to guarantee.
Low likelihood given the shipped default order is English-first, but it's a real seam gap between
S12a's state initialization and S12c's live-preview contract.

**Fix:** in `routes.tsx`'s `JoinScreen` (or inside `LanguageChooser` via a mount-time
`useEffect`), call `setLocale(language)` once on mount so the chrome is synced to whatever value
is actually selected, not only to changes made after mount.

### Minor — `config.ts` maintains the locale→bundle mapping twice, with nothing keeping them in lock-step

`localeResources` (a `Record<SupportedLocale, unknown>`) and the inline `resources` object passed
to `i18n.init()` both list `{ en, 'pt-BR': ptBR, es }`, hand-duplicated rather than one derived
from the other. `SUPPORTED_LOCALES`, `SupportedLocale` and `localeResources` are also all
currently unused outside this file (`grep -rn "localeResources\|SUPPORTED_LOCALES" src` outside
`i18n/config.ts` returns nothing) — `LanguageChooser`'s `locales` prop comes from the server's
`/health` response, not from this constant. A future locale addition that updates one structure
and forgets the other (nothing tests that they match) would silently register a locale i18next
never uses, or vice versa.

**Fix:** derive `resources` from `localeResources` (`Object.fromEntries(Object.entries(localeResources).map(([k, v]) => [k, { translation: v }]))`)
so there is one map to maintain, and either give `SUPPORTED_LOCALES`/`SupportedLocale` a real
consumer (e.g. typing `LanguageChooserProps.locales` more tightly where it's safe to) or drop them
if the server-seeded design genuinely makes them unnecessary.

### Nit — `LanguageChooserProps.id` is declared but never passed by its only caller

`routes.tsx:87` calls `<LanguageChooser value={language} onChange={setLanguage} locales={locales} />`
with no `id`, so `htmlFor={id}`/`id={id}` both resolve to `undefined` — harmless today because the
wrapping `<label>` still associates implicitly, but the prop is dead weight. Either wire an `id`
from the caller (useful once `views/**` needs multiple instances on one screen) or drop the prop
until something needs it.

## What's solid

- **`config.ts`'s `i18n.isInitialized` guard is sound**, not a bug-hider: it protects against a
  real failure mode (a second `.init()` call resetting `lng` back to `DEFAULT_LOCALE`, e.g. under
  HMR or a duplicate module instance), and ordinary same-registry re-imports never re-run the
  guarded block at all (standard ES module caching), so the guard is pure defense-in-depth.
- **The `fallbackLng` strategy is correctly reasoned and documented in place** — a
  deployment-widened locale with no bundle renders English chrome rather than throwing, mirroring
  the server's own `welcome`-line fallback (§5.2), and `LanguageChooser`'s `NATIVE_NAMES[locale]
  ?? locale` fallback for the same scenario is genuinely tested (`'de'` renders as `'de'`).
- **`locales.test.ts`'s key-coverage guard is well-built**, not merely present: it has its own
  non-vacuity check (every bundle has ≥1 key) and I independently confirmed teco's mutation
  (deleting a key from one bundle) reds it with an exact `{locale: [key]}` diagnostic.
- **`format.ts`/`format.test.ts` exercise real `Intl` behavior**, not the test's own fixture — I
  mutated `formatCurrency` to hardcode `'en'` regardless of the `locale` argument and the
  decimal-separator test caught it immediately (`$1,234.50` vs. the expected comma-decimal
  pattern), then restored clean.
- **`LanguageChooser`'s live-preview mechanism itself is correctly wired** — I mutated
  `handleChange` to call `setLocale(value)` (the stale prop) instead of `setLocale(next)` (the
  newly picked one) — precisely the bug the "live preview" contract exists to prevent — and the
  suite's own third test caught it (`findByText('Idioma')` never resolved), then restored clean.
- **Scope discipline is otherwise exact**: the `routes.tsx` and `tsconfig.app.json` diffs match
  the brief's description to the character (verified via `git diff bbd9eb7`), and nothing in
  S12c's own subtree reaches into a file another step owns.

## Open questions

- Should `architect` land the v1.35 `routes.tsx`-row amendment (this review's Major) before or
  after S13/S12d are dispatched? Either order is safe for S12c's own delivery (already landed,
  correct), but S13/S12d's implementers need the corrected baseline note before their swaps are
  reviewed for "touched only their placeholder."
