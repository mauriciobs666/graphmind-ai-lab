# salesperson — agent context

Static storefront SPA for the workflow-engine-backed `salesperson` agent. Read
`salesperson/README.md` first for what it is and how to build it; this file carries the parts an
agent working *inside* the component needs.

Authoritative specs live at repo root, not here: `docs/requirements/salesperson-ui.md` and
`docs/plans/salesperson-ui.md`. **This component deliberately has no `docs/` tree** — the plan's
§4.1 keeps the requirement/plan document family at repo-root `docs/`, but delivery entries and the
backlog go to `falkor-chat/docs/HISTORY.md`/`BACKLOG.md` (root `docs/HISTORY.md`/`BACKLOG.md` are
scoped to the CPG component only, not this feature). `salesperson/` adopts its own `docs/` tree
only if it acquires topics of its own.

> `deprecated/salesperson/` is a *different, retired* Streamlit app that used to occupy this
> path. It is not a precedent for anything here (`deprecated/README.md`). Do not copy patterns
> or dependencies out of it. The plan cites it only as a behavioural parity reference (§2.4).

## Hard constraints

1. **`vite.config.ts`'s `base: "/shop/"` is a contract with the server**, not a preference.
   `falkor-chat`'s FastAPI app mounts this bundle at `/shop`; every hashed asset URL is emitted
   under that prefix. Changing it breaks the deployment. `build.sh` asserts the built
   `index.html` still contains `/shop/` and fails the build if it does not.
2. **`dist/` is gitignored and never committed** (plan OQ-6). `./build.sh` is the only supported
   way to produce it. `falkor-chat/scripts/start_demo.sh` (S11, delivered) invokes it during
   bring-up; `./build.sh` run by hand and the server pointed at `dist/` by hand is the alternative
   for iterating on the SPA alone (`README.md`, "Running the demo").
3. **No secrets in the bundle.** Everything Vite inlines is public. The participant bearer token
   is issued by the server at join time and lives in browser storage — it is never a build-time
   value, and no `VITE_*` variable may carry a credential.
4. **Never render agent or participant output as markup.** Plan §4.2 commits to `textContent`
   only — no `dangerouslySetInnerHTML` anywhere in this tree.
5. **Node is a build-time dependency only.** Nothing here may require a Node process at runtime;
   the demo is one Python process.

## The toolchain trap on this dev box

`node` is **not** on `PATH` on the usual WSL2 dev box, but `npm` **is** — the Windows one at
`/mnt/c/Program Files/nodejs/npm`. Running that `npm` installs Windows-native binaries that a
Linux Vite build cannot load, and it fails later with an error that does not look like a
toolchain problem.

- Provision with `./scripts/install_node.sh` (per-user, no sudo, checksum-verified).
- `./build.sh` resolves the toolchain itself, in order: `$NODE_BIN_DIR` → `$NODE_PREFIX/current/bin`
  (default `~/.local/node/current/bin`) → whatever is on `PATH`. It rejects a `node` resolved
  under `/mnt/` outright.
- For an ad-hoc `npm`/`npx`, prepend the bin dir: `export PATH="$HOME/.local/node/current/bin:$PATH"`.
  *Prepend* — appending leaves the Windows shim winning.

Pinned `v24.20.0`; minimum supported major is `22` (`package.json` `engines`, and `build.sh`'s
floor). The two differ on purpose: an existing Node 22 is acceptable, we just don't install one.

## File ownership (from `docs/plans/salesperson-ui.md` §5.0)

**The plan is delivered — S16 closes it, no further steps are dispatched.** This table is kept as
a structural map of which subtree covers which feature area; the step labels below are history
(cite `falkor-chat/docs/HISTORY.md` for what shipped when), not live parallel-work coordination.

| Path | Owner |
|---|---|
| `package.json`, `package-lock.json`, `vite.config.ts`, `build.sh`, `scripts/install_node.sh`, `.gitignore`, `.node-version` | S5 (this scaffold) |
| `playwright.config.ts` | S5 scaffolded → S12b extends |
| `src/{main.tsx,index.css}` — the SPA's shared entry files | S5 scaffolded → **S12a owns thereafter; no later step edits them** |
| `src/App.tsx` | S5 scaffolded → S12a → **S12b** (v1.36, §4.11: reorders the three providers to wrap `RouterProvider` normally, one narrow structural edit — closes the review's Major); no step edits it after |
| `src/api/**`, `src/session/**` | S12a — **except two named files**: `src/session/SessionContext.tsx` and `src/api/hooks.ts` gain one narrow, additive S13 grant (`welcomeMessage` field + `useJoin()` one-liner; see `docs/plans/salesperson-ui.md` §4.12, v1.37); every other file in both subtrees stays S12a-only |
| `src/routes.tsx` | S12a (builds it) → S12c → **S12b** (v1.36, §4.11: wraps the three routes in one pathless layout route, `element: <LayoutShell/>`) → S13, S12d — each adds one narrow, additive swap; see `docs/plans/salesperson-ui.md` §5.0 |
| `src/layout/**`, `src/components/sheets/**` | S12b → one-time additive **S17** (i18n chrome sweep; `docs/plans/salesperson-ui.md` §4.13, v1.38) |
| `src/i18n/**` | S12c |
| `src/locales/**` | S12c → **S17** (new `chat`/`layout`/`cart`/`order`/`profile`/`catalog` namespaces) → **S12d** (adds its own `presenter` namespace once built) |
| `src/views/Chat*`, `src/components/message/**` | S13 (incl. its own fix-back and the welcome-turn follow-up, §4.12) → one-time additive **S17** |
| `src/views/{Cart,Order,Profile,Catalog}*` | S14 → one-time additive **S17** |
| `tests/e2e/**` | S12b |
| `public/products/**` | S14 |
| `scripts/load_demo.py` | S15 |
| `README.md`, `AGENTS.md` | S5 → S16 |

**Adding a dependency is the sanctioned exception** to `package.json`'s S5 ownership: any step may
`npm install` what it needs. The scaffold pre-installs everything the plan's §4.2 names by
name — TanStack Query v5, `i18next` + `react-i18next`, Tailwind, Vitest, Testing Library,
Playwright — so the common case needs no such edit. A **router library is deliberately not
chosen**: routing is S12a's design call (`src/routes.tsx`), so S12a picks and installs one.

## Application overview

`src/routes.tsx` (`createBrowserRouter`) wraps three routes in one pathless `LayoutShell` layout
route. `src/App.tsx` composes the session/query/i18n providers around `RouterProvider`.
`src/session/**` + `src/api/**` are the join/auth/dispatch layer (S12a); `src/layout/**` +
`src/components/sheets/**` are the mobile shell — sticky header, four sheet-triggering icons
(S12b); `src/views/{Chat,Cart,Order,Profile,Catalog,Presenter*}*` are the panels (S13/S14/S12d);
`src/i18n/**` + `src/locales/{en,pt-BR,es}.json` carry the three-locale translation bundles
(S12c, swept across the remaining chrome at S17). No `VITE_*` environment configuration —
everything the client needs is either public (per hard constraint 3) or read at runtime from
`/shop/api`.

`tests/e2e/` holds the two live Playwright specs (`mobile-shell.spec.ts`, `presenter.spec.ts`),
driven against a running server — see `README.md`, "Test" / "Running the demo". `vitest`'s
`passWithNoTests` carve-out (needed only while the suite was empty) is gone.

## Conventions

- TypeScript strict-ish flags come from `tsconfig.app.json` as Vite generated them, including
  `noUnusedLocals` / `noUnusedParameters` / `verbatimModuleSyntax` — `tsc -b` runs as part of
  `npm run build`, so an unused import breaks the build, not just the lint.
- Unit tests sit beside their subject as `src/**/*.{test,spec}.{ts,tsx}`; that glob is what
  Vitest includes, and `tests/e2e/**` is explicitly excluded so the two runners never overlap.
- Lint is `oxlint` (`.oxlintrc.json`), as Vite's scaffold set up. It is not wired into the build.
