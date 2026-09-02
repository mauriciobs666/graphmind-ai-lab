# salesperson — agent context

Static storefront SPA for the workflow-engine-backed `salesperson` agent. Read
`salesperson/README.md` first for what it is and how to build it; this file carries the parts an
agent working *inside* the component needs.

Authoritative specs live at repo root, not here: `docs/requirements/salesperson-ui.md` and
`docs/plans/salesperson-ui.md`. **This component deliberately has no `docs/` tree** — the plan's
§4.1 keeps the document family at repo-root `docs/`, and delivery entries go to root
`docs/HISTORY.md`. It adopts `salesperson/docs/` only if it acquires topics of its own.

> `deprecated/salesperson/` is a *different, retired* Streamlit app that used to occupy this
> path. It is not a precedent for anything here (`deprecated/README.md`). Do not copy patterns
> or dependencies out of it. The plan cites it only as a behavioural parity reference (§2.4).

## Hard constraints

1. **`vite.config.ts`'s `base: "/shop/"` is a contract with the server**, not a preference.
   `falkor-chat`'s FastAPI app mounts this bundle at `/shop`; every hashed asset URL is emitted
   under that prefix. Changing it breaks the deployment. `build.sh` asserts the built
   `index.html` still contains `/shop/` and fails the build if it does not.
2. **`dist/` is gitignored and never committed** (plan OQ-6). `./build.sh` is the only supported
   way to produce it. `falkor-chat/scripts/start_demo.sh` invokes it during bring-up.
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

The plan splits this tree across steps so that parallel steps never collide. Respect it.

| Path | Owner |
|---|---|
| `package.json`, `package-lock.json`, `vite.config.ts`, `build.sh`, `scripts/install_node.sh`, `.gitignore`, `.node-version` | S5 (this scaffold) |
| `playwright.config.ts` | S5 scaffolded → S12b extends |
| `src/{main.tsx,App.tsx,index.css}` — the SPA's shared entry files | S5 scaffolded → **S12a owns thereafter; no later step edits them** |
| `src/api/**`, `src/session/**`, `src/routes.tsx` | S12a |
| `src/layout/**`, `src/components/sheets/**` | S12b |
| `src/i18n/**`, `src/locales/**` | S12c |
| `src/views/Chat*`, `src/components/message/**` | S13 |
| `src/views/{Cart,Order,Profile,Catalog}*` | S14 |
| `tests/e2e/**` | S12b |
| `public/products/**` | S14 |
| `scripts/load_demo.py` | S15 |
| `README.md`, `AGENTS.md` | S5 → S16 |

**Adding a dependency is the sanctioned exception** to `package.json`'s S5 ownership: any step may
`npm install` what it needs. The scaffold pre-installs everything the plan's §4.2 names by
name — TanStack Query v5, `i18next` + `react-i18next`, Tailwind, Vitest, Testing Library,
Playwright — so the common case needs no such edit. A **router library is deliberately not
chosen**: routing is S12a's design call (`src/routes.tsx`), so S12a picks and installs one.

## What the scaffold does and does not contain

**Wired and verified working:**

- Vite 8 production build with `base: "/shop/"`, verified to emit `/shop/`-prefixed assets.
- Tailwind CSS v4 via `@tailwindcss/vite`; `src/index.css` opens with `@import "tailwindcss";`
  and Tailwind's output is confirmed present in the built stylesheet.
- Vitest 4 + jsdom + Testing Library + `@testing-library/jest-dom` matchers (`vitest.setup.ts`),
  verified by a throwaway render-and-assert probe that was then removed.
- Playwright with one `Pixel 7` mobile project; the `chromium-headless-shell` binary is installed
  under `~/.cache/ms-playwright` and verified to launch on this WSL2 box with no extra system
  libraries.
- `tsc -b` typechecking across three project references (`app`, `node`, root).

**Deliberately not here:**

- Any application code. `src/App.tsx`, `src/main.tsx`, `src/index.css` and `src/App.css` are
  **Vite's generated demo content**, kept exactly as the tooling emitted them (plus the one
  Tailwind import). S12a replaces them and lands the i18n-provider slot, the layout-shell slot
  and the Tailwind layer entry. Do not treat any of it as design intent.
- Tests. `vitest` is configured with `passWithNoTests: true` so `npm test` exits 0 on the empty
  scaffold — **remove that line once S12a lands the first real test**, so an empty suite starts
  failing again.
- `tests/e2e/` holds only a `.gitkeep`; S12b writes the specs.
- A router, and any `VITE_*` environment configuration.

## Conventions

- TypeScript strict-ish flags come from `tsconfig.app.json` as Vite generated them, including
  `noUnusedLocals` / `noUnusedParameters` / `verbatimModuleSyntax` — `tsc -b` runs as part of
  `npm run build`, so an unused import breaks the build, not just the lint.
- Unit tests sit beside their subject as `src/**/*.{test,spec}.{ts,tsx}`; that glob is what
  Vitest includes, and `tests/e2e/**` is explicitly excluded so the two runners never overlap.
- Lint is `oxlint` (`.oxlintrc.json`), as Vite's scaffold set up. It is not wired into the build.
