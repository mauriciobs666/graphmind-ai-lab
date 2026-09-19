# salesperson

Business-facing, mobile-first **storefront UI** for the workflow-engine-backed `salesperson`
agent hosted by `falkor-chat/`. Around 50 audience members each join with nothing but a display
name and then hold their own independent, isolated conversation — own cart, own order, own
profile — in the language each of them picks.

It is a **pure static bundle**. There is no Node server at runtime: `falkor-chat`'s single
FastAPI process serves the built assets at `/shop`, so the demo stays one process, one port, no
CORS. Node is a **build-time dependency only**.

Delivered per `docs/requirements/salesperson-ui.md` (FR-1…FR-11, AC-1…AC-11) and
`docs/plans/salesperson-ui.md` (the stack rationale is §4.2); delivery record, the S15 QA pass and
defect-fix wave, and the one still-open pre-live-demo gate (`K-065`, a language swap under
LLM-serving-layer concurrency) are in `falkor-chat/docs/HISTORY.md`/`BACKLOG.md`, not here.

> This directory previously held a retired Streamlit chatbot. That app now lives at
> `deprecated/salesperson/` and is unrelated to the code here — see `deprecated/README.md`.

## Stack

| Concern | Choice |
|---|---|
| Build / dev server | Vite 8 |
| UI | React 19 + TypeScript 6 |
| Routing | `react-router-dom` v7 (`createBrowserRouter`, HTML5 history) |
| Styling | Tailwind CSS v4 (via `@tailwindcss/vite`) |
| Server state / polling | TanStack Query v5 |
| Localization | `i18next` + `react-i18next` — English (default), Brazilian Portuguese, Spanish |
| Unit / component tests | Vitest 4 + Testing Library + jsdom |
| End-to-end tests | Playwright (one mobile-viewport project) |

## Prerequisites — the Node toolchain

**Pinned version: Node `v24.20.0` (LTS "Krypton"), npm `11.19.0`.** The pin lives in
`.node-version` (and `.nvmrc`); `package.json`'s `engines` records the *minimum* supported
runtime, Node `>= 22.12.0`.

Install it — per-user, no sudo, checksum-verified against `nodejs.org`'s `SHASUMS256.txt`:

```bash
./scripts/install_node.sh
```

That unpacks the official Linux tarball to `~/.local/node/node-v24.20.0-linux-x64` and points
`~/.local/node/current` at it. `./build.sh` finds it there automatically — **you do not need it
on your `PATH`** to build. To use `node`/`npm` interactively:

```bash
export PATH="$HOME/.local/node/current/bin:$PATH"
```

### Why a tarball rather than apt or nvm

On this dev box (WSL2) there is no passwordless sudo, so a system package install is out. More
importantly, **`node` is absent from `PATH` while `npm` is present** — that `npm` is the
*Windows* one under `/mnt/c/Program Files/nodejs`, inherited from the Windows `PATH`. Using it
installs Windows-native binaries (`esbuild`, `rollup`, `lightningcss`) that a Linux Vite build
cannot load, and the failure surfaces deep inside the bundler as something that looks unrelated.
`build.sh` detects exactly that situation and refuses with the fix rather than letting it happen.

A per-user tarball also keeps the toolchain reproducible: one pinned version, one documented
command, upgradeable by editing `.node-version` and re-running the script.

## Build

```bash
./build.sh                 # install deps if needed (npm ci), then build
./build.sh --skip-install  # build against the node_modules already present
./build.sh --help
```

Output lands in `dist/` — `index.html` plus content-hashed assets.

**`dist/` is gitignored and never committed** (`docs/plans/salesperson-ui.md` OQ-6). This script
is the reproducible way to regenerate it. `falkor-chat/scripts/start_demo.sh` is the supported
one-command bring-up and calls it for you (see "Running the demo" below); running `./build.sh`
by hand is for iterating on the SPA alone.

### The `/shop` base path is load-bearing

`vite.config.ts` sets `base: "/shop/"` because `falkor-chat`'s FastAPI app mounts this bundle at
`/shop`. Every hashed asset URL is emitted as `/shop/assets/…`; serving the bundle from any
other prefix 404s all of them. `build.sh` asserts the built `index.html` still references
`/shop/` and fails if it does not, so the two halves cannot silently drift apart.

## Test

```bash
npm test          # Vitest, single run
npm run test:watch
npm run test:e2e  # Playwright, against a running server
npm run typecheck # tsc -b, no emit
npm run lint      # oxlint
```

Unit and component tests live beside the code they cover, as `src/**/*.test.tsx`. The Playwright
suite lives in `tests/e2e/` (`mobile-shell.spec.ts`, `presenter.spec.ts`) and drives a **running**
storefront rather than starting one — it is pointed at `http://127.0.0.1:8000/shop/` unless
`SALESPERSON_E2E_BASE_URL` says otherwise; bring the stack up first (below).

## Running the demo

`falkor-chat/scripts/start_demo.sh` is the supported one-command, from-cold-box bring-up: FalkorDB
→ schema → seed demo/catalog/salesperson → preflight checks → `./build.sh` → uvicorn as the
**storefront** deployment (`FALKORCHAT_WS_ID=demo`, trigger pinned to `salesperson@v8`, responder
fall-through off, `--reload` off). Run it from `falkor-chat/`:

```bash
./scripts/start_demo.sh
```

It is **not** `start_server.sh` with different flags — `start_server.sh` also seeds the `triage`/
`access-request` defs the demo doesn't need and defaults to `--reload`, which a live demo doesn't
want. Env overrides live in the script's own header comment.

To iterate on the SPA alone without the full bring-up script, the equivalent manual sequence
(server venv installed as `start_server.sh` does it — `python3 -m venv server/.venv &&
server/.venv/bin/pip install -e 'server[dev]'`):

```bash
./scripts/start_falkordb.sh -d
EMBEDDING_DIM=1024 ./scripts/bootstrap_schema.sh demo   # ws:demo, NOT the "acme" default
./scripts/seed_demo.sh demo                             # the Agent the storefront posts to
./scripts/seed_catalog.sh                               # products live in `reference`
./scripts/seed_salesperson.sh demo                      # publishes salesperson@v8
./scripts/verify_salesperson.sh demo && ./scripts/verify_catalog.sh   # read-only checks

cd salesperson && ./build.sh && cd ../server && \
  FALKORCHAT_WS_ID=demo FALKORCHAT_EMBEDDING_DIM=1024 \
  FALKORCHAT_ENABLE_AGENT=1 FALKORCHAT_WORKFLOW_ENABLED=1 \
  FALKORCHAT_TRIGGER_DEF_KEY=salesperson FALKORCHAT_TRIGGER_DEF_VERSION=v8 \
  FALKORCHAT_TRIGGER_RESPONDER_FALLTHROUGH=0 \
  FALKORCHAT_STOREFRONT_ENABLED=1 \
  FALKORCHAT_STOREFRONT_DIR="$PWD/../../salesperson/dist" \
  FALKORCHAT_OPENCODE_CONFIG="$HOME/.config/opencode/opencode.json" \
  .venv/bin/uvicorn falkorchat.app:app
```

`FALKORCHAT_STOREFRONT_DIR` must name an **existing** directory or the `/shop` mount is skipped
**silently** while `/shop/api` serves normally (`falkor-chat/docs/SERVER.md` §1.3) — so the build
really does come before uvicorn.

Playwright's browser binary is not installed by `npm ci`; get it once with:

```bash
npx playwright install chromium
```

## Layout

```
salesperson/
├── build.sh                 # the supported build entry point
├── scripts/install_node.sh  # per-user Node provisioning
├── .node-version / .nvmrc   # the toolchain pin
├── index.html               # Vite's HTML entry
├── vite.config.ts           # base "/shop/", Tailwind, Vitest config
├── playwright.config.ts     # one mobile-viewport project
├── vitest.setup.ts          # jest-dom matchers
├── public/                  # copied verbatim into dist/
├── src/                     # the application
└── tests/e2e/               # Playwright specs
```

## Product images

The catalog's 15 `Product` nodes (`falkor-chat/scripts/seed_catalog.sh`) carry **no image field**
of any kind, so images are client-side assets keyed by the deterministic `productId` slug, under
`public/products/<productId>.jpg`. `storefront.build_image_manifest()` (`falkor-chat` §4.7) turns
presence/absence into `imageUrl: string | null` per product; the catalog panel (`src/views/
CatalogPanel.tsx`) renders an `<img>` only when it is non-null.

**Source and licence (OQ-6):** 14 of the 15 products carry a photo sourced from **[Lorem
Picsum](https://picsum.photos)** — Picsum's own photos are drawn from Unsplash and distributed
under the **[Unsplash License](https://unsplash.com/license)** (free for commercial and
non-commercial use, no permission or attribution required). Each file is a deterministic,
per-product photo fetched at build time from `https://picsum.photos/seed/<productId>/640/480.jpg`
(the `seed` makes the pick stable across re-fetches, not a fresh-random image per run) and
committed verbatim — no further processing. The photos are generic stock photography, not literal
product shots (Picsum has no product/keyword search), consistent with the rest of this storefront
being a simulated demo catalog rather than a real one.

**`smart-home-hub` is deliberately left without a file** — one product with no asset is kept on
purpose so the catalog panel's text-only card path is exercised by the running app itself, not
only by a mocked unit test. Adding an image for it later is a drop-in: name the file
`public/products/smart-home-hub.<ext>` and the manifest picks it up on the next server start
(`falkor-chat` §4.7's accepted extensions are `.webp`, `.jpg`, `.jpeg`, `.png`, first match in that
order).

## Status

Delivered — every FR/AC in `docs/requirements/salesperson-ui.md` is built and QA'd
(`docs/test-reports/salesperson-ui-report.md`; `docs/plans/salesperson-ui2-coordination.md`
carries the full unit-by-unit ledger). One still-open item gates the *first live, audience-facing*
demo specifically, not ordinary use: `K-065` (`falkor-chat/docs/BACKLOG.md`), an occasional
wrong-language reply under concurrency, confirmed to reproduce at LM Studio's own serving layer.
Mitigation D (`systemPrompt` language salience, `v8`) is implemented and reviewed but **not yet
live-verified** — see `falkor-chat/docs/BACKLOG.md`'s `K-065` entry for current status. See
`AGENTS.md` for file ownership and conventions.
