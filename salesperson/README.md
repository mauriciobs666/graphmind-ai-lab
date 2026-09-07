# salesperson

Business-facing, mobile-first **storefront UI** for the workflow-engine-backed `salesperson`
agent hosted by `falkor-chat/`. Around 50 audience members each join with nothing but a display
name and then hold their own independent, isolated conversation — own cart, own order, own
profile — in the language each of them picks.

It is a **pure static bundle**. There is no Node server at runtime: `falkor-chat`'s single
FastAPI process serves the built assets at `/shop`, so the demo stays one process, one port, no
CORS. Node is a **build-time dependency only**.

Requirements: `docs/requirements/salesperson-ui.md`. Plan: `docs/plans/salesperson-ui.md`
(the stack rationale is §4.2).

> This directory previously held a retired Streamlit chatbot. That app now lives at
> `deprecated/salesperson/` and is unrelated to the code here — see `deprecated/README.md`.

## Stack

| Concern | Choice |
|---|---|
| Build / dev server | Vite 8 |
| UI | React 19 + TypeScript 6 |
| Styling | Tailwind CSS v4 (via `@tailwindcss/vite`) |
| Server state / polling | TanStack Query v5 |
| Localization | `i18next` + `react-i18next` |
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
is the reproducible way to regenerate it, and `falkor-chat/scripts/start_demo.sh` will call it as
part of demo bring-up — **that script is not built yet (plan step S11)**, so today you run
`./build.sh` yourself before starting the server (see "Test" below).

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
suite lives in `tests/e2e/` and drives a **running** storefront rather than starting one — it is
pointed at `http://127.0.0.1:8000/shop/` unless `SALESPERSON_E2E_BASE_URL` says otherwise. (The
suite itself is **not written yet — plan step S12b**; `tests/e2e/` holds only its `.gitkeep`, so
`npm run test:e2e` has nothing to run today.)

**Bringing the stack up: manual until S11.** `falkor-chat/scripts/start_demo.sh` — the
one-command bring-up — **does not exist yet**; it is plan step S11
(`docs/plans/salesperson-ui.md`), which also defines the sequence below. Until it lands, do it by
hand: build the bundle with `./build.sh`, then, from `falkor-chat/` (server venv installed as
`start_server.sh` does it — `python3 -m venv server/.venv && server/.venv/bin/pip install -e
'server[dev]'`):

```bash
./scripts/start_falkordb.sh -d
EMBEDDING_DIM=1024 ./scripts/bootstrap_schema.sh demo   # ws:demo, NOT the "acme" default
./scripts/seed_demo.sh demo                             # the Agent the storefront posts to
./scripts/seed_catalog.sh                               # products live in `reference`
./scripts/seed_salesperson.sh demo                      # publishes salesperson@v7
./scripts/verify_salesperson.sh demo && ./scripts/verify_catalog.sh   # read-only checks

cd server && FALKORCHAT_WS_ID=demo FALKORCHAT_EMBEDDING_DIM=1024 \
  FALKORCHAT_ENABLE_AGENT=1 FALKORCHAT_WORKFLOW_ENABLED=1 \
  FALKORCHAT_TRIGGER_DEF_KEY=salesperson FALKORCHAT_TRIGGER_DEF_VERSION=v7 \
  FALKORCHAT_TRIGGER_RESPONDER_FALLTHROUGH=0 \
  FALKORCHAT_STOREFRONT_ENABLED=1 \
  FALKORCHAT_STOREFRONT_DIR="$PWD/../../salesperson/dist" \
  FALKORCHAT_OPENCODE_CONFIG="$HOME/.config/opencode/opencode.json" \
  .venv/bin/uvicorn falkorchat.app:app
```

Two traps worth knowing before you run it. `FALKORCHAT_STOREFRONT_DIR` must name an **existing**
directory or the `/shop` mount is skipped **silently** while `/shop/api` serves normally
(`falkor-chat/docs/SERVER.md` §1.3) — so `./build.sh` really does come first. And
`falkor-chat/scripts/start_server.sh` is *not* the shortcut: it also seeds the `triage` /
`access-request` defs that S11 deliberately skips, and it defaults to `--reload`, which S11 turns
off for a demo run.

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

Catalog imagery and its licensing are recorded here when they land (`docs/plans/salesperson-ui.md`
OQ-6). The catalog's 15 `Product` nodes carry **no image field** of any kind, so images are
client-side assets keyed by the deterministic `productId` slug, under `public/products/`.

## Status

Scaffold only. The application itself is built by the later steps of
`docs/plans/salesperson-ui.md` §5.1. See `AGENTS.md` for what is wired and what is deliberately
left open.
