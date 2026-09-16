# Review: salesperson-ui U-DEF1 (`/shop` SPA-fallback fix)

> **Status:** active · **Owner:** `analyst` · **Tracks:** U-DEF1 / DEF-1 (`docs/plans/salesperson-ui2-coordination.md`; `docs/test-reports/salesperson-ui-report.md`)

**Deviation from the requested path:** the brief suggested `salesperson-ui-s15-def1.md`. This
fix is not part of S15 (the QA pass that *found* DEF-1) — it is its own follow-up unit, tracked
in the coordination ledger as **U-DEF1**. `salesperson-ui-def1.md` mirrors the ledger's own unit
name and matches the naming already established for its sibling fix,
`falkor-chat/docs/reviews/salesperson-ui-def2.md` (same deviation, same reasoning, same
directory).

## Scope & verdict

Reviewed the uncommitted diff fixing DEF-1 (`git diff -- falkor-chat/server/falkorchat/app.py
falkor-chat/server/tests/test_app.py falkor-chat/docs/SERVER.md`, by `tdd-engineer`,
`a50911d63a92f8558`) against `docs/test-reports/salesperson-ui-report.md`'s DEF-1 section (read in
full) and the real Starlette 1.3.1 source installed in `falkor-chat/server/.venv`
(`starlette/staticfiles.py`, `starlette/routing.py`) rather than the plan's or PR's prose. Verdict:
**approve with suggestions** — no blocker; one major (a confirmed, narrow test-coverage gap) and
two minor/nit notes below.

**CPG:** considered, not relevant — `cpg_falkorchat` exists and indexes `create_app`
(`m.FILENAME = 'falkorchat/app.py'`), but the new `_SPAStaticFiles` class and its `get_response`
override are uncommitted and not yet in the graph, so it cannot answer anything about the diff
itself. The diff's blast radius (whether `_SPAStaticFiles` reaches beyond the `storefront=True`
branch) was instead confirmed directly: `grep -rn "create_app(" falkor-chat/server` shows every
call site is internal to `app.py` (module-level default app, CLI/env factory functions), and the
mount at issue (`app.py:526`) sits inside the `storefront`-gated block — the change cannot affect
the non-storefront chat deployment.

## Findings

### MAJOR — the `exc.status_code != 404` guard has no regression test (confirmed real, non-blocking)

**Evidence:** the coordinator's own mutation (deleting the `!= 404` check, so *any*
`StarletteHTTPException` from the base `get_response` falls back to `index.html`) left all 57
`test_app.py` tests green. I independently confirmed the two live non-404 paths the guard
protects by reading `starlette/staticfiles.py:109-152` directly: `get_response` raises
`HTTPException(status_code=405)` for a non-GET/HEAD method (line 113-114) and
`HTTPException(status_code=401)` on a `PermissionError` from `lookup_path` (line 118-119) — both
`StarletteHTTPException` instances, both currently un-exercised against the `/shop` mount. I also
ran a positive probe against the code *as written* (not mutated) —
`client.post("/shop/assets/index-abc123.js")` and `client.post("/shop/presenter")` both correctly
return `405 {"detail":"Method Not Allowed"}`, proving the guard works today; nothing proves it
keeps working tomorrow.

**Why it matters:** the guard is exactly the kind of one-line condition a future refactor
(simplifying the `try/except`, "cleaning up" the exclusion logic) could silently drop, and no test
would catch the regression — every request to a static asset would start `200`-ing with the SPA
shell on a wrong-method or permission-denied request instead of failing with the correct status.
Blast radius is narrow (no data exposure — `lookup_path`'s traversal guard, `staticfiles.py:154-173`,
is untouched by this diff either way) but the fix is cheap.

**Suggested improvement — a coverage probe over the axis the guard varies on** (the exception's
`status_code`, not a fixed pair of shapes): parametrize a new test in `test_app.py` over the two
non-404 codes the base class can raise against the `/shop` mount:

```python
@pytest.mark.parametrize(("method", "path", "expected_status"), [
    ("POST", "/shop/assets/index-abc123.js", 405),   # real asset, wrong method
    ("POST", "/shop/presenter", 405),                # SPA-route-shaped path, wrong method
])
def test_non_404_static_errors_are_not_swallowed_by_the_spa_fallback(tmp_path, method, path, expected_status):
    served = _built_spa_dir(tmp_path)
    app = _storefront_app(storefront_dir=served)
    client = TestClient(app)
    resp = client.request(method, path)
    assert resp.status_code == expected_status
    assert "SPA-SHELL-MARKER" not in resp.text
```

This alone kills the coordinator's mutation (verified: I ran it against the current, unmutated
code and it passes — see probe above). The `401`/`PermissionError` leg is harder to hit portably
(needs an actual unreadable file, environment-dependent under `chmod`); it's a real hole in the
*coverage* but not one I'd block on — call it out to `tdd-engineer` as optional (e.g. monkeypatch
`_SPAStaticFiles.lookup_path`/`get_response` to raise `PermissionError` directly) rather than
requiring it before merge.

### MINOR — the base class's `404.html` branch bypasses the override entirely, not just the `!= 404` guard

**Evidence:** `starlette/staticfiles.py:147-151` — when `self.html` is `True` (it is here) and no
static file/directory-index matched, the base `get_response` checks for a `404.html` file in the
served directory and, if found, **returns** a `FileResponse(..., status_code=404)` directly rather
than raising. `_SPAStaticFiles.get_response`'s `try/except StarletteHTTPException` only ever
triggers on a raised exception, so if `salesperson`'s built bundle ever ships a `404.html` (e.g.
via the common GitHub-Pages-style "drop a 404.html for SPA fallback" convention some tutorials
recommend), DEF-1's fix silently stops working for any path that reaches this branch — it would
serve that file's content with a `404` status instead of the SPA shell with `200`. I confirmed no
such file exists today (`find salesperson -iname '*404*'`, `salesperson/public/` has no
`404.html`), so this is latent, not live.

**Suggested improvement:** a one-line note in `_SPAStaticFiles`'s docstring flagging that a
`404.html` in the served directory would short-circuit this fallback (so nobody "fixes" DEF-1
again by adding one, the classic SPA-hosting anti-pattern for this exact server). Not worth a
test — it's a documentation gap, not a code gap, since the fix is correct for the shape the actual
Vite build produces.

### NIT — malformed-path 404s (null bytes, `ENAMETOOLONG`) now also get the SPA shell instead of a bare 404

**Evidence:** `staticfiles.py:120-128` raises a plain `HTTPException(404)` for a `ValueError`
(null bytes in the path) or `OSError` with `ENAMETOOLONG` — both now caught by the `!= 404` guard
and re-served as `index.html`/200, same as any other 404. This is a direct, correct consequence of
the intended design (any non-API 404 under `/shop` becomes the shell) rather than an oversight,
and it introduces no new information disclosure (`lookup_path`'s traversal check is untouched).
Flagging only so it isn't mistaken for a gap later — no action needed.

## What's solid

- **The "Mounts are terminal" claim is true**, verified against the installed Starlette 1.3.1
  source rather than taken on faith: `Mount.matches` (`routing.py:387-418`) returns `Match.FULL`
  as soon as its path regex matches the `/shop` prefix, independent of what the mounted app
  eventually returns; `Router.app` (`routing.py:672-692`) calls `route.handle()` and `return`s
  immediately on `Match.FULL`, never continuing to scan later routes. A plain route added after
  the mount would never be reached for anything under `/shop/*` — the subclass-inside-the-mount
  approach was the correct call, not overengineering.
- **The `/shop/api` exclusion is correct and boundary-safe**: `route_path == self._api_prefix or
  route_path.startswith(f"{self._api_prefix}/")` — no false match on a path like `apiextra`, and
  the prefix itself is derived from `storefront_api.API_PREFIX[len(SHOP_MOUNT):]` rather than
  hardcoded (verified: `API_PREFIX = "/shop/api"`, `SHOP_MOUNT = "/shop"`, slice → `"api"`,
  matching the class's own comment).
- **Registration order confirmed**: the API router (`app.py:519-521`) is registered before the
  `/shop` static mount (`app.py:526`), so a real `/shop/api/*` request is matched by the API
  router first and never reaches `_SPAStaticFiles` — the exclusion logic is purely a backstop for
  an unmatched/typo'd API path, exactly as the class's docstring claims and as
  `test_shop_api_paths_are_never_swallowed_by_the_spa_fallback` verifies (including the positive
  check that `/shop/api/health` still works).
- **The reproduction tests test the right thing, not a coincidence**: `_built_spa_dir` gives
  `index.html` and the real asset distinct content markers (`SPA-SHELL-MARKER` /
  `REAL-ASSET-MARKER`), so the four new tests assert *which content* was served, not just a status
  code — they'd fail if the fallback served the wrong file with a 200.
- **No new path-traversal surface**: `lookup_path`'s `os.path.commonpath` escape guard
  (`staticfiles.py:154-173`) is untouched, and the subclass never feeds attacker-influenced input
  into a new lookup — only the literal `"index.html"`.
- **`SERVER.md`'s addition is accurate and well-placed** (right subsection, cites DEF-1, matches
  actual code behavior).
- **Test suite run myself**: `falkor-chat/server`, `.venv/bin/python -m pytest -q` →
  **2825 passed, 14 deselected** (57/57 in `test_app.py`, matching the coordinator's figure; the 4
  new DEF-1 tests plus the pre-existing 2821 baseline). This run wiped the shared `reference`
  graph's data at teardown, exactly as `falkor-chat/AGENTS.md` documents for a default offline
  run — restored afterward (`bootstrap_schema.sh demo`, `seed_demo.sh demo`, `seed_catalog.sh`,
  `seed_workflows.sh demo`, `seed_salesperson.sh demo`), all three verify scripts
  (`verify_workflows.sh demo`, `verify_catalog.sh`, `verify_salesperson.sh demo`) confirm `OK` /
  in-sync afterward.

## Open questions

None blocking. One judgment call for the team: whether to also close the `401`/`PermissionError`
leg of the missing-coverage finding (monkeypatch-based, since a real unreadable-file fixture is
environment-fragile) or accept the `405` test alone as sufficient evidence the guard is
load-bearing — I'd accept the `405`-only test as enough to close this before merge; the `401` leg
is a nice-to-have.
