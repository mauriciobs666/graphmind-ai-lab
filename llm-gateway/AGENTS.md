# llm-gateway

Common LLM gateway (library core, key-holding process, optional service). Design:
`docs/plans/llm-gateway.md` (repo root); requirements: `docs/requirements/llm-gateway.md`. No
live spec yet (it lands at feature close).

- **State:** only the scaffold and `src/llmgateway/types.py` exist; the build is dispatched in
  steps D1..D15 per the plan §4.
- **Stdlib only, Python 3.12, no runtime dependencies** (a test pins it). `src/` layout; `pytest`
  gets `src` via `pythonpath`.
- **Run tests with `llm-gateway/` as the working directory** (`.venv/bin/pytest`); from the repo
  root pytest picks the monorepo `rootdir` and ignores this component's config.
- `live` and `deploy` markers are deselected by default; `deploy` needs a human-run `install.sh`.
- `Request` has no model or provider field and refuses unknown fields; keep it that way (the call
  site's owner configuration chooses both; refusing unknowns is what stops a request loosening a scope).
- Implementers never run `sudo`/`install.sh`; the holder and service run only from the root-owned
  code root, never from this checkout.
