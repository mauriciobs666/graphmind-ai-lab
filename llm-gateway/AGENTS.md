# llm-gateway

Common LLM gateway (library core, key-holding process, optional service). Design:
`docs/plans/llm-gateway.md` (repo root); requirements: `docs/requirements/llm-gateway.md`;
delivery ledger: `docs/plans/llm-gateway-coordination.md`. No live spec yet (it lands at feature
close).

- **State:** the library core is built: `types`, `ownercheck`/`config`/`failures`,
  `records`/`policy`, `engine`/`transport`/`providers/` (lmstudio, ollama, openai_compat,
  anthropic) and `load` (residency). The key holder (`keyholder.py`, `framing.py`, `ledger.py`,
  `bin/keyholder.py`, `deploy/install.sh`) is built and tested against stand-ins; its tier-2 run
  against a real root-owned install awaits a human `sudo`. No holder-backed transport, service
  entry point or client exists yet, so a hosted step cannot be called from a caller process.
- **Stdlib only, Python 3.12, no runtime dependencies** (a test pins it). `src/` layout; `pytest`
  gets `src` via `pythonpath`.
- **Run tests with `llm-gateway/` as the working directory:**
  `../model-bench/.venv/bin/pytest -q -p no:cacheprovider` (or `.venv/bin/pytest` after
  `setup.sh`). From the repo root pytest picks the monorepo `rootdir` and ignores this
  component's config. `live` and `deploy` markers are deselected by default; `deploy` needs a
  human-run `install.sh`.
- `Request` has no model or provider field and refuses unknown fields; keep it that way (the call
  site's owner configuration chooses both; refusing unknowns is what stops a request loosening a
  scope). `run_id` is validated there against the record alphabet, so a bad one fails before a send.
- **Configuration is owner-protected and re-read per call.** A file failing the owner check
  collapses to loopback-only local providers; hosted needs a protected file. `provider_is_hosted`
  (config) is the one definition of hosted: a `key_ref`, a non-local kind, or a non-local origin.
- **No grammar passthrough in first delivery.** `Request.response_format` carries JSON-schema /
  `json_object` only (per adapter); there is no grammar field (FR-18 names one; the plan should say
  so).
- **Floats are value-identical on the wire, not text-identical** (`1e-7` goes out as `1e-07`);
  integers and strings are exact (`providers/base.py`).
- **Records are advisory and metadata-only:** no header, error body or key column; prompt/response
  text only when every resolved scope opts in. A provider-reported model outside the record
  alphabet is stored as `invalid-reported-model`; the reply is kept.
- Implementers never run `sudo`/`install.sh`; the holder and service run only from the root-owned
  code root, never from this checkout.
