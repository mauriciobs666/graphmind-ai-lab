# llm-gateway

Common LLM gateway: library core, key-holding process, service and client. Design:
`docs/plans/llm-gateway.md` (repo root); requirements: `docs/requirements/llm-gateway.md`;
delivery ledger: `docs/plans/llm-gateway-coordination.md`. No live spec yet (it lands at feature
close).

- **Built (D1-D10, in `src/llmgateway/`):** types; ownercheck, config, failures; records, policy;
  engine, transport, providers (lmstudio, ollama, openai_compat, anthropic); load; key holder,
  ledger, framing, `deploy/install.sh`; service, single-file client, `HolderTransport`. The
  ccg side is in `cpg/`: `cpg/ccg/gateway/` (`BrokerTransport`) and the broker's holder forward
  (`cpg/ccg/broker/holderclient.py`, `server.py`).
- **Not done:** the ccg `ModelRun` swap (D11/D12) waits for the ccg P1 phase exit; suites (D13);
  the tier-2 run against a real root-owned install (D14) needs a human `sudo`.
- **Not supported:** embeddings (refused at first delivery), Anthropic `tools` (a call with tools
  fails naming them), grammar passthrough (`Request.response_format` is JSON-schema or
  `json_object` only; FR-18 names a grammar field).
- **Run tests with `llm-gateway/` as the working directory:**
  `../model-bench/.venv/bin/pytest -q -p no:cacheprovider` (or `.venv/bin/pytest` after
  `setup.sh`); from the repo root pytest picks the monorepo `rootdir`. `live` and `deploy` are
  deselected by default. Stdlib only, Python 3.12, `src/` layout.
- **The client is byte-copied** by components that cannot import outside their folder
  (`src/llmgateway/client.py`); `tests/test_client.py` pins its sha256 (`0d1424d2...`) and its
  imports. Change it deliberately and tell every copy's owner.
- `Request` has no model or provider field and refuses unknown fields; keep it so (owner
  configuration chooses both). `run_id` is validated against the record alphabet.
- **Configuration is owner-protected and re-read per call.** A file failing the check collapses
  to loopback-only local providers. `provider_is_hosted` (config) is the one definition of
  hosted: a `key_ref`, a non-local kind or a non-local origin make a provider hosted.
- **Records are advisory and metadata-only:** prompt/response text only when every resolved scope
  opts in. Floats are value-identical on the wire, not text-identical (`1e-7` goes out `1e-07`).
- **Deploy needs a human `sudo`.** The holder and service run only from the root-owned code root
  `/opt/llmgateway`, never from this checkout; implementers never run `sudo`/`install.sh`.
