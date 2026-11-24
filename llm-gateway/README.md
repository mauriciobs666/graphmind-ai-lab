# llm-gateway

One gateway for every component's LLM calls: provider choice (LM Studio, Ollama, Anthropic,
generic OpenAI-compatible), fallback, local-model loading, call records and hosted token
ceilings, all by configuration. Python 3.12, stdlib only.

**Status: under construction.** Design: `docs/plans/llm-gateway.md` (repo root); requirements:
`docs/requirements/llm-gateway.md`; delivery record: `docs/plans/llm-gateway-coordination.md`.

| Part | Where | State |
|---|---|---|
| Shared types, failure classes | `src/llmgateway/types.py`, `failures.py` | built |
| Owner-protected configuration | `ownercheck.py`, `config.py` | built |
| Call records, scope policy, run bound | `records.py`, `policy.py` | built |
| Engine (retry, fallback), transport, providers | `engine.py`, `transport.py`, `providers/` | built; local providers only from a caller process |
| Model loading (on demand, explicit, strict) | `load.py` | built |
| Key holder, ledger, framing, installer | `keyholder.py`, `ledger.py`, `framing.py`, `bin/`, `deploy/` | built; tier-2 real-install run pending a human `sudo` |
| Holder-backed transport, service, client, broker | | not built |

## Run

```bash
llm-gateway/setup.sh                  # create llm-gateway/.venv, install `.[dev]`
cd llm-gateway && .venv/bin/pytest    # run the suite with this folder as the working directory
llm-gateway/run.sh -c 'import llmgateway; print(llmgateway.__version__)'
```

`pytest -m live` (real LM Studio/Ollama) and `-m deploy` (real root-owned install) are opt-in.
