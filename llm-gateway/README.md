# llm-gateway

One gateway for every component's LLM calls: provider choice (LM Studio, Ollama, Anthropic,
generic OpenAI-compatible), fallback, local-model loading, call records and hosted token
ceilings, all by configuration. Python 3.12, stdlib only.

**Status: library, key holder, service and client built; not yet adopted by a caller.** Design:
`docs/plans/llm-gateway.md` (repo root); requirements: `docs/requirements/llm-gateway.md`;
delivery record: `docs/plans/llm-gateway-coordination.md`.

| Part | Where | State |
|---|---|---|
| Shared types, failure classes | `src/llmgateway/types.py`, `failures.py` | built |
| Owner-protected configuration | `ownercheck.py`, `config.py` | built |
| Call records, scope policy, run bound | `records.py`, `policy.py` | built |
| Engine (retry, fallback), transport, providers | `engine.py`, `transport.py`, `providers/` | built; a caller process reaches local providers directly and hosted ones through the holder |
| Model loading (on demand, explicit, strict) | `load.py` | built |
| Key holder, ledger, framing, installer | `keyholder.py`, `ledger.py`, `framing.py`, `bin/keyholder.py`, `deploy/install.sh` | built |
| Service and single-file client | `service.py`, `bin/service.py`, `client.py` | built |
| Holder-backed transport | `transport.py` (`HolderTransport`) | built |
| ccg egress-broker transport and holder forward | `cpg/ccg/gateway/`, `cpg/ccg/broker/holderclient.py` | built (in `cpg/`) |
| ccg `ModelRun` swap | `cpg/` | not built; waits for the ccg P1 phase exit |
| Suites | | not built |
| Tier-2 run against a real root-owned install | `deploy/install.sh` | pending a human `sudo` |

Not supported at first delivery: embeddings, Anthropic `tools`, a grammar field on `Request`.

## Run

```bash
llm-gateway/setup.sh                  # create llm-gateway/.venv, install `.[dev]`
cd llm-gateway && .venv/bin/pytest    # run the suite with this folder as the working directory
llm-gateway/run.sh -c 'import llmgateway; print(llmgateway.__version__)'
```

Without `setup.sh`, `cd llm-gateway && ../model-bench/.venv/bin/pytest -q -p no:cacheprovider`
works. `pytest -m live` (real LM Studio/Ollama) and `-m deploy` (real root-owned install) are
opt-in. The holder and service run from the root-owned code root `/opt/llmgateway` after a human
runs `deploy/install.sh` with `sudo`, never from this checkout.
