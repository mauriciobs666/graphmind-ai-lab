# llm-gateway

One gateway for every component's LLM calls: provider choice (LM Studio, Ollama, Anthropic,
generic OpenAI-compatible), fallback, local-model loading, call records and hosted token
ceilings, all by configuration. Python 3.12, stdlib only.

**Status: under construction.** Only the scaffold and the shared types (`src/llmgateway/types.py`)
exist; later dispatches follow `docs/plans/llm-gateway.md` (repo root, §4). Requirements:
`docs/requirements/llm-gateway.md`.

## Run

```bash
llm-gateway/setup.sh                  # create llm-gateway/.venv, install `.[dev]`
cd llm-gateway && .venv/bin/pytest    # run the suite with this folder as the working directory
llm-gateway/run.sh -c 'import llmgateway; print(llmgateway.__version__)'
```

`pytest -m live` (real LM Studio/Ollama) and `-m deploy` (real root-owned install) are opt-in.
