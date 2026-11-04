# Compose service to code unit binding (finding F-5)

> **Status:** active · **Owner:** `architect` · **Tracks:** C-1001 (P0)

Decides `docs/reviews/connected-code-graph-fixtures-trees.md` F-5. Touches neither the fixtures plan (`docs/test-plans/connected-code-graph-fixtures.md` v1.3) nor the rule-pack spec (`docs/plans/connected-code-graph-rule-pack-spec.md` v1.3).

## Findings (read in the files)
- The only Compose-to-unit rule in the plans is `docs/plans/connected-code-graph.md:536`: "Compose sets it per service → service build context → unit". No `image:`-name rule exists anywhere in the spec or plans.
- The spec's `launch` block (rule-pack spec §1.6 / pack format) has five keys, `frameworkDefaults`, `portFlags`, `targetArg`, `envPorts`, `mountPaths`, none of them Compose. Compose mappings are core behavior (spec line ~303), so an image rule would be a core/spec change, not a pack matter.
- `cpg/fixtures/system1/fx-svc/deploy/compose.yaml`: `api` has `build.context: ..`; `web` and `job` are `image: fx-web:latest` / `fx-job:latest` only. `job` publishes `8300:8300`. `fx-job/scripts/dev.sh` runs uvicorn on 8000 (this is LNK-16's competing launch); `fx-job` has no Dockerfile.
- No URL in any tree targets host `web` (grep `//web` finds nothing), so `web` is only an env-var setter (`VITE_API_BASE`, `VITE_JOB_BASE`), never a binding target. LNK-13 and OV-2 resolve their hosts (`api`, `job`) to services, not `web` to a unit.

## Decision: option (b), only for `job`
`job` gets a build context and a minimal Dockerfile, so the existing "service build context → unit" rule binds it. `web` stays image-only. Option (a) is rejected: it needs a new core rule (spec v1.4, a revision of the accepted spec) plus planned launch rows for `web` and `job`, one of which (`web`) nothing consumes. (b) needs no spec edit and no new plan row.

Derivation with (b):
- **LNK-15**: consumer `svc:clients/jobs.py` has `FX_JOB_URL` default `http://localhost:8300`. Port 8300 matches service `job`'s mapping `8300:8300`. Service `job` resolves through its build context to unit `job` (as LNK-22 does for `api` to `svc`). The path `/job/trigger` is on `job:job/trigger_server.py::trigger`. Result: confirmed, binding-confirmed.
- **OV-2**: `VITE_JOB_BASE=http://job:8300` is set in service `web`; host `job` is service `job`, port 8300, so unit `job`. `JobPanel` consumes `GET /job/status`, provided by `trigger_server.py::status`, and the change set `CS-JOB` edits that unit. Result: confirmed, binding-confirmed.
- LNK-16 is unaffected: `localhost:8000` is still claimed by `svc` (Compose `api`) and `job` (`dev.sh`), so it stays `possible/port-shared`. The Dockerfile must therefore use port 8300 explicitly, not 8000. AB-4 (`fx-job` never built, target-not-mapped) is unaffected.

## Edits by owner
- **fx-svc builder** (`cpg/fixtures/system1/fx-svc/deploy/compose.yaml`): in service `job`, keep `image: fx-job:latest` and add `build: {context: ../../fx-job, dockerfile: Dockerfile}`. The path resolves to the sibling repo in both the checked-in tree and S39's materialized layout (`cpg/.ccg-work/fixtures/`, four sibling repositories). Do not touch `web` or `compose.example.yaml`.
- **fx-job builder** (`cpg/fixtures/system1/fx-job/Dockerfile`, new): `FROM python:3.12-slim`, `WORKDIR /app`, `COPY . /app`, `RUN pip install -r requirements.txt`, `CMD ["uvicorn","job.trigger_server:app","--host","0.0.0.0","--port","8300"]`. Size and style mirror `fx-svc/deploy/Dockerfile`.
- **qa-engineer** (`docs/test-plans/connected-code-graph-fixtures.md`, next revision): add the mapping sentence to rows LNK-15 and OV-2 ("`job` Compose service: build context `../../fx-job`, port 8300, unit `job`") and to LNK-22's neighbors if wanted. No row added or removed: the census stays 132 rows, and the P1/P3 gates stay as they are.
- **manifest owners** (`cpg/fixtures/system1/expected/fx-job.yaml` LNK-15 `note`, which today says the port lives in fx-svc, plus OV-2 in `fx-job.yaml` and `fx-web.yaml`): restate the binding in those notes. LNK-15 and OV-2 `st`/`reason`/`gate` are unchanged.
- **Rule-pack spec**: no change. Not a v1.4 matter.

## Unverified, owed to the S30 owner
- Resolving a context that leaves its own repo root (`../../fx-job` is outside `fx-svc`), and a unit lookup that crosses repository boundaries, are both unspecified in the spec and plans; I found nothing that states either. `repoId` is carried on units, which suggests a system-wide unit index, but this is inference. S30 must confirm a Compose context pointing at a sibling repo's root resolves to that unit.
- If S30 cannot, the fallback is option (a), which then requires a core rule `image:` name = `repoId` and a spec v1.4 amendment. That is an escalation, not something to patch in the fixtures.
- Whether a Compose env-var setter service (`web`) needs its own unit binding for LNK-13: untested here. LNK-13 stays image-only and relies on the P1 gate to show it.
