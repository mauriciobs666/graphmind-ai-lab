"""analyst probe (not part of the tree): pins the sweep's version binding to its step-5.1 read."""
import json
from tests.test_workflow_input_cas import _build, _start, TIMED, CTX, _run


def test_a_human_repark_right_after_the_sweeps_read_lands_in_raced(wf_repo):
    services, _, clock = _build(wf_repo)
    run_id = _start(services, key="cas-sweep-read", **TIMED)
    clock.advance(120_000)

    class _AfterRead:
        def __init__(self, inner):
            self._inner, self._fired = inner, False

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def get_run(self, ws, *, run_id):
            row = self._inner.get_run(ws, run_id=run_id)
            if not self._fired:
                self._fired = True
                services._repo = self._inner  # noqa: SLF001 — the human runs unproxied
                out = services.submit_workflow_input(CTX, run_id=run_id,
                                                     input={"note": "from-human"})
                assert out["status"] == "waiting"
                services._repo = self  # noqa: SLF001
            return row

    services._repo = _AfterRead(wf_repo)  # noqa: SLF001
    out = services.sweep_due_workflow_runs(CTX, limit=10)
    services._repo = wf_repo  # noqa: SLF001

    assert out["raced"] == [{"runId": run_id}]
    assert json.loads(_run(wf_repo, run_id)["ctx"]) == {"note": "from-human"}
