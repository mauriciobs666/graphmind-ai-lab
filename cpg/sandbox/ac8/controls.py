#!/usr/bin/env python3
"""The AC-8 harness's in-run controls and witnesses (egress-spec3 §8.1 "Controls inside the main run", P-g).

    controls.py --socket <broker socket> --manifest <system.yaml> --audit-dir <dir> [--stub-model] [--runner <offline-run.sh>]
                [--sandbox-src <dir>] [--witness-parse --repos-root <dir> [--lang <repoId>=<lang>]...] [--summary]

Runs on the host, inside the detector's window, against the production-shape broker. It injects the controls through the real
`BrokerClient` (a denial is a normal Decision, never an exception) and the real runner, and, with ``--stub-model``, serves a stub **local model**
on the loopback port of the manifest's `local` model entry (refused if that port is already in use) so the P-g witnesses exist: an `R1` allow from
`answer.interpret` in a run that touches a local-only repository and one in a run that touches only hosted repositories.
Nothing here connects anywhere but loopback and the broker's socket. It records nothing itself: the verdicts are
`assertions.py`'s, computed from the audit.

With ``--witness-parse`` it also runs one real `joern-parse` per repository through the runner (one non-refused `sandbox` record
each, P-g); P1-40's real refresh supplies those itself and omits the flag. Exit 0 iff every injected call returned the decision it
was injected to produce.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ccg import manifest as manifest_mod
from ccg.broker.client import BrokerClient

HERE = Path(__file__).resolve().parent


class _Stub(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(n)
        body = b'{"choices":[{"message":{"role":"assistant","content":"stub"}}]}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):  # silence
        pass


def _entry(egress: dict, section: str, kind: str) -> dict:
    return next(e for e in egress[section] if e["kind"] == kind)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--socket", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--audit-dir", required=True)
    ap.add_argument("--stub-model", action="store_true", help="serve a stub local model on the manifest's local endpoint port (self-tests; refuses a port already in use)")
    ap.add_argument("--runner", default=str(HERE.parent / "offline-run.sh"))
    ap.add_argument("--sandbox-src", help="a directory with one source file, the forbidden-flag control's --source")
    ap.add_argument("--witness-parse", action="store_true")
    ap.add_argument("--repos-root")
    ap.add_argument("--lang", action="append", default=[])
    ap.add_argument("--summary", action="store_true", help="also request answer.summarize (the summary is enabled)")
    a = ap.parse_args(argv)

    m = manifest_mod.load_manifest(a.manifest)
    eg = m.egress
    local_model, hosted_model, idp = _entry(eg, "local", "model"), _entry(eg, "hosted", "model"), _entry(eg, "outbound", "idp")
    local_repos = [r.repo_id for r in m.repos if r.policy.egress == "local-only"]
    hosted_repos = [r.repo_id for r in m.repos if r.policy.egress == "hosted"]
    if not local_repos or not hosted_repos:
        print("controls: the manifest needs a local-only and a hosted-default repository (P-f)", file=sys.stderr)
        return 2
    srv = None
    if a.stub_model:
        from urllib.parse import urlsplit
        parts = urlsplit(local_model["endpoint"])
        if parts.hostname not in ("127.0.0.1", "localhost") or not parts.port:
            print("controls: --stub-model needs a loopback `local` model endpoint with a port", file=sys.stderr)
            return 2
        try:  # binding a port something already serves (LM Studio's 1234) fails here: the stub never shadows a real model
            srv = ThreadingHTTPServer(("127.0.0.1", parts.port), _Stub)
        except OSError as exc:
            print(f"controls: cannot serve the stub model on 127.0.0.1:{parts.port}: {exc}", file=sys.stderr)
            return 2
        threading.Thread(target=srv.serve_forever, daemon=True).start()
    c = BrokerClient(a.socket)
    problems: list[str] = []

    def req(token, component, purpose, payload_class, method, endpoint, path, codebases, body=None):
        return c.request(token, {"component": component, "purpose": purpose, "payloadClass": payload_class,
                                 "target": {"method": method, "path": path}, "endpoint": endpoint, "codebases": codebases}, body)

    def expect(label, got, decision, rule_prefix, reason=None):
        d, _resp = got
        if d.decision != decision or not d.rule.startswith(rule_prefix) or (reason is not None and d.reason != reason):
            problems.append(f"{label}: expected {decision} {rule_prefix} {reason}, got {d.decision} {d.rule} {d.reason}")

    # -- answer run over the whole system (touches a local-only repository): the 8.1/8.8 witness, the R4 and R0 controls
    g = c.open_run("answer", m.system, "*")
    model_body = {"messages": [{"role": "user", "content": "x"}]}
    expect("R1 local interpret (all repos)", req(g.token, "answer.interpret", "model-call", "repo-derived", "POST",
                                                  local_model["endpoint"], local_model["paths"][0], [local_repos[0]], model_body),
           "allow", "R1", None)
    if a.summary:
        expect("R1 local summarize", req(g.token, "answer.summarize", "model-call", "repo-derived", "POST",
                                         local_model["endpoint"], local_model["paths"][0], [local_repos[0]], model_body), "allow", "R1")
    idp_path = idp["paths"][0]
    ident = {"subject": "alice", "issuer": idp["endpoint"]}
    expect("R4 extra body field", req(g.token, "access.idp", "idp", "repo-names-and-identity", "POST", idp["endpoint"], idp_path,
                                      [local_repos[0]], {"repoNames": [local_repos[0]], "identity": ident, "extra": 1}),
           "deny", "R4", "payload-shape")
    expect("R4 repository outside the system", req(g.token, "access.idp", "idp", "repo-names-and-identity", "POST", idp["endpoint"],
                                                    idp_path, [local_repos[0]], {"repoNames": ["not-in-this-system"], "identity": ident}),
           "deny", "R4", "payload-shape")
    expect("R0 no token", req(None, "answer.interpret", "model-call", "repo-derived", "POST", local_model["endpoint"],
                              local_model["paths"][0], [local_repos[0]], model_body), "deny", "R0", "token-invalid")
    expect("R0 codebases=[]", req(g.token, "answer.interpret", "model-call", "repo-derived", "POST", local_model["endpoint"],
                                  local_model["paths"][0], [], model_body), "deny", "R0", "codebases-empty")
    c.close_run(g.token)

    # -- answer run over a hosted repository only, in a system that has a local-only one: 8.7 (interpreted locally), the R5 control
    g = c.open_run("answer", m.system, [hosted_repos[0]])
    if not g.system_local_only:
        problems.append("the hosted-only answer run is not systemLocalOnly")
    expect("R1 local interpret (hosted-only question)", req(g.token, "answer.interpret", "model-call", "repo-derived", "POST",
                                                            local_model["endpoint"], local_model["paths"][0], [hosted_repos[0]], model_body),
           "allow", "R1")
    expect("R5 hosted model call", req(g.token, "answer.interpret", "model-call", "repo-derived", "POST", hosted_model["endpoint"],
                                       hosted_model["paths"][0], [hosted_repos[0]], model_body), "deny", "R5", "system-local-only-model")
    c.close_run(g.token)

    # -- refresh runs: R2 and R8 denies to a local endpoint with a listed path; the two override policy records
    g = c.open_run("refresh", m.system, "*")
    expect("R2 git-fetch", req(g.token, "build.git", "git-fetch", "none", "GET", local_model["endpoint"], local_model["paths"][0],
                               [local_repos[0]]), "deny", "R2", "not-in-p1:git-fetch")
    expect("R8 dependency-download", req(g.token, "build.pkgmgr", "dependency-download", "none", "GET", local_model["endpoint"],
                                         local_model["paths"][0], [local_repos[0]]), "deny", "R8", "not-in-p1:controlled-download")
    c.close_run(g.token)
    g = c.open_run("refresh", m.system, [local_repos[0]], "hosted")
    if g.effective.get(local_repos[0]) != "local-only":
        problems.append("P2-OVERRIDE-REFUSED: the effective mode is not local-only")
    c.close_run(g.token)
    g = c.open_run("refresh", m.system, [hosted_repos[0]], "local")
    if g.effective.get(hosted_repos[0]) != "local-only":
        problems.append("P3-OVERRIDE-LOCAL: the effective mode is not local-only")
    c.close_run(g.token)

    # -- the runner: the forbidden-flag control (refused, 64, no container) and, optionally, one real parse per repository
    env = {"PATH": "/usr/bin:/bin", "CCG_AUDIT_DIR": a.audit_dir}
    if os.environ.get("CCG_TOOLCHAIN_IMAGE"):
        env["CCG_TOOLCHAIN_IMAGE"] = os.environ["CCG_TOOLCHAIN_IMAGE"]
    work = Path(a.audit_dir).parent / "controls-work"
    src_ctl = Path(a.sandbox_src) if a.sandbox_src else work / "flag-src"
    if not a.sandbox_src:
        src_ctl.mkdir(parents=True, exist_ok=True)
        (src_ctl / "x.kt").write_text("fun main() {}\n")
    cp = subprocess.run([a.runner, "--backend", "container", "--mode", "offline", "--frontend", "joern-parse", "--source",
                         str(src_ctl), "--output", str(work / "flag-out"), "--codebase", local_repos[0], "--", "/src",
                         "--language", "kotlin", "--jar4import-url", "http://import-names.invalid/"],
                        env=env, capture_output=True, text=True, timeout=180)
    if cp.returncode != 64:
        problems.append(f"the --jar4import-url control exited {cp.returncode}, not 64: {cp.stderr[-200:]}")
    if a.witness_parse:
        langs = dict(x.split("=", 1) for x in a.lang)
        root = Path(a.repos_root or Path(a.manifest).parent)
        for r in m.repos:
            cp = subprocess.run([a.runner, "--backend", "container", "--mode", "offline", "--frontend", "joern-parse", "--source",
                                 str(root / r.path), "--output", str(work / f"out-{r.repo_id}"), "--codebase", r.repo_id, "--", "/src",
                                 "--language", langs.get(r.repo_id, "pythonsrc")],
                                env=env, capture_output=True, text=True, timeout=600)
            if cp.returncode != 0:
                problems.append(f"witness parse of {r.repo_id} exited {cp.returncode}: {cp.stderr[-200:]}")
    if srv:
        srv.shutdown()
    for p in problems:
        print(f"controls: {p}", file=sys.stderr)
    print(json.dumps({"controls": "done", "problems": problems}))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
