#!/usr/bin/env python3
"""mutants_ac8.py -- mutation test of the AC-8 harness's Python (P1-31): `window.py` and `assertions.py`.

Each mutant is a COPY of `cpg/sandbox/` (the runner, its lists and `ac8/`) in a fresh scratch directory with one deliberate edit; the
real files are never touched. The copy is tested with `cpg/tests/test_ac8_window.py` and `test_ac8_assertions.py`, pointed at it by
`CCG_AC8_DIR`. A mutant is **killed** iff pytest exits non-zero; every edit must apply exactly once (a missing old text is an error,
so no mutant silently equals the original). Exit 0 iff every selected mutant is killed.

    cpg/.venv/bin/python cpg/sandbox/tests/mutants_ac8.py [id ...]       (no id: all; MUTANT_DRY=1: only check the edits apply)

Scratch copies are `<scratch>/p131_<id>_*`, removed afterwards. Run it under `ulimit -v 3000000` and `timeout -s KILL`.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SBX = Path(__file__).resolve().parent.parent
CPG = SBX.parent
W, A, D, L, R = "ac8/window.py", "ac8/assertions.py", "ac8/detector.pl", "ac8/lib.sh", "ac8/run.sh"

M = [
    # ---- window.py: the markers, the verdict, the control scenario
    ("W01", W, "if start is not None and t < start:", "if start is not None and t <= start:", "a conn on the start instant is 'before'"),
    ("W02", W, "elif flush is not None and t > flush:", "elif False:", "a conn after the flush marker is not flagged"),
    ("W03", W, "if start is not None and t < start:", "if False:", "a conn before the start marker is not flagged"),
    ("W04", W, "            fails.append(f\"conn-in-window {c['srcAddr']}", "            _ = (f\"conn-in-window {c['srcAddr']}", "a conn inside the window does not fail"),
    ("W05", W, "    if start is None:\n        fails.append(\"marker-missing:start\")", "    if False:\n        fails.append(\"marker-missing:start\")", "missing start marker passes"),
    ("W06", W, "    if flush is None:\n        fails.append(\"marker-missing:flush\")", "    if False:\n        fails.append(\"marker-missing:flush\")", "missing flush marker passes"),
    ("W07", W, "if start is not None and flush is not None and flush < start:", "if False:", "an inverted window passes"),
    ("W08", W, "if not conn_file_exists:", "if False:", "a missing conn log counts as zero connections"),
    ("W09", W, "min(early) < start", "min(early) <= start", "a policy record at the start instant is 'late'"),
    ("W10", W, "max(record_times) > flush", "max(record_times) >= flush", "a record at the flush instant is 'early'"),
    ("W11", W, "        errs.append(f\"{Path(path).name}: trailing partial line\")\n", "        pass\n", "a trailing partial conn line is ignored"),
    ("W12", W, "    if set(rec) != set(fields):", "    if False:", "a conn record with other fields is accepted"),
    ("W13", W, "        (\"conn after flush\", [{**real, \"ts\": format_ts(flush + timedelta(seconds=1))}], \"conn-outside-window:after-flush\"),\n", "", "control drops the after-flush case"),
    ("W14", W, "    clean = judge([], start, flush)\n    if not clean.ok:", "    clean = judge([], start, flush)\n    if False:", "control drops the negative control"),
    ("W15", W, "doc[\"marker\"] == name and t is not None", "t is not None", "a marker file naming the other marker is accepted"),
    ("W16", W, "    fails += [f\"detector-record-malformed: {e}\" for e in errors if not e.startswith(\"marker\")]\n", "", "malformed detector lines are ignored"),
    # ---- assertions.py: the conn verdict, rows, preconditions, controls, the trace, the overall verdict
    ("A01", A, "policy_times=pol, record_times=times, conn_file_exists=sc.conn_exists, residue=", "conn_file_exists=sc.conn_exists, residue=", "the window's coverage of the audit is not checked"),
    ("A02", A, "if rid(d.get(\"rule\")) not in (\"R1\", \"R4\"):\n                fails.append(f\"allow {d['auditId']} touching", "if False:\n                fails.append(f\"allow {d['auditId']} touching", "8.1 accepts any rule for a local-only allow"),
    ("A03", A, "verdict_row(\"8.1\", \"A local-only codebase makes zero connections outside the allow-list\", fails, bool(wit),", "verdict_row(\"8.1\", \"A local-only codebase makes zero connections outside the allow-list\", fails, True,", "8.1 passes over an empty witness set"),
    ("A04", A, "row(\"8.2\", FAIL if bad else VACUOUS,", "row(\"8.2\", VACUOUS,", "8.2 vacuous even with an R4 allow"),
    ("A05", A, "f83 = [] if ctl[\"C1\"].ok else", "f83 = [] if True else", "8.3 does not need its controls"),
    ("A06", A, "argv[nets[0] + 1] == \"none\"", "argv[nets[0] + 1] != \"\"", "8.5 accepts any --network value"),
    ("A07", A, "f85 += [f\"allow {d['auditId']} has purpose dependency-download\"", "f85 += [] and [f\"allow {d['auditId']} has purpose dependency-download\"", "8.5 ignores a dependency-download allow"),
    ("A08", A, "if digest != (s.get(\"flagCheck\") or {}).get(\"listDigest\"):", "if False:", "8.6 ignores a changed flag list"),
    ("A09", A, "        if hits:\n            f86.append", "        if False:\n            f86.append", "8.6 ignores the re-run's hits"),
    ("A10", A, "            if FORBIDDEN_ENV.fullmatch(k):", "            if False:", "8.6 ignores forbidden env names"),
    ("A12", A, "                if d.get(\"requestMode\") != \"local-only\":", "                if False:", "8.8 ignores requestMode"),
    ("A13", A, "                if d.get(\"decision\") == \"allow\" and rid(d.get(\"rule\")) not in (\"R1\", \"R4\"):", "                if False:", "8.8 ignores an allow's rule"),
    ("A14", A, "f89 = [f\"model-call allow {d['auditId']} has rule {rid(d.get('rule'))}, not R1\" for d in mc if rid(d.get(\"rule\")) != \"R1\"]", "f89 = []", "8.9 accepts any model-call rule"),
    ("A15", A, "row(\"8.10\", PASS if ctl[\"C8\"].ok else FAIL,", "row(\"8.10\", PASS,", "8.10 passes without its policy record"),
    ("A16", A, "row(\"8.11\", PASS if ctl[\"C9\"].ok else FAIL,", "row(\"8.11\", PASS,", "8.11 passes without its policy record"),
    ("A17", A, "             if JAR4IMPORT in s.get(\"toolArgv\", []) and s.get(\"refused\") is not True]", "             if False]", "8.13 ignores a launched --jar4import-url"),
    ("A18", A, "bad14 = [x for x in sc.invalid]", "bad14 = []", "8.14 ignores schema failures"),
    ("A19", A, "if d.get(\"codebases\") == [] or d.get(\"codebases\") is None:", "if False:", "8.15 ignores an under-declaring allow"),
    ("A20", A, "        if n != 1:\n            f_close.append", "        if n < 1:\n            f_close.append", "an allow with two closes passes"),
    ("A21", A, "    if not st.get(\"provisioned\"):\n        return Row(\"8.16\", UNRUN,", "    if False:\n        return Row(\"8.16\", UNRUN,", "8.16 not provisioned is not reported unrun"),
    ("A22", A, "(x[0], x[1]) == key and ev[\"t\"] >= x[2] - 0.002", "x[0] == key[0] and ev[\"t\"] >= x[2] - 0.002", "8.16 ignores the port"),
    ("A23", A, "(x[0], x[1]) == key and ev[\"t\"] >= x[2] - 0.002", "(x[0], x[1]) == key", "8.16 ignores the decision-before-connect order"),
    ("A24", A, "    if not cprobs or (dialed and not any(dialed in c for c in cprobs)):", "    if False:", "8.16 skips its positive control"),
    ("A26", A, "    for t in probe_args.get(\"pos\", []):\n        if not any", "    for t in []:\n        if not any", "P-a skips the store's positive control"),
    ("A27", A, "    for cid in (\"A1d-tcp\", \"A1d-udp\"):\n        bad = ", "    for cid in (\"A1d-tcp\",):\n        bad = ", "P-a ignores UDP"),
    ("A28", A, "    for cid in (\"A2-ahosts\", \"A2-localhost\", \"A2-embedded\"):", "    for cid in (\"A2-localhost\", \"A2-embedded\"):", "P-b ignores DNS"),
    ("A29", A, "\"A3-connect\", \"A3-readonly\", \"A7\"):", "\"A3-connect\", \"A3-readonly\"):", "P-c ignores AF_VSOCK"),
    ("A29b", A, "\"A3-connect\", \"A3-readonly\", \"A7\"):", "\"A3-connect\", \"A7\"):", "P-c ignores the read-only broker directory"),
    ("A30", A, "    if len(probes_logged) != sent:", "    if False:", "P-d ignores the probe-connection count"),
    ("A31", A, "        if not (parts.hostname or \"\").endswith(\".invalid\"):", "        if False:", "P-e ignores a non-.invalid host"),
    ("A32", A, "        if not any(invalid[0][1][\"id\"] in f for f in flagged):", "        if False:", "P-e skips its positive control"),
    ("A33", A, "            if p.get(\"allowListDigest\") != digest:", "            if False:", "P-e skips the allow-list digest"),
    ("A34", A, "    if \"local-only\" not in modes:", "    if False:", "P-f accepts a system with no local-only repository"),
    ("A35", A, "    if not interp:\n        probs.append(\"no R1 allow from answer.interpret\")", "    if False:\n        probs.append(\"no R1 allow\")", "P-g skips the interpret witness"),
    ("A36", A, "        if not any(repo in s.get(\"codebases\", []) for s in launched):", "        if False:", "P-g skips the per-repository sandbox witness"),
    ("A37", A, "    if p[\"checks\"] < ESCAPE_MIN_CHECKS:", "    if False:", "the escape probe's check count is not required"),
    ("A38", A, "    fail = (any(pre.values()) or any(not c.ok for c in controls) or any(r.status == FAIL for r in rows))", "    fail = (any(r.status == FAIL for r in rows))", "the overall verdict ignores preconditions and controls"),
    ("A39", A, "\"overall\": \"fail\" if fail else (\"incomplete\" if reasons else \"pass\"),", "\"overall\": \"fail\" if fail else \"pass\",", "an unrun row or a self-test stand-in still passes the run"),
    ("A40", A, "run_local_only(d)])\n    add(\"C3\"", "True])\n    add(\"C3\"", "C2 does not need the system to be local-only"),
    ("A41", A, "        if ev.get(\"Type\") == \"container\" and (audit_id in json.dumps(attrs) or audit_id in str(ev.get(\"id\", \"\"))):", "        if False:", "C7 ignores docker events"),
    ("A42", A, "    if n == 0:\n        return False, \"docker-events.jsonl is empty", "    if False:\n        return False, \"docker-events.jsonl is empty", "C7 accepts an empty event capture"),
    ("A44", A, "|PIP_.*|UV_.*|", "|UV_.*|", "PIP_* is not a forbidden env name"),
    ("A45", A, "        if self.start is not None and t < self.start:\n            return False", "        if False:\n            return False", "records before the start marker count as in the window"),
    ("A47", A, "            sent += parse_probe(text)[\"sent\"] or 0", "            sent += 0", "the probe-sent total is not counted"),
    ("A48", A, "    return Row(\"8.16\", FAIL if probs else PASS, title,", "    return Row(\"8.16\", PASS, title,", "8.16 passes with violations"),
    # ---- the analyst's 28-mutant review (P1-31 gate, F5): the survivors, rebuilt here (each ESCAPE_REQUIRED id on its own)
    *[(f"E_{cid}", A, f'"{cid}",', "", f"ESCAPE_REQUIRED loses {cid}") for cid in (
        "A1a", "A1b", "A1c", "A2a", "A2b", "A3a", "A3b", "A3c", "A4a:/src", "A4a:/out", "A4a:/tmp", "A4a:/home/ccg", "A4b", "A4c",
        "A5b", "A5c", "A5d")],
    ("Z01", A, "    if not p[\"targets\"] or p[\"targets\"][0] < min_tcp or p[\"targets\"][1] < min_udp:", "    if False:", "P-a's target floor is gone"),
    ("Y06", A, "p[\"targets\"][0] < min_tcp or", "False or", "P-a's tcp target floor is gone"),
    ("Y07", A, "or p[\"targets\"][1] < min_udp:", "or False:", "P-a's udp target floor is gone"),
    ("X06", A, "verdict_row(\"8.7\", \"A question touching only hosted repositories is still interpreted by a local model\", f87, bool(interp),", "verdict_row(\"8.7\", \"A question touching only hosted repositories is still interpreted by a local model\", f87, True,", "8.7 passes over an empty witness set"),
    ("X08", A, "every model call goes to a local endpoint\", f89, bool(mc),", "every model call goes to a local endpoint\", f89, True,", "8.9 passes over an empty witness set"),
    ("X09", A, "No request can opt out of the sticky rule by under-declaring\", f815, n815 > 0,", "No request can opt out of the sticky rule by under-declaring\", f815, True,", "8.15 passes over an empty witness set"),
    ("X10", A, "handled entirely locally\", f88, n88 > 0,", "handled entirely locally\", f88, True,", "8.8 passes over an empty witness set"),
    ("X03", A, "        if self.start is not None and t < self.start:", "        if self.start is not None and t <= self.start:", "in_window drops a record exactly on the start marker"),
    ("X04", A, "        if self.flush is not None and t > self.flush:", "        if self.flush is not None and t >= self.flush:", "in_window drops a record exactly on the flush marker"),
    ("Y03", A, "    times = [t for t in (_ts(r) for r in sc.egress + sc.sandbox) if t is not None]", "    times = [t for t in (_ts(r) for r in sc.egress) if t is not None]", "the sandbox records do not bound the window"),
    ("Z04", A, "    if p[\"fail\"] != sum(1 for v in p[\"ids\"].values() for s, _ in v if s == \"FAIL\"):", "    if False:", "PROBE-DONE's fail count is not cross-checked"),
    ("Y05", A, "    probs += perrs\n", "", "probe.jsonl's malformed lines are ignored"),
    ("Z03", A, "d.get(\"reason\") == \"token-invalid\" and d.get(\"runId\") is None]", "d.get(\"reason\") == \"token-invalid\"]", "C5 does not need the request to have no run"),
    ("X01", A, "return {\"pass\": 0, \"fail\": 1, \"incomplete\": 3}[rep[\"overall\"]]", "return {\"pass\": 0, \"fail\": 1, \"incomplete\": 0}[rep[\"overall\"]]", "the CLI exits 0 on incomplete"),
    ("X02", A, "return {\"pass\": 0, \"fail\": 1, \"incomplete\": 3}[rep[\"overall\"]]", "return {\"pass\": 0, \"fail\": 0, \"incomplete\": 3}[rep[\"overall\"]]", "the CLI exits 0 on fail"),
    # ---- the fix wave: flush residue, nonces/attribution, self-test stand-ins, P-e first, the trace set, empty rows
    ("R01", W, "    elif residue:\n", "    elif False:\n", "a nonzero flush residue passes"),
    ("R02", W, "    if flush is not None and residue is None:", "    if False:", "a flush marker without a residue passes"),
    ("R03", W, "and (name != \"flush\" or _is_int(doc[\"residue\"], 0))", "", "a flush marker's residue is not validated"),
    ("R04", W, "judge(conns, markers[\"start\"], markers[\"flush\"], errors=errs + [f\"marker {e}\" for e in merrs],\n                 policy_times=policy_times, record_times=record_times, conn_file_exists=exists, residue=markers[\"residue\"])", "judge(conns, markers[\"start\"], markers[\"flush\"], errors=errs + [f\"marker {e}\" for e in merrs],\n                 policy_times=policy_times, record_times=record_times, conn_file_exists=exists)", "verdict_for_dir drops the residue"),
    ("R05", A, "conn_file_exists=sc.conn_exists, residue=sc.markers[\"residue\"])", "conn_file_exists=sc.conn_exists)", "the scenario verdict drops the flush residue"),
    ("R06", W, "    if len(conns) != CONTROL_CONNS:", "    if False:", "the control does not count its conns"),
    ("R07", W, "    if [p[\"probe\"] for p in probes] != [CONTROL_PROBE_LABEL]:", "    if False:", "the control does not check its probe record"),
    ("R08", W, "    if markers[\"residue\"] != 0:", "    if False:", "the control ignores its flush residue"),
    ("R09", W, "    if pend.ok or not any(f.startswith(\"flush-residue\") for f in pend.failures):", "    if False:", "the control skips the residue case"),
    ("R10", A, "    for cid in (\"A2-ahosts\", \"A2-localhost\", \"A2-embedded\"):", "    for cid in (\"A2-ahosts\", \"A2-embedded\"):", "P-b drops the getent localhost leg"),
    ("R11", A, "    for cid in (\"A2-ahosts\", \"A2-localhost\", \"A2-embedded\"):", "    for cid in (\"A2-ahosts\", \"A2-localhost\"):", "P-b drops the embedded-DNS leg"),
    ("R12", A, "        [f\"self-test stand-ins were used: {', '.join(selftest)} (an acceptance run has none)\"] if selftest else [])", "        [])", "a self-test run is not incomplete"),
    ("R13", A, "    st_probs = [] if isinstance(recorded, list)", "    st_probs = [] if True or isinstance(recorded, list)", "a missing selfTest list is accepted"),
    ("R14", A, "        if rec[\"srcAddr\"] != e[\"ip\"]:", "        if False:", "a probe record may come from any address"),
    ("R15", A, "        if t is None or not (e[\"t0\"] - PROBE_SLACK <= t <= e[\"t1\"] + PROBE_SLACK):", "        if False:", "a probe record may fall outside its probe's interval"),
    ("R16", A, "        if got != e[\"ports\"]:", "        if False:", "a probe's per-target records are not checked"),
    ("R17", A, "        if e is None:\n            probs.append(f\"probe record with label", "        if False:\n            probs.append(f\"probe record with label", "a record with an unknown label is accepted"),
    ("R18", A, "    probs += probe_attribution_problems(sc, probes_logged)\n", "", "P-d does not attribute probe records"),
    ("R19", A, "    if not launched:\n        f85.append(", "    if False:\n        f85.append(", "8.5 passes over zero launched records"),
    ("R20", A, "        for sec, e, _ in invalid:\n            try:", "        for sec, e, _ in entries:\n            try:", "P-e resolves hosts that are not under .invalid"),
    ("T01", A, "(x for x in pool if not x[3] and", "(x for x in pool if True and", "8.16: one allow legitimises unlimited connects"),
    ("T02", A, "            if ev[\"family\"] not in (\"AF_UNIX\", \"AF_LOCAL\", \"AF_INET\", \"AF_INET6\", \"AF_NETLINK\"):", "            if False:", "8.16 ignores raw/packet sockets"),
    ("T03", A, "    if not ev:\n        probs.append(", "    if False:\n        probs.append(", "8.16 accepts a trace that captured nothing"),
    ("T04", A, "    elif n_net < n_allow:", "    elif False:", "8.16 accepts fewer connects than allows"),
    ("T05", A, "            if call == \"connect\":\n                ev[\"family\"] = \"UNPARSED\"", "            if False:\n                ev[\"family\"] = \"UNPARSED\"", "8.16 skips an unparsable connect"),
    # ---- detector.pl, tested Docker-free by tests/test_ac8_detector.py (loopback, its own ports)
    ("DT1", D, "exists $label{$1} && !$used{$1}++", "0", "every connection is a conn"),
    ("DT2", D, "exists $label{$1} && !$used{$1}++", "($label{$1} //= \"x\") && !$used{$1}++", "any nonce is accepted"),
    ("DT3", D, "exists $label{$1} && !$used{$1}++", "exists $label{$1}", "a nonce may be replayed"),
    ("DT4", D, "my $res = drain();", "my $res = 0;", "the flush does not drain"),
    ("DT5", D, "while (time < $end) { step(0.02);", "for (1 .. 5) { step(0.02);", "the drain is bounded by a count, not wall-clock"),
    ("DT6", D, "O_WRONLY|O_CREAT|O_EXCL", "O_WRONLY|O_CREAT|O_TRUNC", "a marker is rewritten on every signal"),
    ("DT7", D, "if ($want_term) { my $res = drain(); marker('final', ',\"residue\":' . $res); exit 0; }", "if ($want_term) { exit 0; }", "TERM neither drains nor writes the final marker"),
    ("DT12", D, "if ($want_term) { my $res = drain(); marker('final', ',\"residue\":' . $res); exit 0; }", "if ($want_term) { drain(); exit 0; }", "TERM drains but writes no final marker"),
    ("DT13", D, "marker('final', ',\"residue\":' . $res);", "marker('final', ',\"residue\":0');", "the final marker always reports residue 0"),
    ("DT8", D, "marker('flush', ',\"residue\":' . $res);", "marker('flush', ',\"residue\":0');", "the flush marker always reports residue 0"),
    ("DT9", D, "sub queued {", "sub queued { return 0;", "the kernel accept queue is not read"),
    ("DT10", D, "$p->{in} += $n;", "$p->{in} += 0;", "bytesIn is not counted"),
    ("DT11", D, "finish(fileno $h); }   # EOF", "}   # EOF", "a connection that closes is never logged by EOF (only by the deadline)"),
    # ---- lib.sh (Docker-free): the isolation guard and the traps
    ("L01", L, "    0) die \"the bridge $dev carries an address", "    0) : \"the bridge $dev carries an address", "a bridge with an address passes the guard"),
    ("L02", L, "    *) die \"cannot establish whether", "    *) : \"cannot establish whether", "an unknown bridge passes the guard"),
    ("L03", L, "  out=$(\"$ip\" -j address show dev \"$dev\" 2>/dev/null) || return 2", "  out=$(\"$ip\" -j address show dev \"$dev\" 2>/dev/null)", "a failed ip is not unknown"),
    ("L04", L, "sys.exit(0 if any(i[\"addr_info\"] for i in d) else 1)", "sys.exit(1)", "a bridge with an address reads as none"),
    ("L05", L, "if not isinstance(d, list) or not d or not all(", "if not isinstance(d, list) or not all(", "an empty interface list reads as known"),
    ("L06", L, "  trap 'exit 143' TERM", "  trap ac8_cleanup_once TERM", "TERM cleans up but does not exit"),
    ("L07", L, "  trap 'exit 130' INT", "  trap ac8_cleanup_once INT", "INT cleans up but does not exit"),
    ("L08", L, "  [ \"$AC8_CLEANED\" = 1 ] && return 0", "  :", "the cleanup is not idempotent"),
    ("L09", L, "  trap '' INT TERM HUP\n  [ ", "  [ ", "a second signal can cut the cleanup short"),
    # ---- N1/N3/N4: the final marker, the stop timeout and exit status, the control's strace flags, the credential guard
    ("F01", W, "    if final_residue is None:\n", "    if False:\n", "a missing final marker passes"),
    ("F02", W, "    elif final_residue:\n", "    elif False:\n", "a nonzero final residue passes"),
    ("F03", W, "                and (name not in RESIDUE_MARKERS or _is_int(doc[\"residue\"], 0))):", "                and True):", "a final/flush marker's residue is not validated"),
    ("F04", W, "residue=markers[\"residue\"],\n                 final_residue=markers[\"final_residue\"])", "residue=markers[\"residue\"])", "verdict_for_dir drops the final residue"),
    ("F05", A, "residue=sc.markers[\"residue\"],\n                        final_residue=sc.markers[\"final_residue\"])", "residue=sc.markers[\"residue\"])", "the scenario verdict drops the final residue"),
    ("F06", A, "\"window-\", \"flush-\", \"final-\"))]", "\"window-\", \"flush-\"))]", "row 8.1 ignores a final-residue failure"),
    ("F07", W, " or flush is None or markers[\"final\"] is None:", " or flush is None:", "the control accepts a missing final marker"),
    ("F08", W, "    if markers[\"final_residue\"] != 0:", "    if False:", "the control ignores its final residue"),
    ("F09", W, "    if fin.ok or not any(f.startswith(\"final-residue\")", "    if False and not any(f.startswith(\"final-residue\")", "the control skips the final-residue case"),
    ("F10", W, "    if nofin.ok or not any(", "    if False and not any(", "the control skips the missing-final case"),
    ("F11", R, "docker stop -t 40", "docker stop -t 20", "stop timeout is shorter than the drain bound"),
    ("F12", R, "  [ \"$(docker inspect -f '{{.State.ExitCode}}' \"$1\" 2>/dev/null)\" = 0 ] || die", "  true || die", "the detector's exit status is not checked"),
    ("F13", R, "-f --seccomp-bpf -ttt -e trace=%network -o \"$SCEN/trace/control.strace\"", "-f -ttt -e trace=%network -o \"$SCEN/trace/control.strace\"", "N3: control traced without --seccomp-bpf"),
    ("F14", R, "-f --seccomp-bpf -ttt -e trace=%network -o \"$SCEN/trace/broker.strace\"", "-f -ttt -e trace=%network -o \"$SCEN/trace/broker.strace\"", "N3: broker traced without --seccomp-bpf"),
    ("F15", R, "  CCG_FORBID_REAL_CREDENTIALS=1 \"$PY\" -I \"$HERE/assertions.py\" \"${pe[@]}\"", "  \"$PY\" -I \"$HERE/assertions.py\" \"${pe[@]}\"", "N4: P-e without the credential guard"),
    ("F16", R, "env CCG_FORBID_REAL_CREDENTIALS=1 CCG_AUDIT_DIR=\"$SCEN/audit\" \"$PY\" -I \"$HERE/controls.py\"", "env CCG_AUDIT_DIR=\"$SCEN/audit\" \"$PY\" -I \"$HERE/controls.py\"", "N4: controls.py without the credential guard"),
    ("T06", A, "(connect|sendto|sendmsg|sendmmsg|socket)", "(connect|sendto|sendmsg|socket)", "8.16 does not parse sendmmsg"),
]


def main(argv: list[str]) -> int:
    sel = set(argv)
    dry = bool(os.environ.get("MUTANT_DRY"))
    scratch = Path(tempfile.mkdtemp(prefix="p131_mut_"))
    killed, survived, errors = [], [], []
    try:
        for mid, rel, old, new, desc in M:
            if sel and mid not in sel:
                continue
            work = scratch / f"p131_{mid}"
            shutil.copytree(SBX, work / "sandbox", ignore=shutil.ignore_patterns("tests", "__pycache__"))
            target = work / "sandbox" / rel
            text = target.read_text()
            if text.count(old) != 1:
                errors.append(f"{mid}: old text applies {text.count(old)} times")
                continue
            target.write_text(text.replace(old, new))
            if dry:
                print(f"{mid}: edit applies")
                continue
            env = {**os.environ, "CCG_AC8_DIR": str(work / "sandbox" / "ac8"), "PYTHONDONTWRITEBYTECODE": "1"}
            try:
                tests = ["tests/test_ac8_detector.py", "tests/test_ac8_window.py"] if rel == D else ["tests/test_ac8_window.py", "tests/test_ac8_assertions.py", "tests/test_ac8_lib.py"]
                cp = subprocess.run([sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", *tests],
                                    cwd=CPG, env=env, capture_output=True, text=True, timeout=300)
                ok = cp.returncode != 0
                first = next((ln for ln in cp.stdout.splitlines() if ln.startswith("FAILED")), "")
            except subprocess.TimeoutExpired:
                ok, first = True, "timeout"
            (killed if ok else survived).append(mid)
            print(f"{mid} {'killed ' if ok else 'SURVIVED'} {desc} {('-> ' + first[:110]) if first else ''}", flush=True)
            shutil.rmtree(work, ignore_errors=True)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    for e in errors:
        print("ERROR", e)
    print(f"killed {len(killed)}, survived {len(survived)}: {survived}, edit errors {len(errors)}")
    return 1 if survived or errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
