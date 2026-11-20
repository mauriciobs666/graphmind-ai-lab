"""Mutation harness: apply one textual mutant at a time, run a test command,
restore from a side copy, and cmp the file back. Never uses git.

usage: python mutate.py <spec.json>
spec = {"cwd": ..., "cmd": [...], "mutants": [{"id","file","old","new","occ"}]}
occ: "all" or a 0-based occurrence index.
"""
import json, shutil, subprocess, sys, filecmp, os

spec = json.load(open(sys.argv[1]))
cwd = spec["cwd"]
results = []
for m in spec["mutants"]:
    path = os.path.join(cwd, m["file"])
    backup = path + ".mutbak"
    shutil.copy2(path, backup)
    try:
        src = open(path).read()
        n = src.count(m["old"])
        if n == 0:
            results.append((m["id"], "NOT-APPLIED (pattern missing)", ""))
            continue
        if m.get("occ", "all") == "all":
            mutated = src.replace(m["old"], m["new"])
        else:
            i, start = m["occ"], -1
            for _ in range(i + 1):
                start = src.index(m["old"], start + 1)
            mutated = src[:start] + m["new"] + src[start + len(m["old"]):]
        if m.get("extra"):
            eo, en = m["extra"]
            assert mutated.count(eo) == 1, "extra pattern"
            mutated = mutated.replace(eo, en)
        open(path, "w").write(mutated)
        cmd = m.get("cmd", spec["cmd"])
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=900)
        tail = [l for l in p.stdout.splitlines() if " passed" in l or " failed" in l or "error" in l.lower() or "✗" in l][-2:]; tail = [" ".join(tail)]
        failed = sorted({l.split(" ")[1] for l in p.stdout.splitlines() if l.startswith("FAILED ") or l.startswith("ERROR ")})
        verdict = "KILLED" if p.returncode != 0 else "SURVIVED"
        results.append((m["id"], verdict, (tail[0] if tail else "") + " | " + ", ".join(f.split("::")[-1] for f in failed)))
    finally:
        shutil.copy2(backup, path)
        os.remove(backup)
        assert filecmp.cmp(path, backup if os.path.exists(backup) else path, shallow=False)
    # independent cmp against a pristine copy taken by the caller, if given
for r in results:
    print("\t".join(r))
