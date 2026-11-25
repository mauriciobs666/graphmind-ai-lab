"""CO-3 plumbing: a throwaway git repository of shell and Markdown files -> the real build path and assembler (`script_resolver_helpers`)
-> the real `shell-http-client` pack runtime and linker with a HAND-MADE provider fact (the Python provider needs Joern: none here)
-> the real assembler rows with links -> the real answer builder for a contract anchor. No store, no Docker, no network."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from script_resolver_helpers import CPG, assemble, script_symbol
from engine_answer_helpers import AnswerReader
from ccg.assemble.rows import build_rows
from ccg.engine import answer_build as AB
from ccg.engine import anchors
from ccg.linker import link
from ccg.packrt import run_packs
from ccg.rulepack import load_pack
from linker_helpers import prov

SYSTEM = "co3"


def assemble_http(tmp_path: Path, files: dict, provider_script: str, key: str = "GET /health", *, packs=("shell-http-client",)):
    """`files` (a tree with `provider_script` in it); the provider fact is anchored at that script's `<script>` symbol."""
    base = assemble(tmp_path, files)
    loaded = [load_pack(CPG / "packs" / p / "pack.yaml") for p in packs]
    audit = tmp_path / "audit"
    provider_sym = script_symbol(base, provider_script)["symbolId"]
    facts = run_packs(base.bundles, loaded)
    facts = facts._replace(providers=[prov(provider_sym, "http", key)])
    links = link(facts, base.bundles, packs=loaded)
    rows = build_rows(base.bundles, links, facts, system=SYSTEM, packs=loaded, audit_dir=audit)
    return SimpleNamespace(rows=rows, bundles=base.bundles, facts=facts, links=links)


def ask_contract(world, key_norm: str = "GET /health", kind: str = "http") -> dict:
    cp = [c for c in world.rows.nodes["ContractPoint"] if c["keyNorm"] == key_norm and c["kind"] == kind]
    assert len(cp) == 1, cp
    return AB.build_answer(AnswerReader(world), anchors.ContractAnchor(cp[0]["cpId"], cp[0]["kind"], cp[0]["keyNorm"]))
