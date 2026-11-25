"""CO-3 (DEF-1) end to end on synthetic trees: a `curl` in a script, and an inline `curl` span in a prompt, become `impact` items of the
`GET /health` contract through the real extractors, the real `shell-http-client` pack, the real linker, assembler and answer builder.
The provider is a hand-made fact (the Python provider needs Joern). Plan: `docs/plans/connected-code-graph-p1-closeout.md` section 3.3."""
from __future__ import annotations

from co3_helpers import ask_contract, assemble_http

PROMPT = "claude/demo/demo.md"
TREE = {
    "svc/api.sh": "#!/bin/bash\necho serve\n",
    "ops/probe.sh": '#!/bin/bash\nfor i in 1 2 3; do\n  curl -fsS "$BASE_URL/health" >/dev/null && break\ndone\n',
    "ops/other.sh": "#!/bin/bash\ncurl -sf http://127.0.0.1:8200/channels\n",
    PROMPT: "# Demo\n\nPoll `curl -sf http://127.0.0.1:8200/health` until 200.\n\nElsewhere: `curl -sf http://127.0.0.1:8200/workspaces`.\n",
}


def items(doc):
    return {i["subject"]["path"]: i for i in doc["items"]}


def test_a_shell_curl_with_a_dynamic_origin_and_an_inline_markdown_curl_are_items_of_the_contract(tmp_path):
    got = items(ask_contract(assemble_http(tmp_path, TREE, "svc/api.sh")))
    assert {"ops/probe.sh", PROMPT} <= set(got)
    assert got["ops/probe.sh"]["status"] == "possible" and got[PROMPT]["status"] == "possible"
    assert got["ops/probe.sh"]["subject"]["qname"] == "ops/probe.sh::<script>"
    assert got[PROMPT]["subject"]["qname"] == f"{PROMPT}::<document>"


def test_a_curl_of_another_path_is_not_an_item_of_the_contract(tmp_path):
    got = items(ask_contract(assemble_http(tmp_path, TREE, "svc/api.sh")))
    assert "ops/other.sh" not in got
    assert len(got) == 2


def test_without_the_pack_neither_is_an_item(tmp_path):
    """The decoy of the decoys: the same tree and provider with no `shell-http-client` pack links nothing."""
    world = assemble_http(tmp_path, TREE, "svc/api.sh", packs=())
    assert len(world.facts.providers) == 1 and world.links.links == []
