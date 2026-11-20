import hashlib

from coordgraph.parse.requirements import parse_requirements


def test_requirement_is_its_header_fields_and_nothing_else():
    raw = ("# Req\n\n> **Status:** Ready for design · **Owner:** `tico` · **Tracks:** K-1 (M3) · "
           "**Item:** `GAL-F0002` · **Aliases:** `GAL-F0001`\n\n## Body\n\ntext\n").encode()
    d = parse_requirements("claude/docs/requirements/coordination-graph.md", raw)
    p = d.source_props()
    assert p == {
        "path": "claude/docs/requirements/coordination-graph.md", "kind": "requirement",
        "component": "claude", "slug": "coordination-graph", "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw), "parserVersion": 1, "docStatus": "Ready for design", "owner": "`tico`",
        "tracks": "K-1 (M3)", "itemId": "GAL-F0002", "aliases": ["GAL-F0001"],
    }


def test_each_requirements_status():
    for status in ("Interviewing", "Ready for design", "superseded", "archived", "active"):
        raw = f"# R\n\n> **Status:** {status} · **Owner:** `tico` · **Tracks:** —\n".encode()
        assert parse_requirements("docs/requirements/r.md", raw).source_props()["docStatus"] == status


def test_sha256_is_over_the_raw_bytes_including_invalid_utf8():
    raw = b"# R\n\n> **Status:** active\n\xff\xfe\n"
    d = parse_requirements("docs/requirements/r.md", raw)
    assert d.sha256 == hashlib.sha256(raw).hexdigest() and d.header.doc_status == "active"


def test_headerless_requirement_has_no_status_fields():
    p = parse_requirements("docs/requirements/r.md", b"# R\n\nno header\n").source_props()
    assert not {"docStatus", "owner", "tracks", "itemId", "aliases"} & set(p)
    assert p["component"] == "root"


def test_unusable_item_adds_a_source_warning():
    d = parse_requirements("docs/requirements/r.md", b"# R\n\n> **Status:** active \xc2\xb7 **Item:** `  `\n")
    assert d.warnings and "itemId" not in d.source_props()
