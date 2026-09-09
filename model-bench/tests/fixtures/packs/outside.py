"""A file deliberately living outside any pack root, at `tests/fixtures/packs/` itself, so
`tools_module_outside_root/pack.json` can declare `"module": "../outside.py"` and resolve to a
real file (impl review Pass 12, P12-2, Appendix L.2). Never executed by any test: `validate_pack`
is expected to refuse this manifest before `load_tool_module` is ever asked to import it."""


def build_environment():
    return {"ok": True, "outside": True}
