"""Fixture tool module deliberately importing outside packs.py's allowlist (plan §3.3). Not
executed by any test — `validate_pack`'s AST check parses this file, it never imports it."""

from not_an_allowed_package import something  # noqa: F401 — never executed, only AST-walked
