"""Fixture tool module proving `modelbench.tooling` is on the AST-check allowlist. Not executed
by any test — `modelbench.tooling` does not exist yet (a separate, concurrent S2 unit ships it),
and `validate_pack`'s AST check only parses this file, never imports it."""

from modelbench.tooling import ToolEnvironment  # noqa: F401 — never executed, only AST-walked
