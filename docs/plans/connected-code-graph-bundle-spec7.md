# Connected code graph — fact-bundle spec, `recv` of a shell call site

> **Status:** active · **Owner:** `architect` · **Tracks:** C-1001 (P1 close-out) · **Extends:** `docs/plans/connected-code-graph-bundle-spec3.md` §5.1 (read with `docs/plans/connected-code-graph-bundle-spec4.md` §0), `docs/plans/connected-code-graph-bundle-spec6.md`

Step CO-2 (`docs/plans/connected-code-graph-p1-closeout.md` §3.1). It states only the change, and it is no schema change: `bundle-v1.schema.json` and `ccg/bundle.py` already admit `recv` as a `valueSet` `arg`, and `bundleSchemaVersion` does not move.

## 1. Rule

Replaces the `recv` sentence of `-bundle-spec3.md` §5.1 for a shell call site: for a shell call site whose command word is `<dynamic>` and which is not an `eval`, the shell extractor writes a `valueSet` record with `arg: "recv"` whose `values` are the command word's value, with the same shape and screening as an argument's; no `data` flow records are written for it, and no pack reads it. For every other call site, `recv` is the call's receiver as §5.1 states.
