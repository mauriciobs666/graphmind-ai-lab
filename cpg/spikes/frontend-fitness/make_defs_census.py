#!/usr/bin/env python3
"""S27c step 1: the ML §5.3 definitions census -> `truth/<lang>.defs.jsonl` (no Joern).

    cd cpg && .venv/bin/python spikes/frontend-fitness/make_defs_census.py <lang>|all [--check]

Rule sources (they control; this module implements them and never restates a number):
  * docs/plans/connected-code-graph-p0-ml.md v1.15 §5.3 (census truths), §7.2 (metric definitions), §2 (symbol-ness);
  * docs/test-reports/connected-code-graph-p0-frontend-fitness-report.md §a1.6 (operating rules);
  * docs/plans/connected-code-graph-bundle-spec3.md §4.1 rule 8 as amended by -spec4.md (which bodyless forms are symbols);
  * matcher-table.md + `s27_metrics.spec_names` (the accepted Joern names: extended in `s27_census_metrics.py`, never redefined);
  * `run_fitness.py` (frame file walk `iter_source_files`/`is_test_file`, grammars `parse_file`, `frame_sites`, `Lines`, `ws`).

Records (one JSON object per line; readers filter on `k`):
  {"k":"meta","lang","version":1,"files","parseErrorFiles","defs","args"}
  {"k":"file","path","parseErrors"}            (one per frame file: the parse-success denominator)
  {"k":"def","path","name","nameKey","nameLine","endLine","declStartLine","symbol","decorators":[{"name","node","literalArgs":[{"name","value"}]}],
   "qname","kind","anon","declNode"}
  {"k":"arg","path","line","column","callText","argIndex","argName","valueKind","value","siteKey","via"}
`qname` (extra) is `<path>::<chain>` in the spec's dotted style with constructors as `<init>`/`__init__`/`constructor`, so that
`s27_metrics.spec_names(qname)` yields the accepted Joern names (last segment, plus the enclosing type for a constructor).
`name` is None for an anonymous definition (lambda, unbound function expression); `nameLine` is then the node's start line and the
metrics script matches it by position only.

Judgments where §a1.6 leaves a choice (each consistent with ML §7.2; the residual guesses reported to the caller):
  1. Qname join key. The metrics join is (file, name rule, line); `qname` is carried only to derive the accepted-name set via
     `spec_names`. No cross-file qname identity is used.
  2. `Foo.BAR` references are NOT constant arguments: (d) is a bare identifier, as written in §a1.6 and ML §5.3. A bare identifier
     resolves in this order: an enclosing class's constant (Java/Kotlin/C#/C++; Kotlin companion members count as the class's;
     not Python/JS/Go, where a class attribute is not visible as a bare name), then a file-level constant. Python resolves
     module-level `UPPER_CASE` only; JS/TS top-level `const`; Go package-level `const`; Java `final`/`static final`; Kotlin `val`;
     C# `const`; C++ `const`/`constexpr`. A name bound more than once in the same scope is not a constant.
  3. Parameter/local shadowing: ANY binding of the name (parameter, local, loop/catch/with variable, nested function name,
     destructuring leaf) anywhere in the outermost enclosing function disqualifies the argument. Deliberately flow-insensitive
     and conservative: a use before the local's declaration is still disqualified.
  4. Python literals: a bytes literal (`b"x"`) normalises to its text; a raw literal (`r"\\d"`) keeps its backslashes; an f-string
     with no `{}` substitution is a string literal (value = its text, `{{` -> `{`), one with a substitution is not a literal.
     Implicit concatenation `"a" "b"` is a concatenation (valueKind `concat`), as is C/C++ adjacent-literal concatenation.
  5. Decorator literal arguments are string-valued literals (a)-(c) only, no constant identifiers, numbers or booleans; a Java/
     Kotlin array initialiser contributes each string element under the pair's key. Decorator `node` is the tree-sitter node
     type (Python/TS `decorator`; Java `annotation`/`marker_annotation`; Kotlin `annotation`; C# `attribute`; C++
     `attribute_declaration`; Go none). Python decorators are only recorded on the `function_definition` they decorate.
     Metrics wording: ML's "share" is per DEFINITION for the name share (every decorator name of the definition exposed) and per
     DECORATOR for the literal share (over decorators with >= 1 literal argument, all their literal values exposed); a per-decorator
     name share is reported beside the primary.
  6. Arg index: 0-based over the explicit arguments, receiver excluded; a spread/variadic argument keeps its slot (Joern numbers AST
     children) but gets no record; a keyword/named argument keeps its positional index and sets `argName`.
  7. Out of scope (documented, not silently dropped): property accessors with bodies (Kotlin getters/setters, C# accessors), default
     constructors with no source, Go generic instantiation sites (frame kind `other`), JSX.
  8. Anonymous definitions (lambda, arrow/function expression with no binding, Go `func` literal, C++ lambda, C# lambda/delegate,
     Java lambda, Kotlin lambda/anonymous function) are recorded with `name: None`, `symbol: true` (bundle spec: `<anon>#N` symbols).
     An arrow/function expression bound to a variable, field or object key takes that binding as name (run_fitness `_js_binding`).
  9. Java record header `<init>` (only when the record declares no explicit canonical/compact constructor) and implicit accessors
     (a component with no explicit no-arg method of its name) are `symbol:true`; their spans are the header's component list and
     the component entry. Kotlin/C# primary constructors likewise (span: the parameter list; nameLine: the type name's line).
 10. Probe file location for the S30 decorator-node worked examples: `results/<lang>.decorator-probe.json`, `{lang,node,example}`;
     written by graph-dba's measure.sc unit, not here.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_fitness as rf  # noqa: E402

sys.setrecursionlimit(20000)

LANGS = ["python", "javascript", "go", "java", "csharp", "cpp", "kotlin"]
VERSION = 1


def tx(n) -> str:
    return n.text.decode("utf-8", "replace") if n is not None else ""


def fld(n, name):
    return n.child_by_field_name(name)


def kids(n, *types):
    return [c for c in n.children if c.type in types]


def first(n, *types):
    for c in n.children:
        if c.type in types:
            return c
    return None


# ---------------------------------------------------------------- literal normalisation (shared with the metrics script)

_SIMPLE_ESC = {"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f", "v": "\v", "a": "\a", "0": "\0",
               "\\": "\\", '"': '"', "'": "'", "/": "/", "`": "`", "$": "$", "?": "?"}


def decode_escapes(s: str, keep_unknown: bool = False) -> str:
    """Decode the C-family backslash escapes common to the seven languages."""
    out, i, n = [], 0, len(s)
    while i < n:
        c = s[i]
        if c != "\\" or i + 1 >= n:
            out.append(c)
            i += 1
            continue
        d = s[i + 1]
        if d == "\n":
            i += 2
        elif d in "01234567" and re.match(r"[0-7]{1,3}", s[i + 1:]):
            m = re.match(r"[0-7]{1,3}", s[i + 1:])
            out.append(chr(int(m.group(0), 8)))
            i += 1 + len(m.group(0))
        elif d == "x" and re.match(r"[0-9a-fA-F]{2}", s[i + 2:]):
            out.append(chr(int(s[i + 2:i + 4], 16)))
            i += 4
        elif d == "u" and s[i + 2:i + 3] == "{":
            j = s.find("}", i)
            out.append(chr(int(s[i + 3:j], 16)))
            i = j + 1
        elif d == "u" and re.match(r"[0-9a-fA-F]{4}", s[i + 2:]):
            out.append(chr(int(s[i + 2:i + 6], 16)))
            i += 6
        elif d == "U" and re.match(r"[0-9a-fA-F]{8}", s[i + 2:]):
            out.append(chr(int(s[i + 2:i + 10], 16)))
            i += 10
        elif d in _SIMPLE_ESC:
            out.append(_SIMPLE_ESC[d])
            i += 2
        else:
            out.append("\\" + d if keep_unknown else d)
            i += 2
    return "".join(out)


def _dedent_text_block(body: str) -> str:
    lines = body.split("\n")
    if lines and lines[0].strip() == "":
        lines = lines[1:]
    sig = [l for l in lines if l.strip()] + ([lines[-1]] if lines else [])
    ind = min((len(l) - len(l.lstrip()) for l in sig), default=0)
    return "\n".join(l[ind:].rstrip() if i < len(lines) - 1 else l[ind:] for i, l in enumerate(lines))


def normalise_literal(lang: str, s: str):
    """Source text of one string literal (delimiters included) -> normalised value, or None if it is not a plain literal."""
    s = s.strip()
    if lang == "python":
        m = re.fullmatch(r"([A-Za-z]{0,3})(\"\"\"|'''|\"|')(.*)\2", s, re.S)
        if not m:
            return None
        pre, body = m.group(1).lower(), m.group(3)
        if "f" in pre:
            body = body.replace("{{", "{").replace("}}", "}")
        return body if "r" in pre else decode_escapes(body, keep_unknown=True)
    if lang == "javascript":
        if s[:1] in "\"'`" and s[-1:] == s[:1] and len(s) >= 2:
            return decode_escapes(s[1:-1])
        return None
    if lang == "java":
        if s.startswith('"""') and s.endswith('"""') and len(s) >= 6:
            return decode_escapes(_dedent_text_block(s[3:-3]))
        if s[:1] == '"' and s[-1:] == '"' and len(s) >= 2:
            return decode_escapes(s[1:-1])
        return None
    if lang == "kotlin":
        if s.startswith('"""') and s.endswith('"""') and len(s) >= 6:
            return s[3:-3]
        if s[:1] == '"' and s[-1:] == '"' and len(s) >= 2:
            return decode_escapes(s[1:-1])
        return None
    if lang == "csharp":
        m = re.fullmatch(r'(\$?@?|@?\$?)("{3,})(.*)\2', s, re.S)
        if m and len(m.group(2)) >= 3:
            return m.group(3)
        m = re.fullmatch(r'(\$?@|@\$)"(.*)"', s, re.S)
        if m:
            return m.group(2).replace('""', '"').replace("{{", "{").replace("}}", "}") if "$" in m.group(1) else m.group(2).replace('""', '"')
        m = re.fullmatch(r'\$"(.*)"', s, re.S)
        if m:
            return decode_escapes(m.group(1).replace("{{", "{").replace("}}", "}"))
        if s[:1] == '"' and s[-1:] == '"' and len(s) >= 2:
            return decode_escapes(s[1:-1])
        return None
    if lang == "go":
        if s[:1] == "`" and s[-1:] == "`" and len(s) >= 2:
            return s[1:-1].replace("\r", "")
        if s[:1] == '"' and s[-1:] == '"' and len(s) >= 2:
            return decode_escapes(s[1:-1])
        return None
    if lang == "cpp":
        m = re.fullmatch(r'(?:u8|u|U|L)?R"([^()\\ ]{0,16})\((.*)\)\1"', s, re.S)
        if m:
            return m.group(2)
        m = re.fullmatch(r'(?:u8|u|U|L)?"(.*)"', s, re.S)
        if m:
            return decode_escapes(m.group(1))
        return None
    raise ValueError(lang)


# ---------------------------------------------------------------- literal-valued expressions (a)-(c)

STRING_NODES = {
    "python": {"string"}, "javascript": {"string"}, "java": {"string_literal"}, "kotlin": {"string_literal"},
    "csharp": {"string_literal", "verbatim_string_literal", "raw_string_literal"},
    "go": {"interpreted_string_literal", "raw_string_literal"}, "cpp": {"string_literal", "raw_string_literal"},
}
TEMPLATE_NODES = {"javascript": {"template_string"}, "csharp": {"interpolated_string_expression"}}
CONCAT_NODES = {"python": {"binary_operator"}, "javascript": {"binary_expression"}, "java": {"binary_expression"},
                "kotlin": {"binary_expression", "additive_expression"}, "csharp": {"binary_expression"},
                "go": {"binary_expression"}, "cpp": set()}
PAREN = {"parenthesized_expression", "parenthesized"}
_KT_TEMPLATE = re.compile(r"(?<!\\)\$(?=[A-Za-z_{`])")


def _has_op(n, op="+"):
    o = fld(n, "operator")
    if o is not None:
        return tx(o) == op
    return any((not c.is_named) and tx(c) == op for c in n.children)


def literal_value(lang: str, n):
    """(valueKind, value) when `n` is (a) a string literal, (b) a literal-only concatenation, (c) a substitution-free template
    literal; else None. The value is normalised (decoded escapes, delimiters/prefixes stripped, concatenation folded)."""
    if n is None:
        return None
    t = n.type
    if t in PAREN:
        inner = [c for c in n.named_children if c.type != "comment"]
        return literal_value(lang, inner[0]) if len(inner) == 1 else None
    if t in STRING_NODES[lang]:
        if lang == "python" and first(n, "interpolation"):
            return None
        if lang == "kotlin" and (first(n, "interpolation") or _KT_TEMPLATE.search(tx(n))):
            return None
        v = normalise_literal(lang, tx(n))
        return ("string", v) if v is not None else None
    if t in TEMPLATE_NODES.get(lang, ()):
        if first(n, "template_substitution", "interpolation"):
            return None
        v = normalise_literal(lang, tx(n))
        return ("template", v) if v is not None else None
    if lang == "python" and t == "concatenated_string":
        parts = [literal_value(lang, c) for c in n.named_children if c.type != "comment"]
        if parts and all(p is not None for p in parts):
            return ("concat", "".join(p[1] for p in parts))
        return None
    if lang == "cpp" and t == "concatenated_string":
        parts = [literal_value(lang, c) for c in n.named_children if c.type != "comment"]
        if parts and all(p is not None for p in parts):
            return ("concat", "".join(p[1] for p in parts))
        return None
    if t in CONCAT_NODES[lang] and _has_op(n, "+"):
        a, b = fld(n, "left"), fld(n, "right")
        if a is None or b is None:
            ops = [c for c in n.named_children if c.type != "comment"]
            if len(ops) != 2:
                return None
            a, b = ops
        la, lb = literal_value(lang, a), literal_value(lang, b)
        if la is not None and lb is not None:
            return ("concat", la[1] + lb[1])
    return None


# ---------------------------------------------------------------- calls and arguments

CALL_NODES = {
    "python": {"call"}, "javascript": {"call_expression", "new_expression"},
    "java": {"method_invocation", "object_creation_expression", "explicit_constructor_invocation"},
    "kotlin": {"call_expression", "constructor_invocation"}, "csharp": {"invocation_expression", "object_creation_expression"},
    "go": {"call_expression"}, "cpp": {"call_expression", "new_expression"},
}


def callee_span(lang: str, n):
    """(start, end) bytes of the callee expression, mirroring `run_fitness.frame_sites` (which stays the authority on what is a site)."""
    t = n.type
    if lang in ("python", "go", "cpp") and t == "call_expression" or lang == "python" and t == "call":
        f = fld(n, "function")
        return (f.start_byte, f.end_byte) if f is not None else None
    if lang == "javascript":
        if t == "call_expression":
            f = fld(n, "function")
            return (f.start_byte, f.end_byte) if f is not None else None
        c = fld(n, "constructor")
        return (n.start_byte, c.end_byte) if c is not None else None
    if lang == "java":
        if t == "method_invocation":
            nm = fld(n, "name")
            return (n.start_byte, nm.end_byte) if nm is not None else None
        if t == "object_creation_expression":
            ty = fld(n, "type")
            return (n.start_byte, ty.end_byte) if ty is not None else None
        kw = [c for c in n.children if c.type in ("this", "super")]
        return (n.start_byte, kw[-1].end_byte) if kw else None
    if lang == "kotlin":
        c = n.children[0] if n.children else None
        return (c.start_byte, c.end_byte) if c is not None else None
    if lang == "csharp":
        if t == "invocation_expression":
            f = fld(n, "function")
            return (f.start_byte, f.end_byte) if f is not None else None
        ty = fld(n, "type")
        return (n.start_byte, ty.end_byte) if ty is not None else None
    if lang == "cpp" and t == "new_expression":
        ty = fld(n, "type")
        return (n.start_byte, ty.end_byte) if ty is not None else None
    return None


def call_args(lang: str, n):
    """[(arg_expression_node | None for a spread slot, argName | None)] in source order, receiver excluded."""
    out = []
    if lang == "python":
        al = fld(n, "arguments")
        if al is None or al.type != "argument_list":
            return out
        for c in al.named_children:
            if c.type == "comment":
                continue
            if c.type == "keyword_argument":
                out.append((fld(c, "value"), tx(fld(c, "name"))))
            elif c.type in ("list_splat", "dictionary_splat", "parenthesized_list_splat"):
                out.append((None, None))
            else:
                out.append((c, None))
    elif lang == "kotlin":
        va = first(n, "value_arguments")
        if va is None:
            cs = first(n, "call_suffix")
            va = first(cs, "value_arguments") if cs is not None else None
        if va is None:
            return out
        for c in kids(va, "value_argument"):
            parts = [x for x in c.children if x.type not in ("comment",)]
            spread = any(x.type == "*" for x in parts)
            name = None
            if len(parts) >= 3 and parts[0].type == "identifier" and parts[1].type == "=":
                name = tx(parts[0])
            expr = c.named_children[-1] if c.named_children else None
            if name is not None and len(c.named_children) < 2:
                expr = None
            out.append((None if spread else expr, name))
    elif lang == "csharp":
        al = fld(n, "arguments")
        if al is None:
            al = first(n, "argument_list")
        if al is None:
            return out
        for c in kids(al, "argument"):
            nm = fld(c, "name")
            nc = first(c, "name_colon")
            name = tx(nm) if nm is not None else (tx(nc.named_children[0]) if nc is not None and nc.named_children else None)
            exprs = [x for x in c.named_children if x.type not in ("name_colon", "comment") and x is not nm]
            out.append((exprs[-1] if exprs else None, name))
    else:
        al = fld(n, "arguments")
        if al is None:
            al = first(n, "argument_list", "arguments")
        if al is None or al.type not in ("argument_list", "arguments"):
            return out
        for c in al.named_children:
            if c.type == "comment":
                continue
            if c.type in ("spread_element", "variadic_argument"):
                out.append((None, None))
            else:
                out.append((c, None))
    return out


# ---------------------------------------------------------------- constants (d) and shadowing

FN_NODES = {
    "python": {"function_definition", "lambda"},
    "javascript": {"function_declaration", "generator_function_declaration", "function_expression", "function", "arrow_function",
                   "method_definition", "generator_function"},
    "java": {"method_declaration", "constructor_declaration", "compact_constructor_declaration", "lambda_expression",
             "static_initializer"},
    "kotlin": {"function_declaration", "secondary_constructor", "lambda_literal", "anonymous_function", "getter", "setter",
               "anonymous_initializer"},
    "csharp": {"method_declaration", "constructor_declaration", "destructor_declaration", "operator_declaration",
               "conversion_operator_declaration", "local_function_statement", "lambda_expression", "anonymous_method_expression",
               "accessor_declaration"},
    "go": {"function_declaration", "method_declaration", "func_literal"},
    "cpp": {"function_definition", "lambda_expression"},
}
CLASS_NODES = {
    "python": {"class_definition"},
    "javascript": {"class_declaration", "abstract_class_declaration", "class", "interface_declaration", "enum_declaration"},
    "java": {"class_declaration", "interface_declaration", "enum_declaration", "record_declaration", "annotation_type_declaration"},
    "kotlin": {"class_declaration", "object_declaration", "companion_object"},
    "csharp": {"class_declaration", "struct_declaration", "interface_declaration", "record_declaration",
               "record_struct_declaration", "enum_declaration"},
    "go": set(), "cpp": {"class_specifier", "struct_specifier", "union_specifier"},
}
# lexical scopes whose bodies hold file/class-level constants
_CLASS_SCOPED = {"java", "kotlin", "csharp", "cpp"}


def _idents(n, out, skip_types=()):
    """Every identifier leaf under a binding pattern."""
    if n is None:
        return
    if n.type in ("identifier", "shorthand_property_identifier_pattern", "shorthand_property_identifier"):
        out.add(tx(n))
        return
    if n.type in skip_types:
        return
    for c in n.children:
        _idents(c, out, skip_types)


def _cpp_decl_ident(d, out):
    while d is not None:
        if d.type in ("identifier", "field_identifier"):
            out.add(tx(d))
            return
        if d.type in ("pointer_declarator", "reference_declarator", "array_declarator", "parenthesized_declarator",
                      "init_declarator", "attributed_declarator", "function_declarator", "structured_binding_declarator"):
            if d.type == "structured_binding_declarator":
                _idents(d, out)
                return
            nxt = fld(d, "declarator") or (d.named_children[-1] if d.named_children else None)
            d = nxt
        else:
            return


def locals_of(lang: str, fn) -> set:
    """Every name bound anywhere inside function-like node `fn` (parameters included). Flow-insensitive on purpose (module doc 3)."""
    out: set = set()
    stack = [fn]
    while stack:
        n = stack.pop()
        t = n.type
        if lang == "python":
            if t == "parameters" or t == "lambda_parameters":
                for c in n.named_children:
                    if c.type == "identifier":
                        out.add(tx(c))
                    else:
                        _idents(c if c.type not in ("default_parameter", "typed_default_parameter") else (fld(c, "name") or c), out,
                                skip_types=("type", "string"))
            elif t in ("assignment", "augmented_assignment"):
                _idents(fld(n, "left"), out, skip_types=("attribute", "subscript"))
            elif t in ("for_statement", "for_in_clause"):
                _idents(fld(n, "left"), out, skip_types=("attribute", "subscript"))
            elif t == "named_expression":
                _idents(fld(n, "name"), out)
            elif t in ("as_pattern_target",):
                _idents(n, out)
            elif t == "except_clause" or t == "with_item":
                for c in n.children:
                    if c.type == "as_pattern":
                        _idents(fld(c, "alias") or c.children[-1], out)
            elif t in ("function_definition", "class_definition") and n is not fn:
                nm = fld(n, "name")
                if nm is not None:
                    out.add(tx(nm))
            elif t in ("import_statement", "import_from_statement"):
                for c in n.named_children:
                    if c.type == "aliased_import":
                        out.add(tx(fld(c, "alias")))
                    elif c.type == "dotted_name" and t == "import_statement":
                        out.add(tx(c).split(".")[0])
        elif lang == "javascript":
            if t == "variable_declarator":
                _idents(fld(n, "name"), out)
            elif t in ("required_parameter", "optional_parameter"):
                _idents(fld(n, "pattern"), out)
            elif t in ("formal_parameters",):
                for c in n.named_children:
                    if c.type == "identifier":
                        out.add(tx(c))
                    elif c.type in ("assignment_pattern", "rest_pattern", "object_pattern", "array_pattern"):
                        _idents(fld(c, "left") if c.type == "assignment_pattern" else c, out)
            elif t == "arrow_function" and fld(n, "parameter") is not None:
                _idents(fld(n, "parameter"), out)
            elif t == "catch_clause":
                _idents(fld(n, "parameter"), out)
            elif t == "for_in_statement":
                _idents(fld(n, "left"), out)
            elif t in ("function_declaration", "generator_function_declaration", "class_declaration") and n is not fn:
                nm = fld(n, "name")
                if nm is not None:
                    out.add(tx(nm))
        elif lang == "java":
            if t == "variable_declarator" and (n.parent is None or n.parent.type != "field_declaration"):
                out.add(tx(fld(n, "name")))
            elif t in ("formal_parameter", "spread_parameter", "catch_formal_parameter", "receiver_parameter"):
                nm = fld(n, "name") or first(n, "identifier") or first(n, "variable_declarator")
                if nm is not None:
                    out.add(tx(nm if nm.type != "variable_declarator" else fld(nm, "name")))
            elif t == "enhanced_for_statement":
                nm = fld(n, "name")
                if nm is not None:
                    out.add(tx(nm))
            elif t == "inferred_parameters":
                for c in n.named_children:
                    out.add(tx(c))
            elif t == "lambda_expression":
                p = fld(n, "parameters")
                if p is not None and p.type == "identifier":
                    out.add(tx(p))
            elif t in ("resource", "pattern"):
                nm = fld(n, "name")
                if nm is not None:
                    out.add(tx(nm))
            elif t == "type_pattern":
                for c in n.named_children:
                    if c.type == "identifier":
                        out.add(tx(c))
        elif lang == "kotlin":
            if t == "variable_declaration":
                nm = first(n, "identifier")
                if nm is not None:
                    out.add(tx(nm))
            elif t in ("parameter", "class_parameter"):
                nm = first(n, "identifier")
                if nm is not None:
                    out.add(tx(nm))
            elif t == "function_declaration" and n is not fn:
                nm = fld(n, "name")
                if nm is not None:
                    out.add(tx(nm))
        elif lang == "csharp":
            if t == "variable_declarator" and (n.parent is None or n.parent.parent is None or n.parent.parent.type != "field_declaration"):
                nm = fld(n, "name") or first(n, "identifier")
                if nm is not None:
                    out.add(tx(nm))
            elif t == "parameter":
                nm = fld(n, "name")
                if nm is not None:
                    out.add(tx(nm))
            elif t == "foreach_statement":
                _idents(fld(n, "left"), out)
            elif t == "catch_declaration":
                nm = fld(n, "name")
                if nm is not None:
                    out.add(tx(nm))
            elif t in ("single_variable_designation", "declaration_pattern"):
                _idents(n, out)
            elif t == "local_function_statement" and n is not fn:
                out.add(tx(fld(n, "name")))
            elif t == "lambda_expression":
                p = fld(n, "parameters")
                if p is not None and p.type == "identifier":
                    out.add(tx(p))
        elif lang == "go":
            if t == "short_var_declaration":
                _idents(fld(n, "left"), out)
            elif t in ("var_spec", "const_spec"):
                for c in n.children:
                    if c.type == "identifier":
                        out.add(tx(c))
            elif t in ("parameter_declaration", "variadic_parameter_declaration"):
                for c in n.children:
                    if c.type == "identifier":
                        out.add(tx(c))
            elif t == "range_clause":
                _idents(fld(n, "left"), out)
            elif t == "type_switch_statement":
                _idents(fld(n, "alias"), out)
        elif lang == "cpp":
            if t in ("init_declarator", "parameter_declaration", "optional_parameter_declaration", "declaration",
                     "for_range_loop", "condition_declaration"):
                _cpp_decl_ident(fld(n, "declarator"), out)
            elif t == "structured_binding_declarator":
                _idents(n, out)
        stack.extend(n.children)
    return out


_UPPER = re.compile(r"[A-Z][A-Z0-9_]*")


def _scope_consts(lang: str, scope_node, data) -> dict:
    """{name: (valueKind, value)} for the constants bound directly in `scope_node` (a file root or a class), duplicates dropped."""
    found: dict = {}
    dup: set = set()

    def add(name, expr):
        if name in found or name in dup:
            found.pop(name, None)
            dup.add(name)
            return
        lv = literal_value(lang, expr) if expr is not None else None
        if lv is None:
            dup.add(name)  # bound, but not a literal constant: it still occupies the name
            return
        found[name] = lv

    def body_children(n):
        if lang == "kotlin":
            b = first(n, "class_body", "enum_class_body")
            return list(b.children) if b is not None else (list(n.children) if n.type == "source_file" else [])
        if lang == "java":
            b = fld(n, "body")
            return list(b.children) if b is not None else list(n.children)
        if lang == "csharp":
            b = fld(n, "body")
            return list(b.children) if b is not None else list(n.children)
        if lang == "cpp":
            b = fld(n, "body")
            return list(b.children) if b is not None else list(n.children)
        return list(n.children)

    for c in body_children(scope_node):
        t = c.type
        if lang == "python":
            a = first(c, "assignment") if t == "expression_statement" else None
            if a is not None:
                left = fld(a, "left")
                if left is not None and left.type == "identifier" and _UPPER.fullmatch(tx(left)) and fld(a, "right") is not None:
                    add(tx(left), fld(a, "right"))
        elif lang == "javascript":
            d = c.child_by_field_name("declaration") if t == "export_statement" else c
            if d is not None and d.type == "lexical_declaration" and tx(fld(d, "kind")) == "const":
                for v in kids(d, "variable_declarator"):
                    nm = fld(v, "name")
                    if nm is not None and nm.type == "identifier":
                        add(tx(nm), fld(v, "value"))
        elif lang == "java":
            if t == "field_declaration":
                mods = first(c, "modifiers")
                mt = {x.type for x in mods.children} if mods is not None else set()
                in_iface = scope_node.type in ("interface_declaration", "annotation_type_declaration")
                if "final" in mt or in_iface:
                    for v in kids(c, "variable_declarator"):
                        add(tx(fld(v, "name")), fld(v, "value"))
        elif lang == "kotlin":
            if t == "property_declaration" and first(c, "val") is not None:
                vd = first(c, "variable_declaration")
                nm = first(vd, "identifier") if vd is not None else None
                ex = None
                seen_eq = False
                for x in c.children:
                    if x.type == "=":
                        seen_eq = True
                    elif seen_eq and x.is_named and x.type != "comment":
                        ex = x
                        break
                if nm is not None:
                    add(tx(nm), ex)
            elif t == "companion_object" and scope_node.type == "class_declaration":
                for k, v in _scope_consts(lang, c, data).items():
                    add_pre = found.get(k)
                    if k in found or k in dup:
                        found.pop(k, None)
                        dup.add(k)
                    else:
                        found[k] = v
        elif lang == "csharp":
            if t == "field_declaration":
                mods = {tx(x) for x in kids(c, "modifier")}
                if "const" in mods:
                    vd = first(c, "variable_declaration")
                    for v in (kids(vd, "variable_declarator") if vd is not None else []):
                        nm = fld(v, "name") or first(v, "identifier")
                        ex = [x for x in v.named_children if x is not nm and x.type != "bracketed_argument_list"]
                        add(tx(nm), ex[-1] if ex else None)
        elif lang == "go":
            if t == "const_declaration":
                specs = list(kids(c, "const_spec"))
                for g in kids(c, "const_spec_list"):
                    specs.extend(kids(g, "const_spec"))
                for sp in specs:
                    names = [x for x in sp.children if x.type == "identifier"]
                    val = fld(sp, "value")
                    vals = val.named_children if val is not None and val.type == "expression_list" else ([val] if val is not None else [])
                    for i, nm in enumerate(names):
                        add(tx(nm), vals[i] if i < len(vals) and len(vals) == len(names) else None)
        elif lang == "cpp":
            cc = c
            if t in ("namespace_definition",):
                for k, v in _scope_consts(lang, c, data).items():
                    if k in found or k in dup:
                        found.pop(k, None)
                        dup.add(k)
                    else:
                        found[k] = v
                continue
            if t in ("declaration", "field_declaration"):
                qs = {tx(x) for x in kids(cc, "type_qualifier")}
                if qs & {"const", "constexpr"}:
                    for v in cc.named_children:
                        if v.type == "init_declarator":
                            d = fld(v, "declarator")
                            tmp: set = set()
                            _cpp_decl_ident(d, tmp)
                            for nm in tmp:
                                add(nm, fld(v, "value"))
                    dv = fld(cc, "default_value")
                    if dv is not None:
                        tmp = set()
                        _cpp_decl_ident(fld(cc, "declarator"), tmp)
                        for nm in tmp:
                            add(nm, dv)
    return found


class ConstResolver:
    """Resolves a bare identifier argument to a same-file constant (module doc 2 and 3)."""

    def __init__(self, lang: str, root, data):
        self.lang, self.root, self.data = lang, root, data
        self._scope: dict = {}
        self._locals: dict = {}
        self.file_consts = _scope_consts(lang, root, data)

    def _consts(self, node):
        k = node.id
        if k not in self._scope:
            self._scope[k] = _scope_consts(self.lang, node, self.data)
        return self._scope[k]

    def resolve(self, ident):
        lang = self.lang
        name = tx(ident)
        outer_fn, classes = None, []
        p = ident.parent
        while p is not None:
            if p.type in FN_NODES[lang]:
                outer_fn = p
            if p.type in CLASS_NODES[lang]:
                classes.append(p)
            p = p.parent
        if outer_fn is not None:
            if outer_fn.id not in self._locals:
                self._locals[outer_fn.id] = locals_of(lang, outer_fn)
            if name in self._locals[outer_fn.id]:
                return None
        if lang in _CLASS_SCOPED:
            for c in classes:
                if lang == "kotlin" and c.type == "companion_object":
                    continue
                cc = self._consts(c)
                if name in cc:
                    return cc[name]
        return self.file_consts.get(name)


def build_args(lang: str, rel: str, data: bytes, key: str, root, L) -> list:
    """`arg` records of one file (the frame's siteKey, line and column are reused so the Joern join is by siteKey)."""
    sites = rf.frame_sites(lang, key, root, data)
    sites.sort(key=lambda s: (s.start, s.end))
    seen: dict = {}
    keyed: dict = {}
    for s in sites:
        line, col = L.lc(s.start)
        ct = rf.ws(data[s.start:s.end].decode("utf-8", "replace"))
        sk = f"{rel}:{line}:{col}:{ct}"
        c = seen.get(sk, 0) + 1
        seen[sk] = c
        if c > 1:
            sk = f"{sk}#{c}"
        keyed.setdefault((s.start, s.end, s.node_type), []).append((sk, line, col, ct))
    resolver = ConstResolver(lang, root, data)
    out = []
    taken: dict = {}
    stack = [root]
    nodes = []
    while stack:
        n = stack.pop()
        if n.type in CALL_NODES[lang]:
            nodes.append(n)
        stack.extend(n.children)
    nodes.sort(key=lambda n: n.start_byte)
    for n in nodes:
        span = callee_span(lang, n)
        if span is None:
            continue
        cands = keyed.get((span[0], span[1], n.type))
        if not cands:
            continue
        i = taken.get((span, n.type), 0)
        if i >= len(cands):
            continue
        taken[(span, n.type)] = i + 1
        sk, line, col, ct = cands[i]
        for idx, (expr, name) in enumerate(call_args(lang, n)):
            if expr is None:
                continue
            lv = literal_value(lang, expr)
            via = None
            if lv is None and expr.type == "identifier":
                cv = resolver.resolve(expr)
                if cv is not None:
                    lv, via = ("const", cv[1]), tx(expr)
            if lv is None:
                continue
            out.append({"k": "arg", "path": rel, "line": line, "column": col, "callText": ct, "argIndex": idx, "argName": name,
                        "valueKind": lv[0], "value": lv[1], "siteKey": sk, "via": via})
    return out


# ---------------------------------------------------------------- decorators

def _pair_values(lang, v):
    """String-valued literals under an annotation/attribute argument value (array initialisers flattened one level)."""
    lv = literal_value(lang, v)
    if lv is not None:
        return [lv[1]]
    if v is not None and v.type in ("element_value_array_initializer", "array_creation_expression", "initializer_expression",
                                    "collection_expression", "array", "list"):
        return [x[1] for x in (literal_value(lang, c) for c in v.named_children) if x is not None]
    return []


def _deco(node_type, name, args):
    return {"name": name, "node": node_type, "literalArgs": [{"name": a, "value": v} for a, v in args]}


def _call_deco(lang, node_type, expr):
    """decorator expression -> record (Python/TS: identifier, dotted name or call)."""
    if expr.type in ("call", "call_expression"):
        f = fld(expr, "function")
        args = []
        for a, nm in call_args(lang, expr) if lang == "python" else _ts_args(expr):
            for v in _pair_values(lang, a):
                args.append((nm, v))
        return _deco(node_type, re.sub(r"\s+", "", tx(f)), args)
    return _deco(node_type, re.sub(r"\s+", "", tx(expr)), [])


def _ts_args(call):
    al = fld(call, "arguments")
    return [(c, None) for c in (al.named_children if al is not None else []) if c.type != "comment"]


def decorators_of(lang: str, n) -> tuple:
    """([decorator record], first_line_node) for definition node `n`; the second is the earliest decorating node (declStart)."""
    out, nodes = [], []
    if lang == "python":
        par = n.parent
        if par is not None and par.type == "decorated_definition":
            for d in kids(par, "decorator"):
                expr = d.named_children[0] if d.named_children else None
                if expr is not None:
                    out.append(_call_deco(lang, "decorator", expr))
                    nodes.append(d)
    elif lang == "javascript":
        ds = list(kids(n, "decorator"))
        p = n.prev_sibling
        pre = []
        while p is not None and p.type == "decorator":
            pre.append(p)
            p = p.prev_sibling
        for d in list(reversed(pre)) + ds:
            expr = d.named_children[0] if d.named_children else None
            if expr is not None:
                out.append(_call_deco(lang, "decorator", expr))
                nodes.append(d)
    elif lang == "java":
        mods = first(n, "modifiers")
        for a in (kids(mods, "annotation", "marker_annotation") if mods is not None else []):
            args = []
            al = fld(a, "arguments")
            for c in (al.named_children if al is not None else []):
                if c.type == "element_value_pair":
                    for v in _pair_values(lang, fld(c, "value")):
                        args.append((tx(fld(c, "key")), v))
                else:
                    for v in _pair_values(lang, c):
                        args.append((None, v))
            out.append(_deco(a.type, re.sub(r"\s+", "", tx(fld(a, "name"))), args))
            nodes.append(a)
    elif lang == "kotlin":
        mods = first(n, "modifiers")
        for a in (kids(mods, "annotation") if mods is not None else []):
            ci = first(a, "constructor_invocation")
            ut = first(ci, "user_type") if ci is not None else first(a, "user_type")
            args = []
            va = first(ci, "value_arguments") if ci is not None else None
            for c in (kids(va, "value_argument") if va is not None else []):
                nm = tx(c.children[0]) if len(c.children) >= 3 and c.children[0].type == "identifier" and c.children[1].type == "=" else None
                for v in _pair_values(lang, c.named_children[-1] if c.named_children else None):
                    args.append((nm, v))
            out.append(_deco("annotation", re.sub(r"\s+", "", tx(ut)) if ut is not None else "", args))
            nodes.append(a)
        # tree-sitter-kotlin quirk: an annotation on a declaration that follows a statement with no terminator is parsed as
        # an `annotated_expression` (annotation + parenthesised argument) sibling of the declaration. Names and positional
        # string arguments are recovered; named arguments are lost in the ERROR it produces (documented limitation).
        p = n.prev_sibling
        while p is not None and p.type == "comment":
            p = p.prev_sibling
        if not out and p is not None and p.type == "annotated_expression":
            for a in kids(p, "annotation"):
                ut = first(a, "user_type")
                args = []
                for pe in kids(p, "parenthesized_expression"):
                    for v in _pair_values(lang, pe.named_children[0] if pe.named_children else None):
                        args.append((None, v))
                out.append(_deco("annotation", re.sub(r"\s+", "", tx(ut)) if ut is not None else "", args))
                nodes.append(p)
    elif lang == "csharp":
        for al in kids(n, "attribute_list"):
            for a in kids(al, "attribute"):
                args = []
                aa = first(a, "attribute_argument_list")
                for c in (kids(aa, "attribute_argument") if aa is not None else []):
                    nm = fld(c, "name")
                    nm = tx(nm) if nm is not None else None
                    ex = [x for x in c.named_children if x.type not in ("name_colon", "name_equals", "comment") and not (nm and tx(x) == nm and x.type == "identifier" and x == fld(c, "name"))]
                    for v in _pair_values(lang, ex[-1] if ex else None):
                        args.append((nm, v))
                out.append(_deco("attribute", re.sub(r"\s+", "", tx(fld(a, "name"))), args))
                nodes.append(al)
    elif lang == "cpp":
        for ad in kids(n, "attribute_declaration"):
            for a in kids(ad, "attribute"):
                args = []
                al = first(a, "argument_list")
                for c in (al.named_children if al is not None else []):
                    for v in _pair_values(lang, c):
                        args.append((None, v))
                nm = fld(a, "name")
                pre = fld(a, "prefix")
                out.append(_deco("attribute_declaration", (tx(pre) + "::" if pre is not None else "") + tx(nm), args))
                nodes.append(ad)
    return out, nodes


# ---------------------------------------------------------------- definitions

TYPE_NODES = {
    "python": {"class_definition"},
    "javascript": {"class_declaration", "abstract_class_declaration", "class", "interface_declaration", "enum_declaration"},
    "java": {"class_declaration", "interface_declaration", "enum_declaration", "record_declaration", "annotation_type_declaration"},
    "kotlin": {"class_declaration", "object_declaration", "companion_object"},
    "csharp": {"class_declaration", "struct_declaration", "interface_declaration", "record_declaration",
               "record_struct_declaration", "enum_declaration"},
    "go": set(), "cpp": {"class_specifier", "struct_specifier", "union_specifier"},
}
ANON_NODES = {
    "python": {"lambda"}, "javascript": set(), "java": {"lambda_expression"}, "kotlin": {"lambda_literal", "anonymous_function"},
    "csharp": {"lambda_expression", "anonymous_method_expression"}, "go": {"func_literal"}, "cpp": {"lambda_expression"},
}


def js_bind_node(n):
    par = n.parent
    while par is not None and par.type in ("parenthesized_expression", "as_expression", "satisfies_expression", "non_null_expression"):
        n, par = par, par.parent
    if par is None:
        return None
    if par.type == "variable_declarator" and fld(par, "value") == n:
        nm = fld(par, "name")
        return nm if nm is not None and nm.type == "identifier" else None
    if par.type in ("public_field_definition", "field_definition") and fld(par, "value") == n:
        return fld(par, "name") or fld(par, "property")
    if par.type == "pair" and fld(par, "value") == n:
        k = fld(par, "key")
        if k is not None:
            kt = tx(k).strip("\"'`") if k.type == "string" else tx(k)
            if re.fullmatch(r"[A-Za-z_$][\w$]*", kt):
                return k
    return None


def _strip_targs(seg: str) -> str:
    return re.sub(r"<[^<>]*(<[^<>]*>[^<>]*)*>", "", seg)


def cpp_func_declarator(d):
    while d is not None and d.type in ("pointer_declarator", "reference_declarator", "parenthesized_declarator", "attributed_declarator"):
        d = fld(d, "declarator") or (d.named_children[-1] if d.named_children else None)
    return d if d is not None and d.type == "function_declarator" else None


def cpp_name(fd):
    """(qualifier segments, base name, name node) of a function_declarator; operators as written, destructors with `~`."""
    nn = fld(fd, "declarator")
    if nn is None:
        return [], None, None
    last = nn
    while last is not None and last.type == "qualified_identifier":
        last = fld(last, "name") or (last.named_children[-1] if last.named_children else None)
    while last is not None and last.type in ("template_function", "template_method"):
        last = fld(last, "name") or last.named_children[0]
    segs = [s for s in re.split(r"::", re.sub(r"\s+(?=::)|(?<=::)\s+", "", tx(nn))) if s != ""]
    out = []
    for s in segs:
        s = s.strip()
        if not s.startswith("operator"):
            s = _strip_targs(s)
        else:
            s = re.sub(r"\s+", " ", s)
            s = re.sub(r"(?<=[^\w ]) | (?=[^\w ])", "", s)
        out.append(re.sub(r"\s+", "", s) if not s.startswith("operator") else s)
    if not out:
        return [], None, last
    return out[:-1], out[-1], last


def census_file(lang: str, rel: str, data: bytes) -> tuple[list, list, int]:
    """(def records, arg records, tree-sitter error count) of one file."""
    key, tree = rf.parse_file(lang, rel, data)
    root = tree.root_node
    L = rf.Lines(data)
    defs: list = []
    TN, AN = TYPE_NODES[lang], ANON_NODES[lang]

    def mk(ctx, n, segs, name, nkey, name_node, symbol, kind, end_node=None, start_node=None, decos=(), deco_nodes=(), anon=False):
        sn = start_node or n
        nl = L.lc((name_node or n).start_byte)[0]
        ds = min([L.lc(sn.start_byte)[0], nl] + [L.lc(d.start_byte)[0] for d in deco_nodes])
        en = end_node or n
        el = L.lc(max(en.end_byte - 1, en.start_byte))[0]
        defs.append({"k": "def", "path": rel, "name": None if anon else name, "nameKey": None if anon else nkey, "nameLine": nl,
                     "endLine": el, "declStartLine": ds, "symbol": symbol, "decorators": list(decos),
                     "qname": f"{rel}::{'.'.join(ctx[0] + tuple(segs))}", "kind": kind, "anon": anon, "declNode": n.type,
                     "_s": n.start_byte})

    def type_name(n):
        if lang == "kotlin":
            nm = fld(n, "name") or first(n, "identifier")
            return tx(nm) if nm is not None else ("Companion" if n.type == "companion_object" else "?")
        nm = fld(n, "name")
        return _strip_targs(tx(nm)) if nm is not None else "<anon>"

    def named(ctx, n, name_node, kind, symbol=True, nkey=None, segs=None, decos=None, **kw):
        name = tx(name_node)
        d, dn = decorators_of(lang, n) if decos is None else decos
        mk(ctx, n, segs or [name], name, nkey or name, name_node, symbol, kind, decos=d, deco_nodes=dn, **kw)

    def scan(n, ctx):
        """-> new ctx for the children (None: unchanged)."""
        t = n.type
        chain, types, in_fn = ctx
        if t in AN:
            mk(ctx, n, ["<anon>"], None, None, None, True, "anon", anon=True)
            return (chain + ("<anon>",), types, True)
        if t in TN:
            nm = type_name(n)
            if lang == "kotlin" and t == "class_declaration" and first(n, "primary_constructor") is not None:
                pc = first(n, "primary_constructor")
                nmn = fld(n, "name") or first(n, "identifier")
                mk(ctx, n, [nm, "<init>"], "<init>", "<init>", nmn, True, "ctor", end_node=pc, start_node=pc)
            if lang == "csharp" and fld(n, "name") is not None and first(n, "parameter_list") is not None:
                pl = first(n, "parameter_list")
                mk(ctx, n, [nm, "<init>"], "<init>", "<init>", fld(n, "name"), True, "ctor", end_node=pl, start_node=pl)
            if lang == "java" and t == "record_declaration":
                java_record(ctx, n, nm)
            return (chain + (nm,), types + (nm,), in_fn)
        if lang == "python":
            if t == "function_definition":
                nm = fld(n, "name")
                d, dn = decorators_of(lang, n)
                ov = any(x["name"].split(".")[-1] == "overload" for x in d)
                named(ctx, n, nm, "ctor" if tx(nm) == "__init__" else "fn", symbol=not ov, decos=(d, dn),
                      start_node=n.parent if n.parent is not None and n.parent.type == "decorated_definition" else n)
                return (chain + (tx(nm),), types, True)
        elif lang == "javascript":
            if t in ("function_declaration", "generator_function_declaration"):
                named(ctx, n, fld(n, "name"), "fn")
                return (chain + (tx(fld(n, "name")),), types, True)
            if t == "function_signature":
                amb = n.parent is not None and n.parent.type == "ambient_declaration"
                named(ctx, n, fld(n, "name"), "signature", symbol=False)
                return None
            if t in ("method_definition", "method_signature", "abstract_method_signature"):
                nmn = fld(n, "name")
                if t == "method_definition":
                    sym, kind = True, "fn"
                elif t == "abstract_method_signature":
                    sym, kind = True, "abstract"
                else:
                    p = n.parent
                    sym = p is not None and (p.type == "interface_body" or (
                        p.type == "object_type" and p.parent is not None and p.parent.type == "type_alias_declaration"
                        and fld(p.parent, "value") == p))
                    kind = "signature"
                if nmn is None or nmn.type == "computed_property_name":
                    mk(ctx, n, ["<anon>"], None, None, None, sym, kind, anon=True, decos=decorators_of(lang, n)[0],
                       deco_nodes=decorators_of(lang, n)[1])
                    return (chain + ("<anon>",), types, True) if t == "method_definition" else None
                nm = tx(nmn).strip("\"'")
                if nm == "constructor":
                    kind = "ctor"
                d, dn = decorators_of(lang, n)
                mk(ctx, n, [nm], nm, nm, nmn, sym, kind, decos=d, deco_nodes=dn)
                return (chain + (nm,), types, True) if t == "method_definition" else None
            if t in ("function_expression", "function", "arrow_function", "generator_function"):
                own = fld(n, "name")
                b = own if own is not None else js_bind_node(n)
                if b is None:
                    mk(ctx, n, ["<anon>"], None, None, None, True, "anon", anon=True)
                    return (chain + ("<anon>",), types, True)
                nm = tx(b).strip("\"'`")
                mk(ctx, n, [nm], nm, nm, b, True, "fn")
                return (chain + (nm,), types, True)
        elif lang == "java":
            if t in ("method_declaration", "annotation_type_element_declaration"):
                named(ctx, n, fld(n, "name"), "fn")
                return (chain + (tx(fld(n, "name")),), types, True)
            if t in ("constructor_declaration", "compact_constructor_declaration"):
                named(ctx, n, fld(n, "name"), "ctor", nkey="<init>", segs=["<init>"])
                return (chain + ("<init>",), types, True)
        elif lang == "kotlin":
            if t == "function_declaration":
                nmn = fld(n, "name") or first(n, "identifier")
                mods = first(n, "modifiers")
                expect = mods is not None and any(tx(x) == "expect" for x in _walk(mods))
                nm = tx(nmn).strip("`")
                d, dn = decorators_of(lang, n)
                mk(ctx, n, [nm], nm, nm, nmn, not expect, "fn", decos=d, deco_nodes=dn)
                return (chain + (nm,), types, True)
            if t == "secondary_constructor":
                d, dn = decorators_of(lang, n)
                mk(ctx, n, ["<init>"], "<init>", "<init>", first(n, "constructor"), True, "ctor", decos=d, deco_nodes=dn)
                return (chain + ("<init>",), types, True)
        elif lang == "csharp":
            if t in ("method_declaration", "local_function_statement"):
                body = fld(n, "body")
                partial = any(tx(m) == "partial" for m in kids(n, "modifier"))
                named(ctx, n, fld(n, "name"), "fn", symbol=not (partial and body is None))
                return (chain + (tx(fld(n, "name")),), types, True)
            if t == "constructor_declaration":
                named(ctx, n, fld(n, "name"), "ctor", nkey="<init>", segs=["<init>"])
                return (chain + ("<init>",), types, True)
            if t == "destructor_declaration":
                nm = "~" + tx(fld(n, "name"))
                d, dn = decorators_of(lang, n)
                mk(ctx, n, [nm], nm, nm, fld(n, "name"), True, "fn", decos=d, deco_nodes=dn)
                return (chain + (nm,), types, True)
            if t in ("operator_declaration", "conversion_operator_declaration"):
                op = fld(n, "operator")
                nm = "operator" + (tx(op) if op is not None else " " + tx(fld(n, "type")))
                d, dn = decorators_of(lang, n)
                mk(ctx, n, [nm], nm, nm, op or n, True, "fn", decos=d, deco_nodes=dn)
                return (chain + (nm,), types, True)
        elif lang == "go":
            if t == "function_declaration":
                named(ctx, n, fld(n, "name"), "fn")
                return (chain + (tx(fld(n, "name")),), types, True)
            if t == "method_declaration":
                rt = [x for x in _walk(fld(n, "receiver")) if x.type == "type_identifier"] if fld(n, "receiver") is not None else []
                recv = tx(rt[0]) if rt else "?"
                nm = tx(fld(n, "name"))
                mk(ctx, n, [recv, nm], nm, nm, fld(n, "name"), True, "fn")
                return (chain + (recv, nm), types, True)
            if t == "method_elem" and fld(n, "name") is not None:
                it = n.parent
                owner = it.parent if it is not None else None
                named_iface = it is not None and it.type == "interface_type" and owner is not None and owner.type in ("type_spec", "type_alias")
                iface = tx(fld(owner, "name")) if named_iface else "<iface>"
                nm = tx(fld(n, "name"))
                mk(ctx, n, [iface, nm], nm, nm, fld(n, "name"), named_iface, "spec")
                return None
        elif lang == "cpp":
            return scan_cpp(n, ctx)
        return None

    def java_record(ctx, n, nm):
        params = fld(n, "parameters")
        comps = kids(params, "formal_parameter") if params is not None else []
        body = fld(n, "body")
        members = body.children if body is not None else []
        ctypes = [re.sub(r"\s+", "", tx(fld(c, "type"))) for c in comps]
        explicit = False
        for m in members:
            if m.type == "compact_constructor_declaration":
                explicit = True
            elif m.type == "constructor_declaration":
                mp = fld(m, "parameters")
                mt = [re.sub(r"\s+", "", tx(fld(p, "type"))) for p in (kids(mp, "formal_parameter") if mp is not None else [])]
                if mt == ctypes:
                    explicit = True
        if not explicit and params is not None:
            mk(ctx, n, [nm, "<init>"], "<init>", "<init>", fld(n, "name"), True, "ctor", end_node=params, start_node=params)
        for c in comps:
            cn = tx(fld(c, "name"))
            has = any(m.type == "method_declaration" and tx(fld(m, "name")) == cn and not kids(fld(m, "parameters"), "formal_parameter")
                      for m in members)
            if not has:
                mk(ctx, c, [nm, cn], cn, cn, fld(c, "name"), True, "accessor")

    def scan_cpp(n, ctx):
        chain, types, in_fn = ctx
        t = n.type
        if t == "function_definition":
            fd = cpp_func_declarator(fld(n, "declarator"))
            if fd is None:
                return (chain, types, True)
            quals, base, nn = cpp_name(fd)
            if base is None:
                return (chain, types, True)
            encl = types[-1] if types else None
            ctor = base == (quals[-1] if quals else encl)
            seg = "<init>" if ctor else base
            d, dn = decorators_of(lang, n)
            deleted = first(n, "delete_method_clause") is not None
            par = n.parent
            mk(ctx, n, quals + [seg], base, "<init>" if ctor else base, nn, not deleted, "ctor" if ctor else "fn", decos=d,
               deco_nodes=dn, start_node=par if par is not None and par.type == "template_declaration" else n)
            return (chain + tuple(quals) + (seg,), types, True)
        if t in ("declaration", "field_declaration") and not in_fn:
            fd = cpp_func_declarator(fld(n, "declarator"))
            if fd is not None:
                quals, base, nn = cpp_name(fd)
                if base is not None:
                    dv = fld(n, "default_value")
                    pure = (dv is not None and tx(dv).strip() == "0") or first(n, "pure_virtual_clause") is not None
                    friend = n.parent is not None and n.parent.type == "friend_declaration"
                    encl = types[-1] if types else None
                    ctor = base == (quals[-1] if quals else encl)
                    mk(ctx, n, quals + ["<init>" if ctor else base], base, "<init>" if ctor else base, nn, bool(pure) and not friend,
                       "pure" if pure else ("friend" if friend else "prototype"))
            return None
        if t == "preproc_function_def":
            nm = fld(n, "name")
            mk(ctx, n, [tx(nm)], tx(nm), tx(nm), nm, False, "macro")
            return None
        return None

    def rec(n, ctx):
        if not n.is_named:
            return
        new = scan(n, ctx) or ctx
        for c in n.children:
            rec(c, new)

    rec(root, ((), (), False))
    defs.sort(key=lambda d: d.pop("_s"))
    return defs, build_args(lang, rel, data, key, root, L), rf.count_errors(root)


def _walk(n):
    stack = [n]
    while stack:
        x = stack.pop()
        yield x
        stack.extend(reversed(x.children))


# ---------------------------------------------------------------- driver

def build(lang: str, root: Path | None = None, cfg: dict | None = None) -> tuple[list, dict]:
    root = root or rf.src_dir(lang)
    cfg = cfg if cfg is not None else rf.corpus_cfg(lang)
    recs, nfiles, nerr = [], 0, 0
    for rel, p in rf.iter_source_files(lang, root, cfg):
        if rf.is_test_file(lang, rel):
            continue
        d, a, e = census_file(lang, rel, p.read_bytes())
        recs.append({"k": "file", "path": rel, "parseErrors": e})
        recs.extend(d)
        recs.extend(a)
        nfiles += 1
        nerr += 1 if e else 0
    meta = {"k": "meta", "lang": lang, "version": VERSION, "files": nfiles, "parseErrorFiles": nerr,
            "defs": sum(1 for r in recs if r["k"] == "def"), "args": sum(1 for r in recs if r["k"] == "arg")}
    return [meta] + recs, meta


def out_path(lang: str) -> Path:
    return HERE / "truth" / f"{lang}.defs.jsonl"


def check_truth(lang: str, recs: list) -> dict:
    """Acceptance (b): every (path, defLine) that is a T member in final2 or a reference static/impls target with a body is a
    `symbol:true` def (matched on nameLine or declStartLine). Reports counts and the misses."""
    syms: dict = {}
    for r in recs:
        if r["k"] == "def" and r["symbol"]:
            syms.setdefault(r["path"], set()).update({r["nameLine"], r["declStartLine"]})
    wanted: set = set()
    f2 = HERE / "truth" / f"{lang}.final2.jsonl"
    if f2.exists():
        for l in open(f2, encoding="utf-8"):
            if l.strip():
                for t in json.loads(l).get("T", []):
                    wanted.add((t["path"], t["defLine"], t.get("qname") or ""))
    ref = HERE / "truth" / f"{lang}.ref.jsonl"
    if ref.exists():
        for l in open(ref, encoding="utf-8"):
            if l.strip():
                j = json.loads(l)
                for t in [j.get("static", {})] + list(j.get("impls", [])):
                    if t.get("kind", "internal") == "internal" and "defLine" in t:
                        wanted.add((t["path"], t["defLine"], t.get("qname") or ""))
    root = rf.src_dir(lang)
    type_line = re.compile(r"^\s*(?:[\w@\[\]<>,.() ]+\s+)?(class|struct|interface|record|enum|type|object|union)\s+\w")
    lines_cache: dict = {}

    def is_type_decl(path, line):
        if path not in lines_cache:
            try:
                lines_cache[path] = (root / path).read_text(encoding="utf-8", errors="replace").split("\n")
            except OSError:
                lines_cache[path] = []
        ls = lines_cache[path]
        return 0 < line <= len(ls) and bool(type_line.search(ls[line - 1]))

    miss = sorted(w for w in wanted if w[1] not in syms.get(w[0], ()))
    types = [w for w in miss if is_type_decl(w[0], w[1])]
    real = [w for w in miss if w not in types]
    return {"targets": len(wanted), "typeDeclTargets": len(types), "missingNonType": len(real), "examples": real[:12]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("lang", help="|".join(LANGS) + "|all")
    ap.add_argument("--check", action="store_true", help="also run acceptance (b) against final2/ref")
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args(argv)
    for lang in (LANGS if a.lang == "all" else [a.lang]):
        recs, meta = build(lang)
        if not a.no_write:
            with open(out_path(lang), "w", encoding="utf-8") as f:
                for r in recs:
                    f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
        info = dict(meta)
        info["symbolDefs"] = sum(1 for r in recs if r["k"] == "def" and r["symbol"])
        if a.check:
            info["acceptanceB"] = check_truth(lang, recs)
        print(json.dumps(info, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
