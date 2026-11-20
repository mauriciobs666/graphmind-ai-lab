import hashlib

from coordgraph.parse.kb import claim_hash, duplicate_claims, normalize_text, parse_kb

KB = """# Techniques

Intro, not a section.

## First claim

Body one.
More body.

## Second: with `code` & punctuation

```
## not a heading, inside a fence
```

Body two.
"""


def kb(text, path="claude/a/kb.md"):
    return parse_kb(path, text.encode())


def test_splits_at_h2_outside_fences_and_ignores_the_preamble():
    d = kb(KB)
    assert [s.heading for s in d.sections] == ["First claim", "Second: with `code` & punctuation"]
    assert [s.ord for s in d.sections] == [0, 1]
    assert d.sections[0].body == "Body one.\nMore body."
    assert "## not a heading, inside a fence" in d.sections[1].body
    assert d.sha256 == hashlib.sha256(KB.encode()).hexdigest()


def test_whole_mode_is_one_claim_of_the_body_titled_by_the_heading():
    s = kb(KB).sections[0]
    assert s.mode == "whole" and len(s.claims) == 1
    c = s.claims[0]
    assert (c.text, c.title, c.key) == ("Body one.\nMore body.", "First claim", "whole")
    assert c.claim_hash == hashlib.sha256(b"body one. more body.").hexdigest()


def test_claim_hash_uses_falkor_chats_normaliser():
    assert normalize_text("  Foo \n\tBAR  ") == "foo bar"
    assert claim_hash("Foo   Bar") == claim_hash("\nfoo bar\n")
    assert claim_hash("Straße") == hashlib.sha256("strasse".encode()).hexdigest()  # casefold, not lower


def test_section_hash_changes_with_the_body_only():
    a = kb("## H\n\nbody\n").sections[0].section_hash
    assert a == kb("## H\n\nbody\n\n\n").sections[0].section_hash  # trailing blanks do not count
    assert a != kb("## H\n\nbody!\n").sections[0].section_hash
    assert a == kb("## H renamed\n\nbody\n").sections[0].section_hash


def test_heading_keys_are_slugs_with_a_suffix_for_a_repeated_heading():
    d = kb("## Same Heading!\n\na\n\n## Same Heading!\n\nb\n")
    assert [s.heading_key for s in d.sections] == ["same-heading", "same-heading~2"]
    assert any("duplicate heading" in w for w in d.warnings)
    assert kb("## ???\n\nx\n").sections[0].heading_key == "section"


BULLETS = """## Many claims <!-- kb-split: bullets -->

- First bullet, one claim.
  Continued line.
- Second bullet.

  Still second, after a blank.
1. Numbered.
"""


def test_bullets_mode_one_claim_per_top_level_bullet_keyed_by_hash():
    s = kb(BULLETS).sections[0]
    assert s.heading == "Many claims" and s.mode == "bullets"
    assert [c.text for c in s.claims] == [
        "- First bullet, one claim.\n  Continued line.",
        "- Second bullet.\n\n  Still second, after a blank.",
        "1. Numbered.",
    ]
    assert [c.title for c in s.claims] == ["First bullet, one claim.", "Second bullet.", "Numbered."]
    assert all(c.key == f"b-{c.claim_hash[:12]}" for c in s.claims)
    assert len({c.key for c in s.claims}) == 3


def test_bullet_key_is_the_normalised_claim_hash_so_equal_claims_share_it():
    a = kb("## A <!-- kb-split: bullets -->\n\n- Same   Claim\n").sections[0].claims[0]
    b = kb("## B <!-- kb-split: bullets -->\n\n- same claim\n").sections[0].claims[0]
    assert a.key == b.key == "b-" + hashlib.sha256(b"- same claim").hexdigest()[:12]


def test_marker_alone_under_the_heading_also_selects_bullets_and_is_not_claim_text():
    s = kb("## H\n\n<!-- kb-split: bullets -->\n- a\n- b\n").sections[0]
    assert s.mode == "bullets" and [c.text for c in s.claims] == ["- a", "- b"]
    assert "kb-split" not in s.body  # the marker line is stripped from the body


def test_unknown_split_mode_warns_and_stays_whole():
    d = kb("## H <!-- kb-split: paragraphs -->\n\ntext\n")
    assert d.sections[0].mode == "whole" and any("paragraphs" in w for w in d.warnings)


def test_non_bullet_lines_in_a_bullets_section_are_reported_not_dropped_silently():
    d = kb("## H <!-- kb-split: bullets -->\n\nA stray paragraph.\n- a\n")
    assert any("1 non-bullet line" in w for w in d.warnings)


def test_bullets_inside_a_code_fence_do_not_split():
    s = kb("## H <!-- kb-split: bullets -->\n\n- a\n  ```\n  - not a bullet\n  ```\n- b\n").sections[0]
    assert len(s.claims) == 2


def test_empty_section_warns():
    d = kb("## Empty\n\n## Next\n\ntext\n")
    assert any("empty section" in w for w in d.warnings) and len(d.sections[1].claims) == 1


def test_duplicate_claim_text_across_sections_is_detected_by_normalised_hash():
    a = kb("## A\n\nSame   text.\n", "claude/a/kb.md")
    b = kb("## B\n\nsame text.\n\n## C\n\nOther.\n", "claude/b/kb.md")
    dups = duplicate_claims([a, b])
    assert list(dups.values()) == [[("claude/a/kb.md", "A"), ("claude/b/kb.md", "B")]]
    assert duplicate_claims([b]) == {}
