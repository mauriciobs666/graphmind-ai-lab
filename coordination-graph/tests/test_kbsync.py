"""The KB mirror engine (`kbsync`) against in-memory fakes (`tests/kbfakes.py`): append-before-delete, the
budget and `pending`, the per-section lease with L1-L3, cancel-on-re-edit, duplicate refusal and the receipt
check, split modes, `needs-resplit`, reconcile, status. Each test names the alternative it rules out."""
from __future__ import annotations

import pytest

from coordgraph import kbsync
from coordgraph.kbsync import Context, KbHttpError, section_key, status
from coordgraph.parse.kb import claim_hash, parse_kb
from tests.kbfakes import make_engine

P = "claude/a/kb.md"
ONE = "# T\n\n## Alpha\n\nAlpha body.\n"
TWO = "# T\n\n## Alpha\n\nAlpha body.\n\n## Beta\n\nBeta body.\n"
BULLETS = ("# T\n\n## Facts\n<!-- kb-split: bullets -->\n\n- one fact\n- two fact\n- three fact\n")


def kb(text, path=P):
    return parse_kb(path, text.encode())


def sync(engine, mirror, text, path=P):
    doc = kb(text, path)
    rows, _ = engine.read_all()
    outcomes = engine.sync_file(doc, Context.build([doc], rows), rows)
    return {o.heading: o for o in outcomes}, doc


def row(mirror, doc, heading):
    s = next(s for s in doc.sections if s.heading == heading)
    return mirror.sections[section_key(doc.path, s)]


def test_a_new_section_is_posted_once_under_the_kb_mirror_identity_and_mapped():
    engine, mirror, chat, _ = make_engine()
    out, doc = sync(engine, mirror, ONE)
    assert out["Alpha"].status == "synced"
    assert chat.posts() == ["Alpha"] and chat.deletes() == []
    [d] = chat.current()
    r = row(mirror, doc, "Alpha")
    assert r["documentIds"] == [d["id"]] and r["claimHashes"] == [claim_hash("Alpha body.")]
    assert r["state"] == "synced" and "leaseToken" not in r and "pendingDocumentIds" not in r
    assert mirror.files[P]["sha256"] == doc.sha256


def test_an_unchanged_section_makes_no_falkor_chat_call():
    engine, mirror, chat, _ = make_engine()
    sync(engine, mirror, ONE)
    before = list(chat.calls)
    out, _ = sync(engine, mirror, ONE)
    assert out["Alpha"].status == "unchanged" and chat.calls == before


def test_a_changed_section_posts_before_it_deletes_exactly_the_old_id():
    engine, mirror, chat, _ = make_engine()
    out, doc = sync(engine, mirror, TWO)
    old_alpha = row(mirror, doc, "Alpha")["documentIds"][0]
    old_beta = row(mirror, doc, "Beta")["documentIds"][0]
    chat.calls.clear()
    out, doc = sync(engine, mirror, TWO.replace("Alpha body.", "Alpha body, edited."))
    assert [c[0] for c in chat.calls] == ["post", "delete"]  # append before delete
    assert chat.deletes() == [old_alpha] and old_beta in chat.docs  # not the other section's, not the new one
    assert row(mirror, doc, "Alpha")["documentIds"] == [d["id"] for d in chat.current() if d["title"] == "Alpha"]
    assert len(chat.current()) == 2


def test_a_whitespace_or_case_only_edit_keeps_the_document_and_updates_the_hash():
    engine, mirror, chat, _ = make_engine()
    _, doc = sync(engine, mirror, ONE)
    chat.calls.clear()
    out, doc2 = sync(engine, mirror, ONE.replace("Alpha body.", "ALPHA   body.\n"))
    assert out["Alpha"].status == "synced" and chat.calls == []
    assert row(mirror, doc2, "Alpha")["sectionHash"] == doc2.sections[0].section_hash


# --- split modes -------------------------------------------------------------------------------

def test_bullets_mode_posts_one_document_per_bullet_keeping_the_list_marker():
    engine, mirror, chat, _ = make_engine()
    sync(engine, mirror, BULLETS)
    texts = sorted(d["text"] for d in chat.current())
    assert texts == ["- one fact", "- three fact", "- two fact"]
    assert all(t.startswith("Facts") is False for t in chat.posts()) and chat.posts()[0].startswith("kb — one fact")


def test_editing_one_bullet_posts_one_document_and_deletes_only_that_bullets_old_id():
    engine, mirror, chat, _ = make_engine()
    _, doc = sync(engine, mirror, BULLETS)
    ids = dict(zip(row(mirror, doc, "Facts")["claimHashes"], row(mirror, doc, "Facts")["documentIds"], strict=True))
    chat.calls.clear()
    _, doc = sync(engine, mirror, BULLETS.replace("- two fact", "- two fact, revised"))
    assert len(chat.posts()) == 1 and chat.deletes() == [ids[claim_hash("- two fact")]]
    r = row(mirror, doc, "Facts")
    assert len(r["documentIds"]) == 3 and ids[claim_hash("- one fact")] in r["documentIds"]
    assert sorted(d["text"] for d in chat.current()) == ["- one fact", "- three fact", "- two fact, revised"]


def test_dropping_the_split_marker_reingests_whole_deletes_the_siblings_and_flags_needs_resplit():
    engine, mirror, chat, _ = make_engine()
    _, doc = sync(engine, mirror, BULLETS)
    old = set(row(mirror, doc, "Facts")["documentIds"])
    chat.calls.clear()
    out, doc = sync(engine, mirror, BULLETS.replace("<!-- kb-split: bullets -->\n", ""))
    assert out["Facts"].status == "synced" and set(chat.deletes()) == old
    r = row(mirror, doc, "Facts")
    assert len(r["documentIds"]) == 1 and r["needsResplit"] is True
    report = status([doc], list(mirror.sections.values()), 1)
    assert [line.kind for line in report.lines] == ["coarse"] and report.clean  # coarse, not stale
    out, doc = sync(engine, mirror, BULLETS)  # declared bullets again
    assert "needsResplit" not in row(mirror, doc, "Facts") and len(chat.current()) == 3


# --- budget, pending, finish -------------------------------------------------------------------

def edited_pending(engine, mirror, chat):
    _, doc = sync(engine, mirror, ONE)
    old = row(mirror, doc, "Alpha")["documentIds"][0]
    chat.hold_new = True
    out, doc = sync(engine, mirror, ONE.replace("body.", "body v2."))
    return doc, old, out["Alpha"]


def test_over_budget_the_section_stays_pending_with_the_old_claim_served_and_flagged():
    engine, mirror, chat, clock = make_engine(budget_s=2.0)
    doc, old, outcome = edited_pending(engine, mirror, chat)
    assert outcome.status == "pending" and "not yet searchable" in outcome.detail
    assert old in chat.docs and chat.docs[old]["current"] and chat.deletes() == []
    r = row(mirror, doc, "Alpha")
    assert r["state"] == "pending" and r["documentIds"] == [old] and len(r["pendingDocumentIds"]) == 1
    assert "leaseToken" not in r  # released
    [line] = status([doc], list(mirror.sections.values()), 1, ids=[old]).lines
    assert line.kind == "pending" and "stale — read `claude/a/kb.md` § `Alpha`" in line.detail
    assert clock.t >= 2.0  # it waited the whole budget, polling


def test_finish_completes_the_swap_and_deletes_the_old_id_only_then():
    engine, mirror, chat, _ = make_engine()
    doc, old, _ = edited_pending(engine, mirror, chat)
    pending = row(mirror, doc, "Alpha")["pendingDocumentIds"]
    [early] = engine.finish(once=True)
    assert early.status == "pending" and chat.deletes() == [] and old in chat.docs  # embeds not landed
    chat.release_embeds()
    [done] = engine.finish(once=True)
    assert done.status == "synced" and chat.deletes() == [old]
    r = row(mirror, doc, "Alpha")
    assert r["documentIds"] == pending and r["state"] == "synced" and "pendingDocumentIds" not in r
    assert r["sectionHash"] == doc.sections[0].section_hash


def test_finish_keeps_unchanged_bullets_and_swaps_in_only_the_new_one():
    engine, mirror, chat, _ = make_engine()
    _, doc = sync(engine, mirror, BULLETS)
    r0 = row(mirror, doc, "Facts")
    ids = dict(zip(r0["claimHashes"], r0["documentIds"], strict=True))
    chat.hold_new = True
    _, doc = sync(engine, mirror, BULLETS.replace("- two fact", "- two fact v2"))
    chat.release_embeds()
    engine.finish(once=True)
    r = row(mirror, doc, "Facts")
    assert len(r["documentIds"]) == 3 and ids[claim_hash("- one fact")] in r["documentIds"]
    assert ids[claim_hash("- two fact")] not in chat.docs and ids[claim_hash("- one fact")] in chat.docs
    assert sorted(d["text"] for d in chat.current()) == ["- one fact", "- three fact", "- two fact v2"]


def test_a_failed_embed_deletes_the_new_document_and_leaves_the_old_claim_mapped_and_stale():
    engine, mirror, chat, _ = make_engine()
    _, doc = sync(engine, mirror, ONE)
    old = row(mirror, doc, "Alpha")["documentIds"][0]
    chat.during_post = lambda: chat.failed.update(d for d in chat.docs if d != old)
    out, doc = sync(engine, mirror, ONE.replace("body.", "body v2."))
    assert out["Alpha"].status == "failed"
    assert old in chat.docs and len(chat.docs) == 1  # the new one is gone
    r = row(mirror, doc, "Alpha")
    assert r["documentIds"] == [old] and r["state"] == "synced" and "pendingDocumentIds" not in r
    assert [line.kind for line in status([doc], list(mirror.sections.values()), 1).lines] == ["stale"]


def test_finish_gives_up_past_the_cap_deleting_the_new_document_never_the_old():
    engine, mirror, chat, _ = make_engine(cap_ms=10_000)
    doc, old, _ = edited_pending(engine, mirror, chat)
    pending = row(mirror, doc, "Alpha")["pendingDocumentIds"]
    mirror.now += 11_000
    [res] = engine.finish(once=True)
    assert res.status == "failed" and chat.deletes() == pending and old in chat.docs
    r = row(mirror, doc, "Alpha")
    assert r["documentIds"] == [old] and r["state"] == "synced" and r["sectionHash"] != doc.sections[0].section_hash


# --- cancel on re-edit, the lease ---------------------------------------------------------------

def test_a_re_edit_of_a_pending_section_cancels_the_pending_submission_first():
    engine, mirror, chat, _ = make_engine()
    doc, old, _ = edited_pending(engine, mirror, chat)
    stale_pending = row(mirror, doc, "Alpha")["pendingDocumentIds"][0]
    chat.hold_new = False
    chat.calls.clear()
    out, doc = sync(engine, mirror, ONE.replace("body.", "body v3."))
    assert out["Alpha"].status == "synced"
    assert [c[0] for c in chat.calls][:2] == ["delete", "post"] and chat.calls[0][1] == stale_pending
    assert [d["text"] for d in chat.current()] == ["Alpha body v3."]  # one document per section
    mapped = {i for s in mirror.sections.values() for i in s.get("documentIds", [])}
    assert {d["id"] for d in chat.current()} == mapped  # no unmapped kb-mirror document
    assert old not in chat.docs and stale_pending not in chat.docs


def test_an_unchanged_re_sync_of_a_pending_section_leaves_it_to_finish():
    """F2: the in-flight submission is not cancelled and re-POSTed, and `pendingSince` (the cap's clock) is kept."""
    engine, mirror, chat, clock = make_engine()
    doc, old, _ = edited_pending(engine, mirror, chat)
    before = dict(row(mirror, doc, "Alpha"))
    chat.calls.clear()
    clock.sleep(5)
    out, doc = sync(engine, mirror, ONE.replace("body.", "body v2."))
    assert out["Alpha"].status == "pending" and "kb-finish" in out["Alpha"].detail
    assert chat.calls == []  # no cancel, no re-POST
    r = row(mirror, doc, "Alpha")
    assert r["pendingDocumentIds"] == before["pendingDocumentIds"] and r["pendingSince"] == before["pendingSince"]
    assert "leaseToken" not in r
    chat.release_embeds()
    assert [o.status for o in engine.finish(once=True)] == ["synced"] and old not in chat.docs


def test_a_pending_section_re_synced_in_another_mode_is_still_cancelled():
    engine, mirror, chat, _ = make_engine()
    _, doc = sync(engine, mirror, BULLETS)
    chat.hold_new = True
    sync(engine, mirror, BULLETS.replace("- two fact", "- two fact v2"))
    pending = row(mirror, doc, "Facts")["pendingDocumentIds"]
    chat.hold_new = False
    chat.calls.clear()
    out, _ = sync(engine, mirror, BULLETS.replace("<!-- kb-split: bullets -->\n", "").replace("- two fact", "- two fact v2"))
    assert out["Facts"].status == "synced" and set(pending) <= set(chat.deletes())


def test_a_held_lease_reports_pending_at_once_and_other_sections_proceed():
    engine, mirror, chat, _ = make_engine()
    d = kb(TWO)
    key_alpha = section_key(P, d.sections[0])
    engine._acquire(key_alpha, P, "Alpha")  # another actor holds Alpha
    out, _ = sync(engine, mirror, TWO)
    assert out["Alpha"].status == "pending" and "lease is held" in out["Alpha"].detail
    assert out["Beta"].status == "synced" and chat.posts() == ["Beta"]
    assert mirror.files == {}  # an incomplete file is not stamped as synced


def test_l1_a_lease_lost_before_the_post_makes_no_falkor_chat_call():
    engine, mirror, chat, _ = make_engine()
    key = section_key(P, kb(ONE).sections[0])
    mirror.after["MARK"] = lambda p: mirror.steal(key)
    out, _ = sync(engine, mirror, ONE)
    assert out["Alpha"].status == "pending" and chat.calls == []


def test_l2_a_lease_lost_after_the_post_deletes_that_posts_own_document_and_stops():
    engine, mirror, chat, _ = make_engine()
    key = section_key(P, kb(ONE).sections[0])
    chat.during_post = lambda: mirror.steal(key)
    out, _ = sync(engine, mirror, ONE)
    assert out["Alpha"].status == "pending"
    assert [c[0] for c in chat.calls] == ["post", "delete"] and chat.docs == {}
    assert mirror.sections[key]["leaseToken"] == "other"  # the new holder's lease is untouched


def test_l3_a_stale_holder_never_deletes_old_ids():
    engine, mirror, chat, _ = make_engine()
    _, doc = sync(engine, mirror, ONE)
    key = section_key(P, doc.sections[0])
    old = row(mirror, doc, "Alpha")["documentIds"][0]
    mirror.after["RECORD"] = lambda p: mirror.steal(key)  # expired between the record and the swap
    chat.calls.clear()
    out, _ = sync(engine, mirror, ONE.replace("body.", "body v2."))
    assert out["Alpha"].status == "pending" and chat.deletes() == [] and old in chat.docs


def test_a_lease_lost_during_the_swap_deletes_report_pending_not_synced():
    engine, mirror, chat, _ = make_engine()
    _, doc = sync(engine, mirror, ONE)
    key = section_key(P, doc.sections[0])
    chat.on_delete = lambda i: mirror.steal(key)
    out, _ = sync(engine, mirror, ONE.replace("body.", "body v2."))
    assert out["Alpha"].status == "pending" and "swap" in out["Alpha"].detail


# --- L1: renew before EVERY falkor-chat call, deletes included (review S6 F5) -------------------

def watch_deletes(mirror, chat):
    """Record each DELETE in the mirror's statement log, so its position relative to a RENEW is checkable."""
    chat.on_delete = lambda _id: mirror.log.append("DEL")


def assert_renew_per_delete(mirror, deletes):
    positions = [i for i, n in enumerate(mirror.log) if n == "DEL"]
    assert len(positions) == deletes
    assert all(mirror.log[i - 1] == "RENEW" for i in positions), mirror.log


THREE = "# T\n\n## Facts\n<!-- kb-split: bullets -->\n\n- a1\n- b2\n- c3\n"
THREE_V2 = THREE.replace("a1", "a1x").replace("b2", "b2x").replace("c3", "c3x")


def test_the_swap_renews_before_each_old_id_delete():
    engine, mirror, chat, _ = make_engine()
    sync(engine, mirror, THREE)
    watch_deletes(mirror, chat)
    mirror.log.clear()
    out, _ = sync(engine, mirror, THREE.replace("<!-- kb-split: bullets -->\n", ""))
    assert out["Facts"].status == "synced"
    assert_renew_per_delete(mirror, 3)


def test_cancel_on_re_edit_renews_before_each_pending_delete():
    engine, mirror, chat, _ = make_engine()
    sync(engine, mirror, THREE)
    chat.hold_new = True
    sync(engine, mirror, THREE_V2)
    chat.hold_new = False
    watch_deletes(mirror, chat)
    mirror.log.clear()
    out, _ = sync(engine, mirror, THREE_V2.replace("a1x", "a1y").replace("b2x", "b2y").replace("c3x", "c3y"))
    assert out["Facts"].status == "synced"
    assert_renew_per_delete(mirror, 3 + 3)  # three cancelled, three old swapped out


def test_abort_renews_before_each_delete_of_its_own_submissions():
    engine, mirror, chat, _ = make_engine()
    chat.during_post = lambda: chat.failed.update(chat.docs)
    watch_deletes(mirror, chat)
    out, _ = sync(engine, mirror, THREE)
    assert out["Facts"].status == "failed"
    assert_renew_per_delete(mirror, 3)


def test_removing_a_section_renews_before_each_delete():
    engine, mirror, chat, _ = make_engine()
    sync(engine, mirror, THREE)
    watch_deletes(mirror, chat)
    mirror.log.clear()
    out, _ = sync(engine, mirror, "# T\n")
    assert [o.status for o in out.values()] == ["removed"]
    assert_renew_per_delete(mirror, 3)


def test_finish_renews_before_each_stale_delete_and_each_give_up_delete():
    engine, mirror, chat, clock = make_engine()
    sync(engine, mirror, THREE)
    chat.hold_new = True
    sync(engine, mirror, THREE_V2)
    chat.release_embeds()
    watch_deletes(mirror, chat)
    mirror.log.clear()
    assert [o.status for o in engine.finish(once=True)] == ["synced"]
    assert_renew_per_delete(mirror, 3)
    chat.hold_new = True
    sync(engine, mirror, THREE_V2.replace("a1x", "a1y").replace("b2x", "b2y").replace("c3x", "c3y"))
    mirror.log.clear()
    mirror.now += engine.cap_ms + 1  # past the cap: give up, deleting the three pending documents
    assert [o.status for o in engine.finish(once=True)] == ["failed"]
    assert_renew_per_delete(mirror, 3)


# --- duplicate claims, receipts -----------------------------------------------------------------

DUP = "# T\n\n## Alpha\n\nSame claim.\n\n## Beta\n\nsame   CLAIM.\n"


def test_a_second_heading_with_the_same_claim_text_is_refused_naming_both_and_the_first_stays_current():
    engine, mirror, chat, _ = make_engine()
    out, doc = sync(engine, mirror, DUP)
    assert out["Alpha"].status == "synced" and out["Beta"].status == "refused"
    assert "'Alpha'" in out["Beta"].detail and chat.posts() == ["Alpha"]
    [d] = chat.current()
    assert d["current"] and row(mirror, doc, "Alpha")["documentIds"] == [d["id"]]
    kinds = [(line.kind, line.heading) for line in status([doc], list(mirror.sections.values()), 1).lines]
    assert ("duplicate", "Alpha") in kinds


def test_a_new_heading_duplicating_an_already_mapped_claim_is_refused_naming_the_owner():
    engine, mirror, chat, _ = make_engine()
    sync(engine, mirror, ONE)
    out, _ = sync(engine, mirror, ONE + "\n## Gamma\n\nalpha body.\n")
    assert out["Gamma"].status == "refused" and "'Alpha'" in out["Gamma"].detail
    assert out["Alpha"].status == "unchanged" and chat.posts() == ["Alpha"]


def test_two_headings_exchanging_their_claims_in_one_edit_both_sync():
    """F4: each claim still belongs, by the old mapping, to the other heading; neither may be refused for good."""
    engine, mirror, chat, _ = make_engine()
    sync(engine, mirror, "# T\n\n## One\n\nFirst.\n\n## Two\n\nSecond.\n")
    out, doc = sync(engine, mirror, "# T\n\n## One\n\nSecond.\n\n## Two\n\nFirst.\n")
    assert {h: o.status for h, o in out.items()} == {"One": "synced", "Two": "synced"}
    assert sorted(d["text"] for d in chat.current()) == ["First.", "Second."] and len(chat.docs) == 2
    for heading, text in (("One", "Second."), ("Two", "First.")):
        [i] = row(mirror, doc, heading)["documentIds"]
        assert chat.docs[i]["text"] == text and chat.docs[i]["current"]
    again, _ = sync(engine, mirror, "# T\n\n## One\n\nSecond.\n\n## Two\n\nFirst.\n")
    assert {o.status for o in again.values()} == {"unchanged"}


def test_a_heading_taking_a_claim_another_heading_still_wants_is_refused_and_supersedes_nothing():
    """The guard the F4 change must keep: the claim is still in the other heading's file text."""
    engine, mirror, chat, _ = make_engine()
    sync(engine, mirror, TWO)
    before = {d["id"] for d in chat.current()}
    out, _ = sync(engine, mirror, TWO.replace("Beta body.", "Alpha body."))
    assert out["Beta"].status == "refused" and "'Alpha'" in out["Beta"].detail
    assert {d["id"] for d in chat.current()} == before and chat.posts() == ["Alpha", "Beta"]


def test_a_claim_already_mapped_in_another_file_is_refused_even_when_this_file_is_registered_first():
    engine, mirror, chat, _ = make_engine()
    other = kb("# T\n\n## Owner\n\nShared claim.\n", "claude/z/other.md")
    sync(engine, mirror, "# T\n\n## Owner\n\nShared claim.\n", "claude/z/other.md")
    mine = kb("# T\n\n## Mine\n\nshared claim.\n", "claude/a/mine.md")  # parse order alone would own it
    rows, _ = engine.read_all()
    outcomes = engine.sync_file(mine, Context.build([mine, other], rows), rows)
    assert [(o.status, "'Owner'" in o.detail) for o in outcomes] == [("refused", True)]
    assert len(chat.current()) == 1 and chat.posts() == ["Owner"]


def test_a_receipt_superseding_a_document_this_section_does_not_own_is_reported_not_accepted():
    engine, mirror, chat, _ = make_engine()
    foreign = chat.plant("Alpha body.", producer="cobb")
    chat.supersede_foreign = foreign
    out, doc = sync(engine, mirror, ONE)
    assert out["Alpha"].status == "failed" and foreign in out["Alpha"].detail
    assert chat.docs.keys() == {foreign}  # the new POST was deleted again
    assert "documentIds" not in row(mirror, doc, "Alpha")


def test_a_receipt_superseding_this_sections_own_old_id_proceeds():
    engine, mirror, chat, _ = make_engine()
    _, doc = sync(engine, mirror, ONE)
    old = row(mirror, doc, "Alpha")["documentIds"][0]
    section = doc.sections[0]
    out = engine.sync_section(P, section, Context.build([doc], engine.read_all()[0]), force=True,
                              bad_ids=frozenset({old}))
    assert out.status == "synced" and chat.deletes() == [old] and len(chat.current()) == 1


def test_a_timed_out_post_the_server_still_committed_is_superseded_on_retry_without_an_orphan():
    """Review P4-m1: outcome unknown on a client timeout; the retry's receipt names our own earlier document."""
    engine, mirror, chat, _ = make_engine()
    chat.timeout_mode = "late"
    out, doc = sync(engine, mirror, ONE)
    assert out["Alpha"].status == "failed" and "outcome unknown" in out["Alpha"].detail
    chat.flush_late()  # the server commits after the client gave up
    assert len(chat.current()) == 1 and "documentIds" not in row(mirror, doc, "Alpha")  # an unmapped orphan
    chat.timeout_mode = None
    out, doc = sync(engine, mirror, ONE)
    assert out["Alpha"].status == "synced"
    [d] = chat.current()
    assert row(mirror, doc, "Alpha")["documentIds"] == [d["id"]] and len(chat.docs) == 1  # twin deleted


def test_a_superseded_twin_of_another_producer_is_not_adopted_as_ours():
    engine, mirror, chat, _ = make_engine()
    foreign = chat.plant("Alpha body.", producer="cobb")
    out, _ = sync(engine, mirror, ONE)
    assert out["Alpha"].status == "failed" and foreign in chat.docs and chat.docs[foreign]["producer"] == "cobb"


# --- removal, rename, per-section failure -------------------------------------------------------

def test_a_removed_heading_deletes_its_own_documents_and_drops_its_record():
    engine, mirror, chat, _ = make_engine()
    _, doc = sync(engine, mirror, TWO)
    alpha, beta = (row(mirror, doc, h)["documentIds"][0] for h in ("Alpha", "Beta"))
    chat.calls.clear()
    out, doc = sync(engine, mirror, ONE)
    assert out["Beta"].status == "removed" and chat.deletes() == [beta] and alpha in chat.docs
    assert [s["heading"] for s in mirror.sections.values()] == ["Alpha"]


def test_a_renamed_heading_with_the_same_body_ends_with_one_document():
    engine, mirror, chat, _ = make_engine()
    sync(engine, mirror, ONE)
    out, doc = sync(engine, mirror, ONE.replace("## Alpha", "## Alpha renamed"))
    assert out["Alpha renamed"].status == "synced" and out["Alpha"].status == "removed"
    [d] = chat.current()
    assert [s["documentIds"] for s in mirror.sections.values()] == [[d["id"]]] and len(chat.docs) == 1


def test_one_sections_failure_never_aborts_the_others(monkeypatch):
    engine, mirror, chat, _ = make_engine()
    real = kbsync.section_key

    def key(path, section):
        if section.heading == "Alpha":
            raise kbsync.KeyHygieneError("whitespace edge survived normalisation")
        return real(path, section)

    monkeypatch.setattr(kbsync, "section_key", key)
    out, _ = sync(engine, mirror, TWO)
    assert out["Alpha"].status == "failed" and "whitespace edge" in out["Alpha"].detail
    assert out["Beta"].status == "synced"


def test_a_mirror_error_on_one_section_is_that_sections_failure_only():
    engine, mirror, chat, _ = make_engine()
    mirror.fail["ACQUIRE"] = RuntimeError("boom")
    out, _ = sync(engine, mirror, TWO)
    assert out["Alpha"].status == "failed" and "boom" in out["Alpha"].detail and out["Beta"].status == "synced"


def test_a_post_http_error_fails_the_section_and_leaves_the_old_claim():
    engine, mirror, chat, _ = make_engine()
    _, doc = sync(engine, mirror, ONE)
    old = row(mirror, doc, "Alpha")["documentIds"][0]

    def boom(title, text):
        raise KbHttpError("POST /documents: HTTP 503", status=503)

    chat.post_document = boom
    out, doc = sync(engine, mirror, ONE.replace("body.", "body v2."))
    assert out["Alpha"].status == "failed" and "503" in out["Alpha"].detail
    assert old in chat.docs and row(mirror, doc, "Alpha")["documentIds"] == [old]


# --- a mapped document that ended `failed` is re-submitted ------------------------------------------

def test_sync_resubmits_a_section_whose_mapped_document_is_failed():
    # rules out: `unchanged` decided from the section hash alone, a permanently failed document never retried
    engine, mirror, chat, _ = make_engine()
    _, doc = sync(engine, mirror, ONE)
    old = row(mirror, doc, "Alpha")["documentIds"][0]
    chat.failed.add(old)
    chat.held.add(old)  # failed AND unembedded: not searchable
    chat.calls.clear()
    out, doc = sync(engine, mirror, ONE)
    assert out["Alpha"].status == "synced" and chat.posts() == ["Alpha"] and chat.deletes() == [old]
    [new] = [d["id"] for d in chat.docs.values() if d["id"] != old]
    assert row(mirror, doc, "Alpha")["documentIds"] == [new]
    assert [c[0] for c in chat.calls] == ["post", "delete"]  # append before delete


def test_the_failed_document_is_removed_only_once_its_replacement_is_ready():
    # rules out: deleting the old claim before the new one is submitted and embedded
    engine, mirror, chat, _ = make_engine()
    _, doc = sync(engine, mirror, ONE)
    old = row(mirror, doc, "Alpha")["documentIds"][0]
    chat.failed.add(old)
    chat.held.add(old)  # failed AND unembedded: not searchable
    seen = []
    chat.on_delete = lambda i: seen.append((i, [(d["id"], d["status"]) for d in chat.docs.values() if d["id"] != i
                                                 and d["id"] not in chat.held]))
    sync(engine, mirror, ONE)
    assert [i for i, _ in seen] == [old] and len(seen[0][1]) == 1


def test_a_processing_document_is_not_retried():
    # rules out: treating any not-yet-ready status as failed (it would double-submit a document still embedding)
    engine, mirror, chat, _ = make_engine()
    _, doc = sync(engine, mirror, ONE)
    old = row(mirror, doc, "Alpha")["documentIds"][0]
    chat.docs[old]["status"] = "processing"
    chat.calls.clear()
    out, _ = sync(engine, mirror, ONE)
    assert out["Alpha"].status == "unchanged" and chat.calls == []


def test_when_the_resubmission_fails_too_the_old_mapping_stays_and_the_section_is_not_complete():
    engine, mirror, chat, _ = make_engine()
    _, doc = sync(engine, mirror, ONE)
    old = row(mirror, doc, "Alpha")["documentIds"][0]
    chat.failed.add(old)
    chat.held.add(old)
    chat.during_post = lambda: chat.failed.update(chat.docs)
    out, doc = sync(engine, mirror, ONE)
    assert out["Alpha"].status == "failed" and out["Alpha"].incomplete
    r = row(mirror, doc, "Alpha")
    assert r["documentIds"] == [old] and r["state"] == "synced" and "pendingDocumentIds" not in r
    assert chat.deletes() != [old] and old in chat.docs


def test_only_the_failed_claim_of_a_split_section_is_resubmitted():
    engine, mirror, chat, _ = make_engine()
    _, doc = sync(engine, mirror, BULLETS)
    ids = dict(zip(row(mirror, doc, "Facts")["claimHashes"], row(mirror, doc, "Facts")["documentIds"], strict=True))
    bad = ids[claim_hash("- two fact")]
    chat.failed.add(bad)
    chat.held.add(bad)
    chat.calls.clear()
    out, doc = sync(engine, mirror, BULLETS)
    assert out["Facts"].status == "synced" and chat.posts() == ["kb — two fact"] and chat.deletes() == [bad]
    assert ids[claim_hash("- one fact")] in row(mirror, doc, "Facts")["documentIds"]


def test_a_healthy_unchanged_section_costs_one_observation_read_and_no_http():
    engine, mirror, chat, _ = make_engine()
    sync(engine, mirror, TWO)
    reads = []
    real = chat.observe
    chat.observe = lambda ids: reads.append(list(ids)) or real(ids)
    chat.calls.clear()
    out, _ = sync(engine, mirror, TWO)
    assert [o.status for o in out.values()] == ["unchanged", "unchanged"]
    assert len(reads) == 2 and all(len(r) == 1 for r in reads) and chat.calls == []


# --- a failed-document retry never hides a working claim, and is bounded -------------------------------

def failed_section(text=ONE, *, embedded: bool, **kw):
    """A synced one-claim section whose mapped document is `failed`; `embedded` = still searchable."""
    engine, mirror, chat, clock = make_engine(**kw)
    _, doc = sync(engine, mirror, text)
    old = row(mirror, doc, "Alpha")["documentIds"][0]
    chat.failed.add(old)
    if not embedded:
        chat.held.add(old)
    chat.calls.clear()
    return engine, mirror, chat, clock, doc, old, claim_hash("Alpha body.")


def test_an_embedded_but_failed_document_is_reported_degraded_and_never_reposted():
    # rules out: re-posting it, which auto-supersedes (documentCurrent=false) the only searchable copy
    engine, mirror, chat, _, doc, old, h = failed_section(embedded=True)
    out, _ = sync(engine, mirror, ONE)
    assert out["Alpha"].status == "degraded" and not out["Alpha"].incomplete and old in out["Alpha"].detail
    assert chat.posts() == [] and chat.searchable(h) == 1


@pytest.mark.parametrize("how", ["sync", "reconcile", "finish"])
def test_the_claim_stays_searchable_through_every_run_over_an_embedded_failed_document(how):
    engine, mirror, chat, _, doc, old, h = failed_section(embedded=True)
    chat.hold_new = True
    for _ in range(2):
        if how == "sync":
            sync(engine, mirror, ONE)
        elif how == "reconcile":
            assert [o.status for o in engine.reconcile([doc])] == ["degraded"]
        else:
            engine.finish(once=True)
        assert chat.searchable(h) == 1 and chat.posts() == []
    assert row(mirror, doc, "Alpha")["documentIds"] == [old]


def test_the_retry_of_an_unsearchable_failed_document_that_fails_again_keeps_the_mapping_and_counts():
    engine, mirror, chat, _, doc, old, h = failed_section(embedded=False)
    chat.during_post = lambda: chat.failed.update(d for d in chat.docs if d != old)
    out, _ = sync(engine, mirror, ONE)
    r = row(mirror, doc, "Alpha")
    assert out["Alpha"].status == "failed" and r["documentIds"] == [old] and "pendingDocumentIds" not in r
    assert r["retryCount"] == 1 and chat.searchable(h) == 0 and len(chat.docs) == 1  # the new one is deleted


def test_a_timed_out_post_of_the_retry_leaves_the_mapping_on_the_old_id_and_counts_the_attempt():
    engine, mirror, chat, _, doc, old, h = failed_section(embedded=False)
    chat.timeout_mode = "commit"
    out, _ = sync(engine, mirror, ONE)
    r = row(mirror, doc, "Alpha")
    assert out["Alpha"].status == "failed" and "outcome unknown" in out["Alpha"].detail
    assert r["documentIds"] == [old] and r["retryCount"] == 1 and chat.searchable(h) <= 1


def test_kb_finish_giving_up_on_a_retry_keeps_the_old_mapping_and_the_count():
    engine, mirror, chat, clock, doc, old, h = failed_section(embedded=False, cap_ms=10_000)
    chat.hold_new = True
    out, _ = sync(engine, mirror, ONE)
    assert out["Alpha"].status == "pending" and row(mirror, doc, "Alpha")["state"] == "pending"
    clock.sleep(11)
    [res] = engine.finish(once=True)
    r = row(mirror, doc, "Alpha")
    assert res.status == "failed" and r["documentIds"] == [old] and r["state"] == "synced" and r["retryCount"] == 1
    assert len(chat.docs) == 1


def test_a_successful_retry_clears_the_count_and_the_claim_is_searchable_once():
    engine, mirror, chat, _, doc, old, h = failed_section(embedded=False)
    out, _ = sync(engine, mirror, ONE)
    r = row(mirror, doc, "Alpha")
    assert out["Alpha"].status == "synced" and "retryCount" not in r and chat.searchable(h) == 1


def test_the_retry_limit_stops_a_persistently_failing_document_until_the_section_is_edited():
    # rules out: re-posting on every sync/hook run
    engine, mirror, chat, clock, doc, old, h = failed_section(embedded=False, retry_max=2, retry_backoff_ms=1000)
    chat.during_post = lambda: chat.failed.update(d for d in chat.docs if d != old)
    for _ in range(2):
        assert sync(engine, mirror, ONE)[0]["Alpha"].status == "failed"
        clock.sleep(10)
    chat.calls.clear()
    out, _ = sync(engine, mirror, ONE)
    assert out["Alpha"].status == "failed" and "retry limit" in out["Alpha"].detail and chat.posts() == []
    chat.during_post = None
    out, doc2 = sync(engine, mirror, ONE.replace("body.", "body v2."))  # an edit starts over
    assert out["Alpha"].status == "synced" and "retryCount" not in row(mirror, doc2, "Alpha")


def test_a_retry_waits_out_the_doubling_backoff():
    engine, mirror, chat, clock, doc, old, h = failed_section(embedded=False, retry_backoff_ms=60_000)
    chat.during_post = lambda: chat.failed.update(d for d in chat.docs if d != old)
    sync(engine, mirror, ONE)
    chat.calls.clear()
    clock.sleep(59)
    out, _ = sync(engine, mirror, ONE)
    assert "backoff" in out["Alpha"].detail and chat.posts() == []
    clock.sleep(2)
    sync(engine, mirror, ONE)
    assert chat.posts() != [] and row(mirror, doc, "Alpha")["retryCount"] == 2
    chat.calls.clear()
    clock.sleep(61)  # the second wait is twice the first
    assert "backoff" in sync(engine, mirror, ONE)[0]["Alpha"].detail and chat.posts() == []


def test_one_run_retries_at_most_run_cap_sections():
    engine, mirror, chat, _ = make_engine(retry_run_cap=1)
    _, doc = sync(engine, mirror, TWO)
    for h in ("Alpha", "Beta"):
        i = row(mirror, doc, h)["documentIds"][0]
        chat.failed.add(i)
        chat.held.add(i)
    chat.calls.clear()
    out, _ = sync(engine, mirror, TWO)
    assert out["Alpha"].status == "synced" and out["Beta"].status == "failed" and "cap" in out["Beta"].detail
    assert chat.posts() == ["Alpha"]


# --- reconcile -------------------------------------------------------------------------------------

def test_reconcile_reingests_a_mapped_document_that_is_gone():
    engine, mirror, chat, _ = make_engine()
    _, doc = sync(engine, mirror, TWO)
    gone = row(mirror, doc, "Alpha")["documentIds"][0]
    keep = row(mirror, doc, "Beta")["documentIds"][0]
    del chat.docs[gone]
    chat.calls.clear()
    results = {o.heading: o for o in engine.reconcile([doc])}
    assert results["Alpha"].status == "re-ingested" and chat.posts() == ["Alpha"] and keep in chat.docs
    assert row(mirror, doc, "Alpha")["documentIds"] != [gone]


def test_reconcile_reingests_a_hash_mismatch_and_deletes_the_stale_id():
    engine, mirror, chat, _ = make_engine()
    _, doc = sync(engine, mirror, ONE)
    old = row(mirror, doc, "Alpha")["documentIds"][0]
    chat.docs[old]["hash"] = "0" * 64  # edited behind the mirror's back
    [res] = [o for o in engine.reconcile([doc]) if o.heading == "Alpha"]
    assert res.status == "re-ingested" and old in chat.deletes() and old not in chat.docs


def test_reconcile_reingests_a_failed_mapped_document():
    engine, mirror, chat, _ = make_engine()
    _, doc = sync(engine, mirror, TWO)
    bad = row(mirror, doc, "Alpha")["documentIds"][0]
    good = row(mirror, doc, "Beta")["documentIds"][0]
    chat.failed.add(bad)
    chat.held.add(bad)
    chat.calls.clear()
    results = {o.heading: o for o in engine.reconcile([doc])}
    assert results["Alpha"].status == "re-ingested"
    assert chat.posts() == ["Alpha"] and chat.deletes() == [bad] and good in chat.docs and "Beta" not in results


def test_reconcile_does_not_reingest_a_processing_document():
    engine, mirror, chat, _ = make_engine()
    _, doc = sync(engine, mirror, ONE)
    chat.docs[row(mirror, doc, "Alpha")["documentIds"][0]]["status"] = "processing"
    chat.calls.clear()
    assert [o for o in engine.reconcile([doc]) if o.heading == "Alpha"] == [] and chat.calls == []


def test_reconcile_reports_a_non_current_mapped_id_and_does_not_reingest_it():
    engine, mirror, chat, _ = make_engine()
    _, doc = sync(engine, mirror, ONE)
    old = row(mirror, doc, "Alpha")["documentIds"][0]
    chat.docs[old]["current"] = False
    chat.calls.clear()
    results = engine.reconcile([doc])
    assert [o.status for o in results if o.heading == "Alpha"] == ["non-current"] and chat.calls == []


def test_reconcile_adopts_unmapped_documents_whose_hash_equals_a_claim_without_a_post():
    engine, mirror, chat, _ = make_engine()
    a = chat.plant("Alpha body.", producer="cobb")
    b = chat.plant("Beta body.", producer="cobb")
    doc = kb(TWO)
    results = {o.heading: o for o in engine.reconcile([doc])}
    assert results["Alpha"].status == "adopted" and results["Beta"].status == "adopted" and chat.calls == []
    assert row(mirror, doc, "Alpha")["documentIds"] == [a] and row(mirror, doc, "Beta")["documentIds"] == [b]
    out, _ = sync(engine, mirror, TWO)  # nothing left to do
    assert {o.status for o in out.values()} == {"unchanged"}


def test_reconcile_does_not_adopt_a_partial_match_and_adopts_nothing_twice():
    engine, mirror, chat, _ = make_engine()
    a = chat.plant("- one fact")
    chat.plant("- two fact")
    doc = kb(BULLETS)
    [res] = [o for o in engine.reconcile([doc]) if o.heading == "Facts"]
    assert res.status == "unadopted" and "2 of 3" in res.detail
    assert "Facts" not in {s["heading"] for s in mirror.sections.values()} or "documentIds" not in next(
        iter(mirror.sections.values()))
    assert a in chat.docs


def test_reconcile_never_maps_one_stored_document_to_two_sections():
    """F3a: the `mapped_ids` guard in `_adopt`: two sections cannot both own a document, or one's sync would
    delete the other's claim."""
    engine, mirror, chat, _ = make_engine()
    stored = chat.plant("Same claim.", producer="cobb")
    doc = kb(DUP)
    results = {o.heading: o for o in engine.reconcile([doc])}
    assert results["Alpha"].status == "adopted" and results["Beta"].status == "unadopted"
    owners = [s["heading"] for s in mirror.sections.values() if stored in s.get("documentIds", [])]
    assert owners == ["Alpha"]


def test_reconcile_does_not_adopt_a_document_another_section_already_maps():
    engine, mirror, chat, _ = make_engine()
    _, doc = sync(engine, mirror, ONE)
    mapped = row(mirror, doc, "Alpha")["documentIds"][0]
    two = kb(ONE + "\n## Gamma\n\nalpha body.\n")
    results = {o.heading: o for o in engine.reconcile([two]) if o.heading in ("Alpha", "Gamma")}
    assert results["Gamma"].status == "unadopted"
    assert [s["heading"] for s in mirror.sections.values() if mapped in s.get("documentIds", [])] == ["Alpha"]


def test_a_receipt_superseding_another_sections_pending_twin_is_refused_and_that_document_survives():
    """F3b: `victim not in mapped`. The twin is a `kb-mirror` document of the claim's own text, but another
    live section maps it as a *pending* submission (its claim hashes are not in the owners map), so deleting it
    would destroy that section's in-flight write. Reachable with no other guard in the way."""
    engine, mirror, chat, _ = make_engine()
    other = "# T\n\n## Other\n\nOther v1.\n"
    sync(engine, mirror, other)
    chat.hold_new = True
    sync(engine, mirror, other.replace("v1", "v2"))
    [pending] = next(s for s in mirror.sections.values() if s["heading"] == "Other")["pendingDocumentIds"]
    chat.hold_new = False
    doc = kb(other.replace("v1", "v3") + "\n## Beta\n\nOther v2.\n")
    rows, _ = engine.read_all()
    beta = next(s for s in doc.sections if s.heading == "Beta")
    out = engine.sync_section(P, beta, Context.build([doc], rows))
    assert out.status == "failed" and pending in out.detail
    assert pending in chat.docs  # not deleted as a "twin"


def test_marker_stripped_claims_are_adopted_then_reingested_under_append_before_delete():
    """The 17 of `falkordb-reference.md`: stored without the list marker; the first sync replaces each."""
    engine, mirror, chat, _ = make_engine()
    old = [chat.plant(t, title="falkordb-reference — x") for t in ("one fact", "two fact", "three fact")]
    doc = kb(BULLETS)
    [res] = [o for o in engine.reconcile([doc]) if o.heading == "Facts"]
    assert res.status == "adopted" and "re-ingests" in res.detail
    assert row(mirror, doc, "Facts")["documentIds"] == old and "sectionHash" not in row(mirror, doc, "Facts")
    chat.calls.clear()
    out, doc = sync(engine, mirror, BULLETS)
    assert out["Facts"].status == "synced" and len(chat.posts()) == 3 and sorted(chat.deletes()) == sorted(old)
    first_delete = next(i for i, c in enumerate(chat.calls) if c[0] == "delete")
    assert all(c[0] == "post" for c in chat.calls[:first_delete]) and first_delete == 3  # all appended first
    assert sorted(d["text"] for d in chat.current()) == ["- one fact", "- three fact", "- two fact"]


def test_reconcile_reports_kb_mirror_documents_no_section_maps():
    engine, mirror, chat, _ = make_engine()
    sync(engine, mirror, ONE)
    orphan = chat.plant("stray", producer=kbsync.PRODUCER, title="stray claim")
    results = engine.reconcile([kb(ONE)])
    assert [(o.status, o.heading) for o in results if o.status == "orphan"] == [("orphan", "stray claim")]
    assert orphan in chat.docs  # reported, never deleted


def test_reconcile_gives_up_a_pending_section_past_the_cap():
    engine, mirror, chat, _ = make_engine(cap_ms=10_000)
    doc, old, _ = edited_pending(engine, mirror, chat)
    mirror.now += 11_000
    results = engine.reconcile([doc])
    assert any(o.status == "failed" for o in results) and row(mirror, doc, "Alpha")["state"] == "synced"


# --- status ---------------------------------------------------------------------------------------------

def test_status_flags_stale_unmirrored_unmapped_ids_and_filters_by_ids():
    engine, mirror, chat, _ = make_engine()
    _, doc = sync(engine, mirror, TWO)
    rows = list(mirror.sections.values())
    a = row(mirror, doc, "Alpha")["documentIds"][0]
    edited = kb(TWO.replace("Alpha body.", "changed").replace("Beta body.", "also changed")
                + "\n## Gamma\n\nnew claim\n")
    full = {(line.kind, line.heading) for line in status([edited], rows, 5).lines}
    assert full == {("stale", "Alpha"), ("stale", "Beta"), ("unmirrored", "Gamma")}
    only = status([edited], rows, 5, ids=[a, "ghost"])
    assert {(line.kind, line.heading) for line in only.lines} == {("stale", "Alpha"), ("unmirrored", "Gamma"),
                                                                   ("unmapped-id", "")}
    assert not only.clean and status([doc], rows, 5).clean
    assert status([doc], rows, None).header() == "KB mirror: last sync never"
    assert status([doc], rows, 0).header() == "KB mirror: last sync 1970-01-01T00:00:00Z"


def test_make_plan_removes_what_is_not_wanted_and_excludes_bad_ids():
    section = kb(BULLETS).sections[0]
    row_ = {"documentIds": ["a", "b", "c"], "claimHashes": [claim_hash("- one fact"), claim_hash("- gone"),
                                                              claim_hash("- two fact")]}
    plan = kbsync.make_plan(section, row_, frozenset({"c"}))
    assert [k[0] for k in plan.keep] == ["a"] and plan.remove == ["b", "c"]
    assert [c.text for c in plan.new] == ["- two fact", "- three fact"]


def test_claim_title_cuts_the_claim_part_never_the_family_slug():
    from coordgraph.parse.kb import Claim

    section = kb("# T\n\n## H\n<!-- kb-split: bullets -->\n\n- x\n").sections[0]
    long_claim = Claim("- x", "x" * 400, "h", "b-h")  # `parse_kb` cuts a lead at 80; the bound is the title's
    title = kbsync.claim_title("claude/graph-dba/falkordb-reference.md", section, long_claim)
    assert title.startswith("falkordb-reference — x") and len(title) == kbsync.TITLE_LIMIT
    whole = kb("# T\n\n## " + "h" * 300 + "\n\nbody\n").sections[0]
    assert len(kbsync.claim_title(P, whole, whole.claims[0])) == kbsync.TITLE_LIMIT


@pytest.mark.parametrize("budget", [0.0])
def test_a_zero_budget_still_submits_and_reports_pending_never_blocks(budget):
    engine, mirror, chat, clock = make_engine(budget_s=budget)
    chat.hold_new = True
    out, _ = sync(engine, mirror, ONE)
    assert out["Alpha"].status == "pending" and clock.t == 0 and len(chat.docs) == 1
