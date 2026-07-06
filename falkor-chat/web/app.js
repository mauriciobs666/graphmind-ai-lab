"use strict";
// Minimal browser client for falkor-chat's M1 REST API (DESIGN §14.5).
// Same-origin: the FastAPI process serves this file and the API, so no CORS.
// Flow: channels -> threads -> messages, plus workspace-wide full-text search.

// `lastCreatedAt`/`seen` drive the incremental `?since=&limit=` poll of the open
// thread: `lastCreatedAt` is the high-water timestamp, `seen` de-dupes by msgId
// (the plain `>` since-read may re-deliver a millisecond tie — OQ3). `pollTimer`
// is the background interval handle for the currently open thread.
const state = {
  channelId: null, threadId: null,
  lastCreatedAt: 0, seen: new Set(), pollTimer: null,
};

// Background poll cadence for the open thread. Bounded by `POLL_LIMIT` per fetch
// so a poll never walks the full NEXT* chain past the server's 1000ms TIMEOUT.
const POLL_MS = 3000;
const POLL_LIMIT = 50;

const $ = (id) => document.getElementById(id);

async function api(path, opts) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!res.ok) {
    let detail = res.statusText;
    let code = null;
    try {
      const body = await res.json();
      detail = body.detail || body.error || JSON.stringify(body);
      // ServiceError bodies carry a machine-readable class name in `error`
      // (e.g. "UnknownMemberError"); `detail` is human text and varies.
      code = body.error || null;
    } catch (_) { /* non-JSON error body */ }
    const err = new Error(`${res.status}: ${detail}`);
    err.code = code;
    throw err;
  }
  return res.status === 204 ? null : res.json();
}

const fmtTime = (ms) => (ms ? new Date(ms).toLocaleString() : "");
// Pull "@token" handles out of the text as mention ids (best-effort demo parse).
const parseMentions = (text) =>
  [...text.matchAll(/@([\w:-]+)/g)].map((m) => m[1]);

// ── channels ──────────────────────────────────────────────────────────────────

async function loadChannels() {
  const channels = await api("/channels");
  const box = $("channels");
  box.innerHTML = "";
  if (channels.length === 0) {
    box.innerHTML = '<div class="empty">No channels yet.</div>';
  }
  for (const c of channels) {
    const el = document.createElement("div");
    el.className = "item" + (c.channelId === state.channelId ? " active" : "");
    el.innerHTML = `<div>${escapeHtml(c.name)}</div>
      <div class="sub">${fmtTime(c.createdAt)}</div>`;
    el.onclick = () => selectChannel(c.channelId);
    box.appendChild(el);
  }
}

async function selectChannel(channelId) {
  state.channelId = channelId;
  state.threadId = null;
  stopPolling();
  await loadChannels();
  await loadThreads();
  renderMessages([]);
  $("composer").hidden = true;
  $("thread-heading").textContent = "Messages";
}

// ── threads ───────────────────────────────────────────────────────────────────

async function loadThreads() {
  const box = $("threads");
  box.innerHTML = "";
  if (!state.channelId) {
    box.innerHTML = '<div class="empty">Pick a channel.</div>';
    return;
  }
  const threads = await api(`/channels/${state.channelId}/threads`);
  if (threads.length === 0) {
    box.innerHTML = '<div class="empty">No threads yet.</div>';
  }
  for (const t of threads) {
    const el = document.createElement("div");
    el.className = "item" + (t.threadId === state.threadId ? " active" : "");
    el.innerHTML = `<div>${escapeHtml(t.title)}</div>
      <div class="sub">${fmtTime(t.updatedAt)}</div>`;
    el.onclick = () => selectThread(t.threadId, t.title);
    box.appendChild(el);
  }
}

async function selectThread(threadId, title) {
  state.threadId = threadId;
  await loadThreads();
  $("thread-heading").textContent = title;
  $("composer").hidden = false;
  await loadMessages();
  startPolling();
}

// ── messages ──────────────────────────────────────────────────────────────────

// Initial full read (§4) — establishes the rendered set and the poll high-water
// mark. Subsequent updates arrive through the incremental poll, never a re-read.
async function loadMessages() {
  const msgs = await api(`/threads/${state.threadId}/messages`);
  renderMessages(msgs);
}

function renderMessages(msgs) {
  const box = $("messages");
  box.innerHTML = "";
  state.seen = new Set();
  state.lastCreatedAt = 0;
  if (!msgs || msgs.length === 0) {
    box.innerHTML = '<div class="empty">No messages yet.</div>';
    return;
  }
  for (const m of msgs) appendMessage(m);
  box.scrollTop = box.scrollHeight;
}

// Append one message, tracking it for de-dupe and advancing the high-water mark.
// Returns false if the message was already rendered (poll re-delivery). Note the
// `?since=` poll rows carry `authorId` but no `displayName`, so newly polled
// messages show the id while the initial full read shows the display name — a
// cosmetic gap left for the M2 web pass (K-014), not a server change here.
function appendMessage(m) {
  if (m.msgId && state.seen.has(m.msgId)) return false;
  const box = $("messages");
  const placeholder = box.querySelector(".empty");
  if (placeholder) placeholder.remove();
  const el = document.createElement("div");
  // Reader @-mention highlighting is a since-read (§9 isMention flag) concern,
  // deferred to the M2 web pass (K-014) — not wired here.
  el.className = "msg";
  const who = m.displayName || m.authorId || "unknown";
  el.innerHTML = `<span class="who">${escapeHtml(who)}</span>
    <span class="meta">${fmtTime(m.createdAt)}</span>
    <div>${escapeHtml(m.text)}</div>`;
  box.appendChild(el);
  if (m.msgId) state.seen.add(m.msgId);
  if (typeof m.createdAt === "number" && m.createdAt > state.lastCreatedAt) {
    state.lastCreatedAt = m.createdAt;
  }
  return true;
}

// Incremental, bounded catch-up of the open thread. `since` is the high-water
// timestamp (plain `>`); `limit` caps the window. Never touches a cursor (the
// server's explicit-`since` path is a pure read), so browser polls are free.
async function pollMessages() {
  const tid = state.threadId;
  if (!tid) return;
  const rows = await api(
    `/threads/${tid}/messages?since=${state.lastCreatedAt || 0}&limit=${POLL_LIMIT}`,
  );
  if (tid !== state.threadId) return; // thread switched while the fetch was in flight
  const box = $("messages");
  const atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 40;
  let added = 0;
  for (const m of rows) if (appendMessage(m)) added++;
  if (added && atBottom) box.scrollTop = box.scrollHeight;
}

function stopPolling() {
  if (state.pollTimer) { clearInterval(state.pollTimer); state.pollTimer = null; }
}

// Background poll for the open thread. Errors are swallowed here (a transient
// fetch failure shouldn't spam the toast every tick); user-triggered polls
// surface errors through `guard`.
function startPolling() {
  stopPolling();
  state.pollTimer = setInterval(() => { pollMessages().catch(() => {}); }, POLL_MS);
}

// ── search ────────────────────────────────────────────────────────────────────

async function runSearch(q) {
  const hits = await api(`/search?q=${encodeURIComponent(q)}`);
  const list = $("results-list");
  list.innerHTML = "";
  if (hits.length === 0) {
    list.innerHTML = '<div class="empty">No matches.</div>';
  }
  for (const h of hits) {
    const el = document.createElement("div");
    el.className = "item";
    el.innerHTML = `<div>${escapeHtml(h.text)}</div>
      <div class="sub">${fmtTime(h.createdAt)} · score ${Number(h.score).toFixed(3)}</div>`;
    // Search rows carry `threadId` (denormalized, K-007) — open the thread on click.
    if (h.threadId) el.onclick = () => openThreadFromSearch(h.threadId);
    list.appendChild(el);
  }
  $("results").style.display = "block";
}

// Jump to the thread a search hit belongs to. Only `threadId` is on the row (no
// title/channel), so the heading falls back to the id; the thread read then
// populates the messages. A title lookup would need a new server endpoint (out
// of scope for this client-only change).
function openThreadFromSearch(threadId) {
  if (!threadId) return;
  $("results").style.display = "none";
  guard(() => selectThread(threadId, threadId));
}

// ── helpers & wiring ────────────────────────────────────────────────────────────

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

// Inline, non-blocking notices. One shared toast element; `kind` picks the
// styling ("error" | "notice") and the auto-dismiss delay. Replaces the old
// blocking `alert()` calls.
let toastTimer = null;
function showToast(msg, kind) {
  const el = $("toast");
  el.textContent = msg;
  el.className = kind;
  el.hidden = false;
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, kind === "error" ? 6000 : 5000);
}
const showError = (msg) => showToast(msg, "error");
const showNotice = (msg) => showToast(msg, "notice");

async function guard(fn) {
  try { await fn(); } catch (e) { showError(e.message); }
}

$("new-channel").addEventListener("submit", (e) => {
  e.preventDefault();
  const name = $("channel-name").value.trim();
  if (!name) return;
  guard(async () => {
    const c = await api("/channels", { method: "POST", body: JSON.stringify({ name }) });
    $("channel-name").value = "";
    await selectChannel(c.channelId);
  });
});

$("new-thread").addEventListener("submit", (e) => {
  e.preventDefault();
  const title = $("thread-title").value.trim();
  if (!title || !state.channelId) return;
  guard(async () => {
    const t = await api(`/channels/${state.channelId}/threads`, {
      method: "POST", body: JSON.stringify({ title }),
    });
    $("thread-title").value = "";
    await selectThread(t.threadId, t.title);
  });
});

$("composer").addEventListener("submit", (e) => {
  e.preventDefault();
  const text = $("message-text").value.trim();
  if (!text || !state.threadId) return;
  guard(async () => {
    const mentions = parseMentions(text);
    try {
      await postMessage(text, mentions);
    } catch (err) {
      // The server rejects unknown members (M1 has no roster endpoint to pre-check).
      // Don't lose the message: resend as plain text and tell the user which
      // @-handles didn't resolve. Match on the machine-readable error code, not
      // the message text: the 400 `detail` is the member list, not the class name.
      if (mentions.length && err.code === "UnknownMemberError") {
        await postMessage(text, []);
        showNotice("Message sent, but these @-handles weren't recognised and were " +
                   "not linked as mentions: " + mentions.join(", "));
      } else {
        throw err;
      }
    }
    $("message-text").value = "";
    await pollMessages();
  });
});

async function postMessage(text, mentions) {
  return api(`/threads/${state.threadId}/messages`, {
    method: "POST",
    body: JSON.stringify(mentions.length ? { text, mentions } : { text }),
  });
}

$("search-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const q = $("search-q").value.trim();
  if (!q) return;
  guard(() => runSearch(q));
});

$("results-close").addEventListener("click", () => {
  $("results").style.display = "none";
});

// initial load
guard(loadChannels);
