"use strict";
// Minimal browser client for falkor-chat's M1 REST API (DESIGN §14.5).
// Same-origin: the FastAPI process serves this file and the API, so no CORS.
// Flow: channels -> threads -> messages, plus workspace-wide full-text search.

const state = { channelId: null, threadId: null };

const $ = (id) => document.getElementById(id);

async function api(path, opts) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || body.error || JSON.stringify(body);
    } catch (_) { /* non-JSON error body */ }
    throw new Error(`${res.status}: ${detail}`);
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
}

// ── messages ──────────────────────────────────────────────────────────────────

async function loadMessages() {
  const msgs = await api(`/threads/${state.threadId}/messages`);
  renderMessages(msgs);
}

function renderMessages(msgs) {
  const box = $("messages");
  box.innerHTML = "";
  if (!msgs || msgs.length === 0) {
    box.innerHTML = '<div class="empty">No messages yet.</div>';
    return;
  }
  for (const m of msgs) {
    const el = document.createElement("div");
    el.className = "msg" + (m.isMention ? " mention" : "");
    const who = m.displayName || m.authorId || "unknown";
    el.innerHTML = `<span class="who">${escapeHtml(who)}</span>
      <span class="meta">${fmtTime(m.createdAt)}</span>
      <div>${escapeHtml(m.text)}</div>`;
    box.appendChild(el);
  }
  box.scrollTop = box.scrollHeight;
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
    list.appendChild(el);
  }
  $("results").style.display = "block";
}

// ── helpers & wiring ────────────────────────────────────────────────────────────

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

async function guard(fn) {
  try { await fn(); } catch (e) { alert(e.message); }
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
    await api(`/threads/${state.threadId}/messages`, {
      method: "POST",
      body: JSON.stringify(mentions.length ? { text, mentions } : { text }),
    });
    $("message-text").value = "";
    await loadMessages();
  });
});

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
