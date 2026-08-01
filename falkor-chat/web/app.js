"use strict";
// Minimal browser client for falkor-chat's M1 REST API (DESIGN §14.5).
// Same-origin: the FastAPI process serves this file and the API, so no CORS.
// Flow: channels -> threads -> messages, plus workspace-wide full-text search.

// `lastCreatedAt`/`seen` drive the incremental `?since=&limit=` poll of the open
// thread: `lastCreatedAt` is the high-water timestamp, `seen` de-dupes by msgId
// (the plain `>` since-read may re-deliver a millisecond tie — OQ3). `pollTimer`
// is the background interval handle for the currently open thread.
// `runPanelId`/`runPollTimer`/`runTraceShown` drive the run detail panel's own
// poll/stop lifecycle (K-036 U6, §3.2 — separate from the message poll, started
// on open and stopped on close). `participantsOpen` tracks the collapsed/expanded
// state of the thread participants toggle (U7), reset on every thread switch.
const state = {
  channelId: null, threadId: null,
  lastCreatedAt: 0, seen: new Set(), pollTimer: null,
  runPanelId: null, runPollTimer: null, runTraceShown: false,
  runWaitingKey: null,
  participantsOpen: false,
};

// Background poll cadence for the open thread. Bounded by `POLL_LIMIT` per fetch
// so a poll never walks the full NEXT* chain past the server's 1000ms TIMEOUT.
const POLL_MS = 3000;
const POLL_LIMIT = 50;

// The demo workspace id. Only `ws` path segments the M1 REST surface treats as
// **descriptive** land here (`/workspaces/{ws}/...` routes — tenancy actually
// comes from the server-side `get_context` seam, per `api.py`'s own comments at
// those routes); kept in sync with the demo default (`FALKORCHAT_WS_ID`/
// `scripts/*.sh`, "acme").
const DEMO_WS = "acme";

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
  closeRunPanel();
  collapseParticipants();
  await loadChannels();
  await loadThreads();
  renderMessages([]);
  $("composer").hidden = true;
  $("thread-heading").textContent = "Messages";
  $("run-cue").style.display = "none";
  $("run-cue").innerHTML = "";
  $("participants-toggle").disabled = true;
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
  closeRunPanel();
  collapseParticipants();
  await loadThreads();
  $("thread-heading").textContent = title;
  $("composer").hidden = false;
  $("participants-toggle").disabled = false;
  await loadMessages();
  await updateRunCue();
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
// Returns false if the message was already rendered (poll re-delivery).
//
// K-014 rendering: agent-authored messages (`role:"assistant"`) are styled
// distinctly with an "AI" badge; reader @-mentions light up via the since-read
// `isMention` flag (only the poll path carries it — the initial §4 full read has
// no flag, so highlighting comes alive on the next poll). Both the full read and
// the since-read now carry `displayName` (QUERIES.md §4/§9), so polled rows show
// the member name instead of the raw id.
function appendMessage(m) {
  if (m.msgId && state.seen.has(m.msgId)) return false;
  const box = $("messages");
  const placeholder = box.querySelector(".empty");
  if (placeholder) placeholder.remove();
  const el = document.createElement("div");
  const classes = ["msg"];
  if (m.role === "assistant") classes.push("assistant");
  if (m.isMention) classes.push("mention"); // reader @-mention (since-read flag)
  el.className = classes.join(" ");
  const who = m.displayName || m.authorId || "unknown";
  const badge = m.role === "assistant" ? ' <span class="badge">AI</span>' : "";
  el.innerHTML = `<span class="who">${escapeHtml(who)}</span>${badge}
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
// surface errors through `guard`. The inline run cue (FR-2) piggybacks on this
// same cadence (plan §3.2) — one more lightweight fetch alongside the message
// catch-up, only while a thread is open.
function startPolling() {
  stopPolling();
  state.pollTimer = setInterval(() => {
    pollMessages().catch(() => {});
    updateRunCue().catch(() => {});
  }, POLL_MS);
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

// ── workflow defs viewer (FR-1) ─────────────────────────────────────────────────
// Static content: defs are immutable once published, so there is no polling here
// (§3.2). The header button is always visible; the panel and its content are
// fetched/rendered only once clicked (§3.3's AC-5 reading).

function toggleDefsPanel() {
  if ($("defs-panel").style.display === "block") {
    closeDefsPanel();
  } else {
    guard(openDefsPanel);
  }
}

async function openDefsPanel() {
  closeRunPanel(); // no two right-side overlays open at once
  showDefsList();
  await loadDefsList();
  $("defs-panel").style.display = "block";
}

function closeDefsPanel() {
  $("defs-panel").style.display = "none";
}

function showDefsList() {
  $("defs-list-view").hidden = false;
  $("defs-detail-view").hidden = true;
}

function showDefsDetailView() {
  $("defs-list-view").hidden = true;
  $("defs-detail-view").hidden = false;
}

async function loadDefsList() {
  const defs = await api("/workflow-defs");
  const box = $("defs-list");
  box.innerHTML = "";
  if (defs.length === 0) {
    box.innerHTML = '<div class="empty">No workflow definitions published.</div>';
  }
  for (const d of defs) {
    const el = document.createElement("div");
    el.className = "item";
    el.innerHTML = `<div>${escapeHtml(d.key)} <span class="sub">${escapeHtml(d.version)}</span></div>
      <div class="sub">${escapeHtml(d.name || "")}</div>`;
    el.onclick = () => guard(() => selectDef(d.key, d.version));
    box.appendChild(el);
  }
}

// Fetch one def's full structure and render its steps + transitions as plain
// tables (FR-1: text is sufficient, no graph-drawing library, FR-9 minimalism).
async function selectDef(key, version) {
  const structure = await api(
    `/workflow-defs/${encodeURIComponent(key)}/versions/${encodeURIComponent(version)}`,
  );
  renderDefDetail(structure);
  showDefsDetailView();
}

function renderDefDetail(s) {
  const startKeys = s.startKeys || (s.startKey ? [s.startKey] : []);
  const stepRows = s.steps.map((step) => `
    <tr>
      <td>${escapeHtml(step.key)}${startKeys.includes(step.key) ? " <em>(start)</em>" : ""}</td>
      <td>${escapeHtml(step.type)}</td>
      <td class="config">${escapeHtml(step.config)}</td>
    </tr>`).join("");
  const transitionRows = s.transitions.map((t) => `
    <tr>
      <td>${escapeHtml(t.from)}</td>
      <td>${escapeHtml(t.to)}</td>
      <td>${escapeHtml(t.on)}</td>
      <td class="config">${escapeHtml(t.guard)}</td>
    </tr>`).join("");
  $("defs-detail-content").innerHTML = `
    <div><strong>${escapeHtml(s.name)}</strong>
      <div class="sub">${escapeHtml(s.key)} ${escapeHtml(s.version)} · ${escapeHtml(s.kind)}</div>
    </div>
    <h3>Steps (${s.stepCount})</h3>
    <table><thead><tr><th>Key</th><th>Type</th><th>Config</th></tr></thead>
      <tbody>${stepRows || '<tr><td colspan="3" class="empty">No steps.</td></tr>'}</tbody></table>
    <h3>Transitions (${s.transitionCount})</h3>
    <table><thead><tr><th>From</th><th>To</th><th>On</th><th>Guard</th></tr></thead>
      <tbody>${transitionRows || '<tr><td colspan="4" class="empty">No transitions.</td></tr>'}</tbody></table>`;
}

// ── workflow run cue + detail panel (FR-2/3/4/5/6/7, U6/U9) ─────────────────────
// The inline cue piggybacks on the existing per-thread poll loop (`startPolling`,
// above) and is zero-footprint when the thread has no runs (§3.3's AC-5 reading).
// The panel runs its own poll/stop lifecycle, started on open and stopped on
// close, so a user who never opens it costs nothing beyond the one thread-runs
// check. "Most relevant run" is `run-select.js`'s dependency-free pure function
// (§5.2, review finding m3 — plain `node`-runnable unit tests live at
// `web/tests/run-select.test.js`).

async function updateRunCue() {
  const tid = state.threadId;
  if (!tid) return;
  const runs = await api(`/threads/${tid}/workflow-runs`);
  if (tid !== state.threadId) return; // thread switched while the fetch was in flight
  renderRunCue(runs);
}

function renderRunCue(runs) {
  const cue = $("run-cue");
  const best = selectMostRelevantRun(runs);
  if (!best) {
    cue.style.display = "none";
    cue.innerHTML = "";
    return;
  }
  cue.innerHTML = "";
  const label = document.createElement("span");
  label.textContent = `${best.defKey} ${best.defVersion} — ${best.status}`;
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "mini-btn";
  btn.textContent = "View";
  btn.onclick = () => guard(() => openRunPanel(best.runId));
  cue.appendChild(label);
  cue.appendChild(btn);
  cue.style.display = "flex";
}

function stopRunPolling() {
  if (state.runPollTimer) { clearInterval(state.runPollTimer); state.runPollTimer = null; }
}

function startRunPolling() {
  stopRunPolling();
  state.runPollTimer = setInterval(() => { refreshRunPanel().catch(() => {}); }, POLL_MS);
}

async function openRunPanel(runId) {
  closeDefsPanel(); // no two right-side overlays open at once
  state.runPanelId = runId;
  state.runTraceShown = false;
  $("run-trace").hidden = true;
  $("run-trace").textContent = "";
  $("run-trace-toggle").textContent = "Show trace";
  $("run-panel").style.display = "block";
  await refreshRunPanel();
  startRunPolling();
}

function closeRunPanel() {
  $("run-panel").style.display = "none";
  state.runPanelId = null;
  stopRunPolling();
}

// Re-reads status/step-runs (+ trace, if the toggle is open) and re-renders.
// Called on open, on every poll tick while open, and immediately after a
// successful structured-input submit (mirrors `pollMessages()` right after
// `postMessage()`, `app.js`'s existing composer handler — plan §3.3.3/review m2).
async function refreshRunPanel() {
  const runId = state.runPanelId;
  if (!runId) return;
  const [run, steps] = await Promise.all([
    api(`/workflow-runs/${encodeURIComponent(runId)}`),
    api(`/workflow-runs/${encodeURIComponent(runId)}/step-runs`),
  ]);
  if (runId !== state.runPanelId) return; // panel closed/switched while in flight
  renderRunPanel(run, steps);
  if (state.runTraceShown) await loadRunTrace(runId);
}

function renderRunPanel(run, steps) {
  $("run-summary").innerHTML = `
    <div><strong>${escapeHtml(run.defKey)}</strong> <span class="sub">${escapeHtml(run.defVersion)}</span></div>
    <div class="sub">status: ${escapeHtml(run.status)} · started ${fmtTime(run.startedAt)}${
      run.endedAt ? " · ended " + fmtTime(run.endedAt) : ""
    }</div>`;

  const stepsBox = $("run-steps");
  stepsBox.innerHTML = steps.length
    ? steps.map((s) => `<div class="run-step-row">
        <span>${escapeHtml(s.stepKey)}</span>
        <span class="sub">${escapeHtml(s.status)}${s.endedAt ? " · " + fmtTime(s.endedAt) : ""}</span>
      </div>`).join("")
    : '<div class="empty">No steps yet.</div>';

  renderRunFailure(run);
  guard(() => renderWaitingForm(run));
}

// FR-7: a failed run's `ctx` (opaque JSON string) carries `{"error": "..."}`
// stamped by the executor's fault net on every fail path (budget exhaustion,
// drive fault) — `executor._fail_with_note` per the plan §2.1 table.
function renderRunFailure(run) {
  const box = $("run-failure");
  if (run.status !== "failed") {
    box.hidden = true;
    box.textContent = "";
    return;
  }
  let reason = "unknown reason";
  try {
    const parsed = JSON.parse(run.ctx || "{}");
    if (parsed && parsed.error) reason = parsed.error;
  } catch (_) { /* opaque/unparseable ctx — fall back to the generic reason */ }
  box.textContent = `Failed: ${reason}`;
  box.hidden = false;
}

async function loadRunTrace(runId) {
  const events = await api(`/workflow-runs/${encodeURIComponent(runId)}/trace`);
  if (runId !== state.runPanelId) return;
  const box = $("run-trace");
  box.textContent = events.length
    ? events.map((e) => `[${fmtTime(e.at)}] ${e.stepKey} · ${e.kind}${e.payload ? " " + e.payload : ""}`).join("\n")
    : "No trace events (not a debug run).";
}

function toggleRunTrace() {
  const runId = state.runPanelId;
  if (!runId) return;
  state.runTraceShown = !state.runTraceShown;
  $("run-trace").hidden = !state.runTraceShown;
  $("run-trace-toggle").textContent = state.runTraceShown ? "Hide trace" : "Show trace";
  if (state.runTraceShown) guard(() => loadRunTrace(runId));
}

// FR-5/FR-6: when the run is `waiting`, render the parked step's `config.prompt`
// plus a form built from `config.fields` (one input per top-level key; a
// `config.expects[field]` list renders as a `<select>` instead of free text).
// The step's own snapshot (keyed by `run.atStepKey`) is read from
// `GET /workspaces/{ws}/snapshots/{key}/versions/{version}` — the same
// structure shape the defs viewer already renders, `config` passed through
// verbatim as an opaque JSON string (rule 8).
async function renderWaitingForm(run) {
  const box = $("run-waiting-form");
  if (run.status !== "waiting" || !run.atStepKey) {
    state.runWaitingKey = null;
    box.hidden = true;
    box.innerHTML = "";
    return;
  }

  // Poll ticks call this every POLL_MS while the panel is open (`refreshRunPanel`)
  // — but `box.innerHTML = ...` below destroys and rebuilds the live
  // `<form id="run-input-form">`, wiping any text the user has typed but not
  // yet submitted. If the parked step is unchanged since the last render,
  // there's nothing new to show: skip the rebuild (and the snapshot re-fetch)
  // entirely so in-progress input survives repeated ticks (review M1/m4).
  // `submitRunInput` clears `runWaitingKey` on a successful submit so the very
  // next `refreshRunPanel()` always re-renders, even if the run happens to
  // park again on the same step key (plan §3.3.3).
  const key = `${run.runId}:${run.atStepKey}`;
  if (key === state.runWaitingKey && !box.hidden) return;

  let step = null;
  try {
    const snapshot = await api(
      `/workspaces/${DEMO_WS}/snapshots/${encodeURIComponent(run.defKey)}/versions/${encodeURIComponent(run.defVersion)}`,
    );
    step = (snapshot.steps || []).find((s) => s.key === run.atStepKey) || null;
  } catch (_) { /* snapshot read failed — fall through to the generic form below */ }
  if (run.runId !== state.runPanelId) return; // panel moved on while the fetch was in flight

  let config = {};
  if (step) {
    try { config = JSON.parse(step.config || "{}"); } catch (_) { config = {}; }
  }
  const prompt = config.prompt || `Waiting at step "${run.atStepKey}"`;
  const fields = Array.isArray(config.fields) && config.fields.length
    ? config.fields
    : (config.signal ? [config.signal] : []);
  const expects = config.expects && typeof config.expects === "object" ? config.expects : {};

  const fieldsHtml = fields.length
    ? fields.map((f) => {
        const options = Array.isArray(expects[f]) ? expects[f] : null;
        const input = options
          ? `<select name="${escapeHtml(f)}">${
              options.map((o) => `<option value="${escapeHtml(o)}">${escapeHtml(o)}</option>`).join("")
            }</select>`
          : `<input type="text" name="${escapeHtml(f)}" autocomplete="off" />`;
        return `<div class="run-field"><label>${escapeHtml(f)}</label>${input}</div>`;
      }).join("")
    : '<div class="sub">This step declares no fields — submit sends an empty signal.</div>';

  box.innerHTML = `
    <div class="sub">${escapeHtml(prompt)}</div>
    <form id="run-input-form">${fieldsHtml}<button type="submit">Submit</button></form>`;
  box.hidden = false;
  state.runWaitingKey = key;
  $("run-input-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const data = {};
    for (const el of e.target.elements) {
      if (el.name) data[el.name] = el.value;
    }
    guard(() => submitRunInput(run.runId, data));
  });
}

// A rejected submit (400 `WorkflowInputRejectedError`) propagates out of `api()`
// and surfaces via `guard`'s existing `showError` toast — nothing new to build;
// the form is left exactly as the user filled it (no re-render on the throw
// path). On success, immediately re-poll the run instead of waiting for the
// next scheduled tick (plan §3.3.3 / review m2 — mirrors the composer's
// `await pollMessages()` right after `postMessage()`).
async function submitRunInput(runId, data) {
  await api(`/workflow-runs/${encodeURIComponent(runId)}/input`, {
    method: "POST",
    body: JSON.stringify({ input: data }),
  });
  // Force the next render to rebuild the waiting form even if the run somehow
  // parks again on the same step key, so a successful submit always reflects
  // the new state immediately (plan §3.3.3) rather than being skipped by the
  // unchanged-step guard in `renderWaitingForm` above.
  state.runWaitingKey = null;
  if (runId === state.runPanelId) await refreshRunPanel();
  await updateRunCue();
}

// ── thread participants (FR-8/AC-4, U7) ──────────────────────────────────────
// The toggle itself is always present next to `#thread-heading` (§3.3's AC-5
// reading); only its *content* is fetched/rendered on expand. No polling — a
// roster is stable enough to refresh on open/thread-switch only (§3.2).

function collapseParticipants() {
  $("participants-row").style.display = "none";
  $("participants-row").innerHTML = "";
  $("participants-toggle").textContent = "Participants";
  state.participantsOpen = false;
}

function toggleParticipants() {
  if (!state.threadId) return;
  if (state.participantsOpen) {
    collapseParticipants();
  } else {
    guard(openParticipants);
  }
}

async function openParticipants() {
  const rows = await api(`/threads/${state.threadId}/participants`);
  const box = $("participants-row");
  box.innerHTML = rows.length
    ? rows.map((p) => `<span class="chip">${escapeHtml(p.displayName || p.memberId)}${
        p.kind === "Agent" ? ' <span class="badge">AI</span>' : ""
      }</span>`).join("")
    : '<span class="sub">No participants.</span>';
  box.style.display = "flex";
  state.participantsOpen = true;
  $("participants-toggle").textContent = `Participants (${rows.length})`;
}

// ── ready-to-demo banner (FR-10/AC-6, U8) ────────────────────────────────────
// Fetched once on load; not on the 5s freshness bar (§3.2 — readiness isn't
// run progress), so there is no poll loop here — a manual "Recheck" covers it.

async function loadReadiness() {
  const report = await api(`/workspaces/${DEMO_WS}/readiness`);
  renderReadinessBadge(report);
}

function renderReadinessBadge(report) {
  const badge = $("readiness-badge");
  badge.className = "readiness " + (report.ready ? "readiness--ready" : "readiness--not-ready");
  badge.textContent = report.ready ? "Ready to demo" : "Not ready";
  renderReadinessPanel(report);
}

function renderReadinessPanel(report) {
  const problems = (report.defs || []).flatMap((d) => d.problems || []);
  const defsHtml = problems.length
    ? `<ul>${problems.map((p) => `<li>${escapeHtml(p)}</li>`).join("")}</ul>`
    : '<div class="sub">All expected definitions are published, materialized, and in sync.</div>';
  $("readiness-content").innerHTML = defsHtml + renderPostSuccess(report.postSuccess);
}

// Recent triage post-success (K-039 item 3) — a lagging, production-data
// signal, deliberately separate from `ready`/the badge color (plan
// mention-reply-delivery §3.3): never fold this into renderReadinessBadge.
function renderPostSuccess(postSuccess) {
  if (!postSuccess) return "";
  const { sampleSize, postedCount, status } = postSuccess;
  if (status === "no-data") {
    return '<div class="sub post-success">Recent triage post-success: no runs yet.</div>';
  }
  const cls = status === "degraded" ? "post-success post-success--degraded" : "post-success";
  return `<div class="sub ${cls}">Recent triage post-success: ${postedCount}/${sampleSize} replied</div>`;
}

function toggleReadinessPanel() {
  const panel = $("readiness-panel");
  panel.style.display = panel.style.display === "block" ? "none" : "block";
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

$("defs-btn").addEventListener("click", toggleDefsPanel);
$("defs-close").addEventListener("click", closeDefsPanel);
$("defs-back").addEventListener("click", showDefsList);

$("run-close").addEventListener("click", closeRunPanel);
$("run-trace-toggle").addEventListener("click", toggleRunTrace);

$("participants-toggle").addEventListener("click", toggleParticipants);

$("readiness-badge").addEventListener("click", toggleReadinessPanel);
$("readiness-badge").addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggleReadinessPanel(); }
});
$("readiness-recheck").addEventListener("click", () => guard(loadReadiness));

// initial load
guard(loadChannels);
guard(loadReadiness);
