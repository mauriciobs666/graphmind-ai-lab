"use strict";
// The inline run cue's "most relevant run" tie-break (K-036 web-api-coverage
// plan §3.3.2; review finding m3 — required, not optional, coverage).
//
// Given the list a thread's `GET /threads/{id}/workflow-runs` returns (any
// order the server sends — today newest-`startedAt`-first, but this function
// does not rely on that), pick the one the inline cue should show:
//   - a non-terminal run (`running`/`waiting`) always beats a terminal one
//     (`done`/`failed`), regardless of recency;
//   - among runs of the same terminality class, the most recent `startedAt`
//     wins.
//
// Dependency-free on purpose: loaded via a plain <script> tag in the browser
// (no bundler in `web/`) and via `require()` from a bare `node` test with no
// DOM (§5.2's explicit "no framework, no DOM, no new dependency" bar).

function isTerminalRunStatus(status) {
  return status === "done" || status === "failed";
}

function selectMostRelevantRun(runs) {
  if (!runs || runs.length === 0) return null;
  let best = null;
  for (const run of runs) {
    if (best === null) {
      best = run;
      continue;
    }
    const runTerminal = isTerminalRunStatus(run.status);
    const bestTerminal = isTerminalRunStatus(best.status);
    if (runTerminal !== bestTerminal) {
      if (!runTerminal) best = run; // non-terminal beats terminal, unconditionally
      continue;
    }
    if ((run.startedAt || 0) > (best.startedAt || 0)) best = run; // same class: most recent wins
  }
  return best;
}

// UMD-ish export: CommonJS for the bare-`node` test, `window` for the browser.
if (typeof module !== "undefined" && module.exports) {
  module.exports = { selectMostRelevantRun, isTerminalRunStatus };
}
if (typeof window !== "undefined") {
  window.selectMostRelevantRun = selectMostRelevantRun;
  window.isTerminalRunStatus = isTerminalRunStatus;
}
