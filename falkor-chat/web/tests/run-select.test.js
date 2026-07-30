"use strict";
// Plain-assertion coverage for `run-select.js`'s tie-break (K-036 plan §5.2,
// review finding m3 — required). No framework, no DOM: run with
//   node web/tests/run-select.test.js
// Uses only Node's built-in `assert` module (not a new `web/` dependency).

const assert = require("node:assert/strict");
const { selectMostRelevantRun, isTerminalRunStatus } = require("../run-select.js");

let failures = 0;

function test(name, fn) {
  try {
    fn();
    console.log(`ok - ${name}`);
  } catch (err) {
    failures++;
    console.error(`FAIL - ${name}`);
    console.error(`       ${err.message}`);
  }
}

test("isTerminalRunStatus: done and failed are terminal", () => {
  assert.equal(isTerminalRunStatus("done"), true);
  assert.equal(isTerminalRunStatus("failed"), true);
});

test("isTerminalRunStatus: running and waiting are not terminal", () => {
  assert.equal(isTerminalRunStatus("running"), false);
  assert.equal(isTerminalRunStatus("waiting"), false);
});

test("empty list returns null", () => {
  assert.equal(selectMostRelevantRun([]), null);
});

test("null/undefined input returns null", () => {
  assert.equal(selectMostRelevantRun(null), null);
  assert.equal(selectMostRelevantRun(undefined), null);
});

test("single run is returned regardless of status", () => {
  const run = { runId: "r1", status: "done", startedAt: 100 };
  assert.equal(selectMostRelevantRun([run]), run);
});

test("a running run beats a done run even if the done run is more recent", () => {
  const running = { runId: "r1", status: "running", startedAt: 100 };
  const done = { runId: "r2", status: "done", startedAt: 999 };
  assert.equal(selectMostRelevantRun([done, running]), running);
  assert.equal(selectMostRelevantRun([running, done]), running);
});

test("a waiting run beats a failed run even if the failed run is more recent", () => {
  const waiting = { runId: "r1", status: "waiting", startedAt: 100 };
  const failed = { runId: "r2", status: "failed", startedAt: 999 };
  assert.equal(selectMostRelevantRun([failed, waiting]), waiting);
});

test("among two non-terminal runs, the most recent startedAt wins", () => {
  const older = { runId: "r1", status: "running", startedAt: 100 };
  const newer = { runId: "r2", status: "waiting", startedAt: 200 };
  assert.equal(selectMostRelevantRun([older, newer]), newer);
  assert.equal(selectMostRelevantRun([newer, older]), newer);
});

test("among two terminal runs, the most recent startedAt wins", () => {
  const older = { runId: "r1", status: "done", startedAt: 100 };
  const newer = { runId: "r2", status: "failed", startedAt: 200 };
  assert.equal(selectMostRelevantRun([older, newer]), newer);
});

test("three-way mix: the sole non-terminal run wins over two terminal ones", () => {
  const done1 = { runId: "r1", status: "done", startedAt: 300 };
  const running = { runId: "r2", status: "running", startedAt: 100 };
  const done2 = { runId: "r3", status: "failed", startedAt: 250 };
  assert.equal(selectMostRelevantRun([done1, running, done2]), running);
});

test("all terminal: the most recent one is chosen", () => {
  const done1 = { runId: "r1", status: "done", startedAt: 300 };
  const done2 = { runId: "r2", status: "failed", startedAt: 500 };
  const done3 = { runId: "r3", status: "done", startedAt: 100 };
  assert.equal(selectMostRelevantRun([done1, done2, done3]), done2);
});

test("exact-tie startedAt keeps the first-seen candidate (stable, no crash)", () => {
  const a = { runId: "r1", status: "running", startedAt: 100 };
  const b = { runId: "r2", status: "waiting", startedAt: 100 };
  assert.equal(selectMostRelevantRun([a, b]), a);
});

if (failures > 0) {
  console.error(`\n${failures} test(s) failed.`);
  process.exitCode = 1;
} else {
  console.log("\nall tests passed.");
}
