// oracle_session.mjs - the scripted storefront session of the runtime-oracle capture (P1-47;
// docs/plans/connected-code-graph-p1-ml.md §6.5 source 1). Run with the WINDOWS node.exe (no Linux node here):
//
//   node.exe oracle_session.mjs --base http://127.0.0.1:PORT --out C:\Windows\Temp\ccg-oracle-out\spa-raw.jsonl
//            [--chrome-port 9333] [--turn-timeout-s 150]        (env ORACLE_PRESENTER_KEY = the presenter key)
//
// It launches Windows Chrome headless, drives the built SPA (served at <base>/shop/, built with source maps) over raw CDP
// at a 390x844 mobile viewport, and records every fetch the page's application code issues: {method, url, stack}
// (the stack is mapped to `src/api/endpoints.ts` offline by `oracle_capture.py attribute-spa`). A step that cannot
// complete is recorded as a step outcome, never skipped silently. stdout: one JSON line per step, then a summary.
//
// Coverage (every screen and action of salesperson/src/routes.tsx, App paths '/' and '/presenter'):
//   S01 join screen + 4 header sheets before join (health)   S07 order sheet: cancel the order (order/advance cancel)
//   S02 join (session)                                       S08 round 2: add, place with address (one retry)
//   S03 chat turn: catalog question (messages, state)        S09 order sheet: fulfil, deliver (order/advance)
//   S04 chat turn: add to cart                               S10 profile sheet, session reset (confirm)
//   S05 place the order with an address (one retry turn)     S11 presenter: bad key, right key, roster
//   S06 catalog and cart sheets (state, catalog)             S12 presenter reset-everyone (confirm)
//   S13 the salesperson/tests/e2e/ actions, by mapping: mobile-shell.spec = S01, S10; presenter.spec = S11, S12
// A step that finds no order after the retry fails with "model did not place an order" (recorded, not skipped).
// The post-reset language step is not driven: its join posts displayName '' and the server answers 422 (S10 of v2).
// The tests/e2e specs mock /shop/api through page.route(); they are NOT executed under Playwright here (no Linux node,
// no Playwright browser on the Windows side). Their user actions are the steps named in S14, run against the real server.
import { spawn } from 'node:child_process';
import { appendFileSync, mkdirSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';

const argv = process.argv.slice(2);
const arg = (k, d) => { const i = argv.indexOf(k); return i >= 0 ? argv[i + 1] : d; };
const BASE = arg('--base'); const OUT = arg('--out');
const PORT = Number(arg('--chrome-port', '9333')); const TURN_S = Number(arg('--turn-timeout-s', '150'));
const KEY = process.env.ORACLE_PRESENTER_KEY || '';
if (!BASE || !OUT) { console.error('usage: --base URL --out FILE'); process.exit(2); }
mkdirSync(dirname(OUT), { recursive: true }); writeFileSync(OUT, '');

const CHROME = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const PROFILE = `C:\\Windows\\Temp\\ccg-oracle-prof-${PORT}`;
const chrome = spawn(CHROME, ['--headless=new', '--disable-gpu', `--remote-debugging-port=${PORT}`, `--user-data-dir=${PROFILE}`,
  '--no-first-run', 'about:blank'], { stdio: 'ignore' });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function target() {
  for (let i = 0; i < 60; i++) {
    try {
      const l = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json();
      const p = l.find((t) => t.type === 'page'); if (p) return p;
    } catch { /* not up yet */ }
    await sleep(500);
  }
  throw new Error('chrome did not expose a page target');
}
const t = await target();
const ws = new WebSocket(t.webSocketDebuggerUrl);
await new Promise((r, j) => { ws.onopen = r; ws.onerror = j; });
let id = 0; const pending = new Map();
ws.onmessage = (m) => {
  const d = JSON.parse(m.data);
  if (d.id && pending.has(d.id)) { pending.get(d.id)(d); pending.delete(d.id); return; }
  if (d.method === 'Runtime.bindingCalled' && d.params.name === 'oracleRecord') {
    appendFileSync(OUT, d.params.payload + '\n');
  }
};
const send = (method, params = {}) => new Promise((r) => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
const ev = async (expression) => {
  const r = await send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
  if (r.result?.exceptionDetails) throw new Error(r.result.exceptionDetails.text + ' ' + (r.result.exceptionDetails.exception?.description ?? ''));
  return r.result?.result?.value;
};
const until = async (expr, ms = 15000, what = expr) => {
  const end = Date.now() + ms;
  while (Date.now() < end) { if (await ev(expr)) return true; await sleep(250); }
  throw new Error(`timeout waiting for: ${what}`);
};

await send('Page.enable'); await send('Runtime.enable');
await send('Runtime.addBinding', { name: 'oracleRecord' });
await send('Emulation.setDeviceMetricsOverride', { width: 390, height: 844, deviceScaleFactor: 2, mobile: true });
await send('Page.addScriptToEvaluateOnNewDocument', { source: `(() => {
  const of = window.fetch;
  window.fetch = function (input, init) {
    try {
      const url = typeof input === 'string' ? input : (input && input.url) || String(input);
      const method = ((init && init.method) || (input && input.method) || 'GET').toUpperCase();
      window.oracleRecord(JSON.stringify({ method, url, stack: new Error().stack }));
    } catch (e) { /* never break the app */ }
    return of.apply(this, arguments);
  };
})();` });

// page helpers (evaluated in the page)
const H = `(() => {
  const vis = (e) => !!(e.offsetWidth || e.offsetHeight || e.getClientRects().length);
  window.__o = {
    btn: (label) => [...document.querySelectorAll('button')].filter(vis).find((b) => (b.getAttribute('aria-label') || b.textContent).trim() === label),
    btnRe: (re) => [...document.querySelectorAll('button')].filter(vis).find((b) => re.test((b.getAttribute('aria-label') || b.textContent).trim())),
    click: (label) => { const b = window.__o.btn(label); if (!b || b.disabled) return false; b.click(); return true; },
    text: () => document.body.innerText,
    set: (sel, v) => { const e = document.querySelector(sel); if (!e) return false;
      const proto = e.tagName === 'SELECT' ? HTMLSelectElement.prototype : e.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
      Object.getOwnPropertyDescriptor(proto, 'value').set.call(e, v);
      e.dispatchEvent(new Event(e.tagName === 'SELECT' ? 'change' : 'input', { bubbles: true })); return true; },
  };
})()`;
const goto = async (path) => { await send('Page.navigate', { url: BASE.replace(/\/$/, '') + path }); await sleep(1500); await ev(H); };
const q = (s) => JSON.stringify(s);
const clickL = async (label, ms = 15000) => { await until(`window.__o.btn(${q(label)}) && !window.__o.btn(${q(label)}).disabled`, ms, `button ${label}`); await ev(`window.__o.click(${q(label)})`); };
const closeSheet = async () => { await ev(`(() => { const b=[...document.querySelectorAll('button')].filter(e=>e.offsetWidth&&/^Close$/.test((e.getAttribute('aria-label')||e.textContent).trim())); if(b.length) b[b.length-1].click(); return true; })()`); await sleep(500); };
const sheet = async (label) => { await clickL(label); await sleep(1200); };
const turn = async (text) => {
  await until(`!!document.querySelector('textarea, input[placeholder^="Type a message"]')`, 20000, 'composer');
  await ev(`window.__o.set('textarea, input[placeholder^="Type a message"]', ${q(text)})`);
  await sleep(300); await clickL('Send');
  await sleep(3000);
  let quiet = 0; const end = Date.now() + TURN_S * 1000;
  while (Date.now() < end && quiet < 3) {
    const busy = await ev(`/Thinking…|ahead of you|first in line/.test(window.__o.text())`);
    quiet = busy ? 0 : quiet + 1; await sleep(1500);
  }
  return quiet >= 3 ? 'turn-idle' : 'turn-timeout';
};

const results = [];
async function step(name, fn) {
  let outcome = 'ok', detail = '';
  try { detail = (await fn()) ?? ''; } catch (e) { outcome = 'failed'; detail = String(e.message).slice(0, 200); }
  const r = { step: name, outcome, detail }; results.push(r); console.log(JSON.stringify(r));
}
const join = async (name, lang) => {
  await until(`!!document.querySelector('select')`, 20000, 'join form');
  if (name) await ev(`window.__o.set('input', ${q(name)})`);
  await ev(`window.__o.set('select', ${q(lang)})`); await sleep(300);
  await clickL(name ? 'Join' : 'Continue');
};

await step('S01 join screen and the four sheets before join', async () => {
  await goto('/shop/'); await until(`/Welcome to the store/.test(window.__o.text())`, 20000, 'join screen');
  for (const l of ['Browse catalog', 'Cart', 'Order status', 'Profile']) { await sheet(l); await closeSheet(); }
});
await step('S02 join', async () => { await join('Oracle One', 'en'); await until(`!!document.querySelector('textarea, input[placeholder^="Type a message"]')`, 30000, 'chat view'); });
const ADDR = 'Deliver to Rua das Flores 100, Sao Paulo, Brazil.';
const orderSeen = async () => {                      // opens the Order sheet; true iff an order exists (Cancel order shown)
  await sheet('Order status'); await sleep(2000);
  const seen = await ev(`!!window.__o.btn('Cancel order')`);
  if (!seen) await closeSheet();
  return seen;
};
const placeOrder = async () => {                     // address-bearing turn, then one retry turn if still no order
  const a = await turn(`Please place my order now. ${ADDR} Yes, that address is correct.`);
  if (await orderSeen()) return a;
  const b = await turn(`Yes, please go ahead and place the order for the items in my cart, delivering to ${ADDR}`);
  if (await orderSeen()) return `${a},${b}`;
  throw new Error('model did not place an order');
};
await step('S03 chat turn: catalog question', () => turn('Hi! What products do you sell?'));
await step('S04 chat turn: add to cart', () => turn('Please add one of the cheapest products to my cart.'));
await step('S05 chat turns: place the order with an address (one retry)', placeOrder);   // leaves the Order sheet open
await step('S06 catalog and cart sheets', async () => {
  await closeSheet(); await sheet('Browse catalog'); await sleep(1500); await closeSheet();
  await sheet('Cart'); await sleep(1500); await closeSheet();
});
await step('S07 order sheet: cancel the order (round 1)', async () => {
  await sheet('Order status'); await sleep(1500); await clickL('Cancel order', 8000); await sleep(2500); await closeSheet();
});
await step('S08 round 2: add, place with address (one retry)', async () => {
  const a = await turn('Please add one of the cheapest products to my cart.');
  return a + ',' + await placeOrder();
});
await step('S09 order sheet: fulfil then deliver (round 2)', async () => {
  await clickL('Simulate: mark fulfilled', 8000); await sleep(2500);
  await clickL('Simulate: mark delivered', 8000); await sleep(2500); await closeSheet();
});
await step('S10 profile sheet and session reset (last SPA-participant action)', async () => {
  await sheet('Profile'); await sleep(1500);
  await clickL('Reset my session'); await clickL('Yes, reset'); await sleep(3000);
  if (await ev(`!!window.__o.btn('Close')`)) await closeSheet();
});
await step('S11 presenter key and roster', async () => {
  await goto('/shop/presenter'); await until(`/Presenter key/.test(window.__o.text())`, 20000, 'key screen');
  await ev(`window.__o.set('input', 'definitely-wrong-key')`); await clickL('Enter'); await until(`/not accepted/.test(window.__o.text())`, 15000, 'rejection');
  if (!KEY) throw new Error('ORACLE_PRESENTER_KEY not set: right-key path not run');
  await ev(`window.__o.set('input', ${q(KEY)})`); await clickL('Enter'); await until(`/Participants/.test(window.__o.text())`, 20000, 'roster'); await sleep(2500);
});
await step('S12 presenter reset everyone', async () => {
  await clickL('Reset everyone'); await clickL('Yes, reset everyone'); await sleep(4000);
});
await step('S13 e2e-spec actions covered by S01/S10 (mobile-shell) and S11/S12 (presenter)', async () => 'mapping only');

ws.close(); chrome.kill();
console.log(JSON.stringify({ summary: { steps: results.length, ok: results.filter((r) => r.outcome === 'ok').length, failed: results.filter((r) => r.outcome !== 'ok').length } }));
process.exit(0);
