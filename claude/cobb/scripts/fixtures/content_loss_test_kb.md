# Content-loss checker self-test fixture

> **Synthetic fixture — not a real KB.** Exists only for `check_content_loss_selftest.py`'s
> mutation test: one section that migrates as a single verbatim claim, one bulleted section that
> migrates one claim per bullet (marker stripped), and one section that migrates as two
> paragraph-boundary claims. Do not migrate, ingest, or cite this file as a real agent knowledge
> base.

## A single-claim heading migrates as one verbatim claim

When a whole heading is judged to be one technique, the resulting claim's text is the section
body verbatim, unchanged down to the last character — including this parenthetical aside (κ≤α)
and this inline code span `like_this`.

## Three independent facts about connection pooling

- Reusing a connection across requests avoids the TCP/TLS handshake cost on every call, but only
  if the pool actually validates a connection's liveness before handing it out — a pool that
  trusts a connection blindly will hand out one the peer already closed.
- A pool sized below the concurrency the service actually sees under load queues callers behind
  each other invisibly — the symptom looks like backend latency, not pool starvation, unless the
  pool's own wait-time metric is checked directly.
- Closing a pool on shutdown must drain in-flight checkouts first — a hard close mid-request
  surfaces as a client-side connection-reset error with no server-side log line to explain it.

## A worked incident, told in two parts

The first sighting looked like ordinary backend latency: p99 response time crept up over an hour
with no corresponding change in request volume, CPU, or memory — nothing an ordinary dashboard
would flag as a smoking gun on its own.

Only once someone graphed the connection pool's own wait-time metric next to the latency curve
did the two lines move in lockstep, and the root cause was the pool exhaustion in the heading
above — a fix that took ten minutes once found took most of a day to even suspect.
