# Split-candidate flagging self-test fixture

> **Synthetic fixture — not a real KB.** Exists only for
> `flag_split_candidates_selftest.py`'s mutation test: one section constructed to be clearly
> single-claim, one constructed to be clearly multi-claim, so the tool's heuristic can be checked
> against a known answer rather than only against real files whose "true" claim boundaries are a
> judgment call. Do not migrate, ingest, or cite this file as a real agent knowledge base.

## Retrying a flaky health probe with linear backoff

When a startup health probe fails intermittently on a freshly-started service (the process is
up but hasn't finished binding its listener yet), retry it a fixed number of times with a linear
backoff — 500ms, 1000ms, 1500ms — rather than either a single fixed sleep-then-check or an
unbounded retry loop. A fixed sleep either wastes time on a service that came up quickly or fails
on one that came up slowly; an unbounded loop can hang forever against a service that will never
come up, turning a real startup defect into a silent stall instead of a fast, legible failure.
Cap the retry count (five attempts is enough for any of this repo's dev services) and report the
last failure's actual error, not a generic timeout, so whoever reads the log knows what genuinely
broke.

Origin: `falkor-chat/scripts/start_server.sh`'s own health-check loop before uvicorn is
considered up.

## Three separable failure modes when caching a per-request auth token, each independently attributed

Caching a resolved auth token across requests inside one process (to avoid re-validating it on
every call) looks like a single optimization but actually hides three independently-discovered
failure modes, each requiring its own fix and each found in a separate incident. Treat each as
its own claim rather than one bundled "be careful with token caching" note, because a fix for one
does nothing for the other two and a reader skimming only the heading text would reasonably
assume all three are already covered by whichever one they happen to know about.

1. **Stale-token reuse past expiry.** A cache keyed only on the token's own string value, with no
   expiry check at read time, keeps serving an expired token until something else forces a cache
   miss — every call in between fails downstream with an auth error that looks unrelated to
   caching at all, because the caching layer itself reports success. The fix is to key the cache
   entry on `(token, expiresAt)` and check `expiresAt > now()` on every read, not only on write.
   Origin: a 2026-08-02 incident where a workflow's long-running background task kept using a
   token issued at task start, well past its documented 15-minute expiry, because the token cache
   sat in front of every call and was never invalidated mid-task.

2. **Cross-tenant leakage through a process-global cache.** A cache that is a bare module-level
   dict, not scoped per workspace/tenant, silently serves tenant A's cached token to a request
   that believes it is acting as tenant B whenever the two calls race inside the same process —
   this is a distinct bug from staleness above, and fixing expiry does nothing to close it. The
   fix is to key the cache on the full `(tenantId, token)` pair, never on the token alone, and to
   size the cache so one tenant's churn cannot evict another's live entry immediately before use.
   Origin: found independently, by code review rather than a live incident, when auditing a
   second service that copied the first service's caching code verbatim without also copying the
   later expiry fix — the tenant-scoping gap had been present in the original from the start and
   nobody had previously named it as its own defect.
3. **Thundering-herd revalidation on simultaneous expiry.** When many concurrent requests all
   observe an expired cache entry at once, and nothing serializes the revalidation, every one of
   them independently calls the upstream auth service at the same instant — a load spike the
   caching layer exists to prevent, produced by the caching layer itself at the exact moment it
   is least equipped to absorb it. The fix is a single-flight lock per cache key: the first caller
   past expiry revalidates and every other concurrent caller waits on that same in-flight call
   rather than issuing its own. Origin: reproduced under a synthetic load test that fired 200
   concurrent requests at the instant of a scheduled token expiry and counted 200 upstream auth
   calls instead of one, confirmed by re-running the same load test after adding the single-flight
   lock and counting exactly one.

Each of these three failure modes is independently reproducible, independently fixed, and was
independently discovered — they share only the word "caching" in their heading, not a mechanism,
a fix, or a root cause.
