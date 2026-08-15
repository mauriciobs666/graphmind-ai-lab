#!/usr/bin/env python3
"""Seed (or re-seed) the persistent GraphRAG evaluation workspace `ws:eval` (K-026,
`docs/plans/graphrag-eval.md` §5 Unit 1).

Never collected by pytest — this is a bare script, run with `server/.venv/bin/python`
via `scripts/seed_eval_corpus.sh`. That is why it is correct here (and nowhere in
`server/tests/`) to resolve the embedding model through `ModelGateway.from_env()`
directly: `server/tests/conftest.py`'s autouse fixture that redirects
`ModelGateway.from_env()` to the dim-4 test config only ever touches a pytest
session, and this script is never collected into one (plan §3 D7's asymmetry note).

What this does, in order:

  1. Resolve the real, configured embedding model + its declared dimension via
     `ModelGateway.from_env()` — never a hardcoded 1024.
  2. Bootstrap/reset `ws:eval`'s schema at that dimension if it doesn't exist yet,
     is at the wrong dimension, or `RESEED=1` is set (a deliberate full reset).
  3. Ensure the fixed corpus-author `User`/`Agent` exist.
  4. `MERGE`-idempotent Channel/Thread per topic (fixed ids — never
     `repository.create_channel`/`create_thread`, which are plain non-idempotent
     `CREATE`s).
  5. Post every corpus message with a **fixed msgId**, deciding first-vs-subsequent
     via `Repository.thread_has_head` (the same primitive `services._dispatch_write`
     uses) — reruns are safe no-ops via the `dupMsg` status.
  6. Embed every message that doesn't have an embedding yet, via the real
     `EmbeddingWorker` — never a stub or a fake vector.
  7. Print a provenance line and write a machine-readable sidecar,
     `server/tests/eval/corpus_provenance.json`, for a later eval unit to read.

Idempotency is the whole point: run this twice against an already-seeded `ws:eval`
and the second run does zero redundant embedding calls and zero redundant message
writes (all `dupMsg`/no-op) — that is this script's own correctness proof, verified
by actually re-running it (see the plan's Unit 1 "Done when").
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]  # .../falkor-chat
_SERVER_DIR = _REPO_ROOT / "server"
_BOOTSTRAP = _REPO_ROOT / "scripts" / "bootstrap_schema.sh"
_PROVENANCE_PATH = _SERVER_DIR / "tests" / "eval" / "corpus_provenance.json"

# Make `falkorchat` importable even when this script is invoked with a bare
# `python3` outside the venv's own site-packages resolution (the .sh wrapper
# always uses `server/.venv/bin/python`, which already resolves it — this is a
# defensive fallback for a direct `python3 scripts/seed_eval_corpus.py` run).
sys.path.insert(0, str(_SERVER_DIR))

from falkorchat import config, db, modelconfig  # noqa: E402
from falkorchat.embedding import EmbeddingWorker  # noqa: E402
from falkorchat.repository import Repository  # noqa: E402

EVAL_WS = os.environ.get("EVAL_WS", "eval")
EVAL_USER_ID = os.environ.get("EVAL_USER_ID", "eval-author")
EVAL_USER_NAME = os.environ.get("EVAL_USER_NAME", "Corpus Author")
EVAL_AGENT_ID = os.environ.get("EVAL_AGENT_ID", "eval-assistant")
EVAL_AGENT_NAME = os.environ.get("EVAL_AGENT_NAME", "Eval Assistant")

# ── the corpus (plan §4): 12 topical threads, 8-12 messages each ──────────────
#
# Three-plus near-miss pairs (shared vocabulary/domain, different specifics, so a
# paraphrased query can produce a genuine near-miss retrieval): incidents
# (payment-timeout / search-latency), architecture decisions in the same
# subsystem (notification-service database choice / retry strategy), auth-adjacent
# (OAuth refresh-token rotation / session idle-timeout policy), and on-call
# (support-escalation process / on-call rotation schedule). Plus clearly orthogonal
# topics (HR onboarding, product roadmap, office relocation, backup policy) so the
# golden set also has "easy" queries. Every message carries specific, concrete
# facts (service names, versions, dates, numbers) a later paraphrased query can
# target unambiguously — see `server/tests/test_workflow_live.py`'s small `CORPUS`
# constant for the style reference this scales up from.

_CORPUS: list[dict[str, object]] = [
    {
        "slug": "payment-timeout-incident",
        "channel_id": "eval-ch-incidents",
        "channel_name": "incidents",
        "thread_id": "eval-thread-payment-timeout-incident",
        "thread_title": "payment-service timeout incident",
        "messages": [
            ("user", "The payment-service started timing out on 3% of checkout "
             "calls right after the v3.9 deploy went out on Monday, August 3rd."),
            ("assistant", "Can you confirm whether the timeouts started "
             "immediately after the v3.9 rollout finished, or gradually over the "
             "next few hours?"),
            ("user", "Immediately — within 10 minutes of the v3.9 rollout "
             "completing at 14:20 UTC, our timeout rate jumped from 0.1% to 3%."),
            ("assistant", "That points to a config or connection-pool change in "
             "v3.9 rather than a gradual resource leak. Do you have the diff?"),
            ("user", "Yes — v3.9 lowered the payment-service's outbound HTTP "
             "client timeout from 5000ms to 800ms when calling the "
             "card-processor gateway."),
            ("assistant", "800ms is tight for a card-processor round trip, "
             "especially under load. Rolling back to v3.8 should confirm the "
             "hypothesis."),
            ("user", "We rolled back payment-service to v3.8 at 15:05 UTC and "
             "the timeout rate dropped back to 0.1% within 5 minutes."),
            ("assistant", "Confirmed root cause: the v3.9 outbound timeout of "
             "800ms was too aggressive for the card-processor gateway's typical "
             "1200-1800ms response time."),
            ("user", "Action item: raise the outbound timeout back to at least "
             "2000ms before re-deploying v3.9, and add a p99 latency alert on "
             "the card-processor call."),
            ("assistant", "Agreed. I'll track that as a follow-up: re-deploy "
             "payment-service v3.9.1 with a 2000ms outbound timeout and a p99 "
             "alert threshold of 1800ms."),
        ],
    },
    {
        "slug": "search-latency-incident",
        "channel_id": "eval-ch-incidents",
        "channel_name": "incidents",
        "thread_id": "eval-thread-search-latency-incident",
        "thread_title": "search-service latency incident",
        "messages": [
            ("user", "search-service p99 latency jumped from 180ms to 2.1s "
             "right after we deployed v2.7 on Wednesday, August 5th."),
            ("assistant", "Was the spike immediate after the v2.7 rollout, or "
             "did it build up gradually?"),
            ("user", "Immediate — within 15 minutes of the v2.7 deploy "
             "finishing at 09:40 UTC."),
            ("assistant", "Do you know what changed in v2.7 that could affect "
             "latency?"),
            ("user", "v2.7 switched the search-service's relevance-scoring "
             "index from an in-memory cache to a remote Redis lookup per "
             "query."),
            ("assistant", "A remote lookup per query would explain a jump like "
             "that under load. Rolling back should confirm."),
            ("user", "We rolled back search-service to v2.6 at 10:15 UTC and "
             "p99 latency returned to 190ms within 10 minutes."),
            ("assistant", "Confirmed root cause: v2.7's per-query remote Redis "
             "lookup for relevance scoring added roughly 1.9s of network "
             "overhead under production load."),
            ("user", "Action item: cache the relevance-scoring data locally "
             "again, or batch the Redis lookups, before re-attempting v2.7."),
            ("assistant", "Tracked. Follow-up: re-deploy search-service v2.7.1 "
             "with a local LRU cache in front of the Redis relevance lookup, "
             "target p99 under 250ms."),
        ],
    },
    {
        "slug": "notification-service-database-choice",
        "channel_id": "eval-ch-architecture",
        "channel_name": "architecture",
        "thread_id": "eval-thread-notification-service-database-choice",
        "thread_title": "notification-service database choice",
        "messages": [
            ("user", "We need to pick a database for the new notification-"
             "service — it has to handle roughly 40,000 writes/minute at "
             "peak."),
            ("assistant", "The two candidates on the table are PostgreSQL with "
             "partitioned tables, and DynamoDB. What are the read patterns "
             "like?"),
            ("user", "Reads are mostly point-lookups by notificationId, plus a "
             "per-user 'last 50 notifications' query."),
            ("assistant", "DynamoDB handles point-lookups and single-partition "
             "range queries well at that write volume with less operational "
             "overhead than sharded Postgres."),
            ("user", "The concern with DynamoDB is our team's limited "
             "experience with it compared to Postgres, which we run in "
             "production for six other services."),
            ("assistant", "That's a fair operational-risk argument. A "
             "partitioned Postgres table on notificationId's hash, with "
             "monthly partition rotation, can likely sustain 40,000 "
             "writes/minute on our current RDS instance class."),
            ("user", "Let's benchmark both: a partitioned Postgres table vs. "
             "DynamoDB on-demand capacity, using a synthetic 40,000 "
             "writes/minute load test."),
            ("assistant", "Agreed. I'll set up the load test for both this "
             "week."),
            ("user", "Benchmark results: partitioned Postgres sustained "
             "42,000 writes/minute at p99 write latency of 18ms; DynamoDB "
             "on-demand sustained 45,000 writes/minute at p99 write latency "
             "of 9ms."),
            ("user", "Decision: we're going with partitioned PostgreSQL for "
             "notification-service — the latency difference doesn't justify "
             "the operational-experience gap, and 42,000 writes/minute clears "
             "our 40,000 target with headroom."),
            ("assistant", "Recorded. I'll draft the partitioning scheme: hash "
             "partition on notificationId into 16 partitions, monthly "
             "detach-and-archive for partitions older than 90 days."),
        ],
    },
    {
        "slug": "notification-service-retry-strategy",
        "channel_id": "eval-ch-architecture",
        "channel_name": "architecture",
        "thread_id": "eval-thread-notification-service-retry-strategy",
        "thread_title": "notification-service retry strategy",
        "messages": [
            ("user", "Now that notification-service's database is settled, we "
             "need a retry strategy for failed push-notification deliveries "
             "to the mobile provider APIs."),
            ("assistant", "What's the current failure rate from the mobile "
             "push providers (APNs and FCM)?"),
            ("user", "About 2.5% of push attempts fail, mostly with transient "
             "503s from FCM during their own peak hours."),
            ("assistant", "For transient 503s, exponential backoff with "
             "jitter is the standard approach. What retry ceiling do you "
             "want?"),
            ("user", "Let's cap it at 5 retries, with a base delay of 2 "
             "seconds doubling each attempt, plus up to 500ms of random "
             "jitter."),
            ("assistant", "That gives a worst-case delay sequence of roughly "
             "2s, 4s, 8s, 16s, 32s before giving up — about a minute total. Is "
             "that acceptable for push notifications?"),
            ("user", "A minute is too long for time-sensitive alerts. Let's "
             "cap the retry ceiling at 4 attempts and a 10-second max delay "
             "per attempt instead."),
            ("assistant", "Updated: 4 retries, base delay 2s, doubling, "
             "capped at 10s per attempt, plus jitter — worst case about 24 "
             "seconds before we mark the delivery as failed."),
            ("user", "Decision: 4-retry exponential backoff (2s base, 10s "
             "cap, jitter) for push-notification delivery, permanent "
             "failures logged to the dead-letter queue for manual review."),
            ("assistant", "Recorded. I'll implement the backoff policy and "
             "wire the dead-letter queue this sprint."),
        ],
    },
    {
        "slug": "oauth-token-refresh-bug",
        "channel_id": "eval-ch-security",
        "channel_name": "security",
        "thread_id": "eval-thread-oauth-token-refresh-bug",
        "thread_title": "OAuth refresh-token rotation bug",
        "messages": [
            ("user", "Users are getting logged out unexpectedly about once "
             "every 20 minutes on the mobile app since last Thursday's "
             "release."),
            ("assistant", "Does the logout coincide with an access-token "
             "expiry? What's the access-token TTL?"),
            ("user", "Access tokens expire after 15 minutes. The refresh "
             "should be silent, but it looks like the refresh call is "
             "failing."),
            ("assistant", "Can you check whether the refresh-token itself is "
             "being rotated (single-use) or reused across refresh calls?"),
            ("user", "We switched to single-use rotating refresh tokens in "
             "last Thursday's release — each refresh call issues a new "
             "refresh token and invalidates the old one."),
            ("assistant", "A common bug with rotation: if two refresh calls "
             "fire concurrently (e.g. two tabs, or a race on the mobile "
             "client), the second one uses an already-invalidated refresh "
             "token and fails, logging the user out."),
            ("user", "That matches what we're seeing — it's worse on the "
             "mobile app because it opens a background refresh timer that "
             "can race with a foreground request."),
            ("assistant", "Root cause: concurrent refresh calls racing "
             "against single-use token rotation. The fix is a refresh-mutex "
             "so only one refresh-token exchange is in flight per client at "
             "a time."),
            ("user", "We added a client-side refresh-mutex on the mobile app "
             "and deployed it Monday. Unexpected logouts dropped from "
             "roughly 1 in 20 minutes to zero over the following 48 hours."),
            ("assistant", "Confirmed fixed. I'll note this as a required "
             "pattern for any client integrating rotating refresh tokens "
             "going forward."),
        ],
    },
    {
        "slug": "session-timeout-policy",
        "channel_id": "eval-ch-security",
        "channel_name": "security",
        "thread_id": "eval-thread-session-timeout-policy",
        "thread_title": "web session idle-timeout policy",
        "messages": [
            ("user", "Support is getting complaints that the web dashboard "
             "logs users out after only 10 minutes of inactivity — that "
             "feels too short."),
            ("assistant", "10 minutes is indeed on the aggressive side for a "
             "non-financial dashboard. What's driving that current value?"),
            ("user", "It was set conservatively when the dashboard first "
             "shipped, before we had granular audit logging. We now log "
             "every session action with a timestamp."),
            ("assistant", "Given the audit logging is in place, what's the "
             "actual security requirement here — is this covered by a "
             "compliance policy?"),
            ("user", "Our SOC 2 policy requires session termination after "
             "inactivity, but doesn't mandate a specific number — 30 minutes "
             "is a common industry default."),
            ("assistant", "Recommend extending the idle timeout to 30 "
             "minutes, with a warning modal at the 25-minute mark so users "
             "can extend the session with one click."),
            ("user", "Let's go with 30 minutes idle timeout, 25-minute "
             "warning modal, and a hard absolute session cap of 8 hours "
             "regardless of activity."),
            ("assistant", "Decision recorded: 30-minute idle timeout, "
             "25-minute warning, 8-hour absolute session cap. I'll update "
             "the session-management config."),
            ("user", "Deployed the new session-timeout policy Friday. "
             "Complaint volume about unexpected logouts dropped to near "
             "zero the following week."),
            ("assistant", "Good outcome. I'll add the 8-hour absolute cap "
             "and 30-minute idle timeout to the security policy doc."),
        ],
    },
    {
        "slug": "hr-onboarding-checklist",
        "channel_id": "eval-ch-people-ops",
        "channel_name": "people-ops",
        "thread_id": "eval-thread-hr-onboarding-checklist",
        "thread_title": "new-hire onboarding checklist",
        "messages": [
            ("user", "We're onboarding four new engineers starting September "
             "2nd — can we finalize the onboarding checklist?"),
            ("assistant", "Sure. Current checklist has 12 items: laptop "
             "provisioning, badge access, VPN setup, Slack/email accounts, "
             "repo access, and eight training modules."),
            ("user", "Laptop provisioning has been the bottleneck — it took "
             "9 business days for the last cohort in June."),
            ("assistant", "IT says the delay was a vendor backorder on the "
             "specific laptop model. They're recommending we order 6 weeks "
             "ahead of any known start date going forward."),
            ("user", "Let's add a rule: submit laptop orders at least 6 "
             "weeks before a confirmed start date, and flag any cohort where "
             "that lead time isn't met."),
            ("assistant", "Added. I'll also add a day-1 checklist item: "
             "badge access and VPN must be provisioned by 9am on the start "
             "date, not 'sometime that week'."),
            ("user", "Good. What about the eight training modules — is a "
             "week enough time to complete them all?"),
            ("assistant", "Historically new hires finish the eight modules "
             "in an average of 3.5 days, so a 5-business-day first week has "
             "comfortable slack."),
            ("user", "Let's finalize: 12-item checklist, laptops ordered 6 "
             "weeks ahead, badge/VPN ready by 9am day one, 5 business days "
             "allotted for the 8 training modules."),
            ("assistant", "Finalized and posted to the onboarding wiki. I'll "
             "flag the September 2nd cohort's laptop orders today to hit the "
             "6-week lead time."),
        ],
    },
    {
        "slug": "q3-product-roadmap-planning",
        "channel_id": "eval-ch-planning",
        "channel_name": "planning",
        "thread_id": "eval-thread-q3-product-roadmap-planning",
        "thread_title": "Q3 product roadmap planning",
        "messages": [
            ("user", "For Q3 we have three candidate initiatives: the mobile "
             "redesign, the analytics-export feature, and the bulk-import "
             "tool. We can only fully staff two."),
            ("assistant", "What's the estimated engineering effort for each, "
             "in sprint-weeks?"),
            ("user", "Mobile redesign is roughly 10 sprint-weeks, "
             "analytics-export is 4 sprint-weeks, bulk-import is 6 "
             "sprint-weeks."),
            ("assistant", "With 14 sprint-weeks of capacity this quarter, "
             "mobile redesign plus analytics-export fits at 14; bulk-import "
             "plus analytics-export fits at 10 with slack."),
            ("user", "Sales has been asking for bulk-import the most — it's "
             "blocking two enterprise deals worth roughly $180,000 ARR "
             "combined."),
            ("assistant", "That changes the calculus. Bulk-import (6) plus "
             "analytics-export (4) is 10 sprint-weeks, leaving 4 weeks of "
             "buffer or a partial start on the mobile redesign."),
            ("user", "Let's prioritize bulk-import and analytics-export for "
             "Q3, and push the mobile redesign to Q4 with a design-only "
             "spike in the last 4 weeks of Q3."),
            ("assistant", "Recorded: Q3 roadmap is bulk-import (6 "
             "sprint-weeks) + analytics-export (4 sprint-weeks) + a 4-week "
             "mobile-redesign design spike."),
            ("user", "Let's also set success metrics: bulk-import should "
             "close both pending enterprise deals by end of Q3, "
             "analytics-export should ship to at least 3 beta customers."),
            ("assistant", "Added those as the two Q3 success criteria in the "
             "roadmap doc. I'll share it with sales and the beta customer "
             "list this week."),
        ],
    },
    {
        "slug": "office-relocation-logistics",
        "channel_id": "eval-ch-logistics",
        "channel_name": "logistics",
        "thread_id": "eval-thread-office-relocation-logistics",
        "thread_title": "office relocation logistics",
        "messages": [
            ("user", "The lease on our current office ends November 30th, "
             "and the new space on 5th floor of the same building is ready "
             "December 1st."),
            ("assistant", "That's a same-building move, which simplifies "
             "logistics. How many desks need to move?"),
            ("user", "68 desks, plus the server closet with 4 racks that "
             "can't be down for more than a 6-hour window."),
            ("assistant", "Recommend moving the server closet over a weekend "
             "to stay well inside that 6-hour window with buffer. Which "
             "weekend works?"),
            ("user", "The weekend of November 22-23 works — that's a full "
             "week before the lease ends, giving us slack if anything goes "
             "wrong."),
            ("assistant", "Good. For the 68 desks, do you want a single "
             "moving day or a phased move?"),
            ("user", "Phased — let's move by floor-team in three batches of "
             "roughly 23 desks each, on November 25th, 26th, and 27th, to "
             "avoid disrupting everyone at once."),
            ("assistant", "That leaves November 28-29 as buffer before the "
             "December 1st ready date. I'll book the moving company for "
             "those three dates plus the server-closet weekend."),
            ("user", "Also confirm the new floor has enough conference "
             "rooms — we had 4 on the old floor and need at least that "
             "many."),
            ("assistant", "Confirmed: the 5th floor has 5 conference rooms, "
             "one more than the current floor. I'll send the full "
             "relocation schedule to the team today."),
        ],
    },
    {
        "slug": "database-backup-policy",
        "channel_id": "eval-ch-infra",
        "channel_name": "infra",
        "thread_id": "eval-thread-database-backup-policy",
        "thread_title": "database backup retention policy",
        "messages": [
            ("user", "Our current backup policy keeps daily snapshots for 7 "
             "days and nothing beyond that — is that enough for compliance?"),
            ("assistant", "For most compliance frameworks, 7 days of daily "
             "backups is thin. What does your data-retention policy "
             "require?"),
            ("user", "Finance needs point-in-time recovery for any date "
             "within the last 90 days, for audit purposes."),
            ("assistant", "That needs a tiered policy: daily snapshots for 7 "
             "days, weekly snapshots for the following 12 weeks (about 84 "
             "days), which covers 91 days total."),
            ("user", "Storage cost is a concern — what's the estimated size "
             "increase moving from 7-day retention to that tiered scheme?"),
            ("assistant", "At current database size of about 2.1TB, moving "
             "to 7 daily + 12 weekly snapshots is roughly 4x the snapshot "
             "storage, an estimated extra 380GB/month at current growth."),
            ("user", "That's acceptable. Let's also add a monthly snapshot "
             "retained for 1 year, for year-over-year audit comparisons."),
            ("assistant", "Updated policy: 7 daily, 12 weekly, 12 monthly (1 "
             "year) snapshots. Estimated total extra storage is about "
             "420GB/month."),
            ("user", "Approved. Let's also schedule a quarterly restore-"
             "drill to confirm the backups are actually restorable, not "
             "just present."),
            ("assistant", "Added a quarterly restore-drill to the ops "
             "calendar, first one scheduled for the first week of next "
             "quarter. I'll update the backup policy doc with all of "
             "this."),
        ],
    },
    {
        "slug": "customer-support-escalation-process",
        "channel_id": "eval-ch-people-ops",
        "channel_name": "people-ops",
        "thread_id": "eval-thread-customer-support-escalation-process",
        "thread_title": "customer support escalation process",
        "messages": [
            ("user", "We're seeing escalations to engineering take an "
             "average of 6 hours to get a first response — that's too slow "
             "for P1 customer issues."),
            ("assistant", "What does the current escalation path look like, "
             "step by step?"),
            ("user", "Support files a ticket, it sits in a shared queue, an "
             "on-call engineer picks it up whenever they check the queue — "
             "no active paging."),
            ("assistant", "The lack of active paging is likely the main "
             "driver of the 6-hour delay. Do you have an on-call rotation "
             "already, for something like incidents?"),
            ("user", "Yes, we have an on-call rotation for production "
             "incidents that pages within 5 minutes. Support escalations "
             "aren't wired into that same system."),
            ("assistant", "Recommend routing P1 support escalations into the "
             "same paging system, with a stricter response SLA than a P1 "
             "incident since it's already been triaged by a human."),
            ("user", "Let's target a 15-minute page-to-acknowledgment SLA "
             "for P1 support escalations, and 2 hours for P2."),
            ("assistant", "Recorded: P1 escalations page on-call within 15 "
             "minutes, P2 within 2 hours, both routed through the existing "
             "incident-paging system instead of the shared queue."),
            ("user", "We rolled this out starting last Monday. Average "
             "first-response time for P1 escalations dropped from 6 hours "
             "to 22 minutes over the first week."),
            ("assistant", "Good result. I'll keep tracking the metric for a "
             "full month before we call the new process validated."),
        ],
    },
    {
        "slug": "on-call-rotation-schedule",
        "channel_id": "eval-ch-infra",
        "channel_name": "infra",
        "thread_id": "eval-thread-on-call-rotation-schedule",
        "thread_title": "on-call rotation schedule redesign",
        "messages": [
            ("user", "Our on-call rotation currently has 6 engineers on a "
             "weekly rotation, and two of them have said the load is "
             "uneven."),
            ("assistant", "Uneven how — page volume, or calendar fairness "
             "(weekends vs weekdays)?"),
            ("user", "Calendar fairness — the same two engineers keep "
             "landing on Christmas week and July 4th week because the "
             "rotation order never changes."),
            ("assistant", "A fixed rotation order will always collide with "
             "the same calendar weeks. Randomizing the assignment order "
             "each quarter would fix that."),
            ("user", "Let's also add a rule: no one covers a major holiday "
             "week two years in a row."),
            ("assistant", "That needs a bit of manual tracking, but is "
             "doable — I can maintain a holiday-coverage history and "
             "exclude repeat coverage when generating each quarter's "
             "schedule."),
            ("user", "How many engineers do we need per shift, given "
             "current page volume?"),
            ("assistant", "Current volume is about 3.2 pages/week on "
             "average, comfortably handled by one primary plus one "
             "secondary on-call engineer."),
            ("user", "Let's keep it at one primary and one secondary, "
             "1-week shifts, randomized quarterly ordering, and no repeat "
             "major-holiday coverage within 2 years."),
            ("assistant", "Recorded. I'll regenerate Q4's schedule with the "
             "new randomized ordering and the holiday-repeat exclusion rule "
             "this week."),
        ],
    },
]


def _distinct_channels() -> list[tuple[str, str]]:
    seen: dict[str, str] = {}
    for topic in _CORPUS:
        seen[str(topic["channel_id"])] = str(topic["channel_name"])
    return list(seen.items())


def _merge_channel(graph, *, channel_id: str, name: str, created_at: int) -> None:
    graph.query(
        "MERGE (c:Channel {channelId: $channelId}) "
        "ON CREATE SET c.name = $name, c.createdAt = $createdAt "
        "RETURN c.channelId",
        {"channelId": channel_id, "name": name, "createdAt": created_at},
    )


def _merge_thread(
    graph, *, channel_id: str, thread_id: str, title: str, created_at: int
) -> None:
    res = graph.query(
        "MATCH (c:Channel {channelId: $channelId}) "
        "MERGE (t:Thread {threadId: $threadId}) "
        "ON CREATE SET t.title = $title, t.createdAt = $createdAt, "
        "               t.updatedAt = $createdAt "
        "MERGE (c)-[:HAS_THREAD]->(t) "
        "RETURN t.threadId",
        {
            "channelId": channel_id, "threadId": thread_id,
            "title": title, "createdAt": created_at,
        },
    )
    if not res.result_set:
        raise RuntimeError(
            f"thread MERGE was a no-op — channel not found "
            f"(channel={channel_id!r}, thread={thread_id!r})"
        )


def _has_embedding(graph, *, msg_id: str) -> bool:
    res = graph.ro_query(
        "MATCH (m:Message {msgId: $msgId}) RETURN m.embedding IS NOT NULL",
        {"msgId": msg_id},
    )
    return bool(res.result_set and res.result_set[0][0])


def main() -> None:
    reseed = os.environ.get("RESEED", "").strip().lower() in {"1", "true", "yes", "on"}

    print("Resolving embedding model via ModelGateway.from_env()...")
    gateway = modelconfig.ModelGateway.from_env()
    resolution = gateway.resolve("embedding")
    model_ref = resolution.primary.ref
    dim = resolution.primary.dim or config.EMBEDDING_DIM
    print(f"  embedding model: {model_ref}  dim={dim}")

    conn = db.connect()
    repo = Repository(conn)
    graph = db.workspace_graph(conn, EVAL_WS)

    current_dim = repo.read_index_dimension(EVAL_WS, label="Message")
    if reseed or current_dim != dim:
        print(
            f"(Re)bootstrapping ws:{EVAL_WS} at dim={dim} "
            f"(current index dim: {current_dim!r}, RESEED={reseed})..."
        )
        try:
            db.connect().select_graph(f"ws:{EVAL_WS}").delete()
        except Exception:
            pass  # graph may not exist yet — bootstrap creates it fresh
        subprocess.run(
            ["bash", str(_BOOTSTRAP), EVAL_WS],
            check=True,
            env={**os.environ, "EMBEDDING_DIM": str(dim)},
        )
    else:
        print(f"ws:{EVAL_WS} already at dim={dim} — no reseed needed.")

    print(f"Ensuring corpus-author User {EVAL_USER_ID!r} and Agent {EVAL_AGENT_ID!r}...")
    repo.ensure_user(EVAL_WS, user_id=EVAL_USER_ID, display_name=EVAL_USER_NAME)
    repo.ensure_agent(EVAL_WS, agent_id=EVAL_AGENT_ID, name=EVAL_AGENT_NAME)

    base_ts = int(datetime.now(timezone.utc).timestamp() * 1000)
    ts_counter = 0

    print(f"Seeding {len(_CORPUS)} channels' worth of threads...")
    for channel_id, channel_name in _distinct_channels():
        _merge_channel(graph, channel_id=channel_id, name=channel_name, created_at=base_ts)

    written_count = 0
    dup_count = 0
    all_msg_ids: list[tuple[str, str]] = []  # (msgId, text) — for the embed pass

    for topic in _CORPUS:
        slug = str(topic["slug"])
        thread_id = str(topic["thread_id"])
        _merge_thread(
            graph, channel_id=str(topic["channel_id"]), thread_id=thread_id,
            title=str(topic["thread_title"]), created_at=base_ts,
        )

        messages = topic["messages"]
        assert isinstance(messages, list)
        for n, (role, text) in enumerate(messages, start=1):
            msg_id = f"eval-{slug}-{n:03d}"
            author_id = EVAL_AGENT_ID if role == "assistant" else EVAL_USER_ID
            ts_counter += 1
            created_at = base_ts + ts_counter

            use_first = not repo.thread_has_head(EVAL_WS, thread_id=thread_id)
            write = repo.post_first_message if use_first else repo.post_subsequent_message
            status = write(
                EVAL_WS, thread_id=thread_id, msg_id=msg_id, author_id=author_id,
                text=text, role=role, created_at=created_at,
            )
            if status is None:
                raise RuntimeError(
                    f"write for {msg_id!r} hit a missing anchor "
                    f"(thread={thread_id!r}, use_first={use_first}) — the thread "
                    f"disappeared mid-seed, which should be impossible for a "
                    f"single-writer script"
                )
            if not (status.written or status.dup_msg):
                raise RuntimeError(
                    f"unexpected write status for {msg_id!r}: {status!r} "
                    f"(thread={thread_id!r}, use_first={use_first})"
                )
            if status.dup_msg:
                dup_count += 1
            written_count += 1
            all_msg_ids.append((msg_id, text))

    print(
        f"Messages processed: {written_count} "
        f"({written_count - dup_count} newly written, {dup_count} already present)"
    )

    print("Embedding messages that don't have a vector yet...")
    worker = EmbeddingWorker(repo, models=gateway, expected_dim=dim)
    new_embeds = 0
    for msg_id, text in all_msg_ids:
        if _has_embedding(graph, msg_id=msg_id):
            continue
        worker.embed_message(EVAL_WS, msg_id=msg_id, text=text)
        new_embeds += 1
    print(f"New embeddings computed this run: {new_embeds}")

    thread_count = len(_CORPUS)
    message_count = len(all_msg_ids)
    seeded_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    provenance = {
        "message_count": message_count,
        "thread_count": thread_count,
        "embedding_model_ref": model_ref,
        "embedding_dim": dim,
        "seeded_at": seeded_at,
    }
    _PROVENANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _PROVENANCE_PATH.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

    print(
        f"\nws:eval provenance — messages={message_count} threads={thread_count} "
        f"embeddingModel={model_ref} dim={dim} seededAt={seeded_at} "
        f"newEmbeds={new_embeds} dupMessages={dup_count}"
    )
    print(f"Sidecar written: {_PROVENANCE_PATH.relative_to(_REPO_ROOT)}")


if __name__ == "__main__":
    main()
