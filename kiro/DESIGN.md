# Kiro Multi-Agent Ecosystem — Design Document

**Status**: Draft | **Last updated**: 2026-06-20

---

## What is this?

A team of specialized AI agents that collaborate on a shared project. Agents talk freely through a chat MCP server backed by FalkorDB, and deliver their work as structured artifacts tracked by Git.

Two layers, each doing what it's best at:

| Layer | Purpose | Stack |
|---|---|---|
| **Chat** | Free coordination, questions, decisions, @mentions | Chat MCP → FalkorDB |
| **Artifacts** | Durable deliverables (code, tests, reports, decisions) | Git (`<agent>/` directories) |

```mermaid
graph TD
    subgraph Agents
        A[architect]
        D[developer]
        T[tester]
        AM[analyst-market]
        AT[analyst-technical]
        RS[reviewer-security]
        RU[reviewer-ux]
    end

    subgraph "Chat MCP Server"
        MCP[MCP Tools\nsend_message\nread_messages\nget_room_state]
        FDB[(FalkorDB\nGraph Store)]
        MCP <--> FDB
    end

    subgraph "Git Repository"
        SPEC[SPEC.md]
        ARTS[agent directories\narchitect/ developer/\ntester/ analyst-*/\nreviewer-*/]
    end

    Agents <-->|tool calls| MCP
    Agents -->|commit artifacts| ARTS
    A -->|owns| SPEC
```

---

## Agent Roles

Agents come in two kinds: **singletons** and **multiples** (many instances, each with a different perspective).

| Agent | Directory | Instances | Responsibility |
|---|---|---|---|
| Architect | `architect/` | Singleton | Owns SPEC.md, makes technical decisions |
| Developer | `developer/` | Singleton | Implements code |
| Tester | `tester/` | Singleton | Writes and runs tests |
| Analyst | `analyst-<perspective>/` | Multiple | e.g. `analyst-market`, `analyst-technical`, `analyst-security` |
| Reviewer | `reviewer-<perspective>/` | Multiple | e.g. `reviewer-security`, `reviewer-ux`, `reviewer-performance` |

Each agent reads freely from all directories. Each agent writes artifacts only to its own directory. All agents communicate through the Chat MCP.

---

## Project Layout

```
project/
│
├── SPEC.md                        ← living spec, owned by architect
│
├── architect/
│   ├── decisions/                 ← architecture decision records
│   └── tasks.md
│
├── developer/
│   └── src/
│
├── tester/
│   ├── tests/
│   └── reports/
│
├── analyst-market/
│   └── research/
│
├── analyst-technical/
│   └── research/
│
├── reviewer-security/
│   └── reviews/
│
├── reviewer-ux/
│   └── reviews/
│
└── .git/
```

No `chat/` directory. Chat lives in FalkorDB, not the filesystem.

---

## The Chat MCP Server

A small MCP server that wraps FalkorDB. Every Kiro agent gets it in its config. Sending and reading messages are just tool calls — no sockets, no file watchers, no extra libraries per agent.

### Tools exposed

```
send_message(from, body, mentions[], re)
  → writes message node to FalkorDB
  → returns message id and timestamp

read_messages(agent_id, since?, limit?, re?)
  → returns messages since last read (or since timestamp)
  → prioritizes messages where agent_id is in mentions

get_room_state(re?)
  → returns active agents, their current task and status
  → optionally filtered by task id
```

### Message format (graph node)

```
(Message {
  id:       uuid,
  from:     "developer",
  body:     "@reviewer-security login.ts is ready for review",
  mentions: ["reviewer-security"],
  re:       "task-002",
  at:       "2026-06-20T11:30:00Z"
})
```

### FalkorDB graph schema

```mermaid
erDiagram
    Agent {
        string id
        string role
        string status
        string current_task
        string last_artifact
        datetime updated_at
    }
    Message {
        string id
        string from
        string body
        datetime at
        string re
    }
    Task {
        string id
        string title
        string status
        string owner
        datetime created_at
    }
    Artifact {
        string path
        string type
        datetime committed_at
    }

    Agent ||--o{ Message : SENT
    Message }o--o{ Agent : MENTIONS
    Message }o--o| Task : ABOUT
    Task ||--o{ Agent : OWNED_BY
    Task ||--o{ Task : DEPENDS_ON
    Agent ||--o{ Artifact : PRODUCED
    Artifact }o--|| Task : PART_OF
```

This enables graph queries impossible with flat files:

```cypher
-- Full thread for task-002, chronological
MATCH (m:Message {re: 'task-002'})
RETURN m.from, m.body, m.at ORDER BY m.at

-- Who has been mentioned most today?
MATCH (m:Message)-[:MENTIONS]->(a:Agent)
WHERE m.at > '2026-06-20T00:00:00Z'
RETURN a.id, count(m) as mentions ORDER BY mentions DESC

-- What is blocking right now?
MATCH (a:Agent {status: 'blocked'})
RETURN a.id, a.blocked_on

-- All approvals for task-002
MATCH (a:Agent)-[:SENT]->(m:Message {re: 'task-002'})
WHERE m.body CONTAINS 'Approved'
RETURN a.id, m.at
```

---

## Kiro Agent Config

All agents share the same structure. The system prompt is what differentiates them.

```json
{
  "name": "reviewer-security",
  "allowedTools": ["read", "write", "shell", "glob", "grep"],
  "mcpServers": {
    "chat": {
      "command": "python3",
      "args": ["-m", "chat_mcp"],
      "env": { "FALKORDB_URL": "redis://falkordb:6379" }
    }
  }
}
```

System prompt pattern:
> You are `reviewer-security`. At the start of every turn, call `read_messages` to catch up on the conversation.  
> Your artifact directory is `reviewer-security/` — write all review outputs there.  
> Use `send_message` when you have something to say. Use `mentions` to address specific agents.  
> You review code exclusively for security issues: auth, secrets, injection, data exposure.

---

## Metadata Layer

Structured YAML in `meta/` — machine-readable project state that complements the graph.

```
meta/
  tasks/
    task-001.yaml
    task-002.yaml
  agents/
    architect.yaml
    developer.yaml
    ...
```

### Task schema

```yaml
id: task-002
title: Implement auth login endpoint
status: review          # pending | in-progress | review | done | blocked
owner: developer
created_at: 2026-06-20T00:00:00Z
updated_at: 2026-06-20T11:30:00Z
depends_on: []
artifacts:
  - developer/src/auth/login.ts
  - tester/tests/auth/login.test.ts
reviews:
  - reviewer: reviewer-security
    status: pending
  - reviewer: reviewer-ux
    status: pending
blocked_on: null
```

### Task state machine

```mermaid
stateDiagram-v2
    [*] --> pending
    pending --> in_progress : agent picks up task
    in_progress --> review : owner signals done in chat
    in_progress --> blocked : dependency missing
    blocked --> in_progress : blocker resolved
    review --> done : all reviewers approved
    review --> in_progress : changes requested
    done --> [*]
```

### Agent schema

```yaml
id: developer
role: developer
status: active          # active | idle | blocked | offline
current_task: task-002
last_artifact: developer/src/auth/login.ts
blocked_on: null
updated_at: 2026-06-20T11:25:00Z
```

Agents commit their own YAML update in the same commit as the artifact or chat event that triggered the state change.

---

## Worked Flow Examples

### Example 1 — New task from scratch

A human adds a task to SPEC.md. The team picks it up organically.

```mermaid
sequenceDiagram
    participant H as Human
    participant A as architect
    participant AT as analyst-technical
    participant D as developer
    participant T as tester
    participant RS as reviewer-security

    H->>A: updates SPEC.md with new task-002 (auth login)
    A->>MCP: send_message("Starting task-002. @analyst-technical any JWT research?")
    MCP->>AT: delivers message (mention)
    AT->>MCP: read_messages()
    AT->>MCP: send_message("@architect yes — analyst-technical/research/jwt.md. Use JWT.")
    MCP->>A: delivers reply
    A->>Git: update SPEC.md with JWT constraint
    A->>MCP: send_message("@developer JWT confirmed. You can start task-002.")
    D->>MCP: read_messages()
    T->>MCP: read_messages()
    T->>MCP: send_message("@developer ping me when ready, I'll test in parallel.")
    D->>Git: commit developer/src/auth/login.ts
    D->>MCP: send_message("login.ts ready. @reviewer-security @tester please proceed.")
    RS->>Git: reads developer/src/auth/login.ts
    T->>Git: reads developer/src/auth/login.ts
    RS->>MCP: send_message("@developer hardcoded secret at line 12. Needs env var.")
    T->>Git: commit tester/tests/auth/login.test.ts
    D->>Git: fix secret, commit
    D->>MCP: send_message("Fixed. @reviewer-security please recheck.")
    RS->>Git: reads updated login.ts
    RS->>MCP: send_message("Approved.")
    T->>MCP: send_message("All tests passing. Approved.")
    A->>Git: update meta/tasks/task-002.yaml status→done
    A->>MCP: send_message("task-002 done.")
```

---

### Example 2 — Multi-perspective review fan-out

Architect requests reviews from multiple reviewers simultaneously.

```mermaid
sequenceDiagram
    participant A as architect
    participant MCP as Chat MCP
    participant RS as reviewer-security
    participant RU as reviewer-ux
    participant RP as reviewer-performance

    A->>MCP: send_message("Review needed for login.ts", mentions=[RS, RU, RP], re=task-002)

    par All three read simultaneously
        RS->>MCP: read_messages()
        RU->>MCP: read_messages()
        RP->>MCP: read_messages()
    end

    par All three review independently
        RS->>Git: reads developer/src/
        RU->>Git: reads developer/src/
        RP->>Git: reads developer/src/
    end

    RS->>Git: commit reviewer-security/reviews/task-002.md
    RS->>MCP: send_message("@architect changes-requested. See reviewer-security/reviews/task-002.md")

    RU->>Git: commit reviewer-ux/reviews/task-002.md
    RU->>MCP: send_message("@architect approved.")

    RP->>Git: commit reviewer-performance/reviews/task-002.md
    RP->>MCP: send_message("@architect approved.")

    A->>MCP: read_messages()
    Note over A: sees 1 change-request, 2 approvals
    A->>MCP: send_message("@developer security changes needed. See reviewer-security/reviews/task-002.md")
```

---

### Example 3 — Blocked task, self-organized resolution

A developer hits a dependency and self-reports. The relevant agent notices and unblocks without being told.

```mermaid
sequenceDiagram
    participant D as developer
    participant MCP as Chat MCP
    participant AT as analyst-technical
    participant A as architect

    D->>Git: update meta/tasks/task-003.yaml status→blocked
    D->>MCP: send_message("Blocked on task-003: need DB schema before I can write queries.", re=task-003)

    AT->>MCP: read_messages()
    Note over AT: not mentioned, but reads room\nand recognizes this is its domain
    AT->>Git: commit analyst-technical/research/db-schema.md
    AT->>MCP: send_message("@developer schema is in analyst-technical/research/db-schema.md. Unblocked.")

    D->>MCP: read_messages()
    D->>Git: update meta/tasks/task-003.yaml status→in-progress
    D->>Git: commit developer/src/db/queries.ts
    D->>MCP: send_message("Unblocked and done. @architect task-003 ready for review.")

    A->>MCP: read_messages()
    A->>Git: update meta/tasks/task-003.yaml status→review
```

---

## Design Principles

| Principle | Rationale |
|---|---|
| Chat MCP owns communication | Agents use tool calls, not filesystem hacks or sockets |
| FalkorDB owns chat history | Graph queries; no git commit noise from messages |
| Git owns artifacts only | Clean history; every commit is a meaningful deliverable |
| Agents self-organize through chat | No hard assignments; any agent can pick up dropped work |
| @mentions are non-binding | Any agent can respond to anything it finds relevant |
| SPEC.md is always current | Stale specs produce divergent agents |
| Agents update their own metadata | State changes are co-committed with the action that caused them |

---

## Open Questions

- [ ] How does an agent know when to stay quiet? (turn-taking / backoff when multiple agents respond simultaneously)
- [ ] What prevents two agents claiming the same task simultaneously? (optimistic locking on YAML, or a `claim_task` MCP tool?)
- [ ] Should there be multiple chat rooms for different concerns, or one room per project?
- [ ] How does a human participate in the chat? (human as just another agent? separate UI?)
- [ ] Chat MCP: push (SSE) vs pull (polling) for agent wake-up on @mention?
