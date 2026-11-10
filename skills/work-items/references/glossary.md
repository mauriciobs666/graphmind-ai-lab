# Work-item glossary

One formal term per level or type. Each is defined once, on its own `- **term**` line; IDs
use only the segment letter, never a synonym. Source: `docs/requirements/work-item-numbering.md`
FR-1, FR-10, FR-13, FR-14.

## Levels (a hierarchy: initiative, then feature, then story, then task)

- **initiative** (`I`) — a long-running theme that groups features; numbered `I01`, `I02`, ... Two are reserved: `I01` Defects and `I02` Kaizen.
- **feature** (`F`) — a deliverable capability with its own requirements document and family of documents; may sit under an initiative or under the project.
- **story** (`S`) — a slice of one feature; always has a feature as parent; numbered locally from `S01` under it.
- **task** (`T`) — the smallest unit of work; sits under a story, a feature, a defect or a kaizen item; numbered locally from `T01` under its parent.

## Types (not levels)

- **requirement** (`FR`) — a functional requirement of a feature, cited by path (`GAL-I03-F0042-FR2`).
- **acceptance criterion** (`AC`) — a testable criterion of a feature, cited like a requirement.
- **defect** (`D`) — a fault found in delivered work; lives permanently under `I01`; the feature it concerns is a link, not a path segment.
- **kaizen** (`K`) — a continuous-improvement item; lives permanently under `I02`; promoted to a feature when it grows into one (the old ID stays as an alias).

## Aliases

- **epic** — alias of **feature**. Say "feature"; never write `E` or "epic" in an ID.
