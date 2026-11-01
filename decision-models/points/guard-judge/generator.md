# Instructions: write workflow-checkpoint items

You write realistic, synthetic test items for a yes/no judge. At a checkpoint in a support or
sales conversation, the judge reads a condition and the evidence gathered so far, and decides
whether the condition holds. Each seed line in this request describes one item to write; you
write exactly one item per seed.

## The seed fields

- `key`: copy it to your answer line; never write it inside the item.
- `condition`: the condition sentence. Copy it into the item's `condition` field exactly.
- `domain`: the business the conversation happens in. When the condition fits the business
  awkwardly, shape the situation within that business so that the condition arises naturally.
  Never change the condition sentence or the business.
- `evidenceKind`: `understanding` means the evidence is the workflow's summary of the
  conversation; `turns` means the evidence is the conversation's recent messages.
- `turnCount`: for `turns`, the exact number of messages to write; `null` for `understanding`.
- `intendedAnswer`: `true` means a careful reader of the item should conclude that the condition
  holds; `false` means they should conclude that it does not.
- `difficulty`: `clear` means the answer is plain to a careful reader. `near-miss` means the item
  sits close to the line. When `intendedAnswer` is `false`, it looks almost complete but lacks
  exactly one fact the condition needs, or states that fact ambiguously. When `intendedAnswer` is
  `true`, every needed fact is there, but one is given indirectly, late, or informally, where a
  hasty reader could miss it. For `understanding` seeds, `missingForm` says how.
- `missingForm`: for `understanding`, how the summary's `known` and `missing` lists carry the
  answer; `null` for `turns`. Follow its rule exactly:
  - `empty`: every needed fact is in `known`; `missing` is `[]`.
  - `immaterial`: every needed fact is in `known`; `missing` lists one or two details that the
    condition does not need. For a `clear` seed, make the detail plainly unneeded, such as a
    presentation or format preference (gift wrapping, a PDF or a printed copy). For a `near-miss`
    seed, make it look relevant at first sight.
  - `indirect`: `missing` is `[]`; every needed fact is in `known`, but one of them only
    indirectly or informally.
  - `listed`: one needed fact is absent from `known` and is listed in `missing`. For a
    `near-miss` seed, everything else is complete, and the listed fact reads like a detail whose
    importance is arguable.
  - `unlisted`: one needed fact is absent from `known` and is not listed; `missing` is `[]`.
  - `ambiguous`: one needed fact is in `known`, but in an ambiguous form, and it is not listed;
    `missing` is `[]`.

## The item you write

An object with exactly these four fields:

- `condition`: the seed's `condition`, copied exactly.
- `path`: the seed's `evidenceKind`, copied exactly.
- `understanding`: for `understanding`, an object with exactly three fields: `request` (one
  sentence: what the user asked for, in the user's terms), `known` (a list of short strings, each
  one fact the user supplied, such as `"delivery: Saturday morning"`) and `missing` (a list of short
  strings naming facts not yet supplied, as `missingForm` says). For `turns`, an empty object `{}`.
- `turns`: for `turns`, a list of exactly `turnCount` messages, oldest first, each an object with
  exactly `role` (`"user"` for the customer, `"assistant"` for the business's assistant),
  `displayName` and `text` (the message). For `understanding`, an empty list `[]`.
  - The customer, and any staff member, speaks under a person's first name.
  - The business's assistant speaks as `Assistant` or as `<business> Assistant` (for example
    `Bakery Assistant`). It is never a person's first name, and no other speaker uses that name.

Write natural chat language in English. Vary names, tone, length and phrasing across items. In a
`turns` item the conversation can start mid-way, and the evidence must come from what the user
writes or confirms.

## Never reveal the seed

The item must read like a real conversation record. It contains no commentary of any kind:

- never state, hint at or label the intended answer, the difficulty, the missing form, or what
  is missing beyond what the `missing` list itself says;
- never write the words "near-miss", "intended", "difficulty", or a reference code;
- no notes, asides or explanations in brackets or parentheses, and no text such as
  "answer: …", "label: …" or "(the budget is withheld)";
- nothing outside the four fields.
