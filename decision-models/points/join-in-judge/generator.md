# Instructions: write group-chat moments

You write realistic, synthetic test items for a judge that decides whether an AI agent taking part
in a group chat should reply to a new message that does not @mention it. Each seed line in this
request describes one item to write; you write exactly one item per seed.

## The seed fields

- `key`: copy it to your answer line; never write it inside the item.
- `purpose`: what the thread is for, and what the agent does in it.
- `cue`: how the new message relates to the people and the agent in the thread.
- `intendedAnswer`: `true` means a careful reader of the item should conclude that the agent
  should reply to the new message; `false` means it should stay silent.
- `difficulty`: `clear` means the answer is plain to a careful reader. `near-miss` means the item
  sits close to the line (the agent's name appears but the message is for someone else, a room
  question that a person may already be handling, a reply to the agent that only closes the
  exchange) while a careful reader still lands on the intended answer.

## The item you write

An object with exactly these three fields:

- `agentName`: the agent's name, a short name that fits the thread. It is never an @-handle.
- `priorTurns`: the thread's messages before the new one, oldest first, from 0 to 8 messages. Each
  is an object with exactly `role` (`"user"` for a person, `"assistant"` for the agent),
  `displayName` (a person's first name, or `agentName` for the agent's own messages) and `text`.
  Use an empty list `[]` only when the new message opens the thread.
- `newMessage`: an object with exactly `role` (always `"user"`: the new message is from a person),
  `displayName` (a person's first name) and `text`.

Write natural chat language in English, with two to four people besides the agent. Vary names,
tone, length and phrasing across items. The new message never contains `@` followed by the agent's
name; it may contain the agent's name as plain text. The thread must show what the agent does, so a
careful reader can tell what is its job.

## Never reveal the seed

The item must read like a real chat thread. It contains no commentary of any kind:

- never state, hint at or label the intended answer, the difficulty, or whom the message is for
  beyond what the people in the thread would naturally write;
- never write the words "near-miss", "intended", "difficulty", or a reference code;
- no notes, asides or explanations in brackets or parentheses, and no text such as
  "reply: …", "answer: …" or "(the agent should stay out of this)";
- nothing outside the three fields.
