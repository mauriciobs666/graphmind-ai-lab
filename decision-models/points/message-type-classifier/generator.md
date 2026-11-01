# Instructions: write customer chat messages

You write realistic, synthetic test items for a message classifier. A business's chat receives
messages from customers, and the classifier gives each one a type: `order` (places, changes,
confirms or cancels a purchase or booking), `question` (asks for information, with nothing wrong),
`complaint` (says something went wrong), `escalation` (pushes an unresolved problem further:
repeats it, demands a human or manager, or threatens consequences) or `other` (greetings, thanks,
small talk, anything else). Each seed line in this request describes one item to write; you write
exactly one item per seed.

## The seed fields

- `key`: copy it to your answer line; never write it inside the item.
- `domain`: the business whose chat this is.
- `threadPosition`: `opening` means the message starts a thread, with nothing before it;
  `follow-up` means it answers or continues the message just before it.
- `intendedAnswer`: the type a careful reader of the item should give the message.
- `difficulty`: `clear` means the type is plain to a careful reader. `near-miss` means the message
  sits close to a neighbouring type (a question with an edge of dissatisfaction, a complaint that
  nearly becomes a demand, a short reply that is only an order because of what came before) while a
  careful reader still lands on the intended type.

## The item you write

An object with exactly these two fields:

- `text`: the customer's message.
- `priorTurn`: for `follow-up`, the message just before it, as an object with exactly `speaker`
  and `text`. It may come from the business's assistant, from staff or from another customer. For
  `opening`, `null`.
  - Customers and staff speak under a person's first name.
  - The business's assistant speaks as `Assistant` or as `<business> Assistant` (for example
    `Florist Assistant`). It is never a person's first name, and no other speaker uses that name.

Write natural chat language in English. Vary names, tone, length, spelling and phrasing across
items; customers write short, informal and sometimes untidy messages. For `follow-up`,
`priorTurn` is the real preceding message; the message may stand on its own or need it, as in a
real thread.

## Never reveal the seed

The item must read like a real chat message. It contains no commentary of any kind:

- never state, hint at or label the intended type or the difficulty;
- never write the words "near-miss", "intended", "difficulty", or a reference code;
- no notes, asides or explanations in brackets or parentheses, and no text such as
  "type: …", "category: …" or "(this is a complaint)";
- nothing outside the two fields.
