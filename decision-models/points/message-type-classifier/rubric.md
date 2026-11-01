# Rubric: what type of message is this?

Each item is one message that a customer or member posted in a business's chat, such as a shop's
or a service's support thread. You give it exactly one of five types, from the item's text alone.

## What an item looks like

    PREVIOUS MESSAGE:
    <speaker>: <text>

    MESSAGE:
    <text>

`MESSAGE` is the message you classify. `PREVIOUS MESSAGE` is the message posted just before it in
the same thread, shown for context only; it may be from the business's assistant, from staff or
from another customer. When the message opens a thread there is no `PREVIOUS MESSAGE` block.

Classify `MESSAGE`, never the previous message. Use the previous message only to understand what
`MESSAGE` refers to: "yes, the blue one" means nothing alone, but after "Would you like the blue
or the grey case?" it chooses a product.

## Answers

### order

The sender places, changes, confirms or cancels a purchase, booking or reservation, or gives the
details needed to complete one: what, how many, which variant, when, where to deliver, how to pay.

- "I'll take two of the large ones." · "Make it Saturday at 10 instead." · "Cancel my order
  please." · After a product offer: "Yes, the blue one."

### question

The sender asks for information and is not unhappy about anything: product details, prices,
stock, opening hours, delivery options, how something works, or the status of an order when
nothing has gone wrong yet.

- "Do you have this in size 40?" · "How long does delivery take to the north side?" · "Has my
  order shipped yet?"

### complaint

The sender says that something went wrong or is unsatisfactory — a product, a delivery, a charge,
a staff member, the service — whether or not they ask for a fix. It is the first time they raise
it, as far as the item shows, and they make no demand beyond a normal fix.

- "The cake arrived crushed." · "You charged me twice for the same order." · "Why is my parcel two
  days late?"

### escalation

The sender pushes an unresolved problem beyond ordinary handling: asks for a manager or a human
instead of the assistant, says the problem has been raised before without result, sets an
ultimatum, or threatens consequences such as cancelling, a chargeback, a public review or legal
action.

- "Third time I'm asking about this refund." · "Put me through to a real person." · "Fix this
  today or I'm disputing the charge."

### other

Anything that is none of the above: greetings, thanks, goodbyes, small talk, jokes, reactions
("ok", "great"), messages about something unrelated to the business, questions included, and
messages too fragmentary to place.

- "Good morning!" · "Thanks, that's all." · "Lovely weather today."

## Boundary cases

When a message fits more than one type, the first type in this order wins:
**escalation, complaint, order, question, other.** The order applies only among the types a
message fits within the business's scope.

- **Off-topic questions.** A question about something unrelated to the business ("Did you watch
  the match last night?") is `other`, not a `question`.

- **Complaint with a question.** "Why did my order arrive broken?" is a `complaint`: the question
  is the form, the dissatisfaction is the content.
- **Complaint or escalation.** A first, calm report of a problem is a `complaint`, even if it is
  angry in tone. It becomes an `escalation` only with one of the escalation signals: a repeat of an
  unresolved problem, a demand for a human or manager, an ultimatum or a threat. A previous message
  that shows the problem was already raised counts as evidence of a repeat.
- **Order with a question.** "Two of the large ones — do you deliver on Sundays?" is an `order`:
  the sender commits to a purchase. "Do you deliver on Sundays? If so I'd like two" is a
  `question`: the purchase depends on the answer.
- **Asking about an order.** "Where is my order?" with no sign of a problem is a `question`. "Where
  is my order? It was due yesterday" is a `complaint`.
- **Short replies.** "Yes", "that one", "the 14th works" are an `order` when the previous message
  asked the sender to choose or confirm something for a purchase or booking; after a previous
  message that asked for anything else, they are `other`.
- **Thanks with content.** "Thanks! Can you also add a card?" is an `order`; "Thanks, it arrived
  fine" is `other`.
- **Polite requests to change.** "Could you change the delivery to Friday?" is an `order` (it
  changes a booking), not a `question`.
- **Cancelling or changing because of a problem.** Asking to cancel, change or return because
  something went wrong ("Cancel my order, it's two weeks late") is a `complaint`, unless an
  escalation signal is present. Threatening to cancel ("Sort this out or I cancel") is an
  `escalation`. Cancelling with nothing wrong ("Please cancel, I don't need it any more") is an
  `order`.
- **Returns and exchanges.** Returning or exchanging an item with nothing wrong with it (the
  sender changed their mind or chose the wrong size) is an `order`. Returning a faulty or wrongly
  sent item is a `complaint`. Asking how returns work is a `question`.

Answer for every item with one of the five types.
