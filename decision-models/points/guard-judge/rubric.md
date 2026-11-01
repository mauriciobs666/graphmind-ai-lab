# Rubric: does the condition hold?

Each item is a workflow checkpoint in a support or sales conversation. Before the workflow moves
on, it asks one yes/no question: does the stated condition hold, given what the conversation has
established so far? You decide that, item by item, from the item's text alone.

## What an item looks like

Every item starts with one line:

    CONDITION: <a sentence about what the user has done or said>

followed by the evidence, in one of two forms:

- `CURRENT STATE:` and a JSON summary the workflow keeps of the conversation. `request` is what the
  user asked for, `known` lists facts the user has supplied, and `missing` lists facts the summary
  records as not yet supplied.
- `RECENT TURNS (context only):` and the last few messages of the conversation, oldest first, one
  per line as `<speaker>: <text>`. People speak under their first names; the business's assistant
  speaks under an assistant name such as `Assistant` or `<business> Assistant`.

Judge only what is written in the item. Do not assume facts that are not there, and do not use
knowledge of the business, product or world to fill a gap.

## Answers

### true — the condition holds

Answer `true` when the evidence shows that the user has done what the condition says, plainly
enough that the workflow could act on it now without asking again.

- A fact counts as supplied when the user states it, or confirms it when it is offered to them
  ("yes, that address is right").
- A fact stated in other words than the condition's counts ("next Friday afternoon" supplies a time
  window; "anything under 300" supplies a budget).
- Facts may be spread over several turns or listed in `known`.
- When the condition asks for "enough information", it holds when every fact the request depends on
  is present. A detail that would only refine the result, and whose absence would not stop the work
  from starting, is not required.

### false — the condition does not hold

Answer `false` when the evidence does not establish the condition.

- A fact the condition needs is absent.
- A fact is stated ambiguously, so that two different readings would lead the work in different
  directions ("sometime next week or the one after", "the usual address" when no address is on
  record in the item).
- The user has promised the fact but not given it ("I'll send the order number later").
- The user gave the fact and then withdrew or contradicted it, and the latest statement leaves it
  unsettled.
- Only the assistant has stated the fact, and the user has not confirmed it.

## Boundary cases

- **The `missing` list.** An entry in `missing` that the condition needs makes the answer `false`.
  An entry in `missing` that the condition does not need does not change the answer. An empty
  `missing` list is not enough on its own: check `known` against the condition.
- **The latest statement wins.** When the user changes a fact, judge the last version. If the
  last version is clear, the earlier one does not matter.
- **Questions back to the assistant.** A user who answers with a question ("does Tuesday work for
  you?") has proposed Tuesday; a user who asks "which days do you have?" has not chosen one.
- **Partial confirmation.** "Yes" to a question that offered two things at once confirms both only
  if nothing else in the evidence contradicts either.
- **Hedged answers.** "Probably the blue one" does not choose an option; "the blue one, I think,
  yes" does.
- **Facts given before the shown turns.** Only the shown evidence counts. If a turn refers to a fact
  given earlier ("as I said before") without restating it, the fact is not established.
- **Sense of the request.** Do not judge whether the request is sensible, possible or polite; judge
  only whether the condition holds.
- **Condition wording.** Read the condition literally. "Has chosen one of the options" holds only if
  options were offered and one was chosen; it does not hold if the user named something that was
  never offered.

Answer for every item. When the evidence leaves you unsure whether the condition holds, the
condition is not established: answer `false`.
