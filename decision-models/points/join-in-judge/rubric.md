# Rubric: should the agent reply, although it was not @mentioned?

Each item is a moment in a group chat thread where people and one AI agent take part. A new
message has just been posted by a person, and it does not @mention the agent. You decide, from the
item's text alone, whether the agent should post a reply to it. Replying when it should not
interrupts people; staying silent when it should leaves someone waiting. Both mistakes count
equally.

## What an item looks like

    AGENT: <the agent's name>

    RECENT TURNS (context only):
    <speaker>: <text>
    ...

    NEW MESSAGE:
    <speaker>: <text>

`AGENT` names the agent. `RECENT TURNS` are the messages before the new one, oldest first; the
block is absent when the new message opens the thread. Lines whose speaker is the agent's name are
the agent's own earlier turns. `NEW MESSAGE` is the message you judge, and it is always from a
person.

## Answers

### true — the agent should reply

Answer `true` when a reasonable participant would expect the agent to answer this message now:

- the message is addressed to the agent, by its name or by an unmistakable reference ("can the
  bot check…", "assistant, what's…"), without an @mention;
- the message answers a question the agent asked in its most recent turn, or asks the agent a
  follow-up about what it just said, and the conversation is still between the sender and the
  agent;
- the message asks the group for something that is clearly the agent's job in this thread (the
  thread shows what the agent does), nobody else has been asked, and nobody has already answered.

### false — the agent should stay silent

Answer `false` when no reply from the agent is expected:

- the message is addressed to another person, by name, by role or by continuing an exchange
  between people;
- the message is small talk, a joke, a greeting to the room or a social remark that invites no
  answer from the agent;
- the message only acknowledges or closes an exchange ("thanks", "ok, got it", "perfect"), even
  if it follows the agent's turn;
- the message talks about the agent rather than to it ("the assistant said Tuesday, so let's go
  with that");
- the message asks something that a person in the thread has already answered, or that a named
  person has already been asked to handle;
- the message asks the room something outside what the agent does in this thread.

## Boundary cases

- **Name in passing.** The agent's name in a message does not make the message addressed to it.
  "Ask Nova later" is said to a person; "Nova, can you check this?" is said to the agent.
- **After the agent's turn.** A message right after the agent's turn is addressed to the agent
  only if it engages with what the agent said: answers its question, asks about its answer, or
  corrects it. A thank-you or a turn to another person is not.
- **Room questions.** "Does anyone know when the shop opens?" is for the agent when answering it is
  the agent's job in this thread and no person has picked it up; it is not when a person was
  clearly expected to answer ("Ana, you were there, when does it open?").
- **Small talk.** Small talk addressed to the agent by name that invites an answer ("Morning
  Nova, how's your week going?") is `true`. A greeting to the room ("Morning everyone!") is
  `false`.
- **Mixed messages.** "Thanks Nova! Bia, can you bring the cake?" is addressed to Bia: the part for
  the agent is only a thank-you.
- **Several people talking.** When people are discussing among themselves and the message
  continues that discussion, the agent stays silent even if it could add something useful.
- **Unclear addressee.** When the message could be meant for the agent or for a person and nothing
  in the thread settles it, answer `false`: a reply is only expected when the thread makes it so.

Answer for every item.
