## Your answer file

Write your answer to the path in the header's `answerPath`.

- The first line is `{"bodySha256": "<the header's bodySha256, copied exactly>"}`.
- Then write one line per seed, in any order: `{"key": "<the seed's key>", "item": {…}}`. `item`
  holds exactly the fields listed under "The item you write", and no other field.
- Every key gets exactly one line. Write nothing else: no explanation, no blank lines and no other
  fields.

If the answer is too long for one Write, write it in parts. Replace the final `.jsonl` of
`answerPath` with `.1.jsonl`, `.2.jsonl` and so on, in order. Only part 1 starts with the
`bodySha256` line.
