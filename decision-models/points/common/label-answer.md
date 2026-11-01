## Your answer file

The items in this request are unrelated to one another. Judge each one on its own text.

Write your answer to the path in the header's `answerPath`.

- The first line is `{"bodySha256": "<the header's bodySha256, copied exactly>"}`.
- Then write one line per item, in any order: `{"key": "<the item's key>", "answer": <answer>}`.
  `<answer>` is one of the values under "Answers" above, written as JSON: `true` or `false`
  without quotes, or a type name in double quotes.
- Every key gets exactly one line. Write nothing else: no explanation, no blank lines and no other
  fields.

If the answer is too long for one Write, write it in parts. Replace the final `.jsonl` of
`answerPath` with `.1.jsonl`, `.2.jsonl` and so on, in order. Only part 1 starts with the
`bodySha256` line.
