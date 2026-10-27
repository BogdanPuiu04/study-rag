# 0009. Quiz generation design

**Status:** accepted (optional extension; not on the path to Milestone 1)

## Context
The same index that answers questions can generate practice questions with citations. The
risk is low-quality questions: unsupported answers, absurd distractors, trivia.

## Decision
- **Selection is deterministic and diverse** (`quiz/select.py`): round-robin over
  `(source, section)` groups, skipping chunks under 200 characters.
- **One question per excerpt, one call.** Structured output (`Quiz` schema) guarantees the
  shape; each question carries the number of the excerpt it came from, so the citation is
  checked data, not prose.
- **Two gates:** a free structural check (four distinct options, valid index, real excerpt
  number) always runs; an LLM verifier that sees one excerpt and one question runs behind
  `--verify` (an exercise, see `docs/EXERCISES.md` §12).
- Excerpt text is data: a closing `</excerpt>` inside a document is stripped so it cannot end
  the block and inject instructions.
- Needs a real model; the extractive offline backend answers with a clear error (HTTP 501).

## Consequences
- Rejected questions are reported, not hidden; the rejection rate is a quality signal.
- No persistence or per-user history: a quiz is generated on request.
- The writer prompt is empty until exercise 12a is done; `quiz` fails with a message saying so.
