# Spec 002 — Quiz from course materials

**Status:** implemented (plumbing); exercises 12a/12b open. **Milestone:** 3, optional extension.

## Problem
Asking questions only helps once you know what to ask. Before an exam I want the material to
quiz me: multiple-choice questions, each tied to the lecture it came from, so a wrong answer
tells me what to re-read.

## User stories
1. As a student, I run `study-rag quiz --n 5 [--course X]` and get questions with four
   options, the correct answer, an explanation and the source file/section.
2. As a client of the API, I `POST /quiz` and get the same as JSON.
3. As the developer, I can see how many generated questions were rejected, and why.

## Acceptance criteria
- [x] Chunks are chosen to spread across sections; chunks too short to ask about are skipped.
- [x] One question per excerpt; each question names the excerpt it is based on.
- [x] Malformed questions (not 4 distinct options, bad answer index, nonexistent excerpt,
      empty fields) are dropped and reported with reasons.
- [x] Document text cannot close the excerpt block and inject instructions.
- [x] Needs a real model: the offline extractive backend gives a clear error (CLI exit 2,
      HTTP 501), and so does an empty writer prompt.
- [x] Refusal and truncated output surface as errors (422, 502), not as an empty quiz.
- [x] Cost per quiz is reported when the model's price is known.
- [ ] Writer prompt written (exercise 12a) and judged on 5+ quizzes.
- [ ] LLM verifier written (exercise 12b) and checked against hand labels.

## Non-goals
Per-user history, scoring, spaced repetition, free-text answers, a UI.

## Constraints
Constitution 2 (grounded), 3 (no framework), 7 (excerpts are data).
