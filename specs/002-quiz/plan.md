# Plan 002 — Quiz

Design rationale: [ADR 0009](../../docs/adr/0009-quiz-generation.md).

```text
store.all_chunks() -> select_chunks(n, course) -> generate_quiz (1 structured-output call)
                   -> structural_issues (free) -> [verify_question (LLM, --verify)] -> QuizResult
```

| Piece | File | Notes |
|---|---|---|
| Schema | `quiz/models.py` | `QuizQuestion`, `Quiz`: also the structured-output schema |
| Selection | `quiz/select.py` | pure, deterministic, round-robin over (source, section) |
| Generation | `quiz/generator.py` | `generate_quiz`, `make_quiz`, `QuizResult`; error types |
| Gates | `quiz/verifier.py` | `structural_issues` (done), `verify_question` (LLM) |
| Entry points | `pipeline.StudyRAG.quiz`, `cli quiz`, `api POST /quiz` | |
| Store | `VectorStore.all_chunks()` | added to the protocol; memory and pgvector implement it |

Errors: `QuizUnavailableError` (501) · `NoMaterialError` (404) · `RefusalError` (422) · other
`QuizError` (502).
