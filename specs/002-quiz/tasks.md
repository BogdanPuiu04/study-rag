# Tasks 002 — Quiz

- [x] `VectorStore.all_chunks()` on the protocol, `MemoryStore` and `PgVectorStore`
- [x] `quiz/models.py`, `quiz/select.py`
- [x] `quiz/verifier.py`: `structural_issues`
- [x] `quiz/generator.py`: prompt assembly with escaped delimiters, structured output, `make_quiz`
- [x] `StudyRAG.quiz`, CLI `quiz`, API `POST /quiz`
- [x] `tests/test_quiz.py` (29 plumbing tests, fake client)
- [x] `tests/test_quiz_verifier.py`
- [x] ADR 0009
- [x] Writer prompt, `prompts/quiz_system.md`
- [x] `verify_question` + `prompts/quiz_verifier_system.md`
- [ ] Run on real course material, record the bad-question rate before/after the prompt
