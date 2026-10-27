# Spec 005 — Fine-tune the embedding model (draft)

**Status:** not started. **Milestone:** 3, optional. Only after the retrieval eval shows
where retrieval fails: this spec is a hypothesis, not a plan.

## Problem
The off-the-shelf BGE model may rank course-specific vocabulary poorly (lecturer's notation,
abbreviations). Fine-tuning on question/passage pairs from the user's own courses could help.
It is the part of the project that trains a model rather than calling one.

## Hypothesis
Fine-tuning `BAAI/bge-small-en-v1.5` on synthetic (question, chunk) pairs raises hybrid
recall@5 / MRR on the golden set versus the base model. It may not: say so if it doesn't.

## Sketch (to be specified properly when started)
1. Generate questions per chunk with Claude; keep a pair only if the base model finds the
   chunk but not at rank 1 (hard but valid).
2. Split **by document**, never by pair; the golden set is never training data.
3. `sentence-transformers` v3 trainer with `MultipleNegativesRankingLoss`; Colab/Kaggle GPU
   is enough for a small model.
4. Push to the Hugging Face Hub as a **private** model (the data is copyrighted), with a
   model card: data, method, results, limits.
5. Add a row to the results table; the index metadata check forces a reindex for the new model.

## Open questions
Is there enough text in one student's courses to fine-tune without overfitting?
What does a 384-dim model gain over BGE-base on this corpus, measured?

## Constraints
Constitution 1 (measured: no claim without a before/after), 8 (no course data in git or on a
public Hub repo).
