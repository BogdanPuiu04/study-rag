# Build guide — the parts you write

Everything else in the repo is plumbing. These seven pieces are the AI/ML core:
the parts an interviewer will ask you to explain, whiteboard, or extend.

**Rules of the game**
- No LangChain / LlamaIndex. numpy and the Anthropic SDK only.
- Run one exercise's tests at a time: `pytest tests/test_metrics.py -q`.
- After each exercise, run `study-rag eval eval/golden.jsonl` and paste the table in
  your notes. The point is to *see* each component move the numbers.
- Stuck for more than 30 minutes? Read the hint ladder below before searching online.
  Each hint reveals a bit more. Stop at the first one that unblocks you.

| # | What | File | Tests | Size |
|---|---|---|---|---|
| 1 | Retrieval metrics | `src/study_rag/eval/metrics.py` | `test_metrics.py` | ~10 lines |
| 2 | Chunking | `src/study_rag/ingest/chunking.py` | `test_chunking.py` | ~50 lines |
| 3a | Vector search | `src/study_rag/store/memory.py` | `test_store.py` | ~3 lines |
| 3b | BM25 keyword search | `src/study_rag/store/memory.py` | `test_store.py` | ~30 lines |
| 4 | Reciprocal Rank Fusion | `src/study_rag/retrieval.py` | `test_store.py` | ~10 lines |
| 5 | Prompt + Claude call | `prompts/answer_system.md`, `src/study_rag/llm.py` | `test_pipeline.py` | ~40 lines |
| 6 | Cross-encoder reranking | `src/study_rag/retrieval.py` | `test_rerank_judge.py` | ~5 lines |
| 7 | LLM-as-judge | `prompts/judge_system.md`, `src/study_rag/eval/judge.py` | `test_rerank_judge.py` | ~20 lines |

Once 1–4 and 5b are done, `test_pipeline.py` turns green and the CLI works end to end.

---

## 1. Retrieval metrics

**Goal.** Turn "does search work?" into numbers, before touching anything else.

**Concepts**
- *Rank*: the position (1-based) of the first relevant chunk in the result list.
- *recall@k*: share of questions where that rank is ≤ k. Answers "was the right
  chunk in what we sent to the LLM?"
- *MRR* (Mean Reciprocal Rank): mean of 1/rank, a miss counts 0. Rewards getting
  the right chunk *near the top*, not just somewhere in the top k.

**Hint ladder**
1. `first_relevant_rank`: loop with `enumerate(results, start=1)`; `is_relevant` already exists.
2. `recall_at_k`: a miss is `None`, so check that before comparing with `k`.
3. Guard the empty list in both aggregate functions, or you'll divide by zero.

**Pitfall.** Off-by-one: rank 1 is the first result, not rank 0. The tests check this.

**Interview questions**
- Two systems have recall@5 = 1.0. One always ranks the answer 1st, the other 5th. MRR of each?
- Why not just measure the final answer quality? *(Hint: when an answer is wrong, is the
  problem retrieval or generation? You can't tell from the answer alone.)*
- Why label the golden set by `(source, phrase)` instead of `chunk_id`?
- What's nDCG and when would you prefer it over MRR? *(Multiple relevant chunks per question.)*

---

## 2. Chunking

**Goal.** Split documents into pieces that are each *about one thing*, small enough
to be precise and big enough to be self-contained.

**Decide first, then code.** Write 3–4 sentences in the README design table:
- What do your real course materials look like? Markdown notes with headings?
  Slide PDFs (short, fragmented text)? Dense papers?
- Options: fixed-size windows with overlap · split on headings then by size ·
  split on paragraphs/sentences · semantic chunking (split where embeddings shift).

**Hint ladder (heading-aware approach)**
1. Walk the text line by line. A line matching `^(#{1,6})\s+(.*)$` is a heading;
   its `#` count is its level.
2. Keep a stack of headings: on a level-*n* heading, truncate the stack to *n−1* and
   push. `" > ".join(stack)` is the section path.
3. On every heading, *flush* the lines collected so far as one section.
4. Sections longer than `size`: split on blank lines (paragraphs), greedily pack
   paragraphs into pieces ≤ `size`, start each new piece with the last `overlap`
   characters of the previous one.
5. A single paragraph longer than `size` needs a hard split too, or that test fails.

**Pitfalls**
- Text before the first heading: don't drop it.
- Empty sections (heading followed directly by another heading): skip them.
- PDFs: `loaders.py` inserts `[page N]` markers. Keep them inside chunks so citations can say the page.

**Experiment (after exercise 4 works).** Run the eval with
`STUDY_RAG_CHUNK_SIZE=300`, `800`, `1500`. Write down recall@1 and MRR for each.
Explain the trend in one sentence.

**Interview questions**
- Small chunks vs large chunks: effect on precision, recall, and token cost per query?
- Why overlap? What does doubling the overlap cost?
- Your PDFs are slides with 20 words each. What changes?
- What's "parent document retrieval" / "small-to-big" and what problem does it solve?

---

## 3a. Vector search

**Goal.** Return the k chunks whose embeddings are closest to the query embedding.

**Concepts.** Cosine similarity = dot product divided by both norms. The vectors are
already normalised to length 1, so cosine similarity is just the dot product.

**Hint ladder**
1. `self.vectors` is an (n, d) matrix and `query_vec` is (d,). One matrix-vector product gives all n scores.
2. `np.argsort(-scores)[:k]` gives the indices of the top k.
3. Return `ScoredChunk(self.chunks[i], float(scores[i]))`. The `float()` matters for JSON later.

**Interview questions**
- Why cosine and not Euclidean distance? *(For unit vectors they produce the same ranking. Why?)*
- This is O(n·d) per query. At what corpus size does it stop being OK, and what
  do you use then? *(ANN indexes: HNSW, IVF. What do you trade?)*
- Why must the query and the documents use the *same* embedding model?
- BGE models recommend a query prefix ("Represent this sentence for searching…").
  Why might asymmetric query/document encoding help?

---

## 3b. BM25

**Goal.** Classic keyword ranking: rewards rare terms that appear in the chunk,
with diminishing returns for repetition, normalised by chunk length.

**Formula.** For query terms *t* and chunk *D*:

```text
score(D) = Σ_t  IDF(t) · f(t,D)·(k1+1) / ( f(t,D) + k1·(1 − b + b·|D|/avgdl) )
IDF(t)   = ln(1 + (N − n_t + 0.5) / (n_t + 0.5))
```

`f(t,D)` = count of t in D · `|D|` = tokens in D · `avgdl` = mean chunk length ·
`N` = number of chunks · `n_t` = chunks containing t · `k1≈1.5`, `b≈0.75`.

**Hint ladder**
1. In `add()`: for each chunk store a `collections.Counter` of its tokens (that's *f*
   and |D|), and update a corpus-wide Counter of document frequencies (*n_t*,
   count each term once per chunk).
2. In `keyword_search`: tokenize the query, deduplicate terms, accumulate the
   formula per chunk. A numpy array of n zeros is a fine accumulator.
3. Drop chunks with score 0. The test checks that unknown terms return `[]`.

**Pitfalls**
- Computing document frequencies at query time: correct but O(corpus) per query. Do it in `add()`.
- `clear()` must reset your new fields too. Look at how it's implemented.
- Pickling: your new attributes are saved automatically because `save()` pickles `__dict__`.

**Interview questions**
- What does `k1` control? What happens with k1 = 0? And with very large k1?
- What does `b` control? Why should long chunks be penalised?
- Name 3 kinds of queries where BM25 beats embeddings, and 3 where it loses.
- Indexing `section + text` vs only `text`: what's your hypothesis? Measure it.

---

## 4. Reciprocal Rank Fusion

**Goal.** Combine the vector ranking and the BM25 ranking into one.

**Concept.** Cosine scores sit in [−1, 1] and BM25 scores are unbounded, so adding
them is meaningless. RRF ignores the scores and uses only positions:
`score = Σ 1/(k + rank)` over every ranking the chunk appears in, with k = 60.

**Hint ladder**
1. A dict `chunk_id → fused score` and another `chunk_id → Chunk`.
2. Iterate each ranking with `enumerate(..., start=1)`.
3. Sort the ids by fused score, descending.

**Experiment.** Now `study-rag eval` shows all three modes. Does hybrid beat both?
With the `hash` embedder it may not. Install `.[local]`, set
`STUDY_RAG_EMBEDDER=hf`, re-ingest, re-run. **This before/after table goes in
your README**: it's the single best portfolio artifact in the project.

**Interview questions**
- Why k = 60? What happens with k = 1? With k = 1000?
- Alternative: normalise both scores to [0, 1] and take a weighted sum. Pros and cons?
- A chunk appears only in the BM25 list at rank 1. Another appears in both lists
  at rank 3. Which wins?

---

## 5. Prompt + Claude call

### 5a. System prompt (`prompts/answer_system.md`)
Must make Claude answer **only** from the excerpts, cite `[n]`, and **say so
when the excerpts don't cover the question**. Keep it short and specific.
Explain *why* each rule exists: Claude follows reasons better than shouted rules.

### 5b. `build_user_message`
Numbered, clearly delimited excerpts (XML-style tags work well with Claude), each
with source and section, then the question **last**.

### 5c. `ClaudeLLM.generate`

**Hint ladder**
1. `self.client.messages.create(model=..., max_tokens=..., system=..., messages=[{"role": "user", "content": ...}], output_config={"effort": self.effort})`
2. `response.content` is a *list of blocks*. With thinking on, there are thinking blocks before the text. Keep `b.type == "text"` only.
3. Check `response.stop_reason`: `"refusal"` → raise `RefusalError`;
   `"max_tokens"` → the answer is truncated. Decide: raise, or warn?
4. Store `response.usage` on `self.last_usage` (as a dict with input/output
   tokens): the pipeline puts it in the trace automatically.
5. Timeout: `anthropic.Anthropic(timeout=60.0)`. The SDK retries 429/5xx twice by
   default; read its docs on `max_retries` and explain why you'd keep or change it.

**Try it** (needs an API key; costs cents):
`STUDY_RAG_LLM=claude study-rag ask "What is dropout?"`, then ask something *not*
in the notes. Does it abstain? Look at `.traces/queries.jsonl`.

**Interview questions**
- Why put the question after the excerpts?
- How do you stop the model from answering from general knowledge? How do you *test* that?
- The answer is wrong. Walk me through how you debug it. *(Trace → was the right chunk retrieved?)*
- Where does the cost of a query come from? Estimate it for top_k = 5, 800-char chunks.
- What's prompt caching and would it help here? *(Which part of the prompt is stable?)*

---

## 6. Cross-encoder reranking

**Concept.** The embedding model is a *bi-encoder*: it embeds the query and the
chunk separately, so chunks can be embedded once in advance. A *cross-encoder*
reads the (query, chunk) pair *together*. It's more precise, but it must run per
pair at query time, so it's applied only to the top ~20 fused candidates.

**Hint.** One `self.model.predict([...pairs...])` call (batched!), zip with the
candidates, sort, wrap the new scores in `ScoredChunk`.

**Experiment.** `STUDY_RAG_RERANKER=hf study-rag eval eval/golden.jsonl`. Compare
recall@1 and MRR with and without. Also time it: what does it cost in latency?

**Interview questions**
- Why not use the cross-encoder for the whole corpus?
- Bi-encoder vs cross-encoder vs ColBERT (late interaction): one sentence each.
- Reranking improved MRR but not recall@20. Why is that expected?

---

## 7. LLM-as-judge

**Concept.** Retrieval metrics say whether the right excerpt was *found*. A judge
checks whether the *answer* sticks to it: it extracts claims and verifies each
against the excerpts. Structured outputs make the verdict machine-readable with no
parsing.

**Hint ladder**
1. Write `prompts/judge_system.md` first. Be strict: "general knowledge does not count as support".
2. User message: reuse `build_user_message`, then append the answer under test, clearly delimited.
3. `client.messages.parse(model=..., max_tokens=..., system=..., messages=[...], output_format=Verdict)` → `response.parsed_output`.
4. Check `stop_reason` before trusting the output.

**Run it** (costs cents; `--limit` caps it):
`STUDY_RAG_LLM=claude study-rag eval-answers eval/golden.jsonl --limit 5`

**Calibrate the judge.** Read 10 verdicts in `.traces/answer_eval.jsonl` yourself.
Do you agree? An unvalidated judge is just another unmeasured model.

**Interview questions**
- How do you know the judge is right? *(Human agreement on a sample.)*
- Known LLM-judge biases: position, verbosity, self-preference. How would you mitigate each?
- Why is abstention on *unanswerable* questions as important as faithfulness?
- Faithfulness 100% but users still unhappy. What metric is missing? *(Answer relevance / completeness.)*

---

## When all seven are done

1. Replace `data/sample` usage with your real courses (locally, in `data/<course>/`).
2. Write 30+ golden questions from questions you *actually had* while studying.
3. Fill the README results table: hash vs BGE · ± reranker · chunk sizes.
4. Deploy (see `infra/README.md`) and add the CloudWatch screenshot to the README.
