# study-rag constitution

Principles every spec, plan and change is checked against. Derived from decisions already
made in this repo (`docs/adr/`), not new rules.

1. **Measured before optimised.** A change to chunking, embeddings, retrieval, ranking or a
   prompt comes with eval numbers before and after. "Looks better" is not a result, and an
   unmeasured improvement is reported as *unknown*.
2. **Grounded or silent.** Answers come from the user's materials, with citations that map to
   a real excerpt. When the materials don't cover the question the system says so.
3. **No RAG framework.** Chunking, search, fusion, reranking, prompts and the judge are
   implemented directly on numpy and the Anthropic SDK, each with its own tests.
4. **Idle cost is zero.** Nothing that bills by the hour without traffic: no NAT gateway,
   OpenSearch Serverless, always-on SageMaker endpoint or provisioned database. Every AWS
   resource ships with a budget or alarm.
5. **Least privilege.** Separate roles for deploy and runtime, explicit actions, resource ARNs
   wherever the service supports them. No long-lived AWS keys anywhere.
6. **Fail loudly on mismatch.** A config that makes results wrong (different embedding model
   than the index, unknown LLM name, out-of-range setting) stops at startup, not at query time.
7. **Untrusted text stays data.** Course documents are never treated as instructions;
   delimiters are escaped; each defence has a test that attacks it.
8. **Copyrighted material stays out of git.** Only `data/sample/` is committed. Real courses
   and `docs/private/` are ignored.
9. **Offline by default.** The default path (hash embedder, extractive answers, in-memory
   store) runs with no network and no key, so CI is free and deterministic.
10. **Infrastructure code is tested like code.** Plumbing tests always pass in CI; core
    component tests run in their own CI job.
