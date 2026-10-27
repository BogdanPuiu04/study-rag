# 0004. Index without pickle, with build metadata

**Status:** accepted

## Context
The index travels through S3. Loading a pickle executes code, so a tampered object would be
remote code execution inside the Lambda. Separately, vectors from two embedding models
compare as plausible-looking garbage, not as an error.

## Decision
Save vectors as `.npz` (loaded with `allow_pickle=False`) plus chunks and metadata as JSON.
Record embedder, model, dimension and chunking parameters in the index; refuse to start when
embedder, model or dimension differ from the config. Writes are atomic (temp file + rename).

## Consequences
- A tampered index can at worst be wrong, not executable.
- Changing the embedding model requires a reindex, and says so loudly.
- BM25 structures aren't stored; they're rebuilt on load by replaying `add()`, which keeps the
  file format independent of the BM25 implementation (an exercise).
