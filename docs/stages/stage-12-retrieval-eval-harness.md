# Stage 12 - Retrieval Eval Harness Findings

## 1. Stage Metadata

- VERIFIED - Stage name: Stage 12 - Retrieval Eval Harness.
- VERIFIED - Branch inspected: `stage-11-summary-episode-memory` from `git branch --show-current`.
- VERIFIED - Date of inspection: 2026-06-11 from the session context.
- VERIFIED - Status: inspection only.
- VERIFIED - Project instructions require retrieval to remain scoped by `scope_id` and treat scopes as the user-controlled privacy boundary (`AGENTS.md:105-124`).
- VERIFIED - Project instructions identify `eval_*` as eval harness scaffolding and say the eval harness should tune the starting retrieval weights (`AGENTS.md:242-277`).
- VERIFIED - Frontend rules were read because repo instructions require `FRONTEND.md` for frontend-related work, but this Stage 12 inspection did not identify frontend implementation work in scope (`AGENTS.md:406-415`, `FRONTEND.md:225-239`).
- VERIFIED - Stage 11 explicitly kept retrieval SQL and scoring/ranking unchanged, while documenting that summary and raw message episodes share the existing retrieval mix (`docs/stages/stage-11-summary-episode-memory.md:384-408`).
- UNKNOWN - `docs/stages/stage-12-retrieval-eval-harness.md` was not present in the initial `rg --files` output. Absence has no file path/line range; this file is newly created for findings only.

## 2. Repo Findings

### A. Existing Eval Harness

- VERIFIED - The existing eval harness implementation lives in `src/smriti/memory/eval.py` and imports the memory service rather than owning a separate runner or CLI (`src/smriti/memory/eval.py:1-9`).
- VERIFIED - The harness input is `RetrievalEvalCase` with `name`, `user_id`, `scope_id`, `query`, `expected_episode_ids`, and `top_k` (`src/smriti/memory/eval.py:12-20`).
- VERIFIED - The harness result and summary models include hit@k, precision@k, recall@k, and reciprocal rank / mean reciprocal rank fields (`src/smriti/memory/eval.py:22-40`).
- VERIFIED - The harness is invoked by calling `run_retrieval_eval(service, cases, now=...)`; it calls `service.retrieve_scoped_episodes(...)` for each case and returns in-memory results plus a summary (`src/smriti/memory/eval.py:42-69`).

Relevant excerpt:

> `retrieved = await service.retrieve_scoped_episodes(`  
> `user_id=case.user_id,`  
> `scope_id=case.scope_id,`  
> `query=case.query,`  
> `top_k=case.top_k,`

Source: `src/smriti/memory/eval.py:53-60`.

- VERIFIED - Precision@k, recall@k, and reciprocal rank are computed from retrieved episode IDs versus expected episode IDs (`src/smriti/memory/eval.py:79-106`).
- VERIFIED - Summary metrics are total cases, hit rate@k, mean precision@k, mean recall@k, and mean reciprocal rank (`src/smriti/memory/eval.py:109-126`).
- VERIFIED - `RetrievalEvalCase`, result types, and `run_retrieval_eval` are exported from the memory package (`src/smriti/memory/__init__.py:18-23`, `src/smriti/memory/__init__.py:90-104`).
- VERIFIED - There is no project script entry for the eval harness; the only `[project.scripts]` entry is `migrate` (`pyproject.toml:30-31`).
- VERIFIED - The Makefile exposes setup/start/stop/restart/logs/status targets and does not show an eval target (`Makefile:1-19`).
- VERIFIED - Current eval test invocation is direct Python test code: `tests/test_memory_eval.py` imports `run_retrieval_eval`, creates cases inline, and awaits the helper (`tests/test_memory_eval.py:14-26`, `tests/test_memory_eval.py:110-131`).
- VERIFIED - Current eval fixture data is synthetic: it creates messages named `eval memory {index}` and a wrong-scope message named `eval memory from the wrong scope` (`tests/test_memory_eval.py:71-87`).
- VERIFIED - Current eval test uses `FakeEmbedder(dimensions=768)` when constructing the `MemoryService` (`tests/test_memory_eval.py:49-50`).
- VERIFIED - Current eval test manufactures retrieval geometry by embedding `"eval query"` and updating seeded episode embeddings to that vector (`tests/test_memory_eval.py:89-100`, `tests/test_memory_eval.py:321-335`).
- VERIFIED - Current eval test also directly sets scoring fields such as `created_at`, `last_accessed_at`, `importance`, and `access_count` (`tests/test_memory_eval.py:90-108`, `tests/test_memory_eval.py:295-318`).
- VERIFIED - `FakeEmbedder` is documented as a deterministic, local-only embedder for tests and eval fixtures (`src/smriti/embeddings/fake.py:15-17`).

Relevant excerpt:

> `class FakeEmbedder:`  
> `"""Deterministic, local-only embedder for tests and eval fixtures."""`

Source: `src/smriti/embeddings/fake.py:15-17`.

### B. Real Embedder

- VERIFIED - The embedder boundary is an async `Embedder` protocol with `dimensions`, `embed_text`, and `embed_texts` (`src/smriti/embeddings/base.py:9-20`).
- VERIFIED - `OllamaEmbedder` is the real local embedder; it defaults to model `nomic-embed-text`, base URL `http://127.0.0.1:11434`, optional dimensions, and `num_ctx=8192` (`src/smriti/embeddings/ollama.py:21-30`).
- VERIFIED - `OllamaEmbedder.embed_text()` delegates to `embed_texts([text])`; `embed_texts()` posts `model`, `input`, and `options.num_ctx` to Ollama (`src/smriti/embeddings/ollama.py:44-67`).
- VERIFIED - `OllamaEmbedder` enforces localhost-only HTTP base URLs and rejects non-localhost hostnames, credentials, query strings, and fragments (`src/smriti/embeddings/ollama.py:96-106`).
- VERIFIED - The request URL is the configured localhost base path plus `/api/embed` (`src/smriti/embeddings/ollama.py:108-112`).
- VERIFIED - The live FastAPI app wires `OllamaEmbedder` by default unless an embedder is injected, passing `ollama_embed_num_ctx` from settings (`src/smriti/api/app.py:34-55`).
- VERIFIED - Runtime settings define local Ollama base URL, chat model, chat context, embedding context, and chat timeout fields (`src/smriti/config.py:26-30`).
- VERIFIED - `.env.example` exposes `SMRITI_OLLAMA_BASE_URL`, `SMRITI_OLLAMA_CHAT_MODEL`, `SMRITI_OLLAMA_CHAT_NUM_CTX`, and `SMRITI_OLLAMA_EMBED_NUM_CTX` (`.env.example:9-13`).
- VERIFIED - The project README lists Ollama as localhost-only at `127.0.0.1:11434` (`README.md:55-62`).
- VERIFIED - The memory service stores the embedder as an injected dependency and defaults the embedding model registry key to `nomic-embed-text` (`src/smriti/memory/service.py:130-135`).
- VERIFIED - The default database migration registers active 768-dimensional `nomic-embed-text` embeddings in `embedding_models` and stores vectors in `embeddings_768` (`src/smriti/db/migrations/001_init.sql:81-96`).
- VERIFIED - The memory service resolves the active embedding model primary key by model ID, 768 dimensions, and `is_active = TRUE` (`src/smriti/memory/service.py:1513-1527`).
- VERIFIED - The live message episode path calls `self.embedder.embed_text(request.content)`, validates the vector, inserts a `kind='message'` episode, and inserts into `embeddings_768` (`src/smriti/memory/service.py:500-574`).
- VERIFIED - The separate stored-message episode path calls `self.embedder.embed_text(message.content)`, validates the vector, inserts a `kind='message'` episode, and inserts into `embeddings_768` (`src/smriti/memory/service.py:576-619`).
- VERIFIED - The Stage 11 summary path calls `self.embedder.embed_text(summary_text)`, validates the vector, inserts a `kind='summary'` episode, and inserts into `embeddings_768` (`src/smriti/memory/service.py:621-680`, `src/smriti/memory/service.py:968-1049`).
- VERIFIED - The live retrieval path embeds the query through the same injected embedder before SQL retrieval (`src/smriti/memory/service.py:694-711`).
- ASSUMED - Making the existing harness use the real embedder looks mostly like service wiring because `run_retrieval_eval` receives a `MemoryService`, `MemoryService` receives any `Embedder`, and the live app already wires `OllamaEmbedder` through that same protocol (`src/smriti/memory/eval.py:42-60`, `src/smriti/memory/service.py:130-135`, `src/smriti/api/app.py:51-55`).
- VERIFIED - The current eval test fixture is structurally tied to fake/synthetic behavior even if the harness function is not: it creates `FakeEmbedder(dimensions=768)` and overwrites stored embeddings with the query vector (`tests/test_memory_eval.py:49-50`, `tests/test_memory_eval.py:89-100`, `tests/test_memory_eval.py:321-335`).

### C. Scoring / Ranking Weights

- VERIFIED - Retrieval weights live as module-level constants in `src/smriti/memory/service.py`: similarity `0.55`, recency `0.20`, access `0.10`, importance `0.10`, frequency `0.05` (`src/smriti/memory/service.py:55-67`).

Relevant excerpt:

> `SIMILARITY_WEIGHT = 0.55`  
> `RECENCY_WEIGHT = 0.20`  
> `ACCESS_WEIGHT = 0.10`  
> `IMPORTANCE_WEIGHT = 0.10`  
> `FREQUENCY_WEIGHT = 0.05`

Source: `src/smriti/memory/service.py:61-65`.

- VERIFIED - Retrieval first fetches a bounded similarity-first candidate pool from SQL, then reranks in Python (`src/smriti/memory/service.py:712-715`, `src/smriti/memory/service.py:724-775`).
- VERIFIED - SQL converts pgvector cosine distance to similarity, orders the candidate pool by similarity descending, created time descending, and episode ID ascending, then applies `LIMIT` (`src/smriti/memory/service.py:741-759`).
- VERIFIED - Final weighted score is computed in Python from similarity, recency score, access score, importance score, and frequency score (`src/smriti/memory/service.py:1681-1711`).

Relevant excerpt:

> `score = (`  
> `SIMILARITY_WEIGHT * similarity`  
> `+ RECENCY_WEIGHT * recency_score`  
> `+ ACCESS_WEIGHT * access_score`  
> `+ IMPORTANCE_WEIGHT * importance_score`  
> `+ FREQUENCY_WEIGHT * frequency_score`

Source: `src/smriti/memory/service.py:1705-1711`.

- VERIFIED - Final ordering uses score descending, then created time descending, then episode ID ascending (`src/smriti/memory/service.py:1768-1773`).
- VERIFIED - The code comments call the candidate set an accepted Stage 5.2 heuristic and say lower-similarity episodes with high recency/importance may be missed until eval tuning improves it (`src/smriti/memory/service.py:712-715`).
- VERIFIED - AGENTS.md describes the same numeric weights as "Starting weights" and says, "These are starting guesses. The eval harness should be used to tune them." (`AGENTS.md:257-277`).
- VERIFIED - Stage 5 docs recommend module-level constants and say runtime configuration should wait until the eval harness can justify tuning (`docs/stages/stage-05-retrieval-and-eval.md:66-79`).
- VERIFIED - Stage 5 docs also describe `candidate_limit = max(top_k * 5, 25)` as a "similarity-first candidate heuristic" that may miss some lower-similarity but high-recency/high-importance episodes until eval tuning improves candidate selection (`docs/stages/stage-05-retrieval-and-eval.md:80-85`).
- VERIFIED - Stage 11 explicitly scoped out retrieval SQL changes and scoring/ranking changes (`docs/stages/stage-11-summary-episode-memory.md:384-408`).

### D. Retrieval Function(s) Under Test

- VERIFIED - The direct retrieval function under test is `MemoryService.retrieve_scoped_episodes` (`src/smriti/memory/service.py:694-703`).
- VERIFIED - Its signature takes `user_id`, `scope_id`, `query`, `top_k`, and optional keyword-only `now` (`src/smriti/memory/service.py:694-702`).
- VERIFIED - It returns `list[ScoredEpisode]` and rejects non-positive `top_k` before embedding (`src/smriti/memory/service.py:694-711`).
- VERIFIED - Retrieval SQL selects episode ID, user ID, scope ID, conversation ID, kind, message ID, message position, range start/end, content, timestamps, importance, access count, embedding model ID, and similarity (`src/smriti/memory/service.py:724-759`).
- VERIFIED - Retrieval filters by `episodes.scope_id = $1` and `conversations.user_id = $2`, with a conversation join requiring matching conversation and scope IDs (`src/smriti/memory/service.py:746-759`).
- VERIFIED - Retrieval does not filter by episode kind; `episodes.kind` is selected, but no `kind` predicate appears in the WHERE clause (`src/smriti/memory/service.py:724-759`).
- VERIFIED - Retrieval uses an `INNER JOIN embeddings_768` for the active embedding model and a `LEFT JOIN messages` for message metadata, so embedded summaries are eligible for retrieval without a message row (`src/smriti/memory/service.py:746-759`).
- VERIFIED - Returned episodes are reranked by `_scored_episode_sort_key`, truncated to `top_k`, and then access metadata is updated for returned rows (`src/smriti/memory/service.py:768-785`).
- VERIFIED - The returned `ScoredEpisode` model contains rank, IDs, kind, optional message and range fields, content, created time, importance/access metadata, embedding model ID, score components, and final score (`src/smriti/memory/models.py:174-197`).
- VERIFIED - The HTTP retrieval endpoint delegates to `MemoryService.retrieve_scoped_episodes` and returns `ScoredEpisodeResponse` objects (`src/smriti/api/routes/retrieval.py:15-29`).
- VERIFIED - The HTTP search body is `scope_id`, non-empty `query`, and `top_k`; the response model mirrors the scored episode fields without exposing vectors (`src/smriti/api/schemas.py:301-362`).
- VERIFIED - Based on the current `RetrievalEvalCase`, one existing labeled eval example contains the query text, user ID, scope ID, expected episode IDs, and top_k (`src/smriti/memory/eval.py:12-20`).
- VERIFIED - Based on the current test fixture, setup outside the eval case currently creates a user, scope, conversation, message episodes, expected episode IDs, and controlled embeddings/scoring fields (`tests/test_memory_eval.py:52-108`, `tests/test_memory_eval.py:226-335`).
- UNKNOWN - The repo does not define a final hand-labeled eval example schema beyond `RetrievalEvalCase`. The existing input model is line-cited above; no separate corpus/schema file was identified during inspection.

### E. Eval Data

- VERIFIED - The current eval data is created inline in `tests/test_memory_eval.py` by appending synthetic messages and constructing `RetrievalEvalCase` objects in test code (`tests/test_memory_eval.py:71-87`, `tests/test_memory_eval.py:110-128`).
- VERIFIED - The synthetic data uses `FakeEmbedder`, then overwrites embeddings and scoring fields to force deterministic ranking (`tests/test_memory_eval.py:49-50`, `tests/test_memory_eval.py:89-108`, `tests/test_memory_eval.py:295-335`).
- VERIFIED - The repository README describes the implemented eval capability as a "Minimal retrieval eval helper" and later lists "Expanded retrieval eval harness" as a next planned area (`README.md:11-31`, `README.md:347-351`).
- VERIFIED - The database schema includes `eval_scenarios`, `eval_runs`, and `eval_run_results` tables (`src/smriti/db/migrations/001_init.sql:104-132`).
- VERIFIED - DB tests assert the `eval_scenarios`, `eval_runs`, and `eval_run_results` tables exist, but the inspected eval helper does not insert into them (`tests/test_db.py:72-85`, `src/smriti/memory/eval.py:42-69`).
- UNKNOWN - No hand-labeled retrieval corpus file was identified during read-only search. Absence has no file path/line range; the closest inspected data source is the inline synthetic fixture cited above.
- ASSUMED - A plausible repo convention for a future real labeled set is not established by current code. The only explicit instruction-like convention says eval harness work belongs in `tests/eval/`, but current repo files use `src/smriti/memory/eval.py` and `tests/test_memory_eval.py` (`AGENTS.md:310-317`, `src/smriti/memory/eval.py:1-9`, `tests/test_memory_eval.py:1-26`).

### F. Baseline Recording

- VERIFIED - The schema has tables capable of recording eval scenarios, runs, and run result metrics (`src/smriti/db/migrations/001_init.sql:104-132`).
- VERIFIED - The current Python eval helper returns results and summary in memory; it does not write eval run rows or files (`src/smriti/memory/eval.py:42-69`, `src/smriti/memory/eval.py:109-126`).
- VERIFIED - The current eval test asserts `message_retrievals` stays empty after eval, so the eval helper does not record used-memory provenance (`tests/test_memory_eval.py:158-173`).
- VERIFIED - The retrieval service path used by the eval helper updates `access_count` and `last_accessed_at` for retrieved episodes (`src/smriti/memory/service.py:776-783`).
- VERIFIED - The eval test confirms that access metadata changes during eval: in-scope retrieved episodes get incremented access counts and `last_accessed_at`, while the wrong-scope episode remains unchanged (`tests/test_memory_eval.py:172-181`).
- VERIFIED - No project script or Makefile target currently records a re-runnable retrieval baseline (`pyproject.toml:30-31`, `Makefile:1-19`).
- UNKNOWN - No committed eval result file pattern was identified during read-only search. Absence has no file path/line range; the closest committed eval structures are the database tables and in-memory result dataclasses cited above.
- ASSUMED - Based on existing repo conventions only, a re-runnable baseline has no established storage location today. The repo has stage docs under `docs/stages/`, eval DB tables in migrations, and no committed eval-results directory or file pattern in inspected files (`docs/stages/stage-05-retrieval-and-eval.md:1-13`, `src/smriti/db/migrations/001_init.sql:104-132`, `README.md:347-351`).

## 3. Conflicts / Scope Issues

- VERIFIED - AGENTS.md says, "Eval harness lives in `tests/eval/` and runs on demand, not by default in CI," but the current helper implementation lives in `src/smriti/memory/eval.py` and its test lives at `tests/test_memory_eval.py` (`AGENTS.md:310-317`, `src/smriti/memory/eval.py:1-9`, `tests/test_memory_eval.py:1-26`).
- VERIFIED - The current harness uses the real retrieval service path, and that path mutates episode access metadata for returned rows. If a later baseline pass requires no database mutation at all, the current path has a scope issue to resolve before design (`src/smriti/memory/service.py:776-783`, `tests/test_memory_eval.py:172-181`).
- VERIFIED - Existing scoring weights are Python constants, but the SQL candidate pool is similarity-first and bounded before Python reranking. Tuning only the existing weights can stay outside SQL structure; changing candidate selection would touch a documented Stage 5 heuristic and Stage 11 non-goal area (`src/smriti/memory/service.py:712-715`, `src/smriti/memory/service.py:724-775`, `docs/stages/stage-11-summary-episode-memory.md:384-408`).
- UNKNOWN - No hand-labeled corpus or committed baseline result pattern was identified. This is a findings limitation, not a design proposal.
