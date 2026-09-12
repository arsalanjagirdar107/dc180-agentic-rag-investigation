# Interview Cheat Sheet

## One-minute explanation

“Solve the Case is an evidence-first Agentic RAG application over a small Enron email corpus. It preprocesses emails into traceable JSONL records, retrieves evidence with TF-IDF and FastEmbed semantic search, combines them transparently, and runs a bounded Investigator loop. A separate Fact-Checker performs new adversarial searches for contradictions and alternatives. Every displayed claim uses document IDs, and Python validates that citations come from retrieved evidence. The UI is Streamlit, the backend is FastAPI, and the live demo defaults to an offline policy so it works without an API key.”

## Core concepts

### What is RAG?

Retrieval-augmented generation is a pattern where a system fetches relevant documents before reasoning or answering. In this project, retrieval evidence is passed to the reasoning policy and shown to the user with stable IDs.

### Why hybrid retrieval?

TF-IDF is strong for exact terms such as “contract approval.” Embeddings help when wording differs, such as “lawyers must approve the deal.” Hybrid retrieval uses both signals instead of relying entirely on either one.

### TF-IDF versus semantic embeddings?

TF-IDF represents word overlap and rarity. Semantic embeddings represent text as dense vectors, so similar meanings can be close even when exact words differ. Neither is a truth score.

### Why use embeddings?

They improve recall for paraphrased or related questions. The project uses FastEmbed with the all-MiniLM-L6-v2 equivalent and cosine similarity over normalized vectors.

### Why not just ask an LLM?

An LLM can produce plausible but unsupported statements. This project retrieves evidence first, exposes IDs and excerpts, validates citations in Python, and keeps deterministic controls outside the LLM.

## Agent design

### How does the Investigator work?

For each attempt it retrieves hybrid evidence, asks a policy for a theory/sufficiency decision/next query, validates theory IDs and verbatim claims against the retrieved evidence, and either stops or retries. The attempt history is stored in Pydantic models.

### What makes it agentic?

It is a bounded loop rather than one search: retrieve, assess, reformulate, and retry. Its state and retry limit are explicit and inspectable.

### How does self-correction work?

If the policy marks evidence insufficient, it provides a more specific query. The offline policy adds targeted hints such as legal, approval, and contract. Python stops once evidence is sufficient, there is no useful next query, or the retry budget is exhausted.

### How does the Fact-Checker differ from the Investigator?

The Investigator tries to answer the question from retrieved evidence. The Fact-Checker takes the report and performs three **fresh** searches for contradictions, timeline changes, and alternatives. It validates its own citations against that new evidence set.

### What happens if the LLM/API fails?

The deployed UI defaults to the offline rule policy, which requires no key. The optional OpenAI policies require a backend-only `OPENAI_API_KEY`; they are not used by the no-key demo.

### Why use an offline fallback?

It makes the live demo reproducible without exposing credentials or depending on account credit. It is deliberately described as a deterministic fallback, not as equivalent to the optional LLM reasoning policy.

## Evidence safety

### How do you prevent hallucinated citations?

Python removes theory IDs that are not in the retrieved evidence. It only retains a grounded claim if the claim is a verbatim excerpt of at least one cited evidence excerpt. The Fact-Checker similarly filters every cited ID to its fresh evidence set.

### What is grounding?

Grounding means tying a statement to actual retrieved source material. Here it means visible document IDs plus validation that claims and citations are supported by the current evidence set.

### Why are some documents unverified?

An email is evidence of what was written, not automatically proof that the event happened. Preprocessed raw emails are marked `unverified`; the models preserve `misleading` status if a record is explicitly marked that way.

## Technology choices

### Why NetworkX?

It provides an in-memory `MultiDiGraph` that makes document/entity relationships easy to inspect. Every evidence-derived edge stores its source document IDs. It is not used as a decorative visualization or as the retrieval engine.

### Why FastAPI?

FastAPI gives typed HTTP endpoints and works naturally with Pydantic request/response models. It keeps the application logic in Python services that both tests and the UI can use.

### Why Streamlit?

It makes a clear demo interface quickly. The UI is intentionally frontend-only: it calls FastAPI using HTTP instead of reimplementing retrieval or agent logic.

### Why Pydantic?

Pydantic validates API inputs and important structured outputs such as investigation attempts, theories, claims, and Fact-Checker findings. It makes malformed policy output easier to reject.

### Why FastEmbed?

The earlier sentence-transformer/PyTorch runtime exceeded Render free-tier memory during semantic work. FastEmbed uses ONNX, supports the chosen MiniLM model, stays lazy-loaded, and avoids importing the model on `/health`.

### How does deployment work?

Render runs two services from `render.yaml`: FastAPI and Streamlit. `API_BASE_URL` tells the UI where the API lives. The committed sample corpus is used in deployment because free-service disk is ephemeral.

## Limitations and next steps

### What are the limitations?

The deployed corpus is only a small demo subset. Retrieval scores are not confidence values. Offline reasoning is rule-based. Graph extraction is conservative pattern/metadata extraction. There is no persistent database, no ground-truth verdict evaluator, and email content remains unverified by default.

### What would you improve next?

I would add a larger curated corpus with a documented evaluation set, persistence for ingestion/verdicts, retrieval-quality metrics, stronger offline reasoning, carefully evaluated reranking, and human-review workflows for verification status. I would keep citation validation and source traceability as non-negotiable constraints.

## AI tools used

OpenAI Codex was used as an AI coding assistant for implementation iterations, debugging, testing support, and documentation. The project's final claims should be defended from the code and QA results, not from the tool alone.
