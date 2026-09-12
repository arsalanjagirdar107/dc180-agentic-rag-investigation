# Technical Report — Solve the Case: Agentic RAG Investigation

## Objective

This project is an interactive, evidence-first RAG application for investigating a bounded set of Enron emails. The objective is not to decide ground truth automatically. Instead, it helps a user retrieve source records, form a cautious theory, challenge that theory, inspect relationships, and record an independent verdict with traceable document IDs.

## System architecture and data

The pipeline begins with `preprocessing/enron_loader.py`, which reads an extracted Enron `maildir` in deterministic order and writes canonical JSONL. Each record includes a stable document ID, source path, selected headers, cleaned text/plain body, searchable text, and an `unverified` status. The repository commits a four-email schema-compatible sample corpus for reproducible local and deployed demonstrations; the preprocessing and integration tests also support public Enron-formatted records.

The FastAPI backend builds an in-memory lexical retriever, one shared semantic retriever, a hybrid retriever, and a NetworkX evidence graph from the JSONL corpus. Streamlit is a separate HTTP client; it does not duplicate retrieval or agent behavior.

## Retrieval pipeline

Lexical retrieval uses an in-memory TF-IDF cosine scorer. It is reliable for direct term overlap, such as “contract approval.” Semantic retrieval uses FastEmbed's ONNX version of `sentence-transformers/all-MiniLM-L6-v2`; vectors are L2-normalized and ranked by dot product, which equals cosine similarity after normalization. The model is lazy-loaded under a lock and shared between direct semantic and hybrid use, so `/health` avoids loading it.

Hybrid retrieval runs both retrievers, min-max normalizes their per-query scores, and combines them as:

`hybrid = lexical_weight × normalized_lexical + (1 - lexical_weight) × normalized_semantic`

The default weight is 0.5. These are ranking scores, not probabilities or factual confidence.

## Investigator and Fact-Checker

The Investigator is a bounded loop: retrieve evidence, ask a reasoning policy for a theory and sufficiency assessment, validate citations, and retry with a reformulated query when evidence is insufficient. Pydantic models record each attempt, query, evidence set, theory, and assessment. Python enforces the retry limit and rejects theory IDs absent from retrieved evidence. It also retains a claim only when the claim is a verbatim excerpt from cited evidence.

The deployed default is an offline rule policy, allowing a no-key demo. It assesses top lexical/semantic scores and adds targeted query hints. An optional OpenAI policy uses structured output, but Python keeps retrieval and grounding control.

The Fact-Checker is separate from the Investigator. It creates fresh contradiction, timeline, and alternative-explanation queries from the investigation report, retrieves new hybrid evidence, and validates every returned citation against that fresh evidence. This separation prevents the checker from merely repeating the Investigator's evidence.

## Grounding, graph, and user interaction

Grounding is central to the design. The system exposes source document IDs, validates Investigator citations against retrieved evidence, requires verbatim claim support, and carries `unverified` or `misleading` document status into grounded claims. It therefore does not silently upgrade email text into verified fact.

The NetworkX `MultiDiGraph` is an inspection layer. It contains documents, email-address people, domains/organizations, subject-derived events, and conservatively recognized objects. Relationships such as sender-to-document, recipient-to-document, document-to-event, and document-to-transaction retain `source_document_ids` for traceability.

The Streamlit UI supports candidate interrogation, evidence search, investigation, independent fact-checking, graph inspection, and a final verdict. A user must submit a prediction before the consolidated Investigator and Fact-Checker reports are revealed; the API returns it as `not_evaluated` rather than claiming correctness.

## Deployment, testing, and tradeoffs

Render deploys FastAPI and Streamlit as two web services. `API_BASE_URL` connects the UI to the backend, and the committed sample corpus avoids dependence on ephemeral service storage. FastEmbed/ONNX replaced the prior PyTorch runtime to make semantic retrieval viable on Render's free memory tier.

The complete test suite covers API flow, ingestion, retrieval, investigation, fact-checking, graph traceability, citation-status propagation, lazy shared semantic loading, UI HTTP calls, and a public Enron integration path when the external dataset is reachable. The live deployed UI and all major endpoints were also QA-tested.

Key tradeoffs are intentional: the deployed corpus is small, the offline policies are deterministic rather than deeply semantic, graph extraction is conservative, and neither retrieval scores nor email records are treated as proof. Future work could add a larger curated corpus, persistence, better offline reasoning, and evaluation data without weakening the current evidence constraints.

## Bonus features and AI tools

Implemented quality features include fresh adversarial retrieval, Python citation validation, unverified/misleading evidence propagation, a traceable evidence graph, bounded retries, lazy shared ONNX embeddings, and separate user-verdict recording. OpenAI Codex was used as an AI coding assistant for iterative development, debugging, tests, and documentation; the final implementation and QA were checked against the repository behavior.
