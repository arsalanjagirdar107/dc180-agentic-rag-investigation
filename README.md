# Solve the Case — An Interactive Agentic RAG Investigation

An evidence-first investigation application built for the 180DC ML recruitment task. It lets a user search a bounded Enron email corpus, run a traceable investigation, challenge the result with a separate Fact-Checker, inspect evidence relationships, and record an independent verdict.

**Live demo:** [Streamlit UI](https://dc180-agentic-rag-investigation-ui.onrender.com) · [FastAPI health endpoint](https://dc180-agentic-rag-investigation.onrender.com/health)

## Problem

Investigative questions are easy to answer confidently and hard to answer responsibly. This project treats email records as evidence rather than truth: every displayed claim is tied to retrieved document IDs, and corpus records remain explicitly unverified unless an external process verifies them.

## What the system does

- Loads a reproducible subset of Enron-style RFC 822 emails into JSONL.
- Retrieves evidence with lexical TF-IDF, semantic embeddings, or transparent hybrid retrieval.
- Runs a bounded Investigator loop that retrieves, forms a theory, assesses evidence, reformulates a query when needed, and validates citations in Python.
- Runs a separate Fact-Checker that performs new adversarial searches for contradiction, timeline changes, and alternative explanations.
- Builds an inspectable NetworkX evidence graph with source-document IDs on every evidence-derived edge.
- Exposes the workflow through FastAPI and a Streamlit UI.

## Architecture

```text
Enron maildir / sample JSONL
        │
        ▼
Preprocessor ──► canonical JSONL documents ──► lexical TF-IDF retriever
                                             └► FastEmbed semantic retriever
                                                        │
                                                        ▼
                                      weighted hybrid evidence retriever
                                             │                    │
                                             ▼                    ▼
                                      Investigator          Fact-Checker
                                      (bounded loop)      (fresh adversarial search)
                                             │                    │
                                             └──── FastAPI ───────┘
                                                       │
                                                  Streamlit UI

JSONL documents ──► NetworkX evidence graph ──► graph inspection endpoint
```

The graph is an inspection/evidence board, not a retrieval index. Retrieval, retry enforcement, score calculation, graph construction, and citation validation remain deterministic Python responsibilities; language reasoning is optional and policy-driven.

## Corpus and preprocessing

The project targets the public CMU/CALO Enron Email Dataset format: an extracted `maildir` where each file is an RFC 822 email. `preprocessing/enron_loader.py` walks files in deterministic path order, optionally filters by path, extracts `text/plain` content, normalizes whitespace, preserves selected headers, and writes one JSON object per line.

Each processed document contains a stable `document_id`, source path, sender/recipient/date metadata, cleaned body, searchable text, and an `unverified` verification status. For raw maildir records, the ID is a SHA-256-derived value of the relative source path. A small committed, schema-compatible corpus at `examples/phase5_sample.jsonl` makes the deployed demo self-contained. The tests also exercise the preprocessing/retrieval path against a public Enron dataset endpoint when network access is available.

```powershell
python -m preprocessing.enron_loader `
  --input-dir C:\data\maildir `
  --include-path allen-p `
  --output-file data\processed\enron_emails.jsonl `
  --max-documents 200
```

## Retrieval and RAG

### Lexical retrieval

`retrieval/lexical.py` implements dependency-free TF-IDF cosine similarity. It is useful when the user's wording overlaps with the source evidence; term frequency is log-scaled so repeated words do not dominate a result.

### Semantic retrieval

`retrieval/semantic.py` uses FastEmbed's ONNX implementation of `sentence-transformers/all-MiniLM-L6-v2`. Document and query vectors are explicitly L2-normalized, so their dot product is cosine similarity. The model is imported and loaded only on a semantic or hybrid request, behind a lock, and the API shares one `SemanticRetriever` instance with hybrid retrieval.

### Hybrid retrieval

`retrieval/hybrid.py` retrieves all corpus documents from both methods, min-max normalizes each score set per query, then combines them:

```text
hybrid = lexical_weight × normalized_lexical
       + (1 - lexical_weight) × normalized_semantic
```

The default lexical weight is `0.5`. Retrieval scores are ranking signals, not probabilities or factual confidence.

## Investigation and grounding

The Investigator receives a question and a hybrid retriever. For each attempt it retrieves evidence, asks a reasoning policy for a theory/assessment/next query, validates cited IDs against the retrieved evidence, and only retains grounded claims that are verbatim excerpts of cited evidence. It stops when evidence is sufficient or the configured retry limit is reached. Every attempt is retained in a Pydantic `InvestigationState` for inspection.

Two policies are available:

- **Offline demo fallback** — the deployed UI default. A deterministic rule policy uses retrieval scores and targeted query hints, so the demo works without an API key.
- **OpenAI policy** — optional. It uses structured Responses API output, while Python still controls retrieval, retry limits, and grounding validation.

The Fact-Checker is intentionally separate. It turns an Investigator report into three new hybrid searches—contradiction, timeline, and alternative explanation—then validates its supporting, contradicting, and alternative IDs against only that fresh evidence. This separation prevents the checker from merely repeating the Investigator's evidence.

## Evidence graph

`evidence_graph/builder.py` constructs a deterministic NetworkX `MultiDiGraph`. Nodes represent documents, email-address people, email domains/organizations, subject-derived events, and conservatively recognized corpus objects such as contracts and energy trades. Edges represent actual sender/recipient/CC, document-event, and document-entity relationships. Each edge stores `source_document_ids`, enabling a user to trace the relationship back to source evidence.

## API and UI

FastAPI is a thin adapter over the existing modules and uses Pydantic schemas for requests and important structured responses.

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Service and corpus status |
| `POST /ingest` | Local maildir-to-JSONL ingestion and index reload |
| `POST /search_evidence` | Lexical, semantic, or hybrid evidence search |
| `POST /interrogate` | Candidate interrogation via evidence search |
| `POST /investigate` | Bounded Investigator workflow |
| `POST /fact_check` | Fresh adversarial Fact-Checker workflow |
| `POST /submit_verdict` | Record a user prediction as `not_evaluated` |
| `GET /graph/{node_id}` | Traceable graph relationships |

The Streamlit UI is frontend-only and communicates with FastAPI over HTTP. It contains Candidate Interrogation, Evidence Search, Investigation, Fact-Checker, Investigation Graph, and Final Verdict sections. It never reads or displays `OPENAI_API_KEY`.

## Local setup

Requires Python `3.12.12` (pinned in `.python-version`).

```powershell
python -m pip install -r requirements.txt

# Terminal 1
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

# Terminal 2
python -m streamlit run ui/app.py
```

Open Streamlit's local URL and use the sidebar connection check. The UI defaults to **Offline demo fallback**. To use the optional LLM policy, configure the key only in the backend environment:

```powershell
$env:OPENAI_API_KEY = "..."
$env:OPENAI_MODEL = "gpt-5-mini"  # optional
```

| Variable | Used by | Default | Purpose |
| --- | --- | --- | --- |
| `EVIDENCE_CORPUS` | API | `examples/phase5_sample.jsonl` | JSONL corpus path |
| `API_BASE_URL` | UI | `http://127.0.0.1:8000` | FastAPI base URL |
| `OPENAI_API_KEY` | API only | unset | Optional OpenAI reasoning/fact-check policy |
| `OPENAI_MODEL` | API only | `gpt-5-mini` | Optional OpenAI model selection |

## Deployment

`render.yaml` defines two Render web services:

- **API:** `python -m uvicorn api.main:app --host 0.0.0.0 --port $PORT`
- **UI:** `python -m streamlit run ui/app.py --server.address 0.0.0.0 --server.port $PORT`

Set the UI service's `API_BASE_URL` to the public API URL. The committed sample corpus is used for the deployed demo because free-service disk is ephemeral; `/ingest` remains primarily a local-development operation. FastEmbed avoids the previous PyTorch/SentenceTransformers runtime cost, but a first semantic request may still load/download the ONNX model on a fresh instance.

## Testing

```powershell
python -m unittest discover -s tests -v
```

The suite covers API health/search/ingestion/verdict flow, Investigator and Fact-Checker flow, graph traceability, citation status propagation, public-Enron pipeline integration (skipped only when the external dataset is unreachable), lazy shared semantic-model lifecycle, and the Streamlit HTTP client.

## Implemented bonus-quality features

- Fresh adversarial Fact-Checker retrieval rather than self-critique only.
- Python-enforced citation/grounding validation and verbatim-claim checks.
- Explicit `unverified` and `misleading` evidence status propagation.
- Traceable NetworkX graph edges retaining source document IDs.
- Bounded retry loop with inspectable investigation state.
- Lazy, shared FastEmbed/ONNX embedding runtime suitable for the deployed demo.
- User verdict recording that is kept separate from system evaluation.

## Limitations and tradeoffs

- The deployed corpus is deliberately tiny and illustrative; it does not establish real-world ground truth.
- Email evidence is marked unverified by default and can be misleading or incomplete.
- The offline policies are deterministic fallbacks, not substitutes for nuanced LLM reasoning.
- Min-max-normalized hybrid scores are query-relative ranking values, not calibrated confidence.
- The graph uses conservative pattern/metadata extraction rather than unrestricted entity extraction.
- There is no persistent database or long-term verdict storage; Render disk is ephemeral.
- The optional OpenAI policy depends on a valid backend-only API key and available account access.

## AI tools used

OpenAI Codex was used as an AI coding assistant during iterative implementation, debugging, test support, and documentation preparation. Architecture decisions, evidence constraints, deployment checks, and final QA were reviewed against the working code and test results.

## Submission companion documents

- [Technical report](docs/TECHNICAL_REPORT.md)
