# Agentic RAG Investigation

## Phase 4: document loading and preprocessing

This phase turns a small, local subset of the CMU/CALO Enron Email Dataset into
a clean JSONL file. No retrieval, embeddings, agents, graph, API, UI, or
deployment code is included yet.

### Run

Extract the dataset locally, choose a deliberately bounded directory (or use
`--include-path` to select one), then run:

```powershell
python -m preprocessing.enron_loader `
  --input-dir C:\data\enron_subset `
  --output-file data\processed\enron_emails.jsonl `
  --max-documents 200
```

To make a reproducible subset from a larger extracted corpus, constrain both
the path and the maximum number of documents:

```powershell
python -m preprocessing.enron_loader `
  --input-dir C:\data\maildir `
  --include-path allen-p `
  --max-documents 200
```

The input data is never modified. The output is one JSON object per email,
with a stable `document_id`, source path, selected headers, cleaned body, and a
single `text` field that later retrieval code can consume.

## Phase 5: lexical retrieval

The lexical retriever reads the Phase 4 JSONL output and ranks emails using
TF-IDF cosine similarity. It uses only exact word overlap; it does not create
embeddings or perform semantic or hybrid retrieval.

```powershell
python -m retrieval.lexical `
  --corpus data\processed\enron_emails.jsonl `
  --query "contract approval" `
  --top-k 5
```

Pass `--query` more than once to run several searches. A small schema-identical
example corpus is available at `examples/phase5_sample.jsonl` for a quick run.

## Phase 6: semantic retrieval

Semantic retrieval converts each email and query into a dense embedding using
the standard `all-MiniLM-L6-v2` sentence-transformer model. It ranks normalized
embeddings by cosine similarity, which can match related ideas even when they
do not use the same words. It is separate from, and does not change, the
Phase 5 lexical retriever.

Install the one dependency, then run:

```powershell
python -m pip install -r requirements.txt
python -m retrieval.semantic `
  --corpus data\processed\enron_emails.jsonl `
  --query "lawyers must approve the deal" `
  --top-k 5
```

The first run downloads the model into the local Hugging Face cache.

## Phase 7: hybrid retrieval

Hybrid retrieval runs the existing lexical TF-IDF and semantic retrievers over
the same Phase 4 corpus. Their per-query scores are min-max normalized to a
common 0–1 scale, then combined transparently:

`hybrid_score = lexical_weight × lexical_normalized + (1 - lexical_weight) × semantic_normalized`

The default `lexical_weight` is `0.5`. The result reports the document ID,
combined score, both raw scores, both normalized scores, and the contributing
retrieval methods.

```powershell
python -m retrieval.hybrid `
  --corpus data\processed\enron_emails.jsonl `
  --query "lawyers must approve the deal" `
  --top-k 5 `
  --lexical-weight 0.5
```

## Phase 8: Investigator Agent

The Investigator is a bounded investigation loop around the hybrid retriever:

1. Retrieve evidence for the current query.
2. Form a cautious theory and grounded claims from the retrieved text.
3. Assess whether the retrieved evidence is sufficient under a configurable score threshold.
4. If it does not, reformulate the query and retry until the retry limit.

It records every attempt in Pydantic state models and returns only claims that
name their supporting document IDs. The initial reasoning policy is deliberately
LLM-based and uses the OpenAI Responses API; Python retains retrieval, retry,
and grounding control. Set `OPENAI_API_KEY` in your environment (and optionally
`OPENAI_MODEL`) before running it. A `--policy rule` fallback exists only for
offline local demos and tests.

```powershell
python -m investigation.investigator `
  --corpus data\processed\enron_emails.jsonl `
  --question "attorney clearance before transaction" `
  --max-retries 2
```

## Phase 9: Fact-Checker

The separate Fact-Checker reads a saved Investigator JSON report and runs fresh
adversarial hybrid searches for reversals, timeline changes, and alternative
explanations. Its structured result cites only documents from those new search
results.

```powershell
python -m fact_checking.fact_checker `
  --corpus data\processed\enron_emails.jsonl `
  --investigation-file investigation.json
```

## Phase 10: Evidence Graph

The NetworkX evidence graph is an inspection board, not a retrieval component.
It adds deterministic nodes for documents, email-address people, email domains,
subject-backed events, and explicitly mentioned corpus entities. Every edge has
`source_document_ids` for traceability.

```powershell
python -m evidence_graph.builder `
  --corpus data\processed\enron_emails.jsonl `
  --document-id <document-id>
```

## Phase 11: FastAPI backend

The API is a thin adapter over the existing preprocessing, retrieval,
Investigator, Fact-Checker, and evidence-graph modules. Configure the corpus
with `EVIDENCE_CORPUS` (it defaults to the sample corpus), then run:

```powershell
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Available endpoints are `/health`, `/ingest`, `/search_evidence`,
`/interrogate`, `/investigate`, `/fact_check`, `/submit_verdict`, and
`/graph/{node_id}`. `/investigate` defaults to the environment-keyed OpenAI
policy; use `"policy": "rule"` only for local offline tests.

## Phase 12: Streamlit frontend

The frontend has no retrieval or agent logic. It calls the FastAPI endpoints
over HTTP and keeps the current investigation and Fact-Checker reports only in
the browser session.

Start the backend in one terminal:

```powershell
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Then start the UI in another terminal:

```powershell
python -m streamlit run ui/app.py
```

Open the URL Streamlit prints (normally `http://localhost:8501`). Use the
sidebar connection check first. For a no-key demo, choose **Offline demo
fallback** in the Investigation tab; production-style investigations use the
backend's `OPENAI_API_KEY` environment variable.

### Manual demo checklist

1. Start FastAPI, then Streamlit, and use **Check backend** in the sidebar.
2. Run an **Evidence Search** and confirm each result shows a document ID.
3. Run **Investigation** (choose the offline fallback for a no-key demo).
4. Run the independent **Fact-Checker** and inspect its cited evidence groups.
5. Enter `document:sample-001` in **Investigation Graph** and inspect source document IDs.
6. Enter and submit your own **Final Verdict**.
7. Confirm the recorded prediction is marked not automatically evaluated before reviewing the consolidated result.

## Render deployment

Deploy the API and UI as two separate Render **Web Services** from this same
repository. The included `render.yaml` defines both services, or create them
manually with the settings below. Render uses the pinned Python version in
`.python-version`.

### FastAPI service

- **Build command:** `python -m pip install -r requirements.txt`
- **Start command:** `python -m uvicorn api.main:app --host 0.0.0.0 --port $PORT`
- **Environment variables:** set `EVIDENCE_CORPUS=examples/phase5_sample.jsonl`.
  Add `OPENAI_API_KEY` as a Render secret only when using the OpenAI reasoning
  policies. `OPENAI_MODEL` is optional and defaults to `gpt-5-mini`.

The small processed sample corpus is committed under `examples/`, so the demo
starts with traceable corpus evidence and does not need to download or invent
data. Never commit an API key; it is read only from the API service environment.

### Streamlit service

- **Build command:** `python -m pip install -r requirements.txt`
- **Start command:** `python -m streamlit run ui/app.py --server.address 0.0.0.0 --server.port $PORT`
- **Environment variables:** set `API_BASE_URL` to the public HTTPS URL of the
  deployed FastAPI service, for example `https://dc180-api.onrender.com`.

`API_BASE_URL` is the only UI-to-API connection setting. Locally, it defaults
to `http://127.0.0.1:8000`; the Streamlit UI never reads or displays
`OPENAI_API_KEY`.

`/ingest` remains intended for local development because Render's service disk
is ephemeral. The deployed demo uses the included sample corpus instead.
