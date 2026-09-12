# 8–9 Minute Demo Script

Use the deployed Streamlit UI and keep **Reasoning policy = Offline demo fallback**. The sample corpus is intentionally small; present it as a traceable demonstration, not as proof of a real Enron allegation.

## 0:00–0:45 — Opening

**Show:** The landing page and its six tabs.

**Say:** “This is Solve the Case, an evidence-first Agentic RAG investigation system. It searches a small Enron email corpus, creates a grounded investigation report, challenges that report with separate retrieval, and lets the user inspect the source relationships.”

**Explain:** RAG means the answer process retrieves documents before forming a response. Here, document IDs are shown throughout.

**Do not say:** “The system proves what happened,” or “the emails are verified facts.”

## 0:45–1:35 — Architecture

**Show:** The architecture diagram in the README or briefly point to the repository structure.

**Say:** “The pipeline is email preprocessing, lexical and semantic retrieval, hybrid ranking, Investigator, independent Fact-Checker, graph inspection, FastAPI, and Streamlit.”

**Explain:** Python owns deterministic work—retrieval, score calculation, retry limits, and citation checks. The optional LLM policy only supplies structured reasoning; today’s deployed demo uses the offline policy.

**Do not say:** “The graph retrieves the answer.” It is an evidence-inspection layer.

## 1:35–2:25 — Candidate interrogation

**Click/show:** **Case / Candidate Interrogation**. Enter: `What evidence exists about the energy-trade approval?` Click **Interrogate evidence**.

**Say:** “This is the quick evidence-first entry point. It returns ranked emails with stable IDs, excerpts, metadata, and the retrieval methods that contributed.”

**Explain:** `sample-001` discusses sending a contract to legal; `sample-004` later says approval was withdrawn and not to execute the trade.

**Do not say:** “The top score is a probability of truth.” It is only a relevance score.

## 2:25–3:15 — Evidence search and hybrid RAG

**Click/show:** **Evidence Search**. Enter `contract approval`, retain **hybrid**, then click **Search evidence**.

**Say:** “Hybrid retrieval combines TF-IDF for exact wording with semantic embeddings for related wording. Both score types are normalized per query and combined with a transparent weighted formula.”

**Explain:** Show a result that lists both `lexical_tfidf` and `semantic_embedding`. Mention that FastEmbed uses an ONNX embedding model and is loaded only when semantic retrieval is needed.

**Do not say:** “Semantic search understands truth.” It finds related text, not verified conclusions.

## 3:15–4:30 — Investigation

**Click/show:** **Investigation**. Confirm **Offline demo fallback** is selected. Enter: `What happened with the energy-trade approval, and was the trade ultimately authorized?` Click **Run investigation**.

**Say:** “The Investigator retrieves evidence, forms a cautious theory, assesses whether it is sufficient, and returns grounded claims only when Python can validate their citations and verbatim source support.”

**Explain:** Open Attempt 1. Point to the evidence IDs, theory citation, score fields, and grounded claim. Explain that the result says the available corpus contains a withdrawal/pending-review message; it does not justify claiming final authorization.

**Do not say:** “Sufficient means objectively proven.” It means the configured policy found sufficient grounded evidence for the bounded report.

## 4:30–5:20 — Self-correction and retry

**Click/show:** Run a second investigation with `attorney clearance before transaction`. Expand the attempts.

**Say:** “This question triggers the retry behavior. The first evidence set is not enough under the offline score thresholds, so the policy reformulates the query with terms such as legal, approval, and contract. Python enforces the retry bound.”

**Explain:** The visible attempt history makes the agentic loop inspectable rather than hidden.

**Do not say:** “It keeps searching forever,” or “the model autonomously changes the database.”

## 5:20–6:20 — Independent Fact-Checker

**Click/show:** **Fact-Checker**, then **Run independent Fact-Checker**.

**Say:** “The Fact-Checker is separate from the Investigator. It does not merely critique the final sentence. It creates fresh searches for contradiction, timeline changes, and alternative explanations.”

**Explain:** Highlight the contradiction result and cited IDs such as `sample-004` and `sample-003`. These IDs must be present in the Fact-Checker's fresh evidence.

**Do not say:** “Contradicted means the entire dataset is false.” It means fresh evidence challenges the current hypothesis.

## 6:20–7:10 — Evidence graph

**Click/show:** **Investigation Graph**. Keep `document:sample-001`, then click **Inspect graph relationship**.

**Say:** “The graph turns supported email metadata and conservative text patterns into inspectable relationships. Every displayed edge includes the source document ID that supports it.”

**Explain:** Point out sender/recipient relationships and document links to the energy-trade and contract entities.

**Do not say:** “The graph inferred a hidden relationship.” Only evidence-supported edges are created.

## 7:10–8:00 — User verdict

**Click/show:** **Final Verdict**. Enter your own prediction, for example: `The available corpus does not establish final authorization.` Click **Submit my verdict**.

**Say:** “The system records the user's prediction separately before revealing the consolidated outputs. It returns `not_evaluated`; it never declares the user's answer correct automatically.”

**Explain:** This separates user judgment from the system's evidence summary.

**Do not say:** “The system grades the case.” There is no hidden ground-truth evaluator.

## 8:00–8:45 — Technical close

**Show:** README testing/deployment section or the live deployment URLs.

**Say:** “The backend is FastAPI with Pydantic schemas; Streamlit is a pure HTTP frontend. The demo is deployed as two Render services. Tests cover API flow, evidence grounding, graph traceability, semantic-model lifecycle, UI client behavior, and public-Enron pipeline support.”

**Explain:** Mention FastEmbed/ONNX as a deployment-oriented choice and the offline fallback as a reliable no-key demo mode.

**Do not say:** “This is production-grade fact verification.” The project is a careful, explainable investigation prototype with explicit limitations.
