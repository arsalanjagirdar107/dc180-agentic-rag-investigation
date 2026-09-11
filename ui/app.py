"""Streamlit frontend for the existing FastAPI backend."""
from __future__ import annotations
import os
from typing import Any
import streamlit as st
from ui.client import BackendError, request_api

st.set_page_config(page_title="Evidence Investigator", page_icon="🔎", layout="wide")

def call_api(base_url: str, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any] | None:
    try:
        with st.spinner("Contacting the investigation backend…"):
            return request_api(base_url, method, path, payload)
    except BackendError as error:
        st.error(str(error))
        return None

def show_evidence(results: list[dict[str, Any]]) -> None:
    if not results:
        st.info("No evidence was returned.")
        return
    for item in results:
        with st.container(border=True):
            st.markdown(f"**{item['document_id']}** · score {item.get('score', 0):.3f}")
            st.caption(" · ".join(item.get("retrieval_methods", [])) or "retrieval method unavailable")
            if item.get("subject"):
                st.write(item["subject"])
            st.write(item.get("excerpt", ""))
            st.caption(f"Source: {item.get('source_path', 'unknown')} · Metadata: {item.get('metadata', {})}")

def show_attempts(report: dict[str, Any]) -> None:
    state = report["state"]
    (st.success if state["status"] == "sufficient" else st.warning)(
        "Evidence was assessed as sufficient." if state["status"] == "sufficient" else "Evidence was insufficient after retries."
    )
    for attempt in state["attempts"]:
        with st.expander(f"Attempt {attempt['attempt_number']} · {attempt['query']}"):
            st.write(attempt["theory"]["statement"])
            st.caption(f"Citations: {', '.join(attempt['theory']['supporting_document_ids']) or 'none'}")
            st.write(attempt["assessment"]["reason"])
            for evidence in attempt["evidence"]:
                st.markdown(f"**{evidence['document_id']}** · hybrid {evidence['hybrid_score']:.3f}")
                st.write(evidence["excerpt"])
    st.subheader("Grounded claims")
    for claim in report.get("grounded_claims", []):
        st.write(claim["claim"])
        st.caption(f"Citations: {', '.join(claim['supporting_document_ids'])}")

st.title("🔎 Evidence Investigator")
st.caption("Traceable email-corpus investigation. This browser calls FastAPI only and never loads API keys.")
with st.sidebar:
    st.header("Connection")
    base_url = st.text_input("FastAPI base URL", value=os.environ.get("API_BASE_URL", "http://127.0.0.1:8000"))
    if st.button("Check backend"):
        health = call_api(base_url, "GET", "/health")
        if health:
            st.success(f"Connected · {health['documents']} documents")

case_tab, search_tab, investigation_tab, fact_tab, graph_tab, verdict_tab = st.tabs(
    ["Case / Candidate Interrogation", "Evidence Search", "Investigation", "Fact-Checker", "Investigation Graph", "Final Verdict"]
)
with case_tab:
    question = st.text_area("Candidate question", placeholder="What evidence exists about the energy-trade approval?")
    if st.button("Interrogate evidence", type="primary", disabled=not question):
        response = call_api(base_url, "POST", "/interrogate", {"query": question, "method": "hybrid", "top_k": 5})
        if response: st.session_state["interrogation"] = response
    if "interrogation" in st.session_state: show_evidence(st.session_state["interrogation"]["results"])

with search_tab:
    col1, col2 = st.columns([3, 1])
    search_query, method = col1.text_input("Evidence search query"), col2.selectbox("Method", ["hybrid", "lexical", "semantic"])
    if st.button("Search evidence", disabled=not search_query):
        response = call_api(base_url, "POST", "/search_evidence", {"query": search_query, "method": method, "top_k": 5})
        if response: st.session_state["search"] = response
    if "search" in st.session_state: show_evidence(st.session_state["search"]["results"])

with investigation_tab:
    investigation_question = st.text_area("Investigation question", key="investigation_question")
    policy_label = st.selectbox("Reasoning policy", ["OpenAI via backend environment", "Offline demo fallback"])
    if st.button("Run investigation", disabled=not investigation_question):
        policy = "openai" if policy_label.startswith("OpenAI") else "rule"
        response = call_api(base_url, "POST", "/investigate", {"question": investigation_question, "policy": policy, "max_retries": 2, "evidence_limit": 3})
        if response:
            st.session_state["investigation_report"] = response
            st.session_state["fact_check_policy"] = policy
            st.session_state.pop("fact_check_report", None); st.session_state.pop("verdict_submitted", None)
    if "investigation_report" in st.session_state: show_attempts(st.session_state["investigation_report"])

with fact_tab:
    report = st.session_state.get("investigation_report")
    if not report: st.info("Run an investigation first.")
    elif st.button("Run independent Fact-Checker"):
        response = call_api(base_url, "POST", "/fact_check", {"investigation": report, "evidence_limit": 3,
            "policy": st.session_state.get("fact_check_policy", "openai")})
        if response: st.session_state["fact_check_report"] = response; st.session_state.pop("verdict_submitted", None)
    finding = st.session_state.get("fact_check_report", {}).get("finding")
    if finding:
        {"supported": st.success, "contradicted": st.error, "inconclusive": st.warning}[finding["verdict"]](
            f"Fact-Checker: {finding['verdict'].upper()} — {finding['explanation']}"
        )
        for label, key in [("Supporting", "supporting_document_ids"), ("Contradicting", "contradicting_document_ids"), ("Alternative", "alternative_document_ids")]:
            st.write(f"**{label} evidence:** {', '.join(finding[key]) or 'none'}")

with graph_tab:
    node_id = st.text_input("Document or entity node ID", value="document:sample-001")
    if st.button("Inspect graph relationship", disabled=not node_id):
        response = call_api(base_url, "GET", f"/graph/{node_id}")
        if response: st.session_state["graph_result"] = response
    if "graph_result" in st.session_state:
        st.dataframe(st.session_state["graph_result"]["relationships"], use_container_width=True)
        st.caption("Every relationship lists the source document IDs that support it.")

with verdict_tab:
    st.info("Record your own prediction before the consolidated result is revealed.")
    verdict, rationale = st.text_area("Your verdict / prediction"), st.text_area("Optional rationale")
    if st.button("Submit my verdict", type="primary", disabled=not verdict):
        response = call_api(base_url, "POST", "/submit_verdict", {"prediction": verdict, "rationale": rationale})
        if response: st.session_state["verdict_submitted"] = response
    if "verdict_submitted" in st.session_state:
        st.success("Your prediction was recorded separately and has not been automatically judged correct.")
        if "investigation_report" in st.session_state:
            st.subheader("Investigator result"); st.json(st.session_state["investigation_report"])
        if "fact_check_report" in st.session_state:
            st.subheader("Fact-Checker result"); st.json(st.session_state["fact_check_report"])
