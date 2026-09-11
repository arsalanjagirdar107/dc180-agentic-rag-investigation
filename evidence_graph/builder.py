"""Build a deterministic, source-traceable NetworkX graph from Phase 4 JSONL."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from email.utils import getaddresses
from pathlib import Path
from urllib.parse import quote

import networkx as nx

from retrieval.lexical import load_documents


OBJECT_PATTERNS = {
    "transaction:energy_trade": (r"\benergy trade\b", "transaction", "energy trade"),
    "document_type:contract": (r"\bcontract\b", "document_type", "contract"),
    "document_type:agreement": (r"\bagreement\b", "document_type", "agreement"),
    "object:natural_gas": (r"\bnatural gas\b", "object", "natural gas"),
    "object:risk_limit": (r"\brisk limits?\b", "object", "risk limit"),
}
PLACE_PATTERN = re.compile(r"\b(?:location|located in|meeting in)\s*[: ]\s*([A-Z][A-Za-z -]{1,50})(?=[.,;\n]|$)")
ORG_PATTERN = re.compile(r"\b(?:company|organization|organisation)\s*:\s*([A-Za-z][A-Za-z &.-]{1,60})(?=[.,;\n]|$)", re.I)


def stable_id(kind: str, value: str) -> str:
    """Stable readable graph ID for a normalized corpus value."""
    return f"{kind}:{quote(value.strip().lower(), safe='@._-')}"


def event_id(subject: str) -> str:
    digest = hashlib.sha256(" ".join(subject.lower().split()).encode("utf-8")).hexdigest()[:16]
    return f"event:{digest}"


class EvidenceGraph:
    """NetworkX MultiDiGraph whose relationships retain source document IDs."""

    def __init__(self) -> None:
        self.graph = nx.MultiDiGraph()

    def add_node(self, node_id: str, node_type: str, label: str, **metadata: object) -> None:
        if node_id not in self.graph:
            self.graph.add_node(node_id, node_type=node_type, label=label, **metadata)

    def add_evidence_edge(self, source: str, target: str, relationship: str, document_id: str) -> None:
        if self.graph.has_edge(source, target, key=relationship):
            sources = self.graph[source][target][relationship]["source_document_ids"]
            if document_id not in sources:
                sources.append(document_id)
            return
        self.graph.add_edge(
            source, target, key=relationship, relationship=relationship, source_document_ids=[document_id]
        )

    def _add_person(self, address: str, document_id: str, document_node: str, relationship: str) -> None:
        normalized = address.lower().strip()
        if not normalized:
            return
        person = stable_id("person", normalized)
        self.add_node(person, "person", normalized, email=normalized)
        self.add_evidence_edge(person, document_node, relationship, document_id)
        if "@" in normalized:
            domain = normalized.rsplit("@", 1)[1]
            organization = stable_id("organization", domain)
            self.add_node(organization, "organization", domain, domain=domain)
            self.add_evidence_edge(person, organization, "uses_email_domain", document_id)

    def add_document(self, document: dict[str, object]) -> None:
        document_id = str(document["document_id"])
        document_node = stable_id("document", document_id)
        metadata = document.get("metadata", {})
        metadata = metadata if isinstance(metadata, dict) else {}
        subject = str(metadata.get("subject") or "")
        body = str(document.get("body") or "")
        self.add_node(
            document_node, "document", subject or document_id, document_id=document_id,
            source_path=str(document.get("source_path") or ""), date=metadata.get("date"),
        )
        for header, relationship in (("from", "sent"), ("to", "received"), ("cc", "copied")):
            for _, address in getaddresses([str(metadata.get(header) or "")]):
                self._add_person(address, document_id, document_node, relationship)

        if subject:
            event = event_id(subject)
            self.add_node(event, "event", subject, normalized_subject=" ".join(subject.lower().split()))
            self.add_evidence_edge(document_node, event, "describes", document_id)
            for person, _, edge in self.graph.in_edges(document_node, keys=True):
                if edge in {"sent", "received", "copied"}:
                    self.add_evidence_edge(person, event, f"{edge}_message_about", document_id)

        searchable_text = f"{subject}\n{body}"
        for node_id, (pattern, node_type, label) in OBJECT_PATTERNS.items():
            if re.search(pattern, searchable_text, re.I):
                self.add_node(node_id, node_type, label)
                self.add_evidence_edge(document_node, node_id, "mentions", document_id)
        for match in PLACE_PATTERN.finditer(searchable_text):
            place = match.group(1).strip()
            place_node = stable_id("place", place)
            self.add_node(place_node, "place", place)
            self.add_evidence_edge(document_node, place_node, "mentions", document_id)
        for match in ORG_PATTERN.finditer(searchable_text):
            organization = match.group(1).strip()
            org_node = stable_id("organization", organization)
            self.add_node(org_node, "organization", organization)
            self.add_evidence_edge(document_node, org_node, "mentions", document_id)

    @classmethod
    def from_documents(cls, documents: list[dict[str, object]]) -> "EvidenceGraph":
        board = cls()
        for document in sorted(documents, key=lambda item: str(item["document_id"])):
            board.add_document(document)
        return board

    def neighbors(self, node_id: str) -> list[dict[str, object]]:
        """Inspect directly connected relationships, including their source documents."""
        if node_id not in self.graph:
            raise KeyError(f"Unknown node: {node_id}")
        records: list[dict[str, object]] = []
        for source, target, relationship, data in self.graph.in_edges(node_id, keys=True, data=True):
            records.append({"direction": "in", "node_id": source, "relationship": relationship,
                            "source_document_ids": data["source_document_ids"]})
        for source, target, relationship, data in self.graph.out_edges(node_id, keys=True, data=True):
            records.append({"direction": "out", "node_id": target, "relationship": relationship,
                            "source_document_ids": data["source_document_ids"]})
        return records

    def subgraph_for(self, node_id: str, radius: int = 1) -> nx.MultiDiGraph:
        """Return a traceable undirected ego-subgraph for a document or entity node."""
        if node_id not in self.graph:
            raise KeyError(f"Unknown node: {node_id}")
        return nx.ego_graph(self.graph, node_id, radius=radius, undirected=True).copy()

    def document_subgraph(self, document_id: str, radius: int = 1) -> nx.MultiDiGraph:
        return self.subgraph_for(stable_id("document", document_id), radius)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and inspect a traceable evidence graph.")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--document-id", help="Print direct graph relationships for one document.")
    args = parser.parse_args()
    board = EvidenceGraph.from_documents(load_documents(args.corpus))
    if args.document_id:
        node_id = stable_id("document", args.document_id)
        print(json.dumps(board.neighbors(node_id), indent=2))
    else:
        print(json.dumps({"nodes": board.graph.number_of_nodes(), "edges": board.graph.number_of_edges()}, indent=2))


if __name__ == "__main__":
    main()
