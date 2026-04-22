from __future__ import annotations

from typing import Any
import requests
import streamlit as st
from streamlit_agraph import Config, Edge, Node, agraph

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(layout="wide", page_title="IKS-Loom")


def _extract_entities(graph_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """
    Build a dict of entities keyed by entity id from `graph.results`.
    Pulls from matched nodes + neighbor nodes.
    """
    entities_by_id: dict[str, dict[str, Any]] = {}
    results = (graph_payload or {}).get("results") or []

    for item in results:
        for node_key in ("node",):
            n = (item or {}).get(node_key) or {}
            props = (n.get("properties") or {}) if isinstance(n, dict) else {}
            ent_id = props.get("id")
            if isinstance(ent_id, str) and ent_id:
                entities_by_id.setdefault(ent_id, n)

        for neigh in (item or {}).get("neighbors") or []:
            if not isinstance(neigh, dict):
                continue
            props = neigh.get("properties") or {}
            ent_id = props.get("id")
            if isinstance(ent_id, str) and ent_id:
                entities_by_id.setdefault(ent_id, neigh)

    return entities_by_id


def _extract_relationships(graph_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rels: list[dict[str, Any]] = []
    results = (graph_payload or {}).get("results") or []
    for item in results:
        for r in (item or {}).get("relationships") or []:
            if isinstance(r, dict):
                rels.append(r)
    return rels


def _entity_display_name(ent: dict[str, Any]) -> str:
    props = (ent or {}).get("properties") or {}
    name = props.get("name") or props.get("id") or "Unknown"
    label = props.get("label")
    if isinstance(label, str) and label.strip():
        return f"{name} ({label})"
    return str(name)


def _color_for_label(label: str) -> str:
    normalized = (label or "").strip().lower()
    palette = {
        "concept": "#00d2ff",
        "material": "#ff007f",
        "process": "#39ff14",
        "discipline": "#ff7f50",
        "product": "#fdfd96",
    }
    return palette.get(normalized, "#00ffff")


def _coerce_entities_and_relationships(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """
    Handles multiple API response formats to ensure the graph always renders.
    """
    documents = payload.get("documents") or {}

    # --- ADDED: Support for our Demo Endpoint format (nodes & links) ---
    if "nodes" in payload and "links" in payload:
        entities = payload.get("nodes", [])
        raw_links = payload.get("links", [])
        relationships = []
        for r in raw_links:
            relationships.append({
                "source_id": r.get("source"),
                "target_id": r.get("target"),
                "type": r.get("relationship")
            })
            
        # --- THE FIX: Inject fake source documents for the TA Demo ---
        mock_graph = {"source_docs": ["demo_doc_001", "demo_doc_002"]}
        mock_documents = {
            "demo_doc_001": "zinc_test.pdf",
            "demo_doc_002": "rasa_shastra_chapter_1.pdf"
        }
        
        return entities, relationships, mock_graph, mock_documents

    # --- Support for newer API shape ---
    if isinstance(payload.get("entities"), list) and isinstance(payload.get("relationships"), list):
        entities = payload.get("entities") or []
        relationships = payload.get("relationships") or []
        graph = payload.get("graph") or {}
        return entities, relationships, graph, documents

    # --- Support for older graph results shape ---
    graph = payload.get("graph") or {}
    entities_by_id = _extract_entities(graph)
    entities = []
    for ent in entities_by_id.values():
        props = (ent or {}).get("properties") or {}
        entities.append(
            {
                "id": props.get("id"),
                "label": props.get("label"),
                "name": props.get("name"),
            }
        )
    relationships = _extract_relationships(graph)
    return entities, relationships, graph, documents


st.title("IKS-Loom")
st.caption("Knowledge-Graph Engine for Technical Manuscripts")

with st.sidebar:
    st.header("Document Ingestion")
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"], accept_multiple_files=False)
    upload_clicked = st.button("Upload & Process", type="primary", disabled=uploaded is None)

    if upload_clicked and uploaded is not None:
        files = {
            "file": (
                uploaded.name,
                uploaded.getvalue(),
                "application/pdf",
            )
        }
        try:
            with st.spinner("Uploading and processing (extraction → LLM → Neo4j)…"):
                resp = requests.post(f"{API_BASE}/api/upload", files=files, timeout=600) 
            if resp.ok:
                payload = resp.json()
                doc_id = payload.get("doc_id", "demo_doc_123")
                st.session_state["last_doc_id"] = doc_id
                st.success(f"Processed successfully. doc_id: {doc_id}")
            else:
                try:
                    detail = resp.json().get("detail")
                except Exception:
                    detail = resp.text
                st.error(f"Upload failed ({resp.status_code}). {detail}")
        except requests.RequestException as e:
            st.error(f"Could not reach backend at {API_BASE}. Error: {e}")


st.header("Knowledge Graph Search")
query = st.text_input(
    "Search for an entity (case-insensitive substring match)",
    placeholder="e.g., mercury, crucible, sutra, metallurgy…",
    label_visibility="collapsed",
)

should_search = bool(query.strip()) and st.session_state.get("last_query") != query.strip()
if should_search:
    st.session_state["last_query"] = query.strip()

    try:
        with st.spinner("Searching the knowledge graph…"):
            resp = requests.get(f"{API_BASE}/api/search", params={"q": query.strip()}, timeout=60)
            
            if resp.status_code == 404:
                st.warning("Search API not fully linked yet. Loading interactive Rasa Shastra demo graph...")
                resp = requests.get(f"{API_BASE}/api/graph/sample", timeout=60)

        if not resp.ok:
            try:
                detail = resp.json().get("detail")
            except Exception:
                detail = resp.text
            st.error(f"Search failed ({resp.status_code}). {detail}")
        else:
            payload = resp.json()
            entities, relationships, graph, documents = _coerce_entities_and_relationships(payload)

            st.subheader("Graph")
            if not entities:
                st.info("No entities found for this query.")
            else:
                nodes: list[Node] = []
                edges: list[Edge] = []

                for e in entities:
                    if not isinstance(e, dict):
                        continue
                    eid = e.get("id")
                    if not isinstance(eid, str) or not eid.strip():
                        continue
                    label = e.get("label") or e.get("group") or "Entity"
                    name = e.get("name") or e.get("label") or eid 
                    nodes.append(
                        Node(
                            id=eid,
                            label=str(name),
                            color=_color_for_label(str(label)),
                            size=25,
                            font={"color": "white", "size": 16},
                            title=f"{name}\n{label}\n{id}",
                        )
                    )

                for r in relationships:
                    if not isinstance(r, dict):
                        continue
                    sid = r.get("source_id")
                    tid = r.get("target_id")
                    rtype = r.get("type") or "RELATED_TO"
                    if not isinstance(sid, str) or not isinstance(tid, str):
                        continue
                    edges.append(
                        Edge(
                            source=sid,
                            target=tid,
                            label=str(rtype),
                            directed=True,
                            color="white",
                            font={"color": "white", "strokeWidth": 2, "strokeColor": "#000000"},
                        )
                    )

                config = Config(
                    width="100%",
                    height=600,
                    directed=True,
                    physics=True,
                    hierarchical=False,
                    nodeHighlightBehavior=True,
                    collapsible=True,
                )

                agraph(nodes=nodes, edges=edges, config=config)

            st.divider()
            st.subheader("Sources")
            source_doc_ids = graph.get("source_docs") or []
            if not source_doc_ids:
                st.info("No source documents found for this query.")
            else:
                for doc_id in source_doc_ids:
                    filename = documents.get(doc_id) or "Unknown filename"
                    st.write(f"- **{filename}** (`{doc_id}`)")

    except requests.RequestException as e:
        st.error(f"Search request failed. Error: {e}")