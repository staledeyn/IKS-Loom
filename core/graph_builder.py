from __future__ import annotations

import json
import os
import re
from typing import Any, Final, List, Optional

from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError
from pydantic import BaseModel, Field, ValidationError

from langchain_google_genai import ChatGoogleGenerativeAI

try:
    from langchain_core.output_parsers import PydanticOutputParser
    from langchain_core.prompts import ChatPromptTemplate
except ImportError:  # pragma: no cover
    from langchain.output_parsers import PydanticOutputParser  # type: ignore[no-redef]
    from langchain.prompts import ChatPromptTemplate  # type: ignore[no-redef]


_SAFE_LABEL_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_SAFE_RELTYPE_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")


class Entity(BaseModel):
    id: str = Field(..., min_length=1)
    label: str = Field(..., min_length=1, description="e.g., Concept, Material, Person")
    name: str = Field(..., min_length=1)


class Relationship(BaseModel):
    source_id: str = Field(..., min_length=1)
    target_id: str = Field(..., min_length=1)
    type: str = Field(..., min_length=1)


class GraphExtraction(BaseModel):
    entities: List[Entity] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)


class KnowledgeExtractionError(RuntimeError):
    """Raised when knowledge extraction fails or returns invalid output."""


def _get_gemini_api_key() -> Optional[str]:
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def extract_knowledge(text: str) -> GraphExtraction:
    """
    Extract entities and relationships from technical manuscript text via Gemini.

    Returns a validated `GraphExtraction` object.
    """
    if not text or not text.strip():
        return GraphExtraction()

    api_key = _get_gemini_api_key()
    if not api_key:
        raise KnowledgeExtractionError(
            "Missing GEMINI_API_KEY (or GOOGLE_API_KEY) in environment."
        )

    parser = PydanticOutputParser(pydantic_object=GraphExtraction)
    format_instructions = parser.get_format_instructions()

    system_prompt = (
        "You are an expert in extracting technical concepts from ancient manuscripts. "
        "Extract a knowledge graph from the user-provided text. "
        "Return ONLY valid JSON that strictly matches the provided schema. "
        "Do not include markdown, commentary, or additional keys."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "Text to analyze:\n\n{text}\n\n{format_instructions}",
            ),
        ]
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        google_api_key=api_key,
    )

    chain = prompt | llm
    try:
        msg = chain.invoke({"text": text, "format_instructions": format_instructions})
        raw = getattr(msg, "content", msg)
        if not isinstance(raw, str):
            raw = str(raw)
    except Exception as e:  # noqa: BLE001
        raise KnowledgeExtractionError("LLM invocation failed.") from e

    try:
        return parser.parse(raw)
    except Exception:
        try:
            candidate = raw.strip()
            if "```" in candidate:
                candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.I | re.S).strip()
            return GraphExtraction.model_validate(json.loads(candidate))
        except (json.JSONDecodeError, ValidationError) as e:
            raise KnowledgeExtractionError(
                "LLM output could not be parsed/validated against schema."
            ) from e


def _sanitize_label(label: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", (label or "").strip())
    if not cleaned:
        return "Entity"
    if not cleaned[0].isalpha() and cleaned[0] != "_":
        cleaned = f"_{cleaned}"
    if _SAFE_LABEL_RE.match(cleaned):
        return cleaned
    return "Entity"


def _sanitize_rel_type(rel_type: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", (rel_type or "").strip().upper())
    if not cleaned:
        return "RELATED_TO"
    if not cleaned[0].isalpha() and cleaned[0] != "_":
        cleaned = f"_{cleaned}"
    if _SAFE_RELTYPE_RE.match(cleaned):
        return cleaned
    return "RELATED_TO"


class Neo4jManager:
    def __init__(self) -> None:
        self.uri = os.getenv("NEO4J_URI")
        self.username = os.getenv("NEO4J_USERNAME")
        self.password = os.getenv("NEO4J_PASSWORD")

        missing = [k for k, v in {
            "NEO4J_URI": self.uri,
            "NEO4J_USERNAME": self.username,
            "NEO4J_PASSWORD": self.password,
        }.items() if not v]
        if missing:
            raise ValueError(f"Missing Neo4j env vars: {', '.join(missing)}")

        self._driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))

    def close(self) -> None:
        self._driver.close()

    def push_to_graph(self, extraction: GraphExtraction, doc_id: str) -> None:
        if not doc_id or not doc_id.strip():
            raise ValueError("doc_id must be a non-empty string.")

        entities = extraction.entities or []
        relationships = extraction.relationships or []

        def _write(tx) -> None:
            for ent in entities:
                node_label = _sanitize_label(ent.label)
                cypher = (
                    f"MERGE (n:Entity:{node_label} {{id: $id}}) "
                    "SET n.name = $name, n.label = $label, n.source_doc = $doc_id"
                )
                tx.run(
                    cypher,
                    id=ent.id,
                    name=ent.name,
                    label=ent.label,
                    doc_id=doc_id,
                )

            for rel in relationships:
                rel_type = _sanitize_rel_type(rel.type)
                cypher = (
                    "MATCH (s:Entity {id: $source_id}) "
                    "MATCH (t:Entity {id: $target_id}) "
                    f"MERGE (s)-[r:{rel_type}]->(t) "
                    "SET r.source_doc = $doc_id"
                )
                tx.run(
                    cypher,
                    source_id=rel.source_id,
                    target_id=rel.target_id,
                    doc_id=doc_id,
                )

        try:
            with self._driver.session() as session:
                if hasattr(session, "execute_write"):
                    session.execute_write(_write)  # neo4j>=5
                else:  # pragma: no cover
                    session.write_transaction(_write)  # neo4j<5
        except Neo4jError as e:
            raise RuntimeError("Neo4j write failed.") from e

    @staticmethod
    def _node_to_dict(node: Any) -> dict[str, Any]:
        props = dict(node) if node is not None else {}
        labels = list(getattr(node, "labels", []))
        return {"labels": labels, "properties": props}

    def search(self, q: str, limit: int = 25) -> dict[str, Any]:
        """
        Search nodes by case-insensitive substring match on `name`.

        Returns:
            {
              "results": [ { "node": {...}, "relationships": [...], "neighbors": [...] }, ... ],
              "source_docs": ["..."]
            }
        """
        if not q or not q.strip():
            return {"results": [], "source_docs": []}

        q = q.strip()
        limit = max(1, min(int(limit), 100))

        cypher = """
        MATCH (n)
        WHERE n.name IS NOT NULL AND toLower(n.name) CONTAINS toLower($q)
        OPTIONAL MATCH (n)-[r]-(m)
        RETURN n,
               collect(DISTINCT r) AS rels,
               collect(DISTINCT m) AS neighbors
        LIMIT $limit
        """

        def _read(tx) -> dict[str, Any]:
            records = tx.run(cypher, q=q, limit=limit)
            results: list[dict[str, Any]] = []
            source_docs: set[str] = set()

            for rec in records:
                n = rec.get("n")
                rels = rec.get("rels") or []
                neighbors = rec.get("neighbors") or []

                node_dict = self._node_to_dict(n)
                node_source = node_dict["properties"].get("source_doc")
                if isinstance(node_source, str) and node_source:
                    source_docs.add(node_source)

                rel_dicts: list[dict[str, Any]] = []
                for r in rels:
                    if r is None:
                        continue
                    start_id = None
                    end_id = None
                    try:
                        start_id = r.start_node.get("id")  # type: ignore[attr-defined]
                        end_id = r.end_node.get("id")  # type: ignore[attr-defined]
                    except Exception:
                        pass

                    rel_props = dict(r)
                    rel_source = rel_props.get("source_doc")
                    if isinstance(rel_source, str) and rel_source:
                        source_docs.add(rel_source)

                    rel_dicts.append(
                        {
                            "type": getattr(r, "type", None) or getattr(r, "__class__", type("x",(object,),{})).__name__,
                            "source_id": start_id,
                            "target_id": end_id,
                            "properties": rel_props,
                        }
                    )

                neighbor_dicts: list[dict[str, Any]] = []
                for m in neighbors:
                    if m is None:
                        continue
                    md = self._node_to_dict(m)
                    m_source = md["properties"].get("source_doc")
                    if isinstance(m_source, str) and m_source:
                        source_docs.add(m_source)
                    neighbor_dicts.append(md)

                results.append(
                    {
                        "node": node_dict,
                        "relationships": rel_dicts,
                        "neighbors": neighbor_dicts,
                    }
                )

            return {"results": results, "source_docs": sorted(source_docs)}

        try:
            with self._driver.session() as session:
                if hasattr(session, "execute_read"):
                    return session.execute_read(_read)  # neo4j>=5
                return session.read_transaction(_read)  # pragma: no cover
        except Neo4jError as e:
            raise RuntimeError("Neo4j search failed.") from e


def build_graph_from_text(doc_id: str, text: str) -> None:
    extraction = extract_knowledge(text)
    manager = Neo4jManager()
    try:
        manager.push_to_graph(extraction, doc_id=doc_id)
    finally:
        manager.close()

