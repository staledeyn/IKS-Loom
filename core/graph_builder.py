"""NLP-based knowledge extraction and Neo4j graph persistence."""

from __future__ import annotations

import os
from typing import Any, List

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from neo4j import GraphDatabase
from pydantic import BaseModel, Field

# --- Pydantic models for structured LLM output ---


class Entity(BaseModel):
    """A node in the technical / manuscript knowledge graph."""

    id: str = Field(..., description="Stable unique identifier for this entity within the extraction.")
    label: str = Field(
        ...,
        description="High-level category, e.g. Concept, Material, Person, Process, Location, Text.",
    )
    name: str = Field(..., description="Human-readable name or short label for the entity.")


class Relationship(BaseModel):
    """A directed edge between two entities by id."""

    source_id: str = Field(..., description="Must match an Entity.id from the same extraction.")
    target_id: str = Field(..., description="Must match an Entity.id from the same extraction.")
    type: str = Field(..., description="Short relationship type, e.g. USES, DESCRIBES, PART_OF, REFERENCES.")


class GraphExtraction(BaseModel):
    """Structured graph payload returned by the LLM."""

    entities: List[Entity] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)


_EXTRACTION_SYSTEM_PROMPT = """You are an expert in extracting structured technical and historical knowledge \
from ancient manuscripts, treatises, and technical texts (including mathematics, astronomy, crafts, and materials).

Given passage text, identify salient entities (concepts, materials, people, processes, places, named works) \
and the relationships between them. Use concise, consistent identifiers for `id` (ASCII, no spaces; use \
underscores if needed). Every `source_id` and `target_id` in relationships must refer to an `id` you listed \
in `entities`. Prefer precision over volume: omit trivial or redundant items.

Respond only with data that fits the required JSON schema; do not add commentary."""


def extract_knowledge(text: str) -> GraphExtraction:
    """
    Run Gemini over the manuscript text and return a validated `GraphExtraction`.

    Uses `GEMINI_API_KEY` from the environment (see `.env`).
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set in the environment.")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0,
    )
    structured = llm.with_structured_output(GraphExtraction)

    messages = [
        SystemMessage(content=_EXTRACTION_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                "Extract entities and relationships from the following manuscript text.\n\n"
                f"{text}"
            )
        ),
    ]

    result = structured.invoke(messages)
    if not isinstance(result, GraphExtraction):
        raise RuntimeError("Structured output did not return a GraphExtraction instance.")
    return result


class Neo4jManager:
    """Neo4j access using the official driver with env-based configuration."""

    def __init__(self) -> None:
        load_dotenv()
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")
        if not uri or not user or password is None:
            raise ValueError("NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD must be set in the environment.")
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self._driver.close()

    def push_to_graph(self, extraction: GraphExtraction, doc_id: str) -> None:
        """
        Persist extracted entities and relationships in a single write transaction.

        Nodes use label `Entity` with `source_doc` set to `doc_id`. Relationships use `RELATES_TO` with
        a `type` property holding the LLM relationship type string.
        """

        def work(tx) -> None:
            for entity in extraction.entities:
                tx.run(
                    """
                    MERGE (e:Entity {id: $id, source_doc: $source_doc})
                    SET e.name = $name, e.label = $label
                    """,
                    id=entity.id,
                    source_doc=doc_id,
                    name=entity.name,
                    label=entity.label,
                )

            for rel in extraction.relationships:
                tx.run(
                    """
                    MATCH (a:Entity {id: $source_id, source_doc: $source_doc})
                    MATCH (b:Entity {id: $target_id, source_doc: $source_doc})
                    MERGE (a)-[r:RELATES_TO {type: $rel_type}]->(b)
                    """,
                    source_id=rel.source_id,
                    target_id=rel.target_id,
                    source_doc=doc_id,
                    rel_type=rel.type,
                )

        with self._driver.session() as session:
            session.execute_write(work)

    def search_by_name_substring(self, q: str) -> dict[str, Any]:
        """
        Find Entity nodes whose `name` contains ``q`` (case-insensitive).

        Returns each matching node with its direct outgoing and incoming ``RELATES_TO`` edges,
        neighbor node properties, and a deduplicated list of ``source_doc`` values seen on the
        match and its neighbors.
        """

        needle = (q or "").strip()
        if not needle:
            return {"matches": [], "source_doc_ids": []}

        def work(tx) -> tuple[list[dict[str, Any]], list[str]]:
            rows = tx.run(
                """
                MATCH (n:Entity)
                WHERE toLower(n.name) CONTAINS toLower($q)
                OPTIONAL MATCH (n)-[r_out:RELATES_TO]->(out:Entity)
                WITH n,
                  collect(DISTINCT CASE WHEN out IS NOT NULL THEN {rel_type: r_out.type, neighbor: out} END)
                    AS outgoing
                OPTIONAL MATCH (inc:Entity)-[r_in:RELATES_TO]->(n)
                RETURN n,
                  outgoing,
                  collect(DISTINCT CASE WHEN inc IS NOT NULL THEN {rel_type: r_in.type, neighbor: inc} END)
                    AS incoming
                """,
                q=needle,
            )

            matches: list[dict[str, Any]] = []
            source_ids: set[str] = set()

            for record in rows:
                n = record["n"]
                node_props = dict(n.items())
                sd = node_props.get("source_doc")
                if isinstance(sd, str) and sd:
                    source_ids.add(sd)

                outgoing: list[dict[str, Any]] = []
                for item in record["outgoing"]:
                    if not item:
                        continue
                    nb = item.get("neighbor")
                    if nb is None:
                        continue
                    nb_props = dict(nb.items())
                    nsd = nb_props.get("source_doc")
                    if isinstance(nsd, str) and nsd:
                        source_ids.add(nsd)
                    outgoing.append({"rel_type": item.get("rel_type"), "neighbor": nb_props})

                incoming: list[dict[str, Any]] = []
                for item in record["incoming"]:
                    if not item:
                        continue
                    nb = item.get("neighbor")
                    if nb is None:
                        continue
                    nb_props = dict(nb.items())
                    nsd = nb_props.get("source_doc")
                    if isinstance(nsd, str) and nsd:
                        source_ids.add(nsd)
                    incoming.append({"rel_type": item.get("rel_type"), "neighbor": nb_props})

                matches.append(
                    {
                        "node": node_props,
                        "outgoing": outgoing,
                        "incoming": incoming,
                    }
                )

            return matches, sorted(source_ids)

        with self._driver.session() as session:
            match_list, doc_ids = session.execute_read(work)

        return {"matches": match_list, "source_doc_ids": doc_ids}
