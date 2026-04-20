"""FastAPI application: upload pipeline and graph search."""

from __future__ import annotations

import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from supabase import Client, create_client

from core.extractor import process_pdf
from core.graph_builder import GraphExtraction, Neo4jManager, extract_knowledge

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_RAW_DIR = _PROJECT_ROOT / "data" / "raw"
_PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"

_neo4j_manager: Neo4jManager | None = None
_supabase_client: Client | None = None


def get_neo4j_manager() -> Neo4jManager:
    global _neo4j_manager
    if _neo4j_manager is None:
        _neo4j_manager = Neo4jManager()
    return _neo4j_manager


def get_supabase() -> Client:
    global _supabase_client
    if _supabase_client is None:
        load_dotenv()
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in the environment.")
        _supabase_client = create_client(url, key)
    return _supabase_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    global _neo4j_manager
    if _neo4j_manager is not None:
        _neo4j_manager.close()
        _neo4j_manager = None


app = FastAPI(
    title="IKS-Loom",
    description="Knowledge-Graph Engine for Technical Manuscripts",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)) -> dict[str, str]:
    """
    Save PDF, record metadata in Supabase, extract text, build knowledge graph, and persist to Neo4j.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")

    doc_id = str(uuid.uuid4())
    filename = file.filename
    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = _RAW_DIR / f"{doc_id}.pdf"

    try:
        body = await file.read()
        pdf_path.write_bytes(body)
    except OSError as exc:
        logger.exception("Failed to save upload for doc_id=%s", doc_id)
        raise HTTPException(status_code=500, detail="Could not save uploaded file.") from exc

    try:
        supabase = get_supabase()
        supabase.table("documents").insert({"id": doc_id, "filename": filename}).execute()
    except Exception as exc:
        logger.exception("Supabase insert failed for doc_id=%s", doc_id)
        try:
            pdf_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise HTTPException(status_code=502, detail="Could not record document metadata.") from exc

    try:
        process_pdf(str(pdf_path), doc_id)
    except Exception as exc:
        logger.exception("PDF extraction failed for doc_id=%s", doc_id)
        raise HTTPException(status_code=500, detail="PDF text extraction failed.") from exc

    txt_path = _PROCESSED_DIR / f"{doc_id}.txt"
    try:
        text = txt_path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.exception("Could not read processed text for doc_id=%s", doc_id)
        raise HTTPException(status_code=500, detail="Processed text file not found.") from exc

    try:
        graph: GraphExtraction = extract_knowledge(text)
    except Exception as exc:
        logger.exception("Knowledge extraction failed for doc_id=%s", doc_id)
        raise HTTPException(status_code=502, detail="Knowledge extraction failed.") from exc

    try:
        get_neo4j_manager().push_to_graph(graph, doc_id)
    except Exception as exc:
        logger.exception("Neo4j push failed for doc_id=%s", doc_id)
        raise HTTPException(status_code=502, detail="Graph database update failed.") from exc

    return {
        "message": "Document ingested successfully.",
        "doc_id": doc_id,
    }


@app.get("/search")
def search(q: str = Query(..., min_length=1, description="Substring to match against entity names")) -> dict[str, Any]:
    """
    Search entities by name and return graph neighborhood plus Supabase document metadata for involved sources.
    """
    neo4j = get_neo4j_manager()
    try:
        graph_part = neo4j.search_by_name_substring(q)
    except Exception as exc:
        logger.exception("Neo4j search failed")
        raise HTTPException(status_code=502, detail="Graph search failed.") from exc

    doc_ids: list[str] = graph_part.get("source_doc_ids") or []
    documents: list[dict[str, Any]] = []
    if doc_ids:
        try:
            supabase = get_supabase()
            res = (
                supabase.table("documents")
                .select("id, filename, created_at")
                .in_("id", doc_ids)
                .execute()
            )
            documents = list(res.data or [])
        except Exception as exc:
            logger.exception("Supabase lookup failed during search")
            raise HTTPException(status_code=502, detail="Document metadata lookup failed.") from exc

    return {
        "query": q,
        "graph": {
            "matches": graph_part.get("matches", []),
            "source_doc_ids": doc_ids,
        },
        "documents": documents,
        "documents_by_id": {row["id"]: row for row in documents},
    }
