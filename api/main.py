import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any

from core.extractor import extract_graph_from_pdf

os.makedirs("data/raw", exist_ok=True)

app = FastAPI(title="IKS-Loom API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- IN-MEMORY DATABASE ---
GLOBAL_GRAPH_DB = {
    "nodes": [],
    "links": []
}

@app.post("/api/upload", tags=["Pipeline"])
async def upload_document(file: UploadFile = File(...)):
    try:
        file_path = f"data/raw/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        extracted_data = extract_graph_from_pdf(file_path)

        if extracted_data:

            GLOBAL_GRAPH_DB["nodes"] = extracted_data.get("nodes", [])
            GLOBAL_GRAPH_DB["links"] = extracted_data.get("links", [])
            
            return {
                "filename": file.filename,
                "status": "Successfully uploaded and processed via LLM",
                "doc_id": file.filename
            }
        else:
            raise HTTPException(status_code=500, detail="LLM Extraction failed! Check your VS Code terminal.")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")

@app.get("/api/search", tags=["Pipeline"])
async def search_knowledge_graph(q: str = ""):
    # If DB is empty
    if not GLOBAL_GRAPH_DB["nodes"]:
        return {
            "nodes": [{"id": "0", "label": "Upload a PDF first!", "group": "concept"}],
            "links": []
        }

    query = q.lower().strip()
    

    if not query:
        return {
            "nodes": GLOBAL_GRAPH_DB["nodes"],
            "links": GLOBAL_GRAPH_DB["links"]
        }

    matched_node_ids = set()
    for node in GLOBAL_GRAPH_DB["nodes"]:
        if query in str(node.get("label", "")).lower() or query in str(node.get("id", "")).lower():
            matched_node_ids.add(node["id"])

    final_node_ids = set(matched_node_ids)
    final_links = []
    
    for link in GLOBAL_GRAPH_DB["links"]:
        source_id = str(link.get("source"))
        target_id = str(link.get("target"))
        
        if source_id in matched_node_ids or target_id in matched_node_ids:
            final_node_ids.add(source_id)
            final_node_ids.add(target_id)
            final_links.append(link)

    final_nodes = [node for node in GLOBAL_GRAPH_DB["nodes"] if node["id"] in final_node_ids]

    if not final_nodes:
        return {"nodes": [], "links": []}

    return {
        "nodes": final_nodes,
        "links": final_links
    }