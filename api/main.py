import os, shutil, sqlite3
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from typing import List
import secrets
from core.extractor import extract_graph_from_pdf

os.makedirs("data/raw", exist_ok=True)

DB_PATH = "data/graph.db"
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "iks-admin")

app = FastAPI(title="IKS-Loom API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

security = HTTPBasic()

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS nodes (
            id TEXT PRIMARY KEY,
            label TEXT,
            grp TEXT,
            source_doc TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS links (
            source TEXT,
            target TEXT,
            relationship TEXT,
            source_doc TEXT,
            PRIMARY KEY (source, target)
        )
    """)
    conn.commit()
    conn.close()

init_db()

def verify_admin(credentials: HTTPBasicCredentials = Depends(security)):
    correct = secrets.compare_digest(credentials.password, ADMIN_PASSWORD)
    if not correct:
        raise HTTPException(status_code=401, detail="Invalid credentials", headers={"WWW-Authenticate": "Basic"})
    return credentials.username

def merge_graph(new_data: dict, source_doc: str):
    conn = get_db()
    for node in new_data.get("nodes", []):
        conn.execute(
            "INSERT OR IGNORE INTO nodes (id, label, grp, source_doc) VALUES (?, ?, ?, ?)",
            (node.get("id"), node.get("label"), node.get("group"), source_doc)
        )
    for link in new_data.get("links", []):
        conn.execute(
            "INSERT OR IGNORE INTO links (source, target, relationship, source_doc) VALUES (?, ?, ?, ?)",
            (link.get("source"), link.get("target"), link.get("relationship"), source_doc)
        )
    conn.commit()
    conn.close()

@app.post("/api/upload")
async def upload_documents(files: List[UploadFile] = File(...), _=Depends(verify_admin)):
    results = []
    for file in files:
        file_path = f"data/raw/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        extracted = extract_graph_from_pdf(file_path)
        if extracted:
            merge_graph(extracted, file.filename)
            results.append({"filename": file.filename, "status": "ok"})
        else:
            results.append({"filename": file.filename, "status": "extraction_failed"})
    conn = get_db()
    total_nodes = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    conn.close()
    return {"results": results, "total_nodes": total_nodes}

@app.delete("/api/graph")
async def clear_graph(_=Depends(verify_admin)):
    conn = get_db()
    conn.execute("DELETE FROM nodes")
    conn.execute("DELETE FROM links")
    conn.commit()
    conn.close()
    return {"status": "cleared"}

@app.get("/api/search")
async def search(q: str = ""):
    conn = get_db()
    if not q.strip():
        nodes = [dict(r) for r in conn.execute("SELECT id, label, grp as group FROM nodes").fetchall()]
        links = [dict(r) for r in conn.execute("SELECT source, target, relationship FROM links").fetchall()]
        conn.close()
        return {"nodes": nodes, "links": links}

    query = f"%{q.lower().strip()}%"
    matched = conn.execute(
        "SELECT id FROM nodes WHERE LOWER(label) LIKE ? OR LOWER(id) LIKE ?",
        (query, query)
    ).fetchall()
    matched_ids = {r["id"] for r in matched}

    connected = conn.execute(
        "SELECT source, target, relationship FROM links WHERE source IN ({}) OR target IN ({})".format(
            ",".join("?" * len(matched_ids)),
            ",".join("?" * len(matched_ids))
        ),
        list(matched_ids) + list(matched_ids)
    ).fetchall() if matched_ids else []

    all_ids = set(matched_ids)
    final_links = []
    for link in connected:
        all_ids.add(link["source"])
        all_ids.add(link["target"])
        final_links.append(dict(link))

    final_nodes = [dict(r) for r in conn.execute(
        "SELECT id, label, grp as \"group\" FROM nodes WHERE id IN ({})".format(",".join("?" * len(all_ids))),
        list(all_ids)
    ).fetchall()] if all_ids else []

    conn.close()
    return {"nodes": final_nodes, "links": final_links}

@app.get("/api/stats")
async def stats():
    conn = get_db()
    nodes = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    links = conn.execute("SELECT COUNT(*) FROM links").fetchone()[0]
    docs = conn.execute("SELECT COUNT(DISTINCT source_doc) FROM nodes").fetchone()[0]
    conn.close()
    return {"nodes": nodes, "links": links, "documents": docs}

@app.get("/api/documents")
async def documents(_=Depends(verify_admin)):
    conn = get_db()
    docs = conn.execute(
        "SELECT source_doc, COUNT(*) as node_count FROM nodes GROUP BY source_doc"
    ).fetchall()
    conn.close()
    return {"documents": [dict(d) for d in docs]}

from fastapi.responses import RedirectResponse

@app.get("/admin")
async def admin_redirect():
    return RedirectResponse(url="/admin/")

app.mount("/admin", StaticFiles(directory="frontend/static/admin", html=True), name="admin")
app.mount("/", StaticFiles(directory="frontend/static/public", html=True), name="public")