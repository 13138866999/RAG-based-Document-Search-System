# api.py
import os, shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import loader, store
from rag import ChromaData, process_data  # uses your code:contentReference[oaicite:3]{index=3}

app = FastAPI(title="Mini RAG")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

DB_ROOT = "./db"
def db_path() -> str:
    os.makedirs(DB_ROOT, exist_ok=True)
    return DB_ROOT

@app.post("/ingest/folder")
def ingest_folder(folder_path: str = Form(...)):
    if not os.path.isdir(folder_path):
        raise HTTPException(400, "folder_path must be a directory")
    # md loader -> split -> append-persist (your process_data):contentReference[oaicite:4]{index=4}
    process_data(folder_path, db_path=db_path())
    return {"ok": True}

@app.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    tmp_dir = "./_tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    fp = os.path.join(tmp_dir, file.filename)
    with open(fp, "wb") as f:
        f.write(await file.read())

    # read one file -> split -> append (using your loader & store):contentReference[oaicite:5]{index=5}:contentReference[oaicite:6]{index=6}
    docs = loader.loader_md(fp)         # supports pdf/txt/docx
    chunks = loader.split_to_chunk(docs)       # splitter:contentReference[oaicite:7]{index=7}
    store.dataset_creation(chunks, db_path())  # append/persist:contentReference[oaicite:8]{index=8}

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return {"ok": True}

@app.get("/ask")
def ask(q: str, k: int = 10, threshold: float = 0.4):
    rag = ChromaData()
    rag.load_data(db_path())                   # load persisted DB:contentReference[oaicite:9]{index=9}
    out = rag.ask(q, k=k, threshold=threshold) # returns answer+sources+latency:contentReference[oaicite:10]{index=10}
    if not out:
        raise HTTPException(400, "No database loaded yet.")
    return out
