import os
import json
import uuid
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import fitz  # PyMuPDF
import faiss
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from groq import Groq
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

print("LOADED api.py FROM:", __file__)

# -----------------------------
# Paths / Config
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

INDEX_PATH = BASE_DIR / "index" / "faiss.index"
META_PATH = BASE_DIR / "index" / "meta.json"

STATIC_DIR = BASE_DIR / "src" / "web" / "static"
TEMPLATES_DIR = BASE_DIR / "src" / "web" / "templates"

UPLOADS_DIR = BASE_DIR / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

load_dotenv(BASE_DIR / ".env", override=True)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# -----------------------------
# App init
# -----------------------------
app = FastAPI(title="RAG Multi-PDF Chatbot")

if not STATIC_DIR.exists():
    raise RuntimeError(f"Static dir not found: {STATIC_DIR}")
if not TEMPLATES_DIR.exists():
    raise RuntimeError(f"Templates dir not found: {TEMPLATES_DIR}")

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# Debug route: see which endpoints are registered
@app.get("/routes")
def routes():
    return sorted([r.path for r in app.routes])


# -----------------------------
# Load base index (3 PDFs)
# -----------------------------
if not INDEX_PATH.exists() or not META_PATH.exists():
    raise RuntimeError(
        "Base index not found. Build it first:\n"
        "  python src/extract_ocr.py\n"
        "  python src/chunk.py\n"
        "  python src/build_index.py"
    )

embed_model = SentenceTransformer(MODEL_NAME)
base_index = faiss.read_index(str(INDEX_PATH))
base_meta = json.load(open(META_PATH, "r", encoding="utf-8"))


# -----------------------------
# RAG helpers
# -----------------------------
def retrieve_from(index, meta, query: str, sources: Optional[list[str]], topk: int):
    qv = embed_model.encode([query], normalize_embeddings=True)
    qv = np.array(qv, dtype=np.float32)

    D, I = index.search(qv, topk * 3)

    hits = []
    for score, idx in zip(D[0], I[0]):
        m = meta[int(idx)]
        if (not sources) or (m.get("source") in sources):
            hits.append((float(score), m))
        if len(hits) >= topk:
            break
    return hits


def build_prompt_base(question: str, passages: list[tuple[float, dict]]):
    blocks = []
    for _, m in passages:
        cite = f"{m['pdf_file']} p.{m['page']}"
        blocks.append(f"---\nCITATION: {cite}\n{m['text']}\n")
    context = "\n".join(blocks)

    return f"""
You are a technical documentation assistant for FastAPI, Pandas, and Docker.

RULES:
- Answer ONLY using the CONTEXT.
- If the answer is not in the context, say: "I don't know based on the provided PDFs."
- End your answer with:
  SOURCES:
  - pdf p.page

QUESTION:
{question}

CONTEXT:
{context}
""".strip()


def build_prompt_upload(question: str, passages: list[tuple[float, dict]]):
    blocks = []
    for _, m in passages:
        cite = f"{m['pdf_file']} p.{m['page']}"
        blocks.append(f"---\nCITATION: {cite}\n{m['text']}\n")
    context = "\n".join(blocks)

    return f"""
You are a PDF assistant.

IMPORTANT:
- You MUST answer ONLY from the uploaded PDF content in the CONTEXT.
- Do NOT mention FastAPI/Pandas/Docker unless they appear in the CONTEXT.
- If the answer is not in the context, say: "I don't know based on the uploaded PDF."

Return a clear answer, and finish with:
SOURCES:
- original.pdf p.X

QUESTION:
{question}

CONTEXT:
{context}
""".strip()


def call_llm(prompt: str) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("Missing GROQ_API_KEY. Put it in .env")

    client = Groq(api_key=GROQ_API_KEY)
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful technical documentation assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=800,
    )
    return resp.choices[0].message.content or ""


# -----------------------------
# Schemas
# -----------------------------
class Citation(BaseModel):
    pdf_file: str
    page: int
    source: str


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    sources: list[str] = Field(default_factory=list)
    top_k: int = Field(default=8, ge=3, le=12)


class ChatDocRequest(BaseModel):
    doc_id: str
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=8, ge=3, le=12)


# -----------------------------
# Basic endpoints
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/sources")
def sources():
    return {"sources": ["fastapi", "pandas", "docker"]}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        passages = retrieve_from(
            base_index, base_meta,
            req.question,
            req.sources if req.sources else None,
            req.top_k
        )
        prompt = build_prompt_base(req.question, passages)
        answer = call_llm(prompt)

        citations = []
        for _, m in passages:
            citations.append(Citation(pdf_file=m["pdf_file"], page=m["page"], source=m["source"]))

        # dedupe
        uniq, seen = [], set()
        for c in citations:
            k = (c.pdf_file, c.page, c.source)
            if k not in seen:
                seen.add(k)
                uniq.append(c)

        return ChatResponse(answer=answer, citations=uniq[: req.top_k])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------
# Upload PDF -> build FAISS per doc
# -----------------------------
def pdf_to_pages(pdf_path: Path):
    doc = fitz.open(str(pdf_path))
    pages = []
    for i in range(len(doc)):
        text = (doc[i].get_text("text") or "").strip()
        if text:
            pages.append({"page": i + 1, "text": text})
    return pages


def chunk_text(s: str, size: int = 1600, overlap: int = 250):
    s = (s or "").strip()
    if not s:
        return []
    out, start = [], 0
    while start < len(s):
        end = min(len(s), start + size)
        out.append(s[start:end])
        if end == len(s):
            break
        start = max(0, end - overlap)
    return out


def build_doc_index(doc_dir: Path, source_name: str = "user_pdf"):
    pdf_path = doc_dir / "original.pdf"
    pages = pdf_to_pages(pdf_path)

    chunks = []
    for p in pages:
        header = f"[SOURCE: {source_name} | PDF: original.pdf | PAGE: {p['page']}]\n"
        for j, ch in enumerate(chunk_text(p["text"])):
            chunks.append({
                "chunk_id": f"{source_name}_p{p['page']}_c{j}",
                "text": header + ch,
                "metadata": {"source": source_name, "pdf_file": "original.pdf", "page": p["page"]},
            })

    texts = [c["text"] for c in chunks]
    metas = [{"chunk_id": c["chunk_id"], **c["metadata"], "text": c["text"]} for c in chunks]

    if not texts:
        raise ValueError("No text extracted from uploaded PDF (maybe scanned). OCR not enabled for uploads.")

    emb = embed_model.encode(texts, normalize_embeddings=True)
    emb = np.array(emb, dtype=np.float32)

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)

    faiss.write_index(index, str(doc_dir / "faiss.index"))
    with open(doc_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(metas, f, ensure_ascii=False)

    return len(chunks)


def load_doc_index(doc_id: str):
    doc_dir = UPLOADS_DIR / doc_id
    idx_path = doc_dir / "faiss.index"
    meta_path = doc_dir / "meta.json"

    if not idx_path.exists() or not meta_path.exists():
        raise FileNotFoundError("Index not found for this doc_id. Upload again.")

    index = faiss.read_index(str(idx_path))
    meta = json.load(open(meta_path, "r", encoding="utf-8"))
    return index, meta


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    doc_id = str(__import__("uuid").uuid4())
    doc_dir = UPLOADS_DIR / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = doc_dir / "original.pdf"
    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        n_chunks = build_doc_index(doc_dir, source_name="user_pdf")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build index: {e}")

    return {"doc_id": doc_id, "chunks": n_chunks}


@app.post("/chat_doc", response_model=ChatResponse)
def chat_doc(req: ChatDocRequest):
    try:
        index, doc_meta = load_doc_index(req.doc_id)
        passages = retrieve_from(index, doc_meta, req.question, sources=None, topk=req.top_k)

        prompt = build_prompt_upload(req.question, passages)
        answer = call_llm(prompt)

        citations = []
        for _, m in passages:
            citations.append(Citation(pdf_file=m["pdf_file"], page=m["page"], source=m["source"]))

        uniq, seen = [], set()
        for c in citations:
            k = (c.pdf_file, c.page, c.source)
            if k not in seen:
                seen.add(k)
                uniq.append(c)

        return ChatResponse(answer=answer, citations=uniq[: req.top_k])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})