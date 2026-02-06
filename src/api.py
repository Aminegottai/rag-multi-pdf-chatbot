import os
import json
from pathlib import Path

import faiss
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from groq import Groq
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer


# -----------------------------
# Config
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # project root
INDEX_PATH = BASE_DIR / "index" / "faiss.index"
META_PATH = BASE_DIR / "index" / "meta.json"

# If your web folder is inside src/web:
STATIC_DIR = BASE_DIR / "src" / "web" / "static"
TEMPLATES_DIR = BASE_DIR / "src" / "web" / "templates"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Load env from project root
load_dotenv(BASE_DIR / ".env", override=True)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")  # fast + good for RAG


# -----------------------------
# App init
# -----------------------------
app = FastAPI(title="RAG Multi-PDF (FastAPI/Pandas/Docker)")

# Serve frontend (only if folders exist)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    # If template is missing, show a useful message
    index_html = TEMPLATES_DIR / "index.html"
    if not index_html.exists():
        return HTMLResponse(
            f"index.html not found at: {index_html}. Create it and restart.",
            status_code=500,
        )
    return templates.TemplateResponse("index.html", {"request": request})


# -----------------------------
# Load retriever resources
# -----------------------------
if not INDEX_PATH.exists():
    raise RuntimeError(f"FAISS index not found: {INDEX_PATH} (run build_index.py)")
if not META_PATH.exists():
    raise RuntimeError(f"Meta file not found: {META_PATH} (run build_index.py)")

embed_model = SentenceTransformer(MODEL_NAME)
faiss_index = faiss.read_index(str(INDEX_PATH))
meta = json.load(open(META_PATH, "r", encoding="utf-8"))


# -----------------------------
# Retrieval + Prompt
# -----------------------------
def retrieve(query: str, sources: list[str] | None, topk: int):
    qv = embed_model.encode([query], normalize_embeddings=True).astype("float32")
    D, I = faiss_index.search(qv, topk * 3)  # take more then filter

    hits = []
    for score, idx in zip(D[0], I[0]):
        m = meta[int(idx)]
        if (not sources) or (m["source"] in sources):
            hits.append((float(score), m))
        if len(hits) >= topk:
            break
    return hits


def build_prompt(question: str, passages: list[tuple[float, dict]]):
    blocks = []
    for _, m in passages:
        cite = f"{m['pdf_file']} p.{m['page']}"
        blocks.append(f"---\nCITATION: {cite}\n{m['text']}\n")

    context = "\n".join(blocks)

    prompt = f"""
You are a technical documentation assistant for FastAPI, Pandas, and Docker.

RULES:
- Answer ONLY using the CONTEXT.
- If the answer is not in the context, say: "I don't know based on the provided PDFs."
- Be concise and practical. Use steps and short code snippets when helpful.
- At the end, output:
  SOURCES:
  - pdf p.page
  - ...

QUESTION:
{question}

CONTEXT:
{context}
""".strip()

    return prompt


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
        max_tokens=700,
    )
    return resp.choices[0].message.content or ""


# -----------------------------
# API Schemas
# -----------------------------
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    sources: list[str] = Field(default_factory=list, description="Any of: fastapi, pandas, docker")
    top_k: int = Field(default=8, ge=3, le=12)


class Citation(BaseModel):
    pdf_file: str
    page: int
    source: str


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]


# -----------------------------
# API Endpoints
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
        passages = retrieve(req.question, req.sources if req.sources else None, req.top_k)
        prompt = build_prompt(req.question, passages)
        answer = call_llm(prompt)

        # citations from retrieved passages
        citations = []
        for _, m in passages:
            citations.append(Citation(pdf_file=m["pdf_file"], page=m["page"], source=m["source"]))

        # Deduplicate citations
        uniq = []
        seen = set()
        for c in citations:
            key = (c.pdf_file, c.page, c.source)
            if key not in seen:
                seen.add(key)
                uniq.append(c)

        return ChatResponse(answer=answer, citations=uniq[: req.top_k])

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))