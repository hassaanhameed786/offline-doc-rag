"""Core RAG pipeline: load -> chunk -> embed -> retrieve -> generate."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # "ollama" or "anthropic"
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-5-5")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
INDEX_DIR = Path("index")

SYSTEM_PROMPT = (
    "You answer questions using ONLY the provided context. "
    "Cite sources inline like [1], [2]. "
    "If the context does not contain the answer, say you don't know."
)

_embedder: SentenceTransformer | None = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


# ---------- 1. Load ----------
def load_documents(data_dir: str = "data") -> list[dict]:
    docs = []
    for path in sorted(Path(data_dir).rglob("*")):
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8")
        elif suffix == ".pdf":
            from pypdf import PdfReader

            text = "\n".join(p.extract_text() or "" for p in PdfReader(str(path)).pages)
        else:
            continue
        if text.strip():
            docs.append({"source": path.name, "text": text})
    return docs


# ---------- 2. Chunk ----------
def chunk_text(text: str, size: int = 100, overlap: int = 20) -> list[str]:
    """Split text into overlapping chunks of ~`size` words."""
    words = text.split()
    step = size - overlap
    return [" ".join(words[i : i + size]) for i in range(0, max(len(words), 1), step)]


# ---------- 3. Embed + index ----------
def build_index(data_dir: str = "data") -> int:
    chunks = []
    for doc in load_documents(data_dir):
        for piece in chunk_text(doc["text"]):
            chunks.append({"source": doc["source"], "text": piece})
    if not chunks:
        raise SystemExit(f"No .txt/.md/.pdf files found in '{data_dir}/'")

    embeddings = get_embedder().encode(
        [c["text"] for c in chunks], normalize_embeddings=True, show_progress_bar=True
    )
    INDEX_DIR.mkdir(exist_ok=True)
    np.save(INDEX_DIR / "embeddings.npy", embeddings)
    (INDEX_DIR / "chunks.json").write_text(json.dumps(chunks, indent=2), encoding="utf-8")
    return len(chunks)


# ---------- 4. Retrieve ----------
class Retriever:
    def __init__(self) -> None:
        if not (INDEX_DIR / "embeddings.npy").exists():
            raise SystemExit("No index found. Run: python ingest.py")
        self.embeddings = np.load(INDEX_DIR / "embeddings.npy")
        self.chunks = json.loads((INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))

    def search(self, query: str, k: int = 4) -> list[dict]:
        q = get_embedder().encode([query], normalize_embeddings=True)[0]
        scores = self.embeddings @ q  # cosine similarity (vectors are normalized)
        top = np.argsort(scores)[::-1][:k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in top]


# ---------- 5. Generate ----------
def generate_answer(question: str, hits: list[dict]) -> str:
    context = "\n\n".join(
        f"[{i}] (source: {h['source']})\n{h['text']}" for i, h in enumerate(hits, 1)
    )
    user_msg = f"Context:\n{context}\n\nQuestion: {question}"

    if LLM_PROVIDER == "anthropic":
        import anthropic

        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        response = client.messages.create(
            model=LLM_MODEL,
            max_tokens=800,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        return response.content[0].text

    import ollama

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        options={"temperature": 0},  # more factual, less creative
    )
    return response["message"]["content"]