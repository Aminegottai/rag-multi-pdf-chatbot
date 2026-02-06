import os, json

IN_PATH = "data/processed/pages.jsonl"
OUT_PATH = "data/processed/chunks.jsonl"
os.makedirs("data/processed", exist_ok=True)

CHUNK_SIZE = 1600
OVERLAP = 250

def chunk_text(s: str, size: int, overlap: int):
    s = (s or "").strip()
    if not s:
        return []
    out = []
    start = 0
    while start < len(s):
        end = min(len(s), start + size)
        out.append(s[start:end])
        if end == len(s):
            break
        start = max(0, end - overlap)
    return out

def main():
    with open(OUT_PATH, "w", encoding="utf-8") as f_out:
        with open(IN_PATH, "r", encoding="utf-8") as f_in:
            for line in f_in:
                rec = json.loads(line)
                src, pdf, page, text = rec["source"], rec["pdf_file"], rec["page"], rec["text"]

                header = f"[SOURCE: {src} | PDF: {pdf} | PAGE: {page}]\n"
                chunks = chunk_text(text, CHUNK_SIZE, OVERLAP)

                for j, ch in enumerate(chunks):
                    out = {
                        "chunk_id": f"{src}_p{page}_c{j}",
                        "text": header + ch,
                        "metadata": {"source": src, "pdf_file": pdf, "page": page},
                    }
                    f_out.write(json.dumps(out, ensure_ascii=False) + "\n")

    print("Saved:", OUT_PATH)

if __name__ == "__main__":
    main()