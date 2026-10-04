"""
ColPali (page retrieval) + Ollama (answer generation) - full version
 
Install:
    pip install colpali-engine torch pymupdf pillow ollama
 
Run:
    python colpali_ollama_full.py mydoc.pdf
 
What it does for every question:
    1. Finds the best PDF pages with ColPali (uses page images)
    2. Saves the retrieved page images in the folder "retrieved_pages"
       (and opens them if OPEN_IMAGES = True)
    3. Prints the answer written by your Ollama model
"""
import os
import sys
 
import fitz  # PyMuPDF
import ollama
import torch
from PIL import Image
from colpali_engine.models import ColIdefics3, ColIdefics3Processor
 
# ---------------- config ----------------
RETRIEVER = "vidore/colSmol-256M"   # small ColPali-style model, OK on CPU
LLM = "llama3.2:latest"             # any model from `ollama list`
TOP_K = 2                           # how many pages to retrieve
DPI = 100                           # page render quality
BATCH = 2                           # pages embedded at once
OPEN_IMAGES = True                  # open retrieved page images automatically
OUT_DIR = "retrieved_pages"         # folder where page images are saved
# ----------------------------------------
 
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32
 
print(f"Loading {RETRIEVER} on {device} ...")
model = ColIdefics3.from_pretrained(RETRIEVER, torch_dtype=dtype).to(device).eval()
processor = ColIdefics3Processor.from_pretrained(RETRIEVER)
 
 
def load_pdf(path):
    """Return a list of (PIL image, page text) for each page."""
    doc = fitz.open(path)
    pages = []
    for page in doc:
        pix = page.get_pixmap(dpi=DPI)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        pages.append((img, page.get_text()))
    return pages
 
 
@torch.no_grad()
def embed_pages(images):
    embs = []
    for i in range(0, len(images), BATCH):
        batch = processor.process_images(images[i:i + BATCH]).to(model.device)
        out = model(**batch)
        embs.extend(list(out.cpu().to(torch.float32)))
        print(f"  embedded {min(i + BATCH, len(images))}/{len(images)} pages")
    return embs
 
 
@torch.no_grad()
def embed_query(q):
    batch = processor.process_queries([q]).to(model.device)
    return list(model(**batch).cpu().to(torch.float32))
 
 
def retrieve(query, page_embs, k=TOP_K):
    q_emb = embed_query(query)
    scores = processor.score_multi_vector(q_emb, page_embs)[0]
    top = torch.topk(scores, k=min(k, len(page_embs)))
    return top.indices.tolist(), top.values.tolist()
 
 
def answer(query, texts, page_nums):
    context = "\n\n".join(
        f"[Page {n + 1}]\n{t[:3000]}" for n, t in zip(page_nums, texts)
    )
    try:
        resp = ollama.chat(
            model=LLM,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Using ONLY the context, answer the question in 2-4 clear "
                        "sentences. Do not mention page numbers. If the answer is "
                        "not in the context, say you don't know."
                    ),
                },
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
            ],
        )
        return resp["message"]["content"].strip()
    except Exception as e:
        return (
            f"[Ollama error: {e}]\n"
            f"Check that Ollama is running and that '{LLM}' is in `ollama list`."
        )
 
 
def save_pages(images, idx):
    os.makedirs(OUT_DIR, exist_ok=True)
    for i in idx:
        fname = os.path.join(OUT_DIR, f"page_{i + 1}.png")
        images[i].save(fname)
        print(f"  saved {fname}")
        if OPEN_IMAGES and hasattr(os, "startfile"):  # Windows only
            os.startfile(fname)
 
 
def main():
    if len(sys.argv) < 2:
        print("Usage: python colpali_ollama_full.py file.pdf")
        sys.exit(1)
 
    pdf = sys.argv[1]
    if not os.path.exists(pdf):
        print(f"File not found: {pdf}")
        sys.exit(1)
 
    cache = pdf + ".colpali.pt"
 
    pages = load_pdf(pdf)
    images = [p[0] for p in pages]
    texts = [p[1] for p in pages]
    print(f"Loaded {len(pages)} pages from {pdf}")
 
    if os.path.exists(cache):
        print("Loading cached embeddings ...")
        page_embs = torch.load(cache)
    else:
        print(f"Embedding {len(images)} pages (first time only) ...")
        page_embs = embed_pages(images)
        torch.save(page_embs, cache)
 
    print("\nReady. Ask questions (empty line to quit).")
    while True:
        q = input("\nQ> ").strip()
        if not q:
            break
 
        idx, scores = retrieve(q, page_embs)
        print(f"Top pages: {[i + 1 for i in idx]}  scores: {[round(s, 2) for s in scores]}")
        save_pages(images, idx)
 
        print("\nA>", answer(q, [texts[i] for i in idx], idx))
 
 
if __name__ == "__main__":
    main()