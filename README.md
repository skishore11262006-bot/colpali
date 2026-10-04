# Visual Document RAG with ColPali + Ollama (100% local)

Most RAG pipelines extract text from PDFs, chunk it, and embed the chunks. That breaks on tables, diagrams, charts, and scanned pages, because the layout is lost before retrieval even starts.

This project uses **ColPali-style visual retrieval** instead: every PDF page is treated as an **image**, and a vision-language retriever finds the right page directly. A local **Ollama** model then writes the answer. No API keys, no cloud.

```
PDF  ->  page images  ->  ColPali (multi-vector embeddings)
                                   |
 question -> query vectors -> late-interaction scoring -> top-K pages
                                                              |
                                      page text + Ollama LLM -> answer
                                      (retrieved page images are saved too)
```

## Why not just normal RAG?

| | Text RAG | ColPali (this repo) |
|---|---|---|
| Needs OCR / text parsing | Yes | No, uses page images |
| Understands tables and diagrams | Often lost | Kept in the page image |
| Chunking required | Yes | No, one page = one unit |
| Embeddings per page | 1 per chunk | Many vectors per page (late interaction) |

## Features

- Visual page retrieval with `vidore/colSmol-256M` (small, runs on CPU)
- Local answer generation through Ollama (default `llama3.2`)
- Saves and opens the retrieved page images so you can verify the result
- Caches page embeddings so later runs start fast
- Clear errors for missing files or Ollama not running

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- A pulled model, for example: `ollama pull llama3.2`

## Setup

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
ollama pull llama3.2
```

## Usage

```bash
python colpali_ollama_full.py mydoc.pdf
```

The first run embeds every page and saves a cache file next to the PDF (`mydoc.pdf.colpali.pt`). After that it starts quickly.

Example session:

```
Q> What does the PDF renderer do?
Top pages: [2, 1]  scores: [13.58, 13.42]
  saved retrieved_pages/page_2.png
A> The PDF renderer converts each PDF page into a raster image.
```

## Configuration

Edit the config block at the top of `colpali_ollama_full.py`:

| Setting | Default | Meaning |
|---|---|---|
| `RETRIEVER` | `vidore/colSmol-256M` | Visual retriever model |
| `LLM` | `llama3.2:latest` | Any model from `ollama list` |
| `TOP_K` | `2` | Pages retrieved per question |
| `DPI` | `100` | Page render quality |
| `OPEN_IMAGES` | `True` | Open retrieved pages automatically (Windows) |

## Limitations

- Retrieval uses the page **image**, but the answer step currently sends the page **text** to the LLM. Scanned PDFs with no text layer need a vision model in Ollama (for example `qwen2.5vl`) so the page image is sent instead.
- Scores are relative ranking scores, not probabilities. On very short documents, neighboring pages can score close together.
- Small models give short or imperfect answers. Use a larger Ollama model for better wording.
- Embedding is slow on CPU for long PDFs.

## Roadmap

- [ ] Send retrieved page images to an Ollama vision model
- [ ] Multi-PDF index
- [ ] Simple web UI

## Tech stack

`colpali-engine` · `transformers` · `PyTorch` · `PyMuPDF` · `Ollama`

## Credits

- [ColPali](https://arxiv.org/abs/2407.01449) by Faysse et al.
- [colpali-engine](https://github.com/illuin-tech/colpali) and the ViDoRe benchmark
- [Ollama](https://ollama.com)

## License

MIT
