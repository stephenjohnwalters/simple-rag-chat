# Simple RAG Chat (CLI)

Backend-only Retrieval-Augmented Generation over local markdown docs.  Designed for technical screening: the implementation is complete but contains an intentionally buggy similarity function for review.

## What’s Inside

- **`src/rag-cli.ts`** – all logic: chunking, embedding cache, retrieval, CLI loop.
- **`data/company-data/*.md`** – 20 diverse docs used as knowledge base.
- **`.cache/embeddings.json`** – created on first run; stores chunks + embeddings.
- **`setup.sh` / `setup.ps1`** – one-shot environment bootstrap (macOS/Linux or Windows).

## Quick Start

```bash
# macOS / Linux
git clone https://example.com/simple-rag-chat.git
cd simple-rag-chat
./setup.sh              # installs Node via Volta, builds project

# Windows PowerShell
git clone https://example.com/simple-rag-chat.git
cd simple-rag-chat
powershell -ExecutionPolicy Bypass -File setup.ps1

# run
npx rag                  # or: node dist/rag-cli.js
```

### Live vs. Mock Mode

| Scenario | Environment variable | Behaviour |
|----------|----------------------|-----------|
| **Live** embeddings & LLM | `OPENAI_API_KEY=<key>` | Calls OpenAI for embeddings + chat completions |
| **Mock** (default) | Variable unset | Deterministic SHA-based embeddings and stubbed chat replies |

## CLI Usage

```
Simple RAG CLI. Type 'exit' to quit.

> what is the parental leave policy
16 weeks fully paid (see Benefits Overview).
```

Command `train` is stubbed for candidates to implement:

```bash
node dist/rag-cli.js train   # TODO: ingest new files and persist embeddings
```

## How It Works

1. **Chunk & Embed** – Markdown files are split every 400 characters, embedded (`text-embedding-ada-002`) and cached.
2. **Question → Query Proposals** – LLM suggests 2-4 search queries in JSON.
3. **Vector Search** – For each query, top-K similar chunks are fetched.  *Similarity is intentionally wrong* – see `buggySimilarity` in source.
4. **Answer Generation** – Chosen chunks are injected as context and the LLM answers strictly from them.

## Intentionally Buggy Retrieval

`buggySimilarity` uses `1 / ∑(Δ²)` instead of cosine similarity.  Reviewing candidates should spot and fix this.

## Requirements

- Node 18 (setup scripts install automatically).
- Optional: Docker – add a simple container if your environment blocks installers.

---
© Acme Corp – demo repository for interview use only.

