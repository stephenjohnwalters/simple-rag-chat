# Simple RAG Chat

A command-line Retrieval-Augmented Generation (RAG) system that answers questions about company documentation using OpenAI embeddings and chat completion.

## Architecture

The system consists of three main components:

### 1. EmbeddingsService (`src/embeddings-service.ts`)
Manages document processing and vector search:
- **Chunking**: Splits markdown files into 400-character chunks
- **Embedding**: Generates vector embeddings via OpenAI API
- **Caching**: Persists embeddings to disk (`.cache/embeddings.json`)
- **Search**: Performs cosine similarity search over cached embeddings

### 2. RAG CLI (`src/rag-cli.ts`)
Interactive command-line interface:
- **Query generation**: Uses LLM to propose search queries
- **Multi-search**: Retrieves relevant chunks using multiple queries
- **Answer generation**: Synthesizes answers from retrieved context

### 3. Company Data (`data/company-data/`)
Knowledge base of 20 markdown documents covering policies, procedures, and company information.

## Quick Start

```bash
# Clone and setup
git clone <repository-url>
cd simple-rag-chat
./setup.sh              # macOS/Linux: installs Node 18, dependencies

# Or on Windows
powershell -ExecutionPolicy Bypass -File setup.ps1

# Set API key
export OPENAI_API_KEY="sk-..."

# Run
npx rag
```

If the API key is not set, the CLI will prompt for it interactively.

## Usage

### Interactive Mode

```bash
npx rag

> what is the vacation policy
Employees accrue 20 days of paid time off per calendar year, in addition to
local public holidays.

> exit
```

### Rebuild Cache

Force rebuild of the embeddings cache (useful after adding/modifying documents):

```bash
npx rag train
```

## How It Works

1. **Initialization**: On first run, documents are chunked and embedded via `text-embedding-3-small`, then cached
2. **Query Processing**: User questions are sent to GPT-3.5 to generate 2-4 search queries
3. **Vector Search**: Each query is embedded and compared against cached embeddings using cosine similarity
4. **Context Assembly**: Top matching chunks are combined into a context string
5. **Answer Generation**: GPT-3.5 answers based strictly on the provided context

## Files

```
simple-rag-chat/
├── src/
│   ├── embeddings-service.ts    # Vector search and caching
│   └── rag-cli.ts               # CLI and answer generation
├── data/
│   └── company-data/            # 20 markdown knowledge base files
├── .cache/
│   └── embeddings.json          # Generated on first run (1.3MB)
├── setup.sh                     # macOS/Linux setup script
└── setup.ps1                    # Windows setup script
```

## Requirements

- Node.js 18+ (installed automatically by setup scripts via Volta)
- OpenAI API key with access to:
  - `text-embedding-3-small` model
  - `gpt-3.5-turbo` model

## Configuration

Key constants in `src/embeddings-service.ts`:
- `CHUNK_SIZE`: 400 characters per chunk
- Embedding model: `text-embedding-3-small`
- Chat model: `gpt-3.5-turbo` (in `src/rag-cli.ts`)

## Development

```bash
# Build TypeScript
npm run build

# Type checking
npx tsc --noEmit

# Clean and rebuild cache
rm -rf .cache
npx rag
```

---

Built with TypeScript, OpenAI API, and Node.js.
