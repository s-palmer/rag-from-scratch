# RAG from Scratch

A minimal Retrieval-Augmented Generation (RAG) pipeline built from scratch in Python.

## How it works

1. **Chunking** — Text files in `documents/` are split into overlapping 500-character chunks
2. **Embedding** — Chunks are embedded locally using `all-MiniLM-L6-v2` via sentence-transformers
3. **Retrieval** — Cosine similarity finds the most relevant chunks for a query
4. **Generation** — Claude answers the question using only the retrieved context

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install anthropic numpy sentence-transformers python-dotenv
```

Create a `.env` file with your Anthropic API key:

```
ANTHROPIC_API_KEY=your-key-here
```

## Usage

```bash
# First run builds the index from documents/
python rag_from_scratch.py "What is Dan's role in the realm?"

# Subsequent runs load the cached index
python rag_from_scratch.py "Tell me about the guilds"
```

To rebuild the index after changing documents, delete `index.json` and `embeddings.npy` and run again.
