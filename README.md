# Document Semantic Analysis Toolkit

A Python toolkit for semantic analysis of PDFs and web pages with vector embeddings and retrieval capabilities.

## Core Components

### Data Ingestion

- **PDF Processing**: Parse documents with `pdfplumber`, maintaining layout and position data
- **Web Processing**: Scrape websites with `playwright` and `readability-lxml`, preserving DOM structure
- **Adaptive Chunking**: Break documents into semantic chunks (350 tokens with 30% overlap)

### Vector Embedding

- Generate embeddings via OpenAI's `text-embedding-3-small`
- Efficient vector storage using FAISS (Facebook AI Similarity Search)
- Maximum Marginal Relevance (MMR) implementation for diverse result retrieval

### Analysis & Visualization

- Interactive PDF viewer with citation highlighting
- Semantic search across document collections
- Position-aware chunking for source attribution

## Installation

```bash
pip install -r requirements.txt
```

Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_key_here
```

## Usage Examples

### Embedding PDF Documents

```python
from utils.pdf_ingestion import PDFDocument
from openai import AsyncOpenAI
import asyncio

async def analyze_pdf():
    client = AsyncOpenAI()
    pdf = PDFDocument("document.pdf")
    chunks = await pdf.process(client)
    print(f"Extracted {len(chunks)} semantic chunks")

asyncio.run(analyze_pdf())
```

### Semantic Search

```python
from utils.vector_store import VectorStore, get_query_embedding

async def search_documents(chunks, query):
    client = AsyncOpenAI()
    
    # Create and populate vector store
    store = VectorStore()
    store.add_chunks(chunks)
    
    # Search with semantic query
    query_embedding = await get_query_embedding(query, client)
    results = store.mmr_search(query_embedding, k=3, lambda_param=0.7)
    
    return results
```

### Example Notebooks

The repository includes Jupyter Notebooks demonstrating key workflows:

- `process_pdf.ipynb`: PDF ingestion and analysis pipeline
- `process_web.ipynb`: Web content processing
- `pdf_highlight_example.ipynb`: Visualization with highlighted citations
- `utils_example.ipynb`: General utility usage examples

## Project Structure

```
└── utils/
    ├── pdf_ingestion.py   # PDF extraction and chunking
    ├── web_ingestion.py   # Web content processing
    ├── vector_store.py    # Embedding storage and retrieval
    ├── pdf_highlighter.py # Interactive PDF viewer
    └── model_costs.py     # Token usage tracking
```

## Sample Data

The repository includes sample documents for testing:
- `basic_laws.pdf`: Sample structured document
- `patent.pdf`: Example technical content