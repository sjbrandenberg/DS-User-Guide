# DesignSafe User Guide RAG System

A Retrieval-Augmented Generation (RAG) system for querying the DesignSafe User Guide documentation.

## Features

- **Dual Processing Methods**: 
  - Process local markdown files from the MkDocs structure
  - Scrape the live website directly
  
- **Comprehensive Metadata Tracking**:
  - Document title
  - File source
  - Section hierarchy
  - Direct URL links with anchors
  
- **ChromaDB Vector Store**: Persistent storage with cosine similarity search

- **OpenAI Embeddings**: Uses text-embedding-ada-002 model for semantic search

## Setup

1. **Install Dependencies**:
   ```bash
   source env/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure OpenAI API Key**:
   Edit `.env` file and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_actual_api_key_here
   ```

## Usage

### Building the RAG Database

Run the main script to build the vector database:

```bash
python create_rag.py
```

You'll be prompted to choose:
1. Process local markdown files (recommended)
2. Scrape website directly

### Querying the RAG System

Use the interactive query script:

```bash
python query_rag.py
```

## File Structure

- `create_rag.py` - Main RAG system implementation
- `query_rag.py` - Interactive query interface
- `requirements.txt` - Python dependencies
- `.env` - Environment variables (API keys)
- `chroma_db/` - ChromaDB vector store (created automatically)

## Key Metadata Tracked

The system tracks the following metadata for each document chunk:

- **title**: Page title from MkDocs navigation
- **file**: Source markdown file path
- **section**: Hierarchical section path (e.g., "Data Depot / Managing Data / Data Transfer")
- **url**: Full URL to the documentation page
- **base_url**: Base URL without anchors
- **anchor**: Section anchor for direct linking

## URL Structure

The system automatically generates URLs following the DesignSafe documentation structure:
- Base URL: `https://www.designsafe-ci.org/user-guide/`
- Section URLs include anchors for direct navigation to specific sections

## Example Queries

- "How do I transfer data to DesignSafe?"
- "What is the Data Depot?"
- "How do I publish my data?"
- "What are the best practices for data curation?"

## Notes

- The system preserves the navigation hierarchy from MkDocs
- Anchors are generated following standard markdown conventions
- Both markdown processing and web scraping methods are available
- The vector database persists between sessions