# Component Documentation

This directory contains detailed documentation for each core component of MiniMe - an identity-conditioned LLM orchestration system that combines personal identity principles, memory from Obsidian vaults, and multi-agent orchestration.

## 📋 Components Overview

### Core Infrastructure

1. **[EXPLANATION_SESSION.md](EXPLANATION_SESSION.md)**
   - Database session management
   - Connection pooling and transactions
   - Async database operations

2. **[EXPLANATION_DATABASE.md](EXPLANATION_DATABASE.md)**
   - AsyncDatabase implementation
   - Compressed embedding storage
   - CRUD operations for nodes, chunks, edges

3. **[EXPLANATION_CHUNK.md](EXPLANATION_CHUNK.md)**
   - Note chunking strategy
   - Overlapping chunk creation
   - Token-based splitting

### Embeddings & Search

4. **[EXPLANATION_EMBEDDINGS_MODEL.md](EXPLANATION_EMBEDDINGS_MODEL.md)**
   - EmbeddingModel wrapper
   - Sentence transformer integration
   - Batch encoding and metadata

5. **[EXPLANATION_FAISS.md](EXPLANATION_FAISS.md)**
   - FAISS vector store
   - Fast approximate nearest neighbor search
   - Index management and persistence

6. **[EXPLANATION_SEARCH.md](EXPLANATION_SEARCH.md)**
   - MemorySearch semantic search
   - Memory interface for write/read/link
   - Query processing and result ranking

7. **[EXPLANATION_EMBEDDING_UTILS.md](EXPLANATION_EMBEDDING_UTILS.md)**
   - Metadata validation
   - Version compatibility checking
   - Embedding metadata utilities

### Graph & Visualization

8. **[EXPLANATION_GRAPH_SERVICE.md](EXPLANATION_GRAPH_SERVICE.md)**
   - GraphService for edge management
   - Wikilink edge creation
   - Similarity proposal generation

9. **[EXPLANATION_VISUALIZER.md](EXPLANATION_VISUALIZER.md)**
   - GraphVisualizer for rendering
   - Interactive HTML visualizations
   - Static image exports

### Note Generation

10. **[EXPLANATION_SUMMARIZER.md](EXPLANATION_SUMMARIZER.md)**
    - NoteSummarizer for auto-generation
    - LLM-powered note creation
    - Template-based content generation

## 🔄 Component Dependencies

```
session.py (foundation)
  └─► db.py (uses session)
      ├─► chunk.py (text processing)
      ├─► embeddings.py (ML foundation)
      └─► embedding_utils.py (validation)
          │
          ├─► search.py (uses db, embeddings, utils)
          │   └─► vector_store_faiss.py (optional)
          │
          ├─► graph.py (uses db, embeddings, utils)
          │   └─► visualizer.py (uses db, graph)
          │
          └─► summarizer.py (uses db, embeddings, vault)
```

## 📚 Quick Reference

### Database Layer
- **Session**: `EXPLANATION_SESSION.md`
- **Database**: `EXPLANATION_DATABASE.md`

### Text Processing
- **Chunking**: `EXPLANATION_CHUNK.md`

### ML Components
- **Embeddings**: `EXPLANATION_EMBEDDINGS_MODEL.md`
- **FAISS**: `EXPLANATION_FAISS.md`
- **Utils**: `EXPLANATION_EMBEDDING_UTILS.md`

### Search & Retrieval
- **Search**: `EXPLANATION_SEARCH.md`

### Graph Operations
- **Graph Service**: `EXPLANATION_GRAPH_SERVICE.md`
- **Visualizer**: `EXPLANATION_VISUALIZER.md`

### Note Generation
- **Summarizer**: `EXPLANATION_SUMMARIZER.md`

## 🔗 Related Documentation

- **Architecture**: `../architecture/ARCHITECTURE.md`
- **ML Concepts**: `../concepts/ML_CONCEPTS.md`
- **Scripts**: `../scripts/SCRIPTS_README.md`
