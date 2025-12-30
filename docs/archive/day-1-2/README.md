# MiniMe Documentation

Welcome to the MiniMe documentation! This folder contains detailed explanations of the codebase, organized by topic.

## 📚 Documentation Index

### Core Concepts

1. **[ML_CONCEPTS.md](ML_CONCEPTS.md)** - Machine Learning Concepts
   - What is Machine Learning?
   - Text Embeddings explained
   - Vector Similarity
   - Sentence Transformers
   - How MiniMe uses ML concepts
   - **Start here if you're new to ML!**

### File Explanations

2. **[EXPLANATION_SCHEMAS.md](EXPLANATION_SCHEMAS.md)** - `minime/schemas.py`
   - All Pydantic data models
   - Identity, Memory, Agent, Tool schemas
   - How data validation works
   - **The foundation of all data structures**

3. **[EXPLANATION_CONFIG.md](EXPLANATION_CONFIG.md)** - `minime/config.py`
   - Configuration loading
   - Default values
   - YAML file structure
   - **How the system is configured**

4. **[EXPLANATION_EMBEDDINGS.md](EXPLANATION_EMBEDDINGS.md)** - `minime/memory/embeddings.py`
   - EmbeddingModel class
   - How text becomes vectors
   - Lazy loading pattern
   - **The ML engine of MiniMe**

5. **[EXPLANATION_IDENTITY.md](EXPLANATION_IDENTITY.md)** - Identity Layer
   - `minime/identity/principles.py` - IdentityManager
   - `minime/identity/loader.py` - Loading from YAML
   - How principles are stored and used
   - **How MiniMe personalizes responses**

6. **[EXPLANATION_MEMORY_DATABASE.md](EXPLANATION_MEMORY_DATABASE.md)** - Database Layer
   - `minime/memory/db.py` - AsyncDatabase class
   - Database schema (4 tables)
   - Node, chunk, edge operations
   - Similarity search support
   - **The storage layer for all memory data**

7. **[EXPLANATION_VAULT_INDEXER.md](EXPLANATION_VAULT_INDEXER.md)** - Vault Indexer
   - `minime/memory/vault.py` - VaultIndexer class
   - How Obsidian vaults are indexed
   - Metadata extraction (wikilinks, tags, frontmatter)
   - Graph edge creation (explicit + similarity)
   - **The ingestion layer that processes your vault**

8. **[EXPLANATION_CHUNKING.md](EXPLANATION_CHUNKING.md)** - Note Chunking
   - `minime/memory/chunk.py` - chunk_note() function
   - Why and how notes are split into chunks
   - Overlapping chunk strategy
   - Sentence boundary preservation
   - **How long notes are made searchable**

9. **[EXPLANATION_NOTE_SUMMARIZER.md](EXPLANATION_NOTE_SUMMARIZER.md)** - Note Summarizer
   - `minime/memory/summarizer.py` - NoteSummarizer class
   - Auto-generating notes from AI conversations
   - LLM vs template-based generation
   - Auto-indexing of generated notes
   - **How MiniMe builds persistent memory**

## 🎯 Quick Start Guide

### If you're new to ML:
1. Read **[ML_CONCEPTS.md](ML_CONCEPTS.md)** first
2. Then read **[EXPLANATION_EMBEDDINGS.md](EXPLANATION_EMBEDDINGS.md)**

### If you want to understand the code structure:
1. Start with **[EXPLANATION_SCHEMAS.md](EXPLANATION_SCHEMAS.md)** (data structures)
2. Then **[EXPLANATION_CONFIG.md](EXPLANATION_CONFIG.md)** (configuration)
3. Then **[EXPLANATION_IDENTITY.md](EXPLANATION_IDENTITY.md)** (identity layer)
4. Then **[EXPLANATION_EMBEDDINGS.md](EXPLANATION_EMBEDDINGS.md)** (ML implementation)
5. Then **[EXPLANATION_MEMORY_DATABASE.md](EXPLANATION_MEMORY_DATABASE.md)** (storage layer)
6. Then **[EXPLANATION_VAULT_INDEXER.md](EXPLANATION_VAULT_INDEXER.md)** (vault indexing)
7. Then **[EXPLANATION_CHUNKING.md](EXPLANATION_CHUNKING.md)** (chunking strategy)
8. Finally **[EXPLANATION_NOTE_SUMMARIZER.md](EXPLANATION_NOTE_SUMMARIZER.md)** (auto-generated notes)

### If you want to understand a specific file:
- Check the corresponding `EXPLANATION_*.md` file
- Each file has detailed explanations, examples, and code flow

## 📖 Document Structure

Each explanation document follows this structure:

1. **Overview** - What the file/module does
2. **File Structure** - High-level organization
3. **Detailed Explanation** - Line-by-line or function-by-function
4. **Examples** - Code examples showing usage
5. **Integration** - How it works with other modules
6. **Key Concepts** - Important ideas explained
7. **Summary** - Takeaways

## 🔍 Finding Information

### Looking for ML concepts?
→ **[ML_CONCEPTS.md](ML_CONCEPTS.md)**

### Want to understand data structures?
→ **[EXPLANATION_SCHEMAS.md](EXPLANATION_SCHEMAS.md)**

### Need to configure the system?
→ **[EXPLANATION_CONFIG.md](EXPLANATION_CONFIG.md)**

### Curious about embeddings?
→ **[EXPLANATION_EMBEDDINGS.md](EXPLANATION_EMBEDDINGS.md)**

### Want to understand personalization?
→ **[EXPLANATION_IDENTITY.md](EXPLANATION_IDENTITY.md)**

### Want to understand memory storage?
→ **[EXPLANATION_MEMORY_DATABASE.md](EXPLANATION_MEMORY_DATABASE.md)**

### Want to understand vault indexing?
→ **[EXPLANATION_VAULT_INDEXER.md](EXPLANATION_VAULT_INDEXER.md)**

### Want to understand note chunking?
→ **[EXPLANATION_CHUNKING.md](EXPLANATION_CHUNKING.md)**

### Want to understand auto-generated notes?
→ **[EXPLANATION_NOTE_SUMMARIZER.md](EXPLANATION_NOTE_SUMMARIZER.md)**

## 💡 Tips for Reading

1. **Start with concepts** - Understanding ML concepts makes the code easier
2. **Read in order** - The documents build on each other
3. **Try the examples** - Code examples help solidify understanding
4. **Check integration sections** - Shows how pieces fit together

## 🚀 Next Steps

After reading these documents, you should understand:
- ✅ How MiniMe uses ML for semantic understanding
- ✅ How data is structured and validated
- ✅ How configuration works
- ✅ How identity principles are stored and used
- ✅ How embeddings enable similarity search
- ✅ How the database stores vault data
- ✅ How Obsidian vaults are indexed and processed
- ✅ How notes are chunked for better retrieval
- ✅ How AI conversations become persistent memory

Happy learning! 🎓

