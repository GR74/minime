# MiniMe Documentation

Welcome to the MiniMe documentation! This directory contains comprehensive documentation for all components, scripts, concepts, and architecture of MiniMe - an identity-conditioned LLM orchestration system.

## 📁 Directory Structure

```
docs/
├── README.md (this file)
├── DOCUMENTATION_SUMMARY.md
│
├── components/          # Core component documentation
│   ├── README.md
│   └── EXPLANATION_*.md (10 component files)
│
├── scripts/             # Script documentation
│   ├── SCRIPTS_README.md
│   └── SCRIPTS_*.md (4 script category files)
│
├── concepts/            # ML concepts and theory
│   └── ML_CONCEPTS.md
│
├── architecture/        # System architecture
│   └── ARCHITECTURE.md
│
└── archive/            # Historical documentation
    └── day-1-2/         # Original documentation files
```

## 🚀 Quick Start

### For Developers
1. Start with **[ARCHITECTURE.md](architecture/ARCHITECTURE.md)** to understand the system design
2. Browse **[components/](components/)** to learn about individual components
3. Check **[scripts/](scripts/)** for available tools and utilities

### For ML/AI Enthusiasts
1. Read **[ML_CONCEPTS.md](concepts/ML_CONCEPTS.md)** for ML theory
2. Explore **[components/EXPLANATION_EMBEDDINGS_MODEL.md](components/EXPLANATION_EMBEDDINGS_MODEL.md)** for embedding details
3. Review **[components/EXPLANATION_FAISS.md](components/EXPLANATION_FAISS.md)** for vector search

### For Users
1. Check **[scripts/SCRIPTS_README.md](scripts/SCRIPTS_README.md)** for available scripts
2. Read **[scripts/SCRIPTS_DEMO.md](scripts/SCRIPTS_DEMO.md)** for demo examples
3. Explore **[scripts/SCRIPTS_UTILITIES.md](scripts/SCRIPTS_UTILITIES.md)** for utility tools

## 📚 Documentation Categories

### Components (`components/`)
Detailed documentation for each core component:
- **Session Management**: `EXPLANATION_SESSION.md`
- **Database Layer**: `EXPLANATION_DATABASE.md`
- **Chunking Strategy**: `EXPLANATION_CHUNK.md`
- **Embeddings**: `EXPLANATION_EMBEDDINGS_MODEL.md`
- **Vector Search**: `EXPLANATION_FAISS.md`
- **Search Engine**: `EXPLANATION_SEARCH.md`
- **Graph Operations**: `EXPLANATION_GRAPH_SERVICE.md`
- **Note Summarization**: `EXPLANATION_SUMMARIZER.md`
- **Visualization**: `EXPLANATION_VISUALIZER.md`
- **Metadata Utils**: `EXPLANATION_EMBEDDING_UTILS.md`

### Scripts (`scripts/`)
Documentation for all scripts and utilities:
- **Testing Scripts**: `SCRIPTS_TESTING.md`
- **Demo Scripts**: `SCRIPTS_DEMO.md`
- **Visualization Tools**: `SCRIPTS_VISUALIZATION.md`
- **Utilities**: `SCRIPTS_UTILITIES.md`
- **Overview**: `SCRIPTS_README.md`

### Concepts (`concepts/`)
Machine learning concepts explained:
- **ML_CONCEPTS.md**: Comprehensive guide to all ML concepts used in MiniMe

### Architecture (`architecture/`)
System design and flow:
- **ARCHITECTURE.md**: Complete system architecture, data flows, and component communication

### Archive (`archive/`)
Historical documentation:
- **day-1-2/**: Original documentation from early development

## 🔍 Finding Information

### By Topic
- **Embeddings**: `components/EXPLANATION_EMBEDDINGS_MODEL.md`, `concepts/ML_CONCEPTS.md`
- **Search**: `components/EXPLANATION_SEARCH.md`, `components/EXPLANATION_FAISS.md`
- **Graph**: `components/EXPLANATION_GRAPH_SERVICE.md`, `components/EXPLANATION_VISUALIZER.md`
- **Database**: `components/EXPLANATION_DATABASE.md`, `components/EXPLANATION_SESSION.md`
- **Testing**: `scripts/SCRIPTS_TESTING.md`

### By Component
- **Session**: `components/EXPLANATION_SESSION.md`
- **Database**: `components/EXPLANATION_DATABASE.md`
- **Chunking**: `components/EXPLANATION_CHUNK.md`
- **Embeddings**: `components/EXPLANATION_EMBEDDINGS_MODEL.md`
- **FAISS**: `components/EXPLANATION_FAISS.md`
- **Search**: `components/EXPLANATION_SEARCH.md`
- **Graph**: `components/EXPLANATION_GRAPH_SERVICE.md`
- **Summarizer**: `components/EXPLANATION_SUMMARIZER.md`
- **Visualizer**: `components/EXPLANATION_VISUALIZER.md`
- **Utils**: `components/EXPLANATION_EMBEDDING_UTILS.md`

## 📖 Documentation Summary

For a complete overview of all documentation, see **[DOCUMENTATION_SUMMARY.md](DOCUMENTATION_SUMMARY.md)**.

## 🏗️ System Overview

MiniMe is an identity-conditioned LLM orchestration system that combines personal identity principles, memory from Obsidian vaults, and multi-agent orchestration to create a personalized AI assistant.

**Current Implementation (Memory Layer)**:
- **Indexes** notes from Obsidian vaults
- **Embeds** text using sentence transformers
- **Searches** semantically using FAISS
- **Connects** notes via graph edges
- **Visualizes** knowledge graphs
- **Generates** notes from conversations

**Planned Features**:
- Identity principle system with hierarchical masks
- Multi-agent orchestration (architect, builder, critic, etc.)
- Task classification and routing
- Context-aware retrieval with identity weighting

For detailed architecture, see **[architecture/ARCHITECTURE.md](architecture/ARCHITECTURE.md)**.

## 🔗 Related Documentation

- **Main README**: `../README.md` (project root)
- **Setup Guide**: `../SETUP.md`
- **Testing Guide**: `../TESTING.md`

## 📝 Contributing

When adding new documentation:
1. Place component docs in `components/`
2. Place script docs in `scripts/`
3. Update relevant README files
4. Update `DOCUMENTATION_SUMMARY.md` if needed

---

**Last Updated**: Documentation organized by category for easy navigation.

