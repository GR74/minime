# Scripts Documentation

This directory contains documentation for all scripts, utilities, demos, and testing tools in MiniMe - an identity-conditioned LLM orchestration system.

## 📋 Scripts Overview

### Testing Scripts

**[SCRIPTS_TESTING.md](SCRIPTS_TESTING.md)**
- `test_memory.py`: Comprehensive memory system tests
- `quick_test.py`: Quick validation tests
- `test_faiss.py`: FAISS vector store tests
- `test_vault_indexing.py`: Vault indexing tests

### Demo Scripts

**[SCRIPTS_DEMO.md](SCRIPTS_DEMO.md)**
- `demo_memory.py`: Interactive memory system demo
- Demonstrates indexing, search, and graph operations

### Visualization Tools

**[SCRIPTS_VISUALIZATION.md](SCRIPTS_VISUALIZATION.md)**
- `viz_graph.py`: Graph visualization tool
- Interactive HTML and static image exports

### Utilities

**[SCRIPTS_UTILITIES.md](SCRIPTS_UTILITIES.md)**
- `rebuild_faiss_index.py`: Rebuild FAISS index from database
- `memory_explorer.py`: Explore memory database contents

## 🚀 Quick Start

### Running Tests
```bash
# Run all tests
python scripts/test_memory.py

# Quick test
python scripts/quick_test.py

# FAISS tests
python scripts/test_faiss.py
```

### Running Demos
```bash
# Interactive demo
python scripts/demo_memory.py
```

### Visualization
```bash
# Generate graph visualization
python scripts/viz_graph.py
```

### Utilities
```bash
# Rebuild FAISS index
python scripts/rebuild_faiss_index.py

# Explore memory
python scripts/memory_explorer.py
```

## 📚 Documentation Files

1. **[SCRIPTS_TESTING.md](SCRIPTS_TESTING.md)**
   - Complete testing suite documentation
   - Test fixtures and utilities
   - Integration test examples

2. **[SCRIPTS_DEMO.md](SCRIPTS_DEMO.md)**
   - Interactive demo walkthrough
   - Example usage patterns
   - CLI interface documentation

3. **[SCRIPTS_VISUALIZATION.md](SCRIPTS_VISUALIZATION.md)**
   - Graph visualization tool
   - Export formats (HTML, PNG, SVG)
   - Interactive features

4. **[SCRIPTS_UTILITIES.md](SCRIPTS_UTILITIES.md)**
   - Utility script documentation
   - Index rebuilding process
   - Memory exploration tools

## 🔗 Related Documentation

- **Components**: `../components/README.md`
- **Architecture**: `../architecture/ARCHITECTURE.md`
- **Testing Guide**: `../../TESTING.md`

