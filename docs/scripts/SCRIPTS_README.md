# Scripts Documentation - Complete Guide

## Overview

The `scripts/` directory contains various scripts for testing, demonstrating, visualizing, and managing the MiniMe memory system.

## Script Categories

### 📋 Testing Scripts
- **quick_test.py** - Quick verification of core components
- **test_memory.py** - Comprehensive test suite
- **test_search.py** - Search functionality test
- **test_faiss.py** - FAISS integration test
- **test_vault_indexing.py** - Vault indexing test

**Documentation**: See [SCRIPTS_TESTING.md](SCRIPTS_TESTING.md)

### 🎬 Demo Scripts
- **demo_memory.py** - Interactive system demonstration

**Documentation**: See [SCRIPTS_DEMO.md](SCRIPTS_DEMO.md)

### 📊 Visualization Scripts
- **viz_graph.py** - Graph visualization generator

**Documentation**: See [SCRIPTS_VISUALIZATION.md](SCRIPTS_VISUALIZATION.md)

### 🔧 Utility Scripts
- **memory_explorer.py** - Interactive memory exploration CLI
- **rebuild_faiss_index.py** - FAISS index maintenance

**Documentation**: See [SCRIPTS_UTILITIES.md](SCRIPTS_UTILITIES.md)

---

## Quick Start

### 1. Verify Installation

```bash
python scripts/quick_test.py
```

### 2. Run Demo

```bash
python scripts/demo_memory.py run
```

### 3. Explore Your Data

```bash
# Search
python scripts/memory_explorer.py search "your query"

# View graph
python scripts/viz_graph.py generate --open-browser

# Check stats
python scripts/memory_explorer.py stats
```

---

## Script Reference

### Testing

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `quick_test.py` | Quick verification | After installation, before full tests |
| `test_memory.py` | Full test suite | Before commits, CI/CD |
| `test_search.py` | Search test | Quick search verification |
| `test_faiss.py` | FAISS test | Verify FAISS integration |
| `test_vault_indexing.py` | Indexing test | Test indexing pipeline |

### Demo

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `demo_memory.py` | System demo | First-time users, showcases |

### Visualization

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `viz_graph.py` | Graph visualization | Explore connections, presentations |

### Utilities

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `memory_explorer.py` | Memory exploration | Daily use, finding content |
| `rebuild_faiss_index.py` | Index maintenance | After bulk updates, optimization |

---

## Common Workflows

### First Time Setup

```bash
# 1. Verify installation
python scripts/quick_test.py

# 2. Run demo
python scripts/demo_memory.py run

# 3. Explore results
python scripts/viz_graph.py generate --open-browser
```

### Daily Use

```bash
# Search for content
python scripts/memory_explorer.py search "topic"

# View specific note
python scripts/memory_explorer.py node "Note Title"

# Check system stats
python scripts/memory_explorer.py stats
```

### After Adding Notes

```bash
# Index vault
python scripts/demo_memory.py run --vault-path "./vault"

# Rebuild search index
python scripts/rebuild_faiss_index.py rebuild

# Visualize new connections
python scripts/viz_graph.py generate
```

### Maintenance

```bash
# Check system health
python scripts/memory_explorer.py stats

# Rebuild index
python scripts/rebuild_faiss_index.py rebuild

# Run full tests
python scripts/test_memory.py
```

---

## File Locations

### Scripts
- Location: `minime/scripts/`
- All scripts are executable Python files

### Generated Files

**Demo Output**:
- `demo_memory.db` - Demo database
- `demo_memory.faiss` - FAISS index
- `demo_graph.html` - Graph visualization

**Test Output**:
- `test_*.db` - Temporary test databases
- `test_vault/` - Temporary test vaults

**Visualization Output**:
- `graph.html` - HTML visualization (default)
- `graph.png` - PNG image (if specified)

---

## Dependencies

### Required
- All core MiniMe dependencies (see `requirements.txt`)

### Optional
- `faiss-cpu` - For FAISS tests and fast search
- `plotly` - For HTML visualizations
- `matplotlib` - For PNG image export

**Install optional dependencies**:
```bash
pip install faiss-cpu plotly matplotlib
```

---

## Troubleshooting

### Common Issues

1. **Import Errors**: Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

2. **FAISS Not Available**: Install FAISS
   ```bash
   pip install faiss-cpu
   ```

3. **Database Not Found**: Index vault first
   ```bash
   python scripts/demo_memory.py run
   ```

4. **Permission Errors**: Check file permissions
   ```bash
   chmod +x scripts/*.py
   ```

---

## Integration

### With CLI

These scripts complement the main CLI:
```bash
# CLI commands
minime init
minime index
minime search "query"

# Script commands (more detailed)
python scripts/memory_explorer.py search "query"
python scripts/viz_graph.py generate
```

### With CI/CD

Test scripts can be integrated:
```yaml
# GitHub Actions example
- name: Run Tests
  run: |
    python scripts/quick_test.py
    python scripts/test_memory.py
```

---

## Best Practices

1. **Start with Quick Test**: Verify setup first
2. **Use Demo**: Understand system capabilities
3. **Regular Maintenance**: Rebuild index periodically
4. **Explore Graph**: Understand your knowledge structure
5. **Check Stats**: Monitor system growth

---

## Documentation Files

- [SCRIPTS_TESTING.md](SCRIPTS_TESTING.md) - Testing scripts
- [SCRIPTS_DEMO.md](SCRIPTS_DEMO.md) - Demo scripts
- [SCRIPTS_VISUALIZATION.md](SCRIPTS_VISUALIZATION.md) - Visualization scripts
- [SCRIPTS_UTILITIES.md](SCRIPTS_UTILITIES.md) - Utility scripts

---

## Summary

The scripts directory provides:
- ✅ Comprehensive testing
- ✅ Interactive demos
- ✅ Visualization tools
- ✅ Exploration utilities
- ✅ Maintenance tools

Use these scripts to test, explore, and maintain your MiniMe memory system!

