# Testing Scripts Documentation

## Overview

This document describes all the testing scripts in the `scripts/` directory that verify the functionality of the MiniMe memory system.

## Test Scripts

### 1. quick_test.py

**Purpose**: Quick verification that core components work

**What it tests**:
- Module imports
- Embedding model (single and batch encoding)
- Chunking functionality
- Basic database operations

**Usage**:
```bash
python scripts/quick_test.py
```

**Output**:
- ✓ PASSED or ✗ FAILED for each component
- Quick feedback on system health

**When to use**:
- After installation to verify setup
- Before running full test suite
- Quick sanity check

---

### 2. test_memory.py

**Purpose**: Comprehensive test suite for the memory layer

**What it tests**:

#### Test 1: Note Chunking
- Splits long notes into overlapping chunks
- Preserves sentence boundaries
- Handles edge cases (empty, short, very long notes)

#### Test 2: Database Operations
- Database initialization
- Node insertion and retrieval
- Chunk insertion and retrieval
- Embedding storage and retrieval
- Node counting

#### Test 3: Embedding Model
- Single text encoding
- Batch encoding
- Similarity computation
- Model loading

#### Test 4: Vault Indexing (Full Integration)
- Creates test vault with sample notes
- Indexes all notes
- Extracts frontmatter, tags, wikilinks
- Creates embeddings for chunks
- Generates similarity proposals
- Verifies graph edges

**Usage**:
```bash
python scripts/test_memory.py
```

**Output**:
```
MiniMe Memory Layer Test Suite
============================================================

TEST 1: Note Chunking
============================================================
✓ Created X chunks from Y words

TEST 2: Database Operations
============================================================
✓ Database initialized
✓ Node inserted
✓ Node retrieved
...

TEST SUMMARY
============================================================
Chunking            ✓ PASSED
Database            ✓ PASSED
Embeddings          ✓ PASSED
Vault Indexing      ✓ PASSED

🎉 All tests passed!
```

**Test Data**:
- Creates temporary test vault with 3 sample notes
- Creates temporary test database
- Cleans up after completion

**When to use**:
- Full system verification
- Before committing changes
- CI/CD pipeline
- Debugging issues

---

### 3. test_search.py

**Purpose**: Test semantic search functionality

**What it tests**:
- MemorySearch instantiation
- Search on empty database (should return empty list)
- Basic search functionality

**Usage**:
```bash
python scripts/test_search.py
```

**Output**:
```
Testing MemorySearch...
[OK] MemorySearch instantiated
[OK] Search on empty DB returned 0 results

[SUCCESS] MemorySearch basic test passed!
```

**When to use**:
- Quick search functionality check
- Verify search doesn't crash on empty DB
- Basic integration test

---

### 4. test_faiss.py

**Purpose**: Test FAISS integration for fast search

**What it tests**:
- FAISS availability
- FAISS index loading
- Search using FAISS
- Results quality

**Usage**:
```bash
python scripts/test_faiss.py
```

**Prerequisites**:
- Requires `demo_memory.db` to exist (run demo first)
- Requires `faiss-cpu` installed

**Output**:
```
Testing FAISS integration...

FAISS enabled: True
Index size: 15 vectors

Testing search: 'Python async programming'

Found 3 results:

1. [Note Title] (similarity: 0.856)
   Chunk content preview...

FAISS is working correctly!
```

**When to use**:
- Verify FAISS is working
- Test fast search performance
- Debug FAISS-related issues

---

### 5. test_vault_indexing.py

**Purpose**: Test vault indexing with metadata validation

**What it tests**:
- Vault indexing with test markdown file
- Frontmatter parsing
- Tag extraction
- Wikilink extraction
- Embedding creation
- Metadata validation
- Chunk creation

**Usage**:
```bash
python scripts/test_vault_indexing.py
```

**Test Data**:
Creates a test note with:
- Frontmatter (title, tags)
- Body content
- Wikilinks
- Inline tags

**Output**:
```
TEST: Vault Indexing
============================================================

Initializing database...
[OK] Database initialized

Indexing vault...
[OK] Indexed 1 nodes
[OK] Node: Test Note
[OK] Node ID: abc123...
[OK] Created 1 chunks
[OK] Chunk has embedding: True
[OK] Chunk metadata keys: ['chunk_index', 'embedding', ...]
[OK] Embedding metadata keys: ['provider', 'model', 'revision', ...]

[SUCCESS] Vault indexing test passed!
```

**When to use**:
- Test indexing pipeline
- Verify metadata validation
- Debug embedding issues
- Test with specific note structure

---

## Running All Tests

### Quick Test First
```bash
python scripts/quick_test.py
```

### Full Test Suite
```bash
python scripts/test_memory.py
```

### Individual Component Tests
```bash
python scripts/test_search.py
python scripts/test_faiss.py
python scripts/test_vault_indexing.py
```

## Test Coverage

### Components Tested

✅ **Chunking** (`chunk.py`)
- Overlapping chunks
- Sentence boundaries
- Edge cases

✅ **Database** (`db.py`)
- Node operations
- Chunk operations
- Embedding storage
- Edge operations

✅ **Embeddings** (`embeddings.py`)
- Single encoding
- Batch encoding
- Model loading

✅ **Vault Indexing** (`vault.py`)
- File parsing
- Metadata extraction
- Embedding creation
- Graph operations

✅ **Search** (`search.py`)
- Semantic search
- FAISS integration
- Empty database handling

### Not Yet Tested

- Graph visualization
- Note summarizer
- Memory API (write/read/link)
- Proposal approval workflow

## Test Data Management

### Temporary Files

All test scripts create temporary files:
- Test databases: `test_*.db`
- Test vaults: `test_vault/`
- FAISS indices: `*.faiss`, `*.faiss.idmap`

**Cleanup**: Scripts automatically clean up temporary files after tests complete.

### Persistent Test Data

- `demo_memory.db`: Created by demo script, used by FAISS test
- `demo_memory.faiss`: FAISS index for demo database

## Best Practices

1. **Run quick_test.py first**: Fast verification before full suite
2. **Run in clean environment**: Avoid conflicts with existing databases
3. **Check dependencies**: Ensure all packages installed
4. **Review output**: Check for warnings or errors
5. **Clean up manually**: If tests crash, manually remove test files

## Troubleshooting

### Test Failures

**Problem**: Tests fail with import errors

**Solution**:
```bash
# Ensure you're in the minime directory
cd minime

# Install dependencies
pip install -r requirements.txt

# Install in editable mode
pip install -e .
```

### Database Locked

**Problem**: "database is locked" errors

**Solution**:
- Close other database connections
- Delete test database files
- Run tests one at a time

### FAISS Not Available

**Problem**: FAISS tests fail

**Solution**:
```bash
pip install faiss-cpu
```

### Model Download

**Problem**: Slow first test run

**Solution**: Normal - model downloads on first use (~80MB). Subsequent runs are fast.

## Integration with CI/CD

These test scripts can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions
- name: Run Quick Tests
  run: python scripts/quick_test.py

- name: Run Full Test Suite
  run: python scripts/test_memory.py
```

## Summary

The testing scripts provide:
- ✅ Quick verification (`quick_test.py`)
- ✅ Comprehensive testing (`test_memory.py`)
- ✅ Component-specific tests (`test_*.py`)
- ✅ Automatic cleanup
- ✅ Clear output and error messages

Use these scripts to verify the memory system is working correctly!

