# Chunk Module - Note Chunking Strategy

## Overview

`chunk.py` provides the `chunk_note()` function, which splits long notes into overlapping chunks for efficient embedding and retrieval. This is essential for handling notes that exceed the optimal size for embedding models.

## Purpose

The chunking strategy addresses several challenges:
- **Model Limits**: Embedding models work best with text of certain lengths (typically 256-512 tokens)
- **Context Preservation**: Overlapping chunks maintain context across boundaries
- **Retrieval Granularity**: Smaller chunks enable more precise semantic search
- **Memory Efficiency**: Chunks can be processed and stored independently

## Key Function

### chunk_note()

```python
def chunk_note(
    note_text: str,
    max_tokens: int = 512,
    overlap: int = 128
) -> list[str]
```

**Parameters:**
- `note_text`: Full note text (body only, no frontmatter)
- `max_tokens`: Maximum tokens per chunk (default: 512)
- `overlap`: Number of tokens to overlap between chunks (default: 128)

**Returns:**
- List of chunk strings

## How It Works

### 1. Tokenization

The function uses simple word-based tokenization:

```python
tokens = note_text.split()  # Split on whitespace
```

**Note**: For production, this could be improved with proper tokenization (tiktoken, etc.), but word-based splitting works well for MVP.

### 2. Single Chunk Handling

If the note is shorter than `max_tokens`, it's returned as a single chunk:

```python
if len(tokens) <= max_tokens:
    return [" ".join(tokens)]
```

### 3. Overlapping Chunks

For longer notes, chunks are created with overlap:

```python
step_size = max_tokens - overlap  # e.g., 512 - 128 = 384

for i in range(0, len(tokens), step_size):
    chunk_tokens = tokens[i : i + max_tokens]
    chunk_text = " ".join(chunk_tokens)
    chunks.append(chunk_text.strip())
```

**Example:**
- Note: 1000 tokens
- `max_tokens`: 512
- `overlap`: 128
- `step_size`: 384

**Chunks created:**
- Chunk 1: tokens 0-511 (512 tokens)
- Chunk 2: tokens 384-895 (512 tokens, overlaps 128 with Chunk 1)
- Chunk 3: tokens 768-999 (232 tokens, overlaps 128 with Chunk 2)

### 4. Sentence Boundary Preservation

The function attempts to preserve sentence boundaries:

```python
if i > 0 and chunk_text:
    # Look for sentence endings near the start
    first_period = chunk_text.find(". ", 0, min(100, len(chunk_text)))
    if first_period > 0:
        chunk_text = chunk_text[first_period + 2 :]  # Start after period
```

**Why?** Starting chunks at sentence boundaries improves:
- **Readability**: Chunks are more coherent
- **Embedding Quality**: Better semantic representation
- **Retrieval**: More meaningful search results

### 5. End Preservation

The function ensures the end of the note is captured:

```python
# Ensure we don't lose the end
if len(tokens) > 0:
    last_chunk_start = max(0, len(tokens) - max_tokens)
    last_chunk = " ".join(tokens[last_chunk_start:])
    if last_chunk.strip() and (not chunks or chunks[-1] != last_chunk.strip()):
        chunks.append(last_chunk.strip())
```

## Usage Examples

### Basic Usage

```python
from minime.memory.chunk import chunk_note

note_body = """
This is a long note that needs to be chunked.
It contains multiple paragraphs and ideas.
Each chunk will be embedded separately.
...
"""

chunks = chunk_note(note_body, max_tokens=512, overlap=128)
# Returns: ['chunk 1 text...', 'chunk 2 text...', ...]
```

### Custom Parameters

```python
# Smaller chunks for more granular search
chunks = chunk_note(note_body, max_tokens=256, overlap=64)

# Larger chunks for better context
chunks = chunk_note(note_body, max_tokens=1024, overlap=256)
```

### Integration with VaultIndexer

The chunking function is used by `VaultIndexer` when processing notes:

```python
from minime.memory.chunk import chunk_note

# In VaultIndexer._process_file()
chunks = chunk_note(body)  # Split note into chunks

for idx, chunk_text in enumerate(chunks):
    # Create embedding for each chunk
    embedding = embedding_model.encode_single(chunk_text)
    
    # Store chunk with metadata
    chunk = MemoryChunk(
        chunk_id=f"{node_id}_chunk_{idx}",
        content=chunk_text,
        embedding=embedding,
        position=idx,
    )
    await db.insert_chunk(chunk)
```

## Design Decisions

### 1. Word-Based Tokenization

**Decision**: Use simple `split()` instead of proper tokenizer

**Rationale**:
- ✅ Simple and fast
- ✅ No external dependencies
- ✅ Works well for English text
- ⚠️ Less accurate for other languages
- ⚠️ Doesn't handle punctuation perfectly

**Future Improvement**: Use `tiktoken` or similar for accurate token counting

### 2. Fixed Overlap

**Decision**: Use fixed overlap (128 tokens) regardless of content

**Rationale**:
- ✅ Simple and predictable
- ✅ Works well for most cases
- ⚠️ May be too much for short notes
- ⚠️ May be too little for very long notes

**Future Improvement**: Dynamic overlap based on content structure

### 3. Sentence Boundary Heuristic

**Decision**: Simple period-finding heuristic

**Rationale**:
- ✅ Fast and lightweight
- ✅ Improves chunk quality
- ⚠️ Doesn't handle all sentence types
- ⚠️ May miss some boundaries

**Future Improvement**: Use NLP library for proper sentence segmentation

## Edge Cases

### Empty Note

```python
chunk_note("")  # Returns: []
chunk_note("   ")  # Returns: []
```

### Very Short Note

```python
chunk_note("Short note")  # Returns: ["Short note"]
```

### Single Long Paragraph

```python
# Note with no sentence boundaries
long_text = "word " * 1000
chunks = chunk_note(long_text)  # Still chunks correctly
```

## Performance Considerations

### Time Complexity

- **Tokenization**: O(n) where n is number of characters
- **Chunking**: O(n) where n is number of tokens
- **Overall**: O(n) linear time

### Space Complexity

- **Chunks**: O(n) where n is total text length
- **Overlap**: Creates some duplication, but bounded by overlap size

## Best Practices

1. **Choose Appropriate Size**: 
   - 256-512 tokens for most use cases
   - Smaller for more granular search
   - Larger for better context

2. **Set Overlap**:
   - 20-30% of max_tokens is a good rule of thumb
   - 128 tokens works well for 512-token chunks

3. **Consider Content**:
   - Technical docs: Larger chunks (more context)
   - Conversational: Smaller chunks (more granular)
   - Code: May need special handling

## Future Improvements

1. **Better Tokenization**: Use `tiktoken` or similar
2. **Dynamic Overlap**: Adjust based on content structure
3. **Semantic Chunking**: Split at semantic boundaries, not just tokens
4. **Code-Aware**: Special handling for code blocks
5. **Multilingual**: Better support for non-English text

## Summary

The `chunk_note()` function is a simple but effective strategy for splitting notes into manageable pieces. It provides:

- ✅ Overlapping chunks for context preservation
- ✅ Sentence boundary awareness
- ✅ Configurable size and overlap
- ✅ Simple, fast implementation

While it could be improved with more sophisticated tokenization and chunking strategies, it works well for the MVP and provides a solid foundation for future enhancements.

