# Explanation: `minime/memory/chunk.py`

This file provides the chunking strategy for splitting long notes into smaller, overlapping segments.

## Overview

**Chunking** is the process of breaking long text documents into smaller pieces. In MiniMe, notes are chunked to:
1. Improve embedding quality (embeddings work better on shorter text)
2. Enable fine-grained retrieval (find specific parts of notes)
3. Manage token limits (fit within context windows)

---

## Function: `chunk_note(note_text, max_tokens=512, overlap=128) -> List[str]`

**Purpose**: Split note body into overlapping chunks.

**Parameters:**
- `note_text`: Full note text (body only, no frontmatter)
- `max_tokens`: Maximum tokens per chunk (default: 512)
- `overlap`: Number of tokens to overlap between chunks (default: 128)

**Returns**: List of chunk strings

---

## How It Works

### Step 1: Simple Tokenization

For MVP, uses word-based tokenization:
```python
tokens = note_text.split()  # Split on whitespace
```

**Why word-based?**
- Simple and fast
- Good enough for MVP
- No dependencies on tokenizers

**Future improvement**: Could use proper tokenizer (tiktoken, transformers) for more accurate token counting.

---

### Step 2: Check if Chunking is Needed

```python
if len(tokens) <= max_tokens:
    return [" ".join(tokens)]  # Return as single chunk
```

If note is short enough, return as-is (no chunking needed).

---

### Step 3: Calculate Step Size

```python
step_size = max_tokens - overlap  # 512 - 128 = 384
```

**Step size** determines how much to advance between chunks:
- Each chunk is 512 tokens
- Overlap is 128 tokens
- So we advance 384 tokens (512 - 128) for the next chunk

**Visual example:**
```
Note: [====chunk1====][==overlap==][====chunk2====][==overlap==][====chunk3====]
      └─────512─────┘ └───128───┘ └─────512─────┘ └───128───┘ └─────512─────┘
      └───────────────────────────────────────────────────────────────────────┘
                          Step size: 384 tokens
```

---

### Step 4: Create Overlapping Chunks

```python
chunks = []
for i in range(0, len(tokens), step_size):
    chunk_tokens = tokens[i : i + max_tokens]
    chunk_text = " ".join(chunk_tokens)
    # ... boundary preservation logic ...
    chunks.append(chunk_text.strip())
```

**Iteration:**
- Start at position 0
- Advance by `step_size` (384) each iteration
- Take `max_tokens` (512) tokens per chunk

**Example:**
```
Tokens: [0, 1, 2, ..., 511, 512, 513, ..., 895, 896, 897, ...]

Chunk 1: tokens[0:512]      → positions 0-511
Chunk 2: tokens[384:896]    → positions 384-895 (overlaps with chunk 1)
Chunk 3: tokens[768:1280]   → positions 768-1279 (overlaps with chunk 2)
```

---

### Step 5: Preserve Sentence Boundaries (Heuristic)

```python
if i > 0 and chunk_text:
    first_period = chunk_text.find(". ", 0, min(100, len(chunk_text)))
    if first_period > 0:
        chunk_text = chunk_text[first_period + 2 :]  # Start after period
```

**What it does:**
- For chunks after the first one
- Looks for first sentence ending (`. `) in first 100 characters
- Starts chunk after the period

**Why?**
- Prevents cutting mid-sentence
- Preserves context better
- More natural chunk boundaries

**Example:**
```
Original chunk: "This is the end of a sentence. This is the start..."
Adjusted chunk: "This is the start..."  (starts after period)
```

---

### Step 6: Ensure Last Chunk

```python
if len(tokens) > 0:
    last_chunk_start = max(0, len(tokens) - max_tokens)
    last_chunk = " ".join(tokens[last_chunk_start:])
    if last_chunk.strip() and (not chunks or chunks[-1] != last_chunk.strip()):
        chunks.append(last_chunk.strip())
```

**Why?**
- Ensures we don't lose the end of the note
- Takes last 512 tokens
- Only adds if different from previous chunk

---

## Example

### Input Note (1500 words)

```
This is a very long note about machine learning. Machine learning is a subset of artificial intelligence. It involves training models on data to make predictions. There are many types of machine learning algorithms. Supervised learning uses labeled data. Unsupervised learning finds patterns without labels. Reinforcement learning learns through trial and error. Deep learning uses neural networks. Neural networks are inspired by the brain. They consist of layers of neurons. Each neuron processes inputs and produces outputs. The network learns by adjusting weights. Backpropagation is the learning algorithm. It propagates errors backwards through the network. This allows the network to learn from mistakes. Gradient descent optimizes the weights. It finds the minimum of the loss function. Many applications use machine learning. Image recognition identifies objects in photos. Natural language processing understands text. Speech recognition converts speech to text. Recommendation systems suggest products. Autonomous vehicles use computer vision. Medical diagnosis assists doctors. Financial fraud detection prevents theft. Machine learning is transforming industries. It's becoming more accessible. Tools like TensorFlow and PyTorch make it easier. Transfer learning reuses pre-trained models. This saves time and resources. The future of ML is exciting.
```

### Output Chunks (512 tokens, 128 overlap)

**Chunk 1** (tokens 0-511):
```
This is a very long note about machine learning. Machine learning is a subset of artificial intelligence. ... [continues for 512 tokens]
```

**Chunk 2** (tokens 384-895, overlaps with chunk 1):
```
[last 128 tokens of chunk 1] ... There are many types of machine learning algorithms. Supervised learning uses labeled data. ... [continues for 384 new tokens]
```

**Chunk 3** (tokens 768-1280, overlaps with chunk 2):
```
[last 128 tokens of chunk 2] ... Deep learning uses neural networks. Neural networks are inspired by the brain. ... [continues for 384 new tokens]
```

**Chunk 4** (tokens 1152-1500, last chunk):
```
[last 128 tokens of chunk 3] ... Machine learning is transforming industries. It's becoming more accessible. Tools like TensorFlow and PyTorch make it easier. Transfer learning reuses pre-trained models. This saves time and resources. The future of ML is exciting.
```

---

## Key Concepts

### 1. Overlap

**Why overlap?**
- Preserves context at chunk boundaries
- Ensures no information is lost
- Helps embeddings capture relationships between adjacent chunks

**Visual:**
```
Chunk 1: [====text====]
Chunk 2:        [====text====]  (overlaps with chunk 1)
Chunk 3:               [====text====]  (overlaps with chunk 2)
```

### 2. Token-Based Chunking

**Tokens ≠ Words**
- Tokens can be words, subwords, or characters
- For MVP, we approximate with words
- Proper tokenizers count tokens more accurately

**Example:**
- Word count: "machine learning" = 2 words
- Token count: "machine learning" = 2-3 tokens (depending on tokenizer)

### 3. Sentence Boundary Preservation

**Problem**: Cutting mid-sentence breaks context.

**Solution**: Start chunks after sentence endings when possible.

**Trade-off**: Slight variation in chunk sizes (usually acceptable).

---

## Edge Cases

### Empty Note
```python
if not note_text or not note_text.strip():
    return []  # Return empty list
```

### Very Short Note
```python
if len(tokens) <= max_tokens:
    return [" ".join(tokens)]  # Single chunk
```

### Exact Multiple
If note length is exactly a multiple of step_size:
- Last chunk might be duplicate
- Guard clause prevents adding duplicate

---

## Configuration

Chunking parameters can be configured via `MiniMeConfig`:

```python
class MiniMeConfig:
    chunk_max_tokens: int = 512  # Max tokens per chunk
    chunk_overlap: int = 128     # Overlap between chunks
```

**Tuning:**
- **Larger chunks** (768 tokens): Better context, fewer chunks, slower embeddings
- **Smaller chunks** (256 tokens): More granular, more chunks, faster embeddings
- **More overlap** (256 tokens): Better context preservation, more redundancy
- **Less overlap** (64 tokens): Fewer chunks, might lose context at boundaries

---

## Integration

### With VaultIndexer

```python
# In VaultIndexer._process_file()
chunks = chunk_note(body)  # Split note into chunks

# Compute embeddings for each chunk
embeddings = embedding_model.encode(chunks)

# Store chunks in database
for chunk_text, embedding in zip(chunks, embeddings):
    chunk = MemoryChunk(...)
    await db.insert_chunk(chunk)
```

---

## Future Improvements

### 1. Proper Tokenization

Use `tiktoken` or transformers tokenizer:
```python
import tiktoken

encoding = tiktoken.encoding_for_model("gpt-4")
tokens = encoding.encode(text)  # Accurate token count
```

### 2. Semantic Chunking

Split on semantic boundaries (paragraphs, sections) rather than fixed sizes:
```python
# Split on headings or paragraphs
chunks = split_on_markdown_headings(note_text)
```

### 3. Adaptive Chunking

Adjust chunk size based on content:
- Code blocks: Larger chunks (preserve context)
- Lists: Smaller chunks (granular retrieval)

### 4. Metadata Preservation

Track chunk metadata:
- Which section of note?
- What's the chunk topic?
- Relationship to adjacent chunks?

---

## Performance

### Time Complexity
- Tokenization: O(n) where n = text length
- Chunking: O(n) - single pass through tokens
- Overall: Linear time, very fast

### Space Complexity
- Stores all chunks in memory
- For very large notes, could stream chunks

---

## Summary

The `chunk_note()` function is a **critical preprocessing step**:

1. **Splits long notes** into manageable pieces
2. **Preserves context** with overlapping chunks
3. **Enables fine-grained retrieval** (find specific parts)
4. **Improves embedding quality** (shorter text = better embeddings)
5. **Simple and fast** (word-based tokenization for MVP)

Without chunking, long notes would have poor embeddings and coarse-grained retrieval. Chunking enables MiniMe to find relevant **parts** of notes, not just whole notes.

