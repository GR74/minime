# Machine Learning Concepts in MiniMe

This document explains the ML concepts used in MiniMe in beginner-friendly terms.

## Table of Contents
1. [What is Machine Learning?](#what-is-machine-learning)
2. [Text Embeddings](#text-embeddings)
3. [Vector Similarity](#vector-similarity)
4. [Sentence Transformers](#sentence-transformers)
5. [How MiniMe Uses These Concepts](#how-minime-uses-these-concepts)

---

## What is Machine Learning?

**Machine Learning (ML)** is a way for computers to learn patterns from data without being explicitly programmed for every scenario.

### Simple Analogy
Think of teaching a child to recognize cats:
- **Traditional Programming**: "If it has whiskers AND pointy ears AND a tail, it's a cat"
- **Machine Learning**: Show the child 1000 pictures of cats and 1000 pictures of dogs, and they learn to recognize the pattern

### In MiniMe
MiniMe uses ML to understand the **meaning** of text, not just match exact words. This allows it to:
- Find similar concepts even if different words are used
- Understand relationships between ideas
- Personalize responses based on your identity principles

---

## Text Embeddings

### What is an Embedding?

An **embedding** is a way to convert text into a list of numbers (a vector) that captures the meaning of the text.

### Visual Analogy

Imagine you're plotting cities on a map:
- Cities close together = similar (e.g., "New York" and "Brooklyn")
- Cities far apart = different (e.g., "New York" and "Tokyo")

Embeddings work the same way:
- Similar meanings = vectors close together
- Different meanings = vectors far apart

### Example

```python
# These texts have similar meanings
text1 = "I love coding"
text2 = "I enjoy programming"

# Their embeddings would be similar (close in vector space)
embedding1 = [0.2, 0.8, -0.1, 0.5, ...]  # 384 numbers
embedding2 = [0.3, 0.7, -0.2, 0.4, ...]  # 384 numbers (similar!)

# This text is different
text3 = "I hate cooking"
embedding3 = [-0.5, 0.1, 0.9, -0.3, ...]  # Very different numbers
```

### Why Numbers?

Computers work with numbers, not words. By converting text to numbers:
- We can do math on meanings (add, subtract, compare)
- We can find similarities quickly
- We can store and search efficiently

### In MiniMe

When you define an identity principle like:
```yaml
name: "Modularity"
description: "Break systems into independent, testable modules"
```

MiniMe converts this to an embedding (384 numbers) that represents the **concept** of modularity. Later, when searching for similar ideas, it compares these number vectors.

---

## Vector Similarity

### What is Vector Similarity?

**Vector similarity** measures how "close" two embeddings are in the mathematical space.

### Common Methods

#### 1. Cosine Similarity (Most Common)
Measures the angle between two vectors:
- **1.0** = Identical meaning (same direction)
- **0.0** = Completely unrelated (perpendicular)
- **-1.0** = Opposite meaning (opposite direction)

**Visual**: Think of two arrows:
- Pointing same direction = high similarity
- Pointing different directions = low similarity

#### 2. Euclidean Distance
Measures straight-line distance between vectors:
- **0.0** = Identical
- **Large number** = Very different

### Example

```python
# Principle: "Modularity"
principle_embedding = [0.2, 0.8, -0.1, ...]

# User query: "How do I split code into modules?"
query_embedding = [0.3, 0.7, -0.2, ...]

# Calculate cosine similarity
similarity = cosine_similarity(principle_embedding, query_embedding)
# Result: 0.95 (very similar!)

# Different query: "What's the weather today?"
weather_embedding = [-0.5, 0.1, 0.9, ...]
similarity = cosine_similarity(principle_embedding, weather_embedding)
# Result: 0.12 (not similar at all)
```

### In MiniMe

When you ask MiniMe a question:
1. Your question is converted to an embedding
2. MiniMe compares it to all stored principles/notes
3. It finds the most similar ones (highest similarity scores)
4. Those are used to personalize the response

---

## Sentence Transformers

### What is a Sentence Transformer?

A **Sentence Transformer** is a pre-trained ML model that converts sentences into embeddings.

### What is "Pre-trained"?

**Pre-trained** means the model was already trained on millions of text examples before you use it. You don't need to train it yourself!

**Analogy**: Like hiring a translator who already knows 50 languages vs. teaching someone from scratch.

### The Model: all-MiniLM-L6-v2

MiniMe uses `all-MiniLM-L6-v2`:
- **all-MiniLM**: Trained on diverse text
- **L6**: 6 layers (relatively small, fast)
- **v2**: Version 2 (improved)

**Why this model?**
- ✅ Fast (processes ~300 texts/second)
- ✅ Small (22 million parameters, ~80MB)
- ✅ Good quality for most tasks
- ✅ Runs on CPU (no GPU needed)

### How It Works (Simplified)

```
Input Text: "I love coding"
    ↓
[Tokenization] → ["I", "love", "coding"]
    ↓
[Neural Network] → Processes through 6 layers
    ↓
[Output] → [0.2, 0.8, -0.1, 0.5, ...] (384 numbers)
```

**The neural network** learned from millions of examples that:
- "love" and "enjoy" should be close
- "coding" and "programming" should be close
- "coding" and "cooking" should be far apart

### In MiniMe

```python
from minime.memory.embeddings import EmbeddingModel

model = EmbeddingModel()  # Loads all-MiniLM-L6-v2
embedding = model.encode_single("Modularity: Break systems into modules")
# Returns: [0.2, 0.8, -0.1, ...] (384 numbers)
```

---

## How MiniMe Uses These Concepts

### 1. Identity Principles → Embeddings

**Step 1**: You define principles in YAML
```yaml
principles:
  - name: "Modularity"
    description: "Break systems into independent modules"
```

**Step 2**: MiniMe converts to embedding
```python
text = "Modularity: Break systems into independent modules"
vector = embedding_model.encode_single(text)
# vector = [0.2, 0.8, -0.1, ...] (384 numbers)
```

**Step 3**: Stored in `IdentityPrinciple.vector`

### 2. Memory Retrieval → Similarity Search

**Step 1**: User asks: "How should I structure my code?"

**Step 2**: Query converted to embedding
```python
query_vector = embedding_model.encode_single("How should I structure my code?")
```

**Step 3**: Compare with all stored notes/principles
```python
for principle in identity_principles:
    similarity = cosine_similarity(query_vector, principle.vector)
    if similarity > 0.7:  # Threshold
        # This principle is relevant!
```

**Step 4**: Most similar principles are used to personalize the response

### 3. Graph Connections → Semantic Relationships

MiniMe can also find relationships:
- Two notes with similar embeddings → likely related
- Proposes graph edges between them
- Creates a knowledge graph of your ideas

---

## Key Takeaways

1. **Embeddings** = Text as numbers that capture meaning
2. **Similarity** = How close two meanings are (using cosine similarity)
3. **Sentence Transformers** = Pre-trained models that create embeddings
4. **MiniMe uses embeddings** to:
   - Store your identity principles
   - Find relevant notes when you ask questions
   - Personalize responses based on your values

### No Training Required!

The best part: You don't need to train any models. The sentence transformer is already trained. You just:
1. Load the model
2. Feed it text
3. Get embeddings
4. Compare embeddings to find similarities

That's it! 🎉

---

---

## Advanced ML Concepts in MiniMe

### 6. FAISS - Approximate Nearest Neighbor Search

#### What is FAISS?

**FAISS** (Facebook AI Similarity Search) is a library for efficient similarity search in high-dimensional vector spaces. It enables fast search over millions of embeddings.

#### Why FAISS?

**Problem**: Linear search is O(n) - slow for large datasets
- 10,000 embeddings: ~100ms per search
- 100,000 embeddings: ~10s per search
- 1,000,000 embeddings: ~100s per search

**Solution**: FAISS uses optimized data structures for O(log n) search
- 10,000 embeddings: ~5ms per search (20x faster)
- 100,000 embeddings: ~50ms per search (200x faster)
- 1,000,000 embeddings: ~500ms per search (200x faster)

#### How FAISS Works

```python
# Traditional linear search
for embedding in all_embeddings:
    similarity = cosine_similarity(query, embedding)
    # O(n) - checks every embedding

# FAISS search
results = faiss_index.search(query, k=5)
# O(log n) - uses optimized index structure
```

**Key Features**:
- **Index Structure**: Pre-computed index for fast lookup
- **Batch Processing**: Handles multiple queries efficiently
- **Persistence**: Save/load index from disk
- **Normalization**: Requires normalized vectors for cosine similarity

#### In MiniMe

```python
from minime.memory.vector_store_faiss import FaissVectorStore

# Create index
store = FaissVectorStore(dim=384, index_path="./index.faiss")

# Add embeddings (normalized automatically)
store.add(embedding1, "chunk_1")
store.add(embedding2, "chunk_2")

# Fast search
results = store.search(query_embedding, k=5)
# Returns: [("chunk_1", 0.95), ("chunk_2", 0.87), ...]
```

**Benefits**:
- ✅ Fast search even with millions of embeddings
- ✅ Scalable to large knowledge bases
- ✅ Persistent index (save/load)
- ✅ Automatic normalization

---

### 7. Vector Normalization

#### What is Normalization?

**Normalization** converts a vector to a unit vector (length = 1) while preserving direction.

#### Why Normalize?

For cosine similarity, normalized vectors simplify computation:

```python
# Without normalization
vec1 = [0.2, 0.8, -0.1]
vec2 = [0.3, 0.7, -0.2]
similarity = dot(vec1, vec2) / (norm(vec1) * norm(vec2))

# With normalization
vec1_norm = normalize(vec1)  # Length = 1.0
vec2_norm = normalize(vec2)  # Length = 1.0
similarity = dot(vec1_norm, vec2_norm)  # Simpler!
```

**Benefits**:
- **Simpler Computation**: Inner product = cosine similarity
- **Consistent Scores**: All similarities in [0, 1] range
- **FAISS Requirement**: IndexFlatIP needs normalized vectors

#### Normalization Formula

```python
def normalize(vector):
    norm = sqrt(sum(x² for x in vector))  # Vector length
    if norm > 0:
        return [x / norm for x in vector]  # Divide by length
    else:
        return [0.0] * len(vector)  # Zero vector
```

**Example**:
```python
vec = [3, 4]  # Length = 5 (3² + 4² = 25, sqrt(25) = 5)
normalized = [3/5, 4/5] = [0.6, 0.8]  # Length = 1.0
```

#### In MiniMe

FAISS automatically normalizes vectors:

```python
# In FaissVectorStore.add()
vec = np.array(embedding, dtype=np.float32)
norm = np.linalg.norm(vec)
if norm > 0:
    vec = vec / norm  # Normalize to unit vector
self.index.add(vec)
```

---

### 8. Embedding Versioning and Metadata

#### Why Versioning?

Embeddings from different model versions are **incompatible**:

```python
# Model v1 embedding
old_embedding = [0.2, 0.8, -0.1, ...]  # Created with v1

# Model v2 embedding (same text, different model)
new_embedding = [0.3, 0.7, -0.2, ...]  # Created with v2

# Can't compare directly - different vector spaces!
```

**Problem**: Upgrading the model makes old embeddings useless

**Solution**: Track embedding metadata and validate compatibility

#### Embedding Metadata Schema

```python
embedding_metadata = {
    "provider": "sentence-transformers",  # Who created it
    "model": "all-MiniLM-L6-v2",          # Which model
    "revision": "v1",                     # Model version
    "encoder_sha": "a1b2c3d4",           # Code version
    "dim": 384,                           # Vector dimension
    "ts": 1234567890.0                    # Timestamp
}
```

#### Version Validation

```python
from minime.memory.embedding_utils import metadata_matches

current_meta = embedding_model.get_embedding_metadata()
stored_meta = chunk.metadata.get("embedding")

if metadata_matches(current_meta, stored_meta):
    # Compatible - can compare
    similarity = cosine_similarity(query, chunk.embedding)
else:
    # Incompatible - skip or re-embed
    pass
```

**Checks**:
- Provider matches (sentence-transformers, openai, etc.)
- Model name matches (all-MiniLM-L6-v2, etc.)
- Revision matches (v1, v2, etc.)
- Encoder SHA matches (code version)

#### In MiniMe

Every chunk stores embedding metadata:

```python
chunk_metadata = {
    "embedding": {
        "provider": "sentence-transformers",
        "model": "all-MiniLM-L6-v2",
        "revision": "v1",
        "encoder_sha": "a1b2c3d4",
        "dim": 384,
        "ts": time.time()
    }
}
```

**Benefits**:
- ✅ Prevents comparing incompatible embeddings
- ✅ Tracks which model created each embedding
- ✅ Enables safe model upgrades
- ✅ Debugging and auditing

---

### 9. Text Chunking Strategy

#### Why Chunking?

Embedding models work best with text of certain lengths:
- **Too Short**: Loses context
- **Too Long**: May exceed model limits or lose focus
- **Optimal**: 256-512 tokens

**Problem**: Notes can be very long (thousands of tokens)

**Solution**: Split notes into overlapping chunks

#### Chunking Strategy

```python
def chunk_note(note_text: str, max_tokens: int = 512, overlap: int = 128):
    tokens = note_text.split()
    
    # If short enough, return as single chunk
    if len(tokens) <= max_tokens:
        return [note_text]
    
    # Create overlapping chunks
    step_size = max_tokens - overlap  # e.g., 512 - 128 = 384
    chunks = []
    
    for i in range(0, len(tokens), step_size):
        chunk_tokens = tokens[i : i + max_tokens]
        chunk_text = " ".join(chunk_tokens)
        chunks.append(chunk_text)
    
    return chunks
```

**Example**:
- Note: 1000 tokens
- `max_tokens`: 512
- `overlap`: 128
- `step_size`: 384

**Chunks**:
- Chunk 1: tokens 0-511 (512 tokens)
- Chunk 2: tokens 384-895 (512 tokens, overlaps 128 with Chunk 1)
- Chunk 3: tokens 768-999 (232 tokens, overlaps 128 with Chunk 2)

#### Why Overlap?

**Overlap preserves context** across chunk boundaries:

```
Note: "I love coding. Programming is fun. I enjoy building apps."

Without overlap:
  Chunk 1: "I love coding. Programming is fun."
  Chunk 2: "I enjoy building apps."
  # Lost connection between "Programming" and "building apps"

With overlap:
  Chunk 1: "I love coding. Programming is fun. I enjoy"
  Chunk 2: "Programming is fun. I enjoy building apps."
  # Maintains context!
```

#### In MiniMe

```python
from minime.memory.chunk import chunk_note

# Chunk note
chunks = chunk_note(note_body, max_tokens=512, overlap=128)

# Embed each chunk
for chunk in chunks:
    embedding = embedding_model.encode_single(chunk)
    # Store chunk with embedding
```

**Benefits**:
- ✅ Handles long notes
- ✅ Preserves context with overlap
- ✅ Enables granular search
- ✅ Works with model limits

---

### 10. Graph-Based Similarity

#### What is Graph-Based Similarity?

Instead of just comparing embeddings directly, we can use **graph structure** to find related content:

```
Note A --[similarity: 0.8]--> Note B --[similarity: 0.9]--> Note C

Query: "Find notes related to Note A"
Results: Note B (direct), Note C (via Note B)
```

#### Two Types of Edges

1. **Explicit Edges (Wikilinks)**
   - User creates: `[[Note B]]` in Note A
   - Confidence: 1.0 (always)
   - Auto-approved

2. **Similarity Edges (Proposed)**
   - System proposes: Similarity > 0.7
   - Confidence: Similarity score (0.0-1.0)
   - Requires user approval

#### Similarity Proposal Generation

```python
# For each new note
for existing_node in all_nodes:
    similarity = cosine_similarity(new_embedding, existing_embedding)
    
    if similarity > 0.7:  # Threshold
        # Propose edge
        proposal = GraphEdge(
            source=new_node,
            target=existing_node,
            edge_type="similar",
            confidence=similarity,
            is_approved=False  # Needs approval
        )
```

#### Graph Traversal

```python
# Find related notes (k-hop neighbors)
def find_related(node, hops=2):
    related = []
    
    # Direct neighbors (1 hop)
    for edge in node.edges:
        related.append(edge.target)
    
    # Neighbors of neighbors (2 hops)
    for neighbor in node.edges:
        for edge in neighbor.edges:
            related.append(edge.target)
    
    return related
```

#### In MiniMe

```python
from minime.memory.graph import GraphService

# Create wikilink edges
wikilink_edges = await graph_service.create_wikilink_edges(node)

# Generate similarity proposals
proposals = await graph_service.generate_similarity_proposals(
    node,
    embedding,
    threshold=0.7,
    max_proposals=10
)
```

**Benefits**:
- ✅ Finds indirect relationships
- ✅ Combines explicit and implicit connections
- ✅ User control (approve/reject proposals)
- ✅ Builds knowledge graph over time

---

### 11. Batch Processing

#### Why Batch Processing?

Processing embeddings one at a time is slow:

```python
# Slow: Individual processing
for text in texts:
    embedding = model.encode_single(text)  # 10ms each
# Total: 10ms × 1000 = 10 seconds

# Fast: Batch processing
embeddings = model.encode(texts)  # 100ms for all
# Total: 100ms (100x faster!)
```

#### Batch Encoding

```python
# Single encoding
embedding = model.encode_single("text")
# Returns: [0.2, 0.8, ...]

# Batch encoding
embeddings = model.encode(["text1", "text2", "text3"])
# Returns: [[0.2, 0.8, ...], [0.3, 0.7, ...], [0.4, 0.6, ...]]
```

**Benefits**:
- **Faster**: Parallel processing
- **Efficient**: Better GPU/CPU utilization
- **Scalable**: Handles large batches

#### Batch FAISS Operations

```python
# Add one embedding
store.add(embedding1, "chunk_1")  # Slow

# Add batch
store.add_batch([emb1, emb2, emb3], ["chunk_1", "chunk_2", "chunk_3"])
# Much faster!
```

#### In MiniMe

```python
# Chunk note
chunks = chunk_note(note_body)

# Batch encode all chunks
embeddings = embedding_model.encode(chunks)  # Fast!

# Batch add to FAISS
store.add_batch(embeddings, chunk_ids)  # Fast!
```

**Best Practices**:
- ✅ Use batch operations when possible
- ✅ Process in chunks (not all at once)
- ✅ Use progress bars for large batches
- ✅ Handle errors gracefully

---

## Summary of New Concepts

1. **FAISS**: Fast approximate nearest neighbor search (O(log n) vs O(n))
2. **Vector Normalization**: Unit vectors for simplified cosine similarity
3. **Embedding Versioning**: Track model versions for compatibility
4. **Text Chunking**: Split long notes into manageable pieces
5. **Graph-Based Similarity**: Use graph structure for relationships
6. **Batch Processing**: Process multiple items efficiently

These concepts enable MiniMe to:
- ✅ Scale to large knowledge bases
- ✅ Handle long notes efficiently
- ✅ Find relationships between ideas
- ✅ Maintain compatibility across model upgrades
- ✅ Provide fast semantic search

---

## Further Reading

- [Sentence Transformers Documentation](https://www.sbert.net/)
- [Understanding Word Embeddings](https://www.tensorflow.org/text/guide/word_embeddings)
- [Cosine Similarity Explained](https://en.wikipedia.org/wiki/Cosine_similarity)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Vector Normalization](https://en.wikipedia.org/wiki/Unit_vector)
- [Graph Neural Networks](https://en.wikipedia.org/wiki/Graph_neural_network)

