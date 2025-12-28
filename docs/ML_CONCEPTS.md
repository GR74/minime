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

## Further Reading

- [Sentence Transformers Documentation](https://www.sbert.net/)
- [Understanding Word Embeddings](https://www.tensorflow.org/text/guide/word_embeddings)
- [Cosine Similarity Explained](https://en.wikipedia.org/wiki/Cosine_similarity)

