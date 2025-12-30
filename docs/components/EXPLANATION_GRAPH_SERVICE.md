# GraphService - Graph Edge Management

## Overview

`graph.py` provides the `GraphService` class, which manages graph edges and similarity proposals in the MiniMe knowledge graph. It creates explicit edges from wikilinks and generates similarity-based edge proposals.

## Purpose

The GraphService handles:
- **Wikilink Edges**: Create edges from explicit [[links]] in notes
- **Similarity Proposals**: Generate edge proposals based on semantic similarity
- **Edge Management**: Create, validate, and store graph connections
- **Separation of Concerns**: Separates graph operations from file I/O

## Key Components

### GraphService Class

```python
class GraphService:
    def __init__(self, db: AsyncDatabase, embedding_model: EmbeddingModel)
    async def create_wikilink_edges(node: VaultNode) -> List[GraphEdge]
    async def generate_similarity_proposals(
        node: VaultNode,
        embedding: Optional[List[float]],
        threshold: float = 0.7,
        max_comparisons: int = 200,
        max_proposals: int = 10,
    ) -> List[GraphEdge]
```

## How It Works

### 1. Wikilink Edge Creation

Creates explicit edges from wikilinks found in notes:

```python
async def create_wikilink_edges(self, node: VaultNode) -> List[GraphEdge]:
    edges = []
    
    for link_target in node.links:
        # Resolve target node by title
        target_node_id = await self.db.get_node_by_title(link_target)
        if not target_node_id:
            continue  # Target doesn't exist yet
        
        # Check if edge already exists
        existing = await self.db.get_edge(
            node.node_id,
            target_node_id,
            edge_type="wikilink",
        )
        if existing:
            continue  # Edge already exists
        
        # Create edge
        edge = GraphEdge(
            edge_id=f"{node.node_id}_{target_node_id}_wikilink",
            source_node_id=node.node_id,
            target_node_id=target_node_id,
            edge_type="wikilink",
            weight=1.0,
            rationale=f"Explicit wikilink: [[{link_target}]]",
            confidence=1.0,
            is_approved=True,  # Wikilinks are auto-approved
            ...
        )
        
        await self.db.insert_edge(edge)
        edges.append(edge)
    
    return edges
```

**Features**:
- **Auto-Approved**: Wikilinks are automatically approved (confidence=1.0)
- **Deduplication**: Checks for existing edges before creating
- **Ghost Node Handling**: Skips links to non-existent nodes

### 2. Similarity Proposal Generation

Generates edge proposals based on semantic similarity:

```python
async def generate_similarity_proposals(
    self,
    node: VaultNode,
    embedding: Optional[List[float]],
    threshold: float = 0.7,
    max_comparisons: int = 200,
    max_proposals: int = 10,
) -> List[GraphEdge]:
    if not embedding:
        return []
    
    # Get all existing node embeddings
    existing_embeddings = await self.db.get_all_node_embeddings_with_metadata()
    
    # Limit comparisons to prevent O(N²) explosion
    if len(existing_embeddings) > max_comparisons:
        existing_embeddings = random.sample(existing_embeddings, max_comparisons)
    
    proposals = []
    for other_node_id, other_embedding, other_metadata in existing_embeddings:
        # Skip self
        if other_node_id == node.node_id:
            continue
        
        # Validate embedding metadata compatibility
        if not metadata_matches(current_meta, other_metadata["embedding"]):
            continue  # Skip incompatible embeddings
        
        # Compute similarity
        similarity = self._compute_similarity(embedding, other_embedding)
        
        if similarity > threshold:
            # Create proposal
            proposal = GraphEdge(
                edge_id=f"{node.node_id}_{other_node_id}_similar_{sha1}_{sha2}",
                source_node_id=node.node_id,
                target_node_id=other_node_id,
                edge_type="similar",
                weight=similarity,
                rationale=f"Semantic similarity: {similarity:.3f}",
                confidence=similarity,
                is_approved=False,  # Proposals need approval
                ...
            )
            
            await self.db.insert_proposal(proposal)
            proposals.append(proposal)
            
            if len(proposals) >= max_proposals:
                break
    
    return proposals
```

**Features**:
- **Similarity Threshold**: Only creates proposals above threshold (default: 0.7)
- **Sampling**: Limits comparisons to prevent O(N²) complexity
- **Metadata Validation**: Ensures embedding compatibility
- **Revisioning**: Includes embedding SHA in edge ID for versioning

### 3. Similarity Computation

Uses cosine similarity to measure semantic similarity:

```python
def _compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    similarity = dot_product / (norm1 * norm2)
    return float(similarity)
```

**Range**: 0.0 to 1.0
- **1.0**: Identical meaning
- **0.7+**: Very similar (good for proposals)
- **0.5-0.7**: Somewhat similar
- **<0.5**: Not similar

## Usage Examples

### Basic Usage

```python
from minime.memory.graph import GraphService
from minime.memory.db import AsyncDatabase
from minime.memory.embeddings import EmbeddingModel

# Initialize
db = AsyncDatabase("./data/minime.db")
model = EmbeddingModel()
graph_service = GraphService(db, model)

# Create wikilink edges
node = await db.get_node("node_id")
wikilink_edges = await graph_service.create_wikilink_edges(node)

# Generate similarity proposals
embedding = model.encode_single("Note content...")
proposals = await graph_service.generate_similarity_proposals(
    node,
    embedding,
    threshold=0.7,
    max_proposals=10
)
```

### Integration with VaultIndexer

The GraphService is used by VaultIndexer:

```python
# In VaultIndexer._process_file()
graph_service = GraphService(db, embedding_model)

# After processing node
node = VaultNode(...)
await db.insert_node(node)

# Create wikilink edges
await graph_service.create_wikilink_edges(node)

# Generate similarity proposals
primary_embedding = ...
await graph_service.generate_similarity_proposals(node, primary_embedding)
```

## Design Decisions

### 1. Separation of Concerns

**Decision**: Separate GraphService from VaultIndexer

**Rationale**:
- **VaultIndexer**: File I/O, parsing, chunking, embedding
- **GraphService**: Graph operations (edges, proposals)
- **Single Responsibility**: Each class has one clear purpose

### 2. Proposal System

**Decision**: Similarity edges are proposals, not auto-approved

**Rationale**:
- **Quality Control**: User can review before approval
- **False Positives**: Some similar notes may not be related
- **User Intent**: User knows best which connections are meaningful

### 3. Sampling for Performance

**Decision**: Limit comparisons to 200 nodes (sampling)

**Rationale**:
- **O(N²) Problem**: Comparing all nodes is O(N²)
- **Good Enough**: Random sample finds most similar nodes
- **Scalability**: Works with large vaults

### 4. Embedding SHA in Edge ID

**Decision**: Include embedding SHA in edge ID

**Rationale**:
- **Versioning**: Tracks which embedding version created edge
- **Revisioning**: Can regenerate edges when model changes
- **Deduplication**: Prevents duplicate edges from same embeddings

## Edge Types

### Wikilink Edges

- **Type**: `wikilink`
- **Source**: Explicit [[link]] in note
- **Confidence**: 1.0 (always)
- **Approved**: True (auto-approved)
- **Weight**: 1.0

### Similarity Edges

- **Type**: `similar`
- **Source**: Semantic similarity computation
- **Confidence**: Similarity score (0.0-1.0)
- **Approved**: False (requires approval)
- **Weight**: Similarity score

### Manual Edges

- **Type**: `manual`
- **Source**: User-created link
- **Confidence**: 1.0
- **Approved**: True
- **Weight**: 1.0

## Performance Considerations

### Time Complexity

- **Wikilink Edges**: O(L) where L is number of links
- **Similarity Proposals**: O(M) where M is max_comparisons (default: 200)
- **Overall**: O(M) per node

### Optimization Strategies

1. **Sampling**: Limit comparisons to prevent O(N²)
2. **Early Termination**: Stop after max_proposals
3. **Metadata Validation**: Skip incompatible embeddings early
4. **Batch Processing**: Process multiple nodes together

## Best Practices

1. **Set Appropriate Threshold**: Balance quality and quantity
   ```python
   threshold=0.7  # Good default
   ```

2. **Limit Comparisons**: Prevent performance issues
   ```python
   max_comparisons=200  # Good default
   ```

3. **Limit Proposals**: Don't overwhelm user
   ```python
   max_proposals=10  # Good default
   ```

4. **Validate Metadata**: Ensure embedding compatibility
5. **Review Proposals**: User should review before approval

## Common Issues

### Issue: Too Many Proposals

**Problem**: Too many similarity proposals

**Solutions**:
- Increase threshold (e.g., 0.8)
- Reduce max_proposals
- Improve embedding quality

### Issue: Missing Connections

**Problem**: Similar notes not connected

**Solutions**:
- Lower threshold
- Increase max_comparisons
- Check embedding quality

### Issue: Performance Issues

**Problem**: Slow proposal generation

**Solutions**:
- Reduce max_comparisons
- Use FAISS for faster similarity search
- Process in batches

## Future Enhancements

1. **Graph Traversal**: K-hop neighbor search
2. **Community Detection**: Find note clusters
3. **Centrality Metrics**: Identify important notes
4. **Temporal Edges**: Track connections over time
5. **Weighted Paths**: Find best paths between notes

## Summary

The `GraphService` provides:

- ✅ Wikilink edge creation
- ✅ Similarity proposal generation
- ✅ Efficient similarity computation
- ✅ Separation of concerns
- ✅ Scalable design

It's the component that builds the knowledge graph connecting related notes in the MiniMe memory system.

