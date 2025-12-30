# GraphVisualizer - Knowledge Graph Visualization

## Overview

`visualizer.py` provides the `GraphVisualizer` class, which creates interactive and static visualizations of the Obsidian vault knowledge graph. It enables users to see relationships between notes and understand the structure of their knowledge base.

## Purpose

The GraphVisualizer addresses the need to:
- **Visual Understanding**: See how notes are connected
- **Graph Exploration**: Discover relationships and clusters
- **Quality Assessment**: Identify isolated or highly connected notes
- **Presentation**: Share knowledge graph structure

## Key Components

### GraphVisualizer Class

```python
class GraphVisualizer:
    def __init__(self, db: AsyncDatabase)
    async def load_graph(include_proposals: bool = True) -> GraphData
    def export_html(output_path: str, title: str = "MiniMe Knowledge Graph") -> None
    def export_image(output_path: str, format: str = "png", dpi: int = 300) -> None
    def get_stats() -> dict
```

### Data Structures

```python
@dataclass
class NodeData:
    node_id: str
    title: str
    tags: List[str]
    domain: Optional[str]
    scope: str
    path: str
    degree: int = 0  # Number of connections

@dataclass
class EdgeData:
    source: str
    target: str
    edge_type: str
    weight: float
    confidence: float
    rationale: str
    is_proposal: bool = False

@dataclass
class GraphData:
    nodes: List[NodeData]
    edges: List[EdgeData]
```

## How It Works

### 1. Loading Graph Data

```python
async def load_graph(self, include_proposals: bool = True) -> GraphData:
    # Load nodes
    vault_nodes = await self.db.get_all_nodes()
    
    # Load edges (including proposals if requested)
    edges = await self.db.get_all_edges(include_proposals=include_proposals)
    
    # Build node data
    node_dict = {}
    for node in vault_nodes:
        node_dict[node.node_id] = NodeData(
            node_id=node.node_id,
            title=node.title or node.path,
            tags=node.tags or [],
            domain=node.domain,
            scope=node.scope,
            path=node.path,
        )
    
    # Count degrees (connections per node)
    for edge in edges:
        if edge.source_node_id in node_dict:
            node_dict[edge.source_node_id].degree += 1
        if edge.target_node_id in node_dict:
            node_dict[edge.target_node_id].degree += 1
    
    # Build edge data
    edge_list = []
    for edge in edges:
        is_proposal = not edge.is_approved or edge.edge_type == "similar"
        edge_list.append(EdgeData(
            source=edge.source_node_id,
            target=edge.target_node_id,
            edge_type=edge.edge_type,
            weight=edge.weight,
            confidence=edge.confidence,
            rationale=edge.rationale or "",
            is_proposal=is_proposal,
        ))
    
    return GraphData(
        nodes=list(node_dict.values()),
        edges=edge_list,
    )
```

**Features**:
- **Node Loading**: Gets all nodes from database
- **Edge Loading**: Includes proposals if requested
- **Degree Calculation**: Counts connections per node
- **Data Structure**: Creates clean data structures for visualization

### 2. HTML Export (Interactive)

Creates interactive HTML visualization using Plotly:

```python
def export_html(self, output_path: str, title: str = "MiniMe Knowledge Graph") -> None:
    # Build networkx graph for layout
    G = nx.Graph()
    
    # Add nodes and edges
    for node in self.graph_data.nodes:
        G.add_node(node.node_id, title=node.title, tags=node.tags, domain=node.domain)
    
    for edge in self.graph_data.edges:
        G.add_edge(edge.source, edge.target, weight=edge.weight, edge_type=edge.edge_type)
    
    # Compute force-directed layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Create Plotly figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=title,
            showlegend=False,
            hovermode="closest",
            ...
        ),
    )
    
    # Save HTML
    pyo.plot(fig, filename=str(output_file), auto_open=False)
```

**Features**:
- **Interactive**: Hover to see node/edge details
- **Force-Directed Layout**: Automatic positioning
- **Color Coding**: Different colors for domains
- **Size by Degree**: Larger nodes = more connections
- **Edge Styling**: Different styles for wikilinks vs similarity

### 3. Image Export (Static)

Creates static image using Matplotlib:

```python
def export_image(self, output_path: str, format: str = "png", dpi: int = 300) -> None:
    # Build networkx graph
    G = nx.Graph()
    for node in self.graph_data.nodes:
        G.add_node(node.node_id, title=node.title)
    for edge in self.graph_data.edges:
        G.add_edge(edge.source, edge.target, weight=edge.weight, edge_type=edge.edge_type)
    
    # Compute layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Draw edges (different styles for different types)
    wikilink_edges = [(e.source, e.target) for e in self.graph_data.edges if e.edge_type == "wikilink"]
    similar_edges = [(e.source, e.target) for e in self.graph_data.edges if e.edge_type == "similar"]
    
    nx.draw_networkx_edges(G, pos, edgelist=wikilink_edges, edge_color="gray", width=2, alpha=0.5)
    nx.draw_networkx_edges(G, pos, edgelist=similar_edges, edge_color="red", width=1, alpha=0.3, style="dashed")
    
    # Draw nodes
    node_sizes = [max(100, min(1000, 100 + node.degree * 50)) for node in self.graph_data.nodes]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="lightblue", alpha=0.7)
    
    # Draw labels
    labels = {node.node_id: node.title for node in self.graph_data.nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    # Save
    plt.savefig(output_file, format=format, dpi=dpi, bbox_inches="tight")
```

**Features**:
- **Static Image**: PNG, SVG, or PDF format
- **High Resolution**: Configurable DPI
- **Legend**: Shows edge type meanings
- **Publication Ready**: Suitable for documents

### 4. Graph Statistics

Provides insights about the graph:

```python
def get_stats(self) -> dict:
    nodes = self.graph_data.nodes
    edges = self.graph_data.edges
    
    # Count by type
    wikilink_count = sum(1 for e in edges if e.edge_type == "wikilink")
    similar_count = sum(1 for e in edges if e.edge_type == "similar")
    proposal_count = sum(1 for e in edges if e.is_proposal)
    
    # Node statistics
    total_degree = sum(node.degree for node in nodes)
    avg_degree = total_degree / len(nodes) if nodes else 0
    max_degree = max((node.degree for node in nodes), default=0)
    hub_nodes = [node.title for node in nodes if node.degree >= max_degree * 0.7]
    
    # Domain breakdown
    domains = {}
    for node in nodes:
        domain = node.domain or "none"
        domains[domain] = domains.get(domain, 0) + 1
    
    return {
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "wikilink_edges": wikilink_count,
        "similarity_edges": similar_count,
        "pending_proposals": proposal_count,
        "average_connections": round(avg_degree, 2),
        "max_connections": max_degree,
        "hub_nodes": hub_nodes[:5],
        "domains": domains,
    }
```

**Statistics Provided**:
- Total nodes and edges
- Edge type breakdown
- Average and max connections
- Hub nodes (highly connected)
- Domain distribution

## Usage Examples

### Basic Usage

```python
from minime.memory.visualizer import GraphVisualizer
from minime.memory.db import AsyncDatabase

# Initialize
db = AsyncDatabase("./data/minime.db")
visualizer = GraphVisualizer(db)

# Load graph
graph_data = await visualizer.load_graph(include_proposals=True)

# Export HTML
visualizer.export_html("./output/graph.html", title="My Knowledge Graph")

# Export image
visualizer.export_image("./output/graph.png", format="png", dpi=300)

# Get statistics
stats = visualizer.get_stats()
print(f"Total nodes: {stats['total_nodes']}")
print(f"Average connections: {stats['average_connections']}")
```

### With Filtering

```python
# Load without proposals
graph_data = await visualizer.load_graph(include_proposals=False)

# Only approved edges will be shown
visualizer.export_html("./output/graph_approved.html")
```

### Statistics Analysis

```python
stats = visualizer.get_stats()

print("Graph Statistics:")
print(f"  Nodes: {stats['total_nodes']}")
print(f"  Edges: {stats['total_edges']}")
print(f"  Wikilinks: {stats['wikilink_edges']}")
print(f"  Similarity: {stats['similarity_edges']}")
print(f"  Pending Proposals: {stats['pending_proposals']}")
print(f"  Average Connections: {stats['average_connections']}")
print(f"  Hub Nodes: {', '.join(stats['hub_nodes'])}")
print(f"  Domains: {stats['domains']}")
```

## Visualization Features

### Node Styling

- **Size**: Based on degree (more connections = larger)
- **Color**: Based on domain (different colors per domain)
- **Label**: Node title
- **Hover**: Shows path, tags, domain, connections

### Edge Styling

- **Wikilinks**: Solid gray lines (explicit connections)
- **Similarity**: Dashed red lines (proposed connections)
- **Width**: Based on weight/confidence
- **Hover**: Shows edge type, confidence, rationale

### Layout Algorithm

Uses force-directed layout (spring layout):
- **Repulsion**: Nodes repel each other
- **Attraction**: Connected nodes attract
- **Iterations**: 50 iterations for convergence
- **Seed**: Fixed seed for reproducibility

## Integration with CLI

Can be used from command line:

```python
# In cli.py
@app.command()
def graph_visualize(
    output: str = typer.Option("./graph.html", help="Output file path"),
    format: str = typer.Option("html", help="Format: html or png"),
    include_proposals: bool = typer.Option(True, help="Include pending proposals"),
):
    db = AsyncDatabase(config.db_path)
    visualizer = GraphVisualizer(db)
    
    await visualizer.load_graph(include_proposals=include_proposals)
    
    if format == "html":
        visualizer.export_html(output)
    else:
        visualizer.export_image(output, format=format)
    
    typer.echo(f"Graph exported to {output}")
```

## Best Practices

1. **Include Proposals**: See full graph including pending connections
2. **High DPI**: Use 300 DPI for publication-quality images
3. **Regular Updates**: Regenerate visualizations as graph grows
4. **Statistics**: Use stats to understand graph health
5. **Interactive HTML**: Better for exploration than static images

## Common Issues

### Issue: Empty Graph

**Problem**: No nodes or edges shown

**Solutions**:
- Check if vault is indexed
- Verify database has data
- Check include_proposals setting

### Issue: Overcrowded Visualization

**Problem**: Too many nodes to see clearly

**Solutions**:
- Filter by domain
- Use larger output size
- Increase node spacing (k parameter)
- Filter by degree (show only highly connected nodes)

### Issue: Missing Dependencies

**Problem**: ImportError for plotly or matplotlib

**Solutions**:
```bash
pip install plotly  # For HTML export
pip install matplotlib  # For image export
```

## Future Enhancements

1. **Interactive Filtering**: Filter nodes/edges in HTML
2. **Cluster Detection**: Identify communities
3. **Temporal Visualization**: Show graph evolution
4. **3D Visualization**: Three-dimensional layout
5. **Export Formats**: JSON, GraphML, GEXF

## Summary

The `GraphVisualizer` provides:

- ✅ Interactive HTML visualizations
- ✅ Static image exports
- ✅ Graph statistics and insights
- ✅ Force-directed layouts
- ✅ Styling by node/edge properties

It's the component that enables users to visualize and understand their knowledge graph structure in MiniMe.

