# Visualization Scripts Documentation

## Overview

This document describes the visualization scripts that help you understand and explore your knowledge graph.

## Visualization Scripts

### 1. viz_graph.py

**Purpose**: Generate graph visualizations from your memory database

**Features**:
- Interactive HTML visualization
- Static image export (PNG)
- Graph statistics
- Configurable options

**Usage**:

#### Generate HTML Visualization

```bash
python scripts/viz_graph.py generate
```

**Options**:
```bash
python scripts/viz_graph.py generate \
  --db-path "memory.db" \
  --output "graph.html" \
  --format "html" \
  --include-proposals \
  --open-browser  # Open in browser after generation
```

#### Generate PNG Image

```bash
python scripts/viz_graph.py generate \
  --output "graph.png" \
  --format "png"
```

#### Show Statistics Only

```bash
python scripts/viz_graph.py stats --db-path "memory.db"
```

---

## Commands

### generate

Generates graph visualization from database.

**Options**:
- `--db-path`: Path to SQLite database (default: `memory.db`)
- `--output`: Output file path (default: `graph.html`)
- `--format`: Output format - `html` or `png` (default: `html`)
- `--include-proposals`: Include pending similarity proposals (default: `True`)
- `--open-browser`: Open HTML in browser after generation (default: `False`)

**Example**:
```bash
python scripts/viz_graph.py generate \
  --db-path "./data/minime.db" \
  --output "./output/my_graph.html" \
  --format "html" \
  --include-proposals \
  --open-browser
```

**Output**:
```
Loading graph data...
[OK] Loaded 15 nodes and 23 edges

Generating HTML visualization...
[OK] Saved to: /path/to/graph.html

Graph Statistics:
  Nodes: 15
  Edges: 23 (Wikilinks: 8, Similarity: 15)
  Pending Proposals: 5
  Avg Connections: 1.5
  Hub Nodes: Machine Learning, Python Basics
```

### stats

Shows graph statistics without generating visualization.

**Options**:
- `--db-path`: Path to SQLite database (default: `memory.db`)

**Example**:
```bash
python scripts/viz_graph.py stats --db-path "memory.db"
```

**Output**:
```
Graph Statistics

Nodes: 15
Edges: 23
  - Wikilinks: 8
  - Similarity: 15
  - Pending Proposals: 5

Connections:
  - Average: 1.5
  - Max: 5

Hub Nodes (most connected):
  - Machine Learning Basics
  - Python Programming

Domains:
  - coding: 10
  - ai: 5
```

---

## Visualization Features

### HTML Visualization

**Interactive Features**:
- **Hover**: See node/edge details
- **Zoom**: Mouse wheel or pinch
- **Pan**: Click and drag
- **Node Selection**: Click nodes to highlight

**Visual Elements**:
- **Node Size**: Based on number of connections (degree)
- **Node Color**: Based on domain
- **Edge Style**: 
  - Solid gray = Wikilinks (explicit)
  - Dashed red = Similarity (proposed)
- **Edge Width**: Based on weight/confidence

**Layout**:
- Force-directed layout (spring layout)
- Automatic positioning
- Reproducible (fixed seed)

### PNG Image

**Features**:
- High resolution (300 DPI default)
- Publication-ready
- Static (no interaction)
- Smaller file size

**Use Cases**:
- Documentation
- Presentations
- Reports
- Sharing

---

## Graph Statistics Explained

### Node Statistics

- **Total Nodes**: Number of notes in vault
- **Hub Nodes**: Most connected notes (top 5)
- **Average Connections**: Mean connections per node
- **Max Connections**: Most connected single node

### Edge Statistics

- **Total Edges**: All connections
- **Wikilinks**: Explicit `[[link]]` connections
- **Similarity**: Proposed semantic connections
- **Pending Proposals**: Unapproved similarity edges

### Domain Breakdown

Shows distribution of notes by domain:
- `coding`: Programming-related notes
- `ai`: AI/ML-related notes
- `none`: No domain specified

---

## Usage Examples

### Basic Visualization

```bash
# Generate HTML from default database
python scripts/viz_graph.py generate
```

### Custom Database

```bash
# Use different database
python scripts/viz_graph.py generate --db-path "./data/my_vault.db"
```

### High-Resolution Image

```bash
# Generate PNG for documentation
python scripts/viz_graph.py generate \
  --output "graph.png" \
  --format "png"
```

### Without Proposals

```bash
# Show only approved edges
python scripts/viz_graph.py generate --no-include-proposals
```

### Quick Statistics

```bash
# Just see stats, no visualization
python scripts/viz_graph.py stats
```

---

## Integration with Other Scripts

### After Indexing

```bash
# Index vault
python scripts/demo_memory.py run

# Visualize results
python scripts/viz_graph.py generate --open-browser
```

### After Search

```bash
# Search for content
python scripts/memory_explorer.py search "machine learning"

# See how results are connected
python scripts/viz_graph.py generate
```

### Regular Updates

```bash
# Regenerate visualization after adding notes
python scripts/viz_graph.py generate --output "graph_$(date +%Y%m%d).html"
```

---

## Customization

### Output Location

```bash
# Save to specific directory
python scripts/viz_graph.py generate --output "./outputs/graph.html"
```

### Include/Exclude Proposals

```bash
# Include pending proposals (default)
python scripts/viz_graph.py generate --include-proposals

# Exclude proposals (only approved edges)
python scripts/viz_graph.py generate --no-include-proposals
```

### Browser Auto-Open

```bash
# Automatically open in browser
python scripts/viz_graph.py generate --open-browser
```

---

## Troubleshooting

### No Nodes Found

**Problem**: "No nodes found in database"

**Solutions**:
- Index your vault first: `python scripts/demo_memory.py run`
- Check database path is correct
- Verify database has data

### Missing Dependencies

**Problem**: ImportError for plotly or matplotlib

**Solutions**:
```bash
# For HTML export
pip install plotly

# For PNG export
pip install matplotlib
```

### Empty Graph

**Problem**: Graph shows no edges

**Solutions**:
- Add wikilinks to notes: `[[Other Note]]`
- Lower similarity threshold (in config)
- Wait for proposals to be generated
- Use `--include-proposals` flag

### Large Graph Performance

**Problem**: Slow rendering with many nodes

**Solutions**:
- Use PNG format (faster)
- Filter by domain (future feature)
- Increase node spacing (future feature)

---

## Best Practices

1. **Regular Updates**: Regenerate after indexing new notes
2. **Include Proposals**: See full graph structure
3. **Save Versions**: Keep snapshots of graph evolution
4. **Review Statistics**: Understand graph health
5. **Share HTML**: Interactive visualization is best for exploration

---

## Future Enhancements

Planned features:
- Filter by domain
- Filter by tags
- Time-based visualization
- 3D layout option
- Export to GraphML/GEXF
- Custom color schemes

---

## Summary

The visualization script provides:
- ✅ Interactive HTML visualizations
- ✅ Static image exports
- ✅ Graph statistics
- ✅ Configurable options
- ✅ Easy integration

Use it to explore and understand your knowledge graph!

