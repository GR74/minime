"""Graph visualization for Obsidian vault knowledge graph."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import networkx as nx

from minime.memory.db import AsyncDatabase
from minime.schemas import GraphEdge, VaultNode


@dataclass
class NodeData:
    """Node data for visualization."""
    node_id: str
    title: str
    tags: List[str]
    domain: Optional[str]
    scope: str
    path: str
    degree: int = 0  # Number of connections


@dataclass
class EdgeData:
    """Edge data for visualization."""
    source: str
    target: str
    edge_type: str
    weight: float
    confidence: float
    rationale: str
    is_proposal: bool = False


@dataclass
class GraphData:
    """Complete graph data for visualization."""
    nodes: List[NodeData]
    edges: List[EdgeData]


class GraphVisualizer:
    """Visualize the Obsidian vault knowledge graph."""

    def __init__(self, db: AsyncDatabase):
        """
        Initialize GraphVisualizer.

        Args:
            db: AsyncDatabase instance
        """
        self.db = db
        self.graph_data: Optional[GraphData] = None

    async def load_graph(self, include_proposals: bool = True) -> GraphData:
        """
        Load graph data from database.

        Args:
            include_proposals: If True, include pending similarity proposals

        Returns:
            GraphData object with nodes and edges
        """
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
            # Check if it's a proposal
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

        self.graph_data = GraphData(
            nodes=list(node_dict.values()),
            edges=edge_list,
        )

        return self.graph_data

    def export_html(self, output_path: str, title: str = "MiniMe Knowledge Graph") -> None:
        """
        Export interactive HTML visualization using plotly.

        Args:
            output_path: Path to save HTML file
            title: Title for the visualization
        """
        if not self.graph_data:
            raise ValueError("Graph data not loaded. Call load_graph() first.")

        try:
            import plotly.graph_objects as go
            import plotly.offline as pyo
        except ImportError:
            raise ImportError("plotly is required for HTML export. Install with: pip install plotly")

        # Build networkx graph for layout
        G = nx.Graph()
        
        # Add nodes
        node_positions = {}
        for node in self.graph_data.nodes:
            G.add_node(node.node_id, title=node.title, tags=node.tags, domain=node.domain)
            node_positions[node.node_id] = node

        # Add edges with weights
        for edge in self.graph_data.edges:
            G.add_edge(
                edge.source,
                edge.target,
                weight=edge.weight,
                edge_type=edge.edge_type,
                confidence=edge.confidence,
                is_proposal=edge.is_proposal,
            )

        # Compute force-directed layout
        if len(G.nodes) > 0:
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        else:
            pos = {}

        # Extract node positions
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        node_color = []
        node_info = []

        for node in self.graph_data.nodes:
            if node.node_id in pos:
                x, y = pos[node.node_id]
                node_x.append(x)
                node_y.append(y)
                node_text.append(node.title)
                # Size based on degree (connections)
                node_size.append(max(10, min(30, 10 + node.degree * 2)))
                # Color based on domain or default
                domain_colors = {
                    "ai-memory": "#FF6B6B",
                    "ephemeral": "#95E1D3",
                    "global": "#F38181",
                }
                color = domain_colors.get(node.domain, "#95A5A6") if node.domain else "#95A5A6"
                node_color.append(color)
                
                # Hover info
                info = f"<b>{node.title}</b><br>"
                info += f"Path: {node.path}<br>"
                if node.tags:
                    info += f"Tags: {', '.join(node.tags)}<br>"
                if node.domain:
                    info += f"Domain: {node.domain}<br>"
                info += f"Connections: {node.degree}"
                node_info.append(info)

        # Extract edge positions
        edge_x = []
        edge_y = []
        edge_info = []
        edge_colors = []
        edge_widths = []
        edge_styles = []

        for edge in self.graph_data.edges:
            if edge.source in pos and edge.target in pos:
                x0, y0 = pos[edge.source]
                x1, y1 = pos[edge.target]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

                # Edge styling
                if edge.edge_type == "wikilink":
                    color = "rgba(50, 50, 50, 0.5)"
                    width = 2
                    style = "solid"
                elif edge.edge_type == "similar":
                    # Color by confidence (red=low, green=high)
                    confidence = edge.confidence
                    r = int(255 * (1 - confidence))
                    g = int(255 * confidence)
                    color = f"rgba({r}, {g}, 0, 0.4)"
                    width = 1
                    style = "dash"
                else:
                    color = "rgba(150, 150, 150, 0.3)"
                    width = 1
                    style = "solid"

                edge_colors.append(color)
                edge_widths.append(width)
                edge_styles.append(style)

                # Edge hover info
                info = f"<b>{edge.edge_type}</b><br>"
                if edge.confidence < 1.0:
                    info += f"Confidence: {edge.confidence:.2f}<br>"
                if edge.rationale:
                    info += f"Reason: {edge.rationale[:50]}..."
                edge_info.append(info)

        # Create edge traces
        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=1, color="rgba(125, 125, 125, 0.5)"),
            hoverinfo="none",
            mode="lines",
        )

        # Create node trace
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="middle center",
            textfont=dict(size=10, color="black"),
            hovertext=node_info,
            hoverinfo="text",
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=2, color="white"),
            ),
        )

        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=dict(
                    text=title,
                    x=0.5,
                    font=dict(size=20),
                ),
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[
                    dict(
                        text="Nodes = Notes | Solid lines = Wikilinks | Dashed lines = Similarity",
                        showarrow=False,
                        xref="paper",
                        yref="paper",
                        x=0.005,
                        y=-0.002,
                        xanchor="left",
                        yanchor="bottom",
                        font=dict(size=12, color="#888"),
                    )
                ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            ),
        )

        # Save HTML
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        pyo.plot(fig, filename=str(output_file), auto_open=False)

    def export_image(self, output_path: str, format: str = "png", dpi: int = 300) -> None:
        """
        Export static image visualization.

        Args:
            output_path: Path to save image file
            format: Image format (png, svg, pdf)
            dpi: Resolution for raster formats
        """
        if not self.graph_data:
            raise ValueError("Graph data not loaded. Call load_graph() first.")

        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            raise ImportError("matplotlib is required for image export. Install with: pip install matplotlib")

        # Build networkx graph
        G = nx.Graph()
        
        for node in self.graph_data.nodes:
            G.add_node(node.node_id, title=node.title)

        for edge in self.graph_data.edges:
            G.add_edge(edge.source, edge.target, weight=edge.weight, edge_type=edge.edge_type)

        # Compute layout
        if len(G.nodes) > 0:
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        else:
            pos = {}

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))

        # Draw edges
        wikilink_edges = [(e.source, e.target) for e in self.graph_data.edges if e.edge_type == "wikilink"]
        similar_edges = [(e.source, e.target) for e in self.graph_data.edges if e.edge_type == "similar"]

        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=wikilink_edges,
            edge_color="gray",
            width=2,
            alpha=0.5,
            ax=ax,
        )

        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=similar_edges,
            edge_color="red",
            width=1,
            alpha=0.3,
            style="dashed",
            ax=ax,
        )

        # Draw nodes
        node_sizes = [max(100, min(1000, 100 + node.degree * 50)) for node in self.graph_data.nodes]
        nx.draw_networkx_nodes(
            G,
            pos,
            node_size=node_sizes,
            node_color="lightblue",
            alpha=0.7,
            ax=ax,
        )

        # Draw labels
        labels = {node.node_id: node.title for node in self.graph_data.nodes}
        nx.draw_networkx_labels(
            G,
            pos,
            labels,
            font_size=8,
            ax=ax,
        )

        ax.set_title("MiniMe Knowledge Graph", fontsize=16, pad=20)
        ax.axis("off")

        # Add legend
        wikilink_patch = mpatches.Patch(color="gray", label="Wikilinks")
        similar_patch = mpatches.Patch(color="red", label="Similarity", linestyle="--")
        ax.legend(handles=[wikilink_patch, similar_patch], loc="upper right")

        # Save
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, format=format, dpi=dpi, bbox_inches="tight")
        plt.close()

    def get_stats(self) -> dict:
        """
        Get graph statistics.

        Returns:
            Dictionary with graph statistics
        """
        if not self.graph_data:
            raise ValueError("Graph data not loaded. Call load_graph() first.")

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
            "hub_nodes": hub_nodes[:5],  # Top 5
            "domains": domains,
        }

