"""
Create network topology visualization with route overlay.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D
from pathlib import Path

from src.data_generation.network_generator import OpticalNetworkGenerator
from src.routing.traditional_routing import TraditionalRouter

# Generate a sample network
generator = OpticalNetworkGenerator(seed=42)
G = generator.generate_random_geometric(30, radius=0.3, area_size=800)

# Find a route
router = TraditionalRouter()
source = 5
target = 20

result = router.dijkstra(G, source, target)

if result.success:
    # Create visualization
    fig, ax = plt.subplots(figsize=(14, 10))

    pos = nx.get_node_attributes(G, 'pos')
    path = result.path

    # Draw all edges
    nx.draw_networkx_edges(G, pos, edge_color='lightgray', width=1, alpha=0.5, ax=ax)

    # Draw path edges
    if len(path) > 1:
        path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
        nx.draw_networkx_edges(G, pos, edgelist=path_edges,
                              edge_color='red', width=3, alpha=0.8, ax=ax)

    # Draw nodes
    node_colors = []
    for node in G.nodes():
        if node == source:
            node_colors.append('green')
        elif node == target:
            node_colors.append('blue')
        elif node in path:
            node_colors.append('orange')
        else:
            node_colors.append('lightblue')

    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                          node_size=300, alpha=0.9, edgecolors='black', linewidths=1.5, ax=ax)

    # Draw labels for source, target, and path nodes
    labels_to_draw = {node: str(node) for node in [source, target] + path}
    nx.draw_networkx_labels(G, pos, labels_to_draw, font_size=8, font_weight='bold', ax=ax)

    # Add legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
               markersize=10, label='Source Node'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
               markersize=10, label='Target Node'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange',
               markersize=10, label='Path Node'),
        Line2D([0], [0], color='red', linewidth=3, label='Selected Route'),
        Line2D([0], [0], color='lightgray', linewidth=1, label='Other Links')
    ]
    ax.legend(handles=legend_elements, loc='upper left', framealpha=0.9)

    ax.set_title(f'Network Topology with Route from Node {source} to Node {target}\n'
                f'Cost: ${result.cost:,.0f} | Distance: {result.distance:.1f} km | Hops: {len(path)-1}',
                fontsize=14, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()

    # Save
    output_path = Path('results/plots/04_network_topology_example.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Network topology visualization created: {output_path}")
else:
    print("Failed to find route")
