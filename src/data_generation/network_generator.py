"""
Synthetic Optical Network Generator

Generates diverse network topologies with realistic optical parameters
for training ML-based routing algorithms.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple
import json


class OpticalNetworkGenerator:
    """Generate synthetic optical fiber networks with realistic parameters."""

    def __init__(self, seed: int = 42):
        """
        Initialize the network generator.

        Args:
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        self.seed = seed

        # Realistic optical network parameters
        self.fiber_attenuation = 0.25  # dB/km (typical for single-mode fiber)
        self.max_span_length = 120  # km (before regeneration needed)
        self.regenerator_cost = 50000  # USD
        self.fiber_cost_per_km = 20000  # USD/km

    def generate_random_geometric(
        self,
        n_nodes: int,
        radius: float = 0.3,
        area_size: float = 1000.0
    ) -> nx.Graph:
        """
        Generate a random geometric graph representing a metropolitan/regional network.

        Args:
            n_nodes: Number of nodes (network locations)
            radius: Connection radius (normalized)
            area_size: Geographic area in km

        Returns:
            NetworkX graph with optical parameters
        """
        # Generate random geometric graph
        G = nx.random_geometric_graph(n_nodes, radius, seed=self.seed)

        # Scale positions to realistic geographic area
        pos = nx.get_node_attributes(G, 'pos')
        scaled_pos = {node: (x * area_size, y * area_size) for node, (x, y) in pos.items()}
        nx.set_node_attributes(G, scaled_pos, 'pos')

        # Add optical parameters to edges
        self._add_optical_parameters(G)

        # Add node attributes
        self._add_node_attributes(G)

        return G

    def generate_scale_free(
        self,
        n_nodes: int,
        m: int = 3,
        area_size: float = 2000.0
    ) -> nx.Graph:
        """
        Generate a scale-free network using Barabási-Albert model.
        Represents backbone/long-haul networks with hub nodes.

        Args:
            n_nodes: Number of nodes
            m: Number of edges to attach from new node
            area_size: Geographic area in km

        Returns:
            NetworkX graph with optical parameters
        """
        # Generate scale-free network
        G = nx.barabasi_albert_graph(n_nodes, m, seed=self.seed)

        # Add random geographic positions
        pos = {i: (np.random.rand() * area_size, np.random.rand() * area_size)
               for i in G.nodes()}
        nx.set_node_attributes(G, pos, 'pos')

        # Add optical parameters
        self._add_optical_parameters(G)
        self._add_node_attributes(G)

        return G

    def generate_grid(
        self,
        rows: int,
        cols: int,
        spacing: float = 100.0
    ) -> nx.Graph:
        """
        Generate a grid network topology.
        Represents structured metro networks or data center interconnects.

        Args:
            rows: Number of rows
            cols: Number of columns
            spacing: Distance between adjacent nodes (km)

        Returns:
            NetworkX graph with optical parameters
        """
        # Generate 2D grid graph
        G = nx.grid_2d_graph(rows, cols)

        # Relabel nodes to integers
        mapping = {node: i for i, node in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping)

        # Create inverse mapping for positions
        inv_mapping = {v: k for k, v in mapping.items()}

        # Add positions based on grid coordinates
        pos = {node: (inv_mapping[node][1] * spacing, inv_mapping[node][0] * spacing)
               for node in G.nodes()}
        nx.set_node_attributes(G, pos, 'pos')

        # Add optical parameters
        self._add_optical_parameters(G)
        self._add_node_attributes(G)

        return G

    def _add_optical_parameters(self, G: nx.Graph) -> None:
        """
        Add realistic optical fiber parameters to edges.

        Args:
            G: NetworkX graph to modify
        """
        pos = nx.get_node_attributes(G, 'pos')

        for u, v in G.edges():
            # Calculate Euclidean distance
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            # Add random variation to distance (account for non-straight paths)
            distance *= np.random.uniform(1.1, 1.3)

            # Calculate signal attenuation
            attenuation = distance * self.fiber_attenuation

            # Add random fiber quality variation
            attenuation *= np.random.uniform(0.95, 1.05)

            # Calculate cost
            fiber_cost = distance * self.fiber_cost_per_km

            # Number of regenerators needed
            n_regenerators = int(np.ceil(distance / self.max_span_length)) - 1
            regenerator_cost = n_regenerators * self.regenerator_cost

            total_cost = fiber_cost + regenerator_cost

            # Add edge attributes
            G[u][v]['distance'] = round(distance, 2)
            G[u][v]['attenuation_db'] = round(attenuation, 3)
            G[u][v]['fiber_cost'] = round(fiber_cost, 2)
            G[u][v]['n_regenerators'] = n_regenerators
            G[u][v]['regenerator_cost'] = regenerator_cost
            G[u][v]['total_cost'] = round(total_cost, 2)

            # Current utilization (for dynamic routing scenarios)
            G[u][v]['utilization'] = round(np.random.uniform(0.1, 0.7), 2)

    def _add_node_attributes(self, G: nx.Graph) -> None:
        """
        Add node attributes (equipment, capacity).

        Args:
            G: NetworkX graph to modify
        """
        for node in G.nodes():
            # Node type (access, aggregation, core)
            degree = G.degree(node)
            if degree <= 2:
                node_type = 'access'
                capacity = 10  # Gbps
            elif degree <= 4:
                node_type = 'aggregation'
                capacity = 40  # Gbps
            else:
                node_type = 'core'
                capacity = 100  # Gbps

            G.nodes[node]['type'] = node_type
            G.nodes[node]['capacity_gbps'] = capacity
            G.nodes[node]['has_regenerator'] = True  # All nodes can regenerate

    def save_network(self, G: nx.Graph, filepath: str) -> None:
        """
        Save network to JSON file.

        Args:
            G: NetworkX graph
            filepath: Output file path
        """
        # Convert to node-link format
        data = nx.node_link_data(G)

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load_network(self, filepath: str) -> nx.Graph:
        """
        Load network from JSON file.

        Args:
            filepath: Input file path

        Returns:
            NetworkX graph
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        return nx.node_link_graph(data)

    def generate_diverse_dataset(
        self,
        n_networks: int = 100,
        output_dir: str = 'data/synthetic'
    ) -> List[nx.Graph]:
        """
        Generate a diverse set of network topologies for training.

        Args:
            n_networks: Number of networks to generate
            output_dir: Directory to save networks

        Returns:
            List of generated networks
        """
        networks = []

        for i in range(n_networks):
            # Vary network size and type
            n_nodes = np.random.randint(20, 101)
            topology_type = np.random.choice(['geometric', 'scale_free', 'grid'])

            if topology_type == 'geometric':
                radius = np.random.uniform(0.2, 0.4)
                area = np.random.uniform(500, 2000)
                G = self.generate_random_geometric(n_nodes, radius, area)
            elif topology_type == 'scale_free':
                m = np.random.randint(2, 5)
                area = np.random.uniform(1000, 3000)
                G = self.generate_scale_free(n_nodes, m, area)
            else:  # grid
                rows = np.random.randint(4, 11)
                cols = np.random.randint(4, 11)
                spacing = np.random.uniform(80, 150)
                G = self.generate_grid(rows, cols, spacing)

            # Store metadata
            G.graph['network_id'] = i
            G.graph['topology_type'] = topology_type
            G.graph['n_nodes'] = G.number_of_nodes()
            G.graph['n_edges'] = G.number_of_edges()

            # Save network
            self.save_network(G, f'{output_dir}/network_{i:04d}.json')
            networks.append(G)

            # Update seed for variety
            self.seed += 1
            np.random.seed(self.seed)

        print(f"Generated {n_networks} diverse networks in {output_dir}/")
        return networks


if __name__ == '__main__':
    # Example usage
    generator = OpticalNetworkGenerator(seed=42)

    # Generate example networks
    print("Generating example networks...")

    G1 = generator.generate_random_geometric(50, radius=0.3, area_size=1000)
    print(f"Random Geometric: {G1.number_of_nodes()} nodes, {G1.number_of_edges()} edges")

    G2 = generator.generate_scale_free(60, m=3, area_size=2000)
    print(f"Scale-Free: {G2.number_of_nodes()} nodes, {G2.number_of_edges()} edges")

    G3 = generator.generate_grid(7, 7, spacing=100)
    print(f"Grid: {G3.number_of_nodes()} nodes, {G3.number_of_edges()} edges")

    print("\nNetwork generation module ready!")
