"""
Real Network Topology Parser

Downloads and parses real network topologies from SNDlib and Topology Zoo.
Augments them with realistic optical fiber parameters.
"""

import requests
import networkx as nx
import numpy as np
import xml.etree.ElementTree as ET
from typing import Dict, List
import json
import os


class RealTopologyParser:
    """Parse and augment real network topologies."""

    def __init__(self, seed: int = 42):
        """
        Initialize parser.

        Args:
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        self.seed = seed

        # SNDlib base URL
        self.sndlib_base_url = "http://sndlib.zib.de/download"

        # Optical parameters (same as synthetic networks)
        self.fiber_attenuation = 0.25  # dB/km
        self.max_span_length = 120  # km
        self.regenerator_cost = 50000  # USD
        self.fiber_cost_per_km = 20000  # USD/km

        # Well-known SNDlib networks (name: expected nodes)
        self.sndlib_networks = {
            'abilene': 12,
            'atlanta': 15,
            'di': 11,
            'france': 25,
            'geant': 22,
            'germany50': 50,
            'nobel-germany': 17,
            'nobel-us': 14,
            'norway': 27,
            'polska': 12,
            'ta1': 24,
            'ta2': 65
        }

    def download_sndlib_network(
        self,
        network_name: str,
        output_dir: str = 'data/raw'
    ) -> str:
        """
        Download network topology from SNDlib.

        Args:
            network_name: Name of the network
            output_dir: Directory to save file

        Returns:
            Path to downloaded file
        """
        os.makedirs(output_dir, exist_ok=True)

        url = f"{self.sndlib_base_url}/{network_name}.xml"
        filepath = f"{output_dir}/{network_name}.xml"

        try:
            print(f"Downloading {network_name} from SNDlib...")
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            with open(filepath, 'wb') as f:
                f.write(response.content)

            print(f"  Saved to {filepath}")
            return filepath

        except requests.exceptions.RequestException as e:
            print(f"  Failed to download {network_name}: {e}")
            return None

    def parse_sndlib_xml(self, filepath: str) -> nx.Graph:
        """
        Parse SNDlib XML format network file.

        Args:
            filepath: Path to XML file

        Returns:
            NetworkX graph
        """
        try:
            tree = ET.parse(filepath)
            root = tree.getroot()

            # Extract namespace
            ns = {'snd': 'http://sndlib.zib.de/network'}

            G = nx.Graph()

            # Parse nodes
            nodes = root.findall('.//snd:node', ns)
            node_coords = {}

            for node in nodes:
                node_id = node.get('id')

                # Try to get coordinates
                coords = node.find('snd:coordinates', ns)
                if coords is not None:
                    x = float(coords.find('snd:x', ns).text)
                    y = float(coords.find('snd:y', ns).text)
                    node_coords[node_id] = (x, y)
                else:
                    # Random position if not provided
                    node_coords[node_id] = (
                        np.random.rand() * 1000,
                        np.random.rand() * 1000
                    )

                G.add_node(node_id, pos=node_coords[node_id])

            # Parse links
            links = root.findall('.//snd:link', ns)

            for link in links:
                link_id = link.get('id')
                source = link.find('snd:source', ns).text
                target = link.find('snd:target', ns).text

                # Get preinstalled capacity if available
                setup = link.find('.//snd:preInstalledModule', ns)
                capacity = 10  # Default
                if setup is not None:
                    capacity = float(setup.find('snd:capacity', ns).text)

                G.add_edge(source, target, link_id=link_id, capacity=capacity)

            print(f"  Parsed: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            return G

        except Exception as e:
            print(f"  Error parsing {filepath}: {e}")
            return None

    def create_simple_topology(self, network_name: str) -> nx.Graph:
        """
        Create a simplified topology based on well-known network structure.
        Used as fallback when SNDlib download fails.

        Args:
            network_name: Name of network

        Returns:
            NetworkX graph
        """
        print(f"Creating simplified {network_name} topology...")

        if network_name == 'abilene':
            # Abilene Network (12 nodes, US backbone)
            G = nx.Graph()
            nodes = ['Seattle', 'Sunnyvale', 'Los Angeles', 'Denver', 'Kansas City',
                     'Houston', 'Atlanta', 'Indianapolis', 'Chicago', 'New York',
                     'Washington DC', 'Salt Lake City']

            # Approximate geographic positions (normalized)
            positions = {
                'Seattle': (50, 950), 'Sunnyvale': (100, 800), 'Los Angeles': (120, 700),
                'Denver': (450, 800), 'Kansas City': (600, 700), 'Houston': (650, 500),
                'Atlanta': (850, 550), 'Indianapolis': (750, 700), 'Chicago': (700, 800),
                'New York': (950, 800), 'Washington DC': (900, 700), 'Salt Lake City': (300, 750)
            }

            edges = [
                ('Seattle', 'Sunnyvale'), ('Seattle', 'Denver'),
                ('Sunnyvale', 'Los Angeles'), ('Sunnyvale', 'Denver'),
                ('Los Angeles', 'Houston'), ('Denver', 'Kansas City'),
                ('Denver', 'Salt Lake City'), ('Kansas City', 'Indianapolis'),
                ('Kansas City', 'Houston'), ('Houston', 'Atlanta'),
                ('Indianapolis', 'Chicago'), ('Indianapolis', 'Atlanta'),
                ('Chicago', 'New York'), ('Atlanta', 'Washington DC'),
                ('New York', 'Washington DC')
            ]

        elif network_name == 'geant':
            # GÉANT Network (simplified European research network)
            G = nx.Graph()
            nodes = ['London', 'Amsterdam', 'Hamburg', 'Copenhagen', 'Stockholm',
                     'Paris', 'Geneva', 'Milan', 'Vienna', 'Prague', 'Berlin',
                     'Madrid', 'Barcelona', 'Rome', 'Athens', 'Budapest',
                     'Warsaw', 'Zagreb', 'Bratislava', 'Dublin', 'Oslo', 'Luxembourg']

            # Approximate European coordinates (scaled)
            positions = {
                'London': (200, 700), 'Amsterdam': (300, 750), 'Hamburg': (400, 800),
                'Copenhagen': (450, 850), 'Stockholm': (500, 950), 'Paris': (250, 650),
                'Geneva': (350, 600), 'Milan': (400, 550), 'Vienna': (550, 650),
                'Prague': (500, 700), 'Berlin': (500, 750), 'Madrid': (100, 450),
                'Barcelona': (250, 500), 'Rome': (450, 480), 'Athens': (700, 400),
                'Budapest': (600, 650), 'Warsaw': (600, 750), 'Zagreb': (550, 580),
                'Bratislava': (570, 680), 'Dublin': (100, 750), 'Oslo': (400, 950),
                'Luxembourg': (320, 700)
            }

            # Create scale-free connectivity
            edges = [
                ('London', 'Amsterdam'), ('London', 'Paris'), ('London', 'Dublin'),
                ('Amsterdam', 'Hamburg'), ('Amsterdam', 'Paris'), ('Amsterdam', 'Luxembourg'),
                ('Hamburg', 'Berlin'), ('Hamburg', 'Copenhagen'), ('Copenhagen', 'Stockholm'),
                ('Copenhagen', 'Oslo'), ('Paris', 'Geneva'), ('Paris', 'Madrid'),
                ('Paris', 'Barcelona'), ('Geneva', 'Milan'), ('Milan', 'Vienna'),
                ('Milan', 'Rome'), ('Vienna', 'Prague'), ('Vienna', 'Budapest'),
                ('Vienna', 'Bratislava'), ('Prague', 'Berlin'), ('Berlin', 'Warsaw'),
                ('Budapest', 'Zagreb'), ('Barcelona', 'Madrid'), ('Rome', 'Athens')
            ]

        else:
            # Generic small network
            n = self.sndlib_networks.get(network_name, 20)
            G = nx.barabasi_albert_graph(n, 3, seed=self.seed)
            nodes = list(G.nodes())
            positions = {i: (np.random.rand() * 1000, np.random.rand() * 1000)
                        for i in nodes}
            edges = list(G.edges())

        # Build graph
        for node in nodes:
            G.add_node(node, pos=positions.get(node, (0, 0)))

        for u, v in edges:
            G.add_edge(u, v)

        return G

    def augment_with_optical_parameters(self, G: nx.Graph) -> nx.Graph:
        """
        Add realistic optical parameters to a topology.

        Args:
            G: NetworkX graph

        Returns:
            Augmented graph
        """
        pos = nx.get_node_attributes(G, 'pos')

        for u, v in G.edges():
            # Calculate distance from coordinates
            if u in pos and v in pos:
                x1, y1 = pos[u]
                x2, y2 = pos[v]
                distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            else:
                distance = np.random.uniform(50, 500)  # Default range

            # Add realism variation
            distance *= np.random.uniform(1.1, 1.3)

            # Calculate optical parameters (same as synthetic)
            attenuation = distance * self.fiber_attenuation
            attenuation *= np.random.uniform(0.95, 1.05)

            fiber_cost = distance * self.fiber_cost_per_km
            n_regenerators = int(np.ceil(distance / self.max_span_length)) - 1
            n_regenerators = max(0, n_regenerators)
            regenerator_cost = n_regenerators * self.regenerator_cost
            total_cost = fiber_cost + regenerator_cost

            # Set edge attributes
            G[u][v]['distance'] = round(distance, 2)
            G[u][v]['attenuation_db'] = round(attenuation, 3)
            G[u][v]['fiber_cost'] = round(fiber_cost, 2)
            G[u][v]['n_regenerators'] = n_regenerators
            G[u][v]['regenerator_cost'] = regenerator_cost
            G[u][v]['total_cost'] = round(total_cost, 2)
            G[u][v]['utilization'] = round(np.random.uniform(0.1, 0.7), 2)

        # Add node attributes
        for node in G.nodes():
            degree = G.degree(node)
            if degree <= 2:
                node_type, capacity = 'access', 10
            elif degree <= 4:
                node_type, capacity = 'aggregation', 40
            else:
                node_type, capacity = 'core', 100

            G.nodes[node]['type'] = node_type
            G.nodes[node]['capacity_gbps'] = capacity
            G.nodes[node]['has_regenerator'] = True

        return G

    def load_real_networks(
        self,
        network_names: List[str],
        output_dir: str = 'data/raw'
    ) -> Dict[str, nx.Graph]:
        """
        Load multiple real network topologies.

        Args:
            network_names: List of network names
            output_dir: Directory for saving/loading

        Returns:
            Dictionary of network_name -> graph
        """
        networks = {}

        for name in network_names:
            print(f"\nProcessing {name}...")

            # Try to download from SNDlib
            filepath = self.download_sndlib_network(name, output_dir)

            if filepath and os.path.exists(filepath):
                G = self.parse_sndlib_xml(filepath)
            else:
                G = None

            # Fallback to simplified topology
            if G is None:
                G = self.create_simple_topology(name)

            # Augment with optical parameters
            G = self.augment_with_optical_parameters(G)

            # Store metadata
            G.graph['network_name'] = name
            G.graph['source'] = 'sndlib' if filepath else 'simplified'
            G.graph['n_nodes'] = G.number_of_nodes()
            G.graph['n_edges'] = G.number_of_edges()

            networks[name] = G

            # Save as JSON
            json_path = f"{output_dir}/{name}.json"
            nx.write_json(G, json_path)
            print(f"  Saved to {json_path}")

        return networks


if __name__ == '__main__':
    # Example usage
    parser = RealTopologyParser(seed=42)

    # Load selected real networks
    network_names = ['abilene', 'geant', 'germany50', 'nobel-us']

    print("Loading real network topologies...")
    networks = parser.load_real_networks(network_names)

    print(f"\n{len(networks)} networks loaded successfully!")
    for name, G in networks.items():
        print(f"  {name}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
