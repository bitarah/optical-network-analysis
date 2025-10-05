"""
Dataset Builder

Generates training datasets for ML algorithms by:
1. Creating network topologies (synthetic + real)
2. Generating source-destination pairs
3. Computing optimal routes using traditional algorithms (labels)
4. Extracting features for ML training
"""

import numpy as np
import networkx as nx
from typing import List, Dict, Tuple
import json
import os
from tqdm import tqdm
import pandas as pd

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_generation.network_generator import OpticalNetworkGenerator
from data_generation.real_topology_parser import RealTopologyParser
from routing.traditional_routing import TraditionalRouter
from optical.signal_quality import OpticalSignalAnalyzer


class DatasetBuilder:
    """Build comprehensive training datasets for ML routing."""

    def __init__(self, seed: int = 42):
        """
        Initialize dataset builder.

        Args:
            seed: Random seed
        """
        np.random.seed(seed)
        self.seed = seed
        self.network_generator = OpticalNetworkGenerator(seed)
        self.topology_parser = RealTopologyParser(seed)
        self.router = TraditionalRouter()
        self.signal_analyzer = OpticalSignalAnalyzer()

    def generate_route_problems(
        self,
        G: nx.Graph,
        n_problems: int = 100
    ) -> List[Tuple[int, int]]:
        """
        Generate source-destination pairs for routing problems.

        Args:
            G: NetworkX graph
            n_problems: Number of routing problems to generate

        Returns:
            List of (source, target) tuples
        """
        nodes = list(G.nodes())
        problems = []

        for _ in range(n_problems):
            source, target = np.random.choice(nodes, size=2, replace=False)
            problems.append((source, target))

        return problems

    def extract_path_features(
        self,
        G: nx.Graph,
        source: int,
        target: int,
        path: List[int]
    ) -> Dict:
        """
        Extract features from a routing problem and solution.

        Args:
            G: NetworkX graph
            source: Source node
            target: Target node
            path: Computed path

        Returns:
            Feature dictionary
        """
        features = {}

        # Network-level features
        features['n_nodes'] = G.number_of_nodes()
        features['n_edges'] = G.number_of_edges()
        features['avg_degree'] = np.mean([d for _, d in G.degree()])
        features['network_density'] = nx.density(G)

        # Source-destination features
        features['source_degree'] = G.degree(source)
        features['target_degree'] = G.degree(target)

        # Geographic features
        pos = nx.get_node_attributes(G, 'pos')
        if source in pos and target in pos:
            x1, y1 = pos[source]
            x2, y2 = pos[target]
            features['euclidean_distance'] = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        else:
            features['euclidean_distance'] = 0

        # Shortest path length (hops)
        try:
            features['shortest_path_hops'] = nx.shortest_path_length(G, source, target)
        except:
            features['shortest_path_hops'] = -1

        # Average edge costs on shortest path
        if len(path) >= 2:
            edge_costs = [G[path[i]][path[i+1]]['total_cost']
                         for i in range(len(path)-1)]
            features['avg_edge_cost'] = np.mean(edge_costs)
            features['max_edge_cost'] = np.max(edge_costs)
            features['min_edge_cost'] = np.min(edge_costs)
        else:
            features['avg_edge_cost'] = 0
            features['max_edge_cost'] = 0
            features['min_edge_cost'] = 0

        # Betweenness centrality of source/target
        try:
            betweenness = nx.betweenness_centrality(G, k=min(20, G.number_of_nodes()))
            features['source_betweenness'] = betweenness.get(source, 0)
            features['target_betweenness'] = betweenness.get(target, 0)
        except:
            features['source_betweenness'] = 0
            features['target_betweenness'] = 0

        # Number of alternative paths
        try:
            alt_paths = list(nx.all_simple_paths(G, source, target, cutoff=features['shortest_path_hops']+2))
            features['n_alternative_paths'] = len(alt_paths)
        except:
            features['n_alternative_paths'] = 0

        return features

    def extract_path_labels(
        self,
        G: nx.Graph,
        path: List[int]
    ) -> Dict:
        """
        Extract labels (ground truth) from optimal path.

        Args:
            G: NetworkX graph
            path: Optimal path

        Returns:
            Label dictionary
        """
        labels = {}

        if len(path) < 2:
            return {
                'path_cost': float('inf'),
                'path_distance': 0,
                'path_hops': 0,
                'path_regenerators': 0,
                'path_found': False
            }

        # Calculate path metrics
        total_cost = 0
        total_distance = 0
        total_regenerators = 0

        for i in range(len(path) - 1):
            edge = G[path[i]][path[i+1]]
            total_cost += edge['total_cost']
            total_distance += edge['distance']
            total_regenerators += edge['n_regenerators']

        # Path encoding (for classification tasks)
        labels['path'] = path
        labels['path_cost'] = total_cost
        labels['path_distance'] = total_distance
        labels['path_hops'] = len(path) - 1
        labels['path_regenerators'] = total_regenerators
        labels['path_found'] = True

        # Signal quality
        signal_metrics = self.signal_analyzer.analyze_network_path(G, path)
        labels['path_osnr'] = signal_metrics.get('min_osnr_db', 0)
        labels['path_viable'] = signal_metrics.get('is_viable', False)

        return labels

    def build_dataset_from_network(
        self,
        G: nx.Graph,
        n_problems: int = 100
    ) -> pd.DataFrame:
        """
        Build dataset from a single network.

        Args:
            G: NetworkX graph
            n_problems: Number of routing problems

        Returns:
            DataFrame with features and labels
        """
        problems = self.generate_route_problems(G, n_problems)
        dataset = []

        for source, target in tqdm(problems, desc="Generating routes"):
            # Solve with Dijkstra (ground truth)
            result = self.router.dijkstra(G, source, target)

            if not result.success:
                continue

            # Extract features
            features = self.extract_path_features(G, source, target, result.path)

            # Extract labels
            labels = self.extract_path_labels(G, result.path)

            # Combine
            sample = {**features, **labels}
            sample['source'] = source
            sample['target'] = target
            sample['network_id'] = G.graph.get('network_id', 0)

            dataset.append(sample)

        return pd.DataFrame(dataset)

    def build_complete_dataset(
        self,
        n_synthetic: int = 100,
        problems_per_network: int = 100,
        output_dir: str = 'data/processed'
    ) -> pd.DataFrame:
        """
        Build complete training dataset with synthetic and real networks.

        Args:
            n_synthetic: Number of synthetic networks to generate
            problems_per_network: Routing problems per network
            output_dir: Output directory

        Returns:
            Complete DataFrame
        """
        os.makedirs(output_dir, exist_ok=True)

        all_data = []

        # Generate synthetic networks
        print(f"Generating {n_synthetic} synthetic networks...")
        networks = self.network_generator.generate_diverse_dataset(
            n_networks=n_synthetic,
            output_dir='data/synthetic'
        )

        # Build dataset from each network
        print("\nBuilding training dataset from synthetic networks...")
        for i, G in enumerate(tqdm(networks[:10])):  # Use subset for speed
            try:
                df = self.build_dataset_from_network(G, problems_per_network)
                all_data.append(df)
            except Exception as e:
                print(f"  Error processing network {i}: {e}")
                continue

        # Load real networks
        print("\nLoading real network topologies...")
        real_networks = self.topology_parser.load_real_networks(
            ['abilene', 'geant'],
            output_dir='data/raw'
        )

        # Build dataset from real networks
        print("\nBuilding validation dataset from real networks...")
        for name, G in real_networks.items():
            try:
                df = self.build_dataset_from_network(G, problems_per_network)
                df['network_name'] = name
                all_data.append(df)
            except Exception as e:
                print(f"  Error processing {name}: {e}")
                continue

        # Combine all data
        complete_df = pd.concat(all_data, ignore_index=True)

        # Save dataset
        train_file = f"{output_dir}/training_data.csv"
        complete_df.to_csv(train_file, index=False)
        print(f"\nDataset saved: {train_file}")
        print(f"Total samples: {len(complete_df)}")

        # Save feature names
        feature_cols = [col for col in complete_df.columns
                       if col not in ['path', 'source', 'target', 'network_id',
                                     'network_name', 'path_cost', 'path_distance',
                                     'path_hops', 'path_regenerators', 'path_found',
                                     'path_osnr', 'path_viable']]

        with open(f"{output_dir}/feature_columns.json", 'w') as f:
            json.dump(feature_cols, f, indent=2)

        # Save label columns
        label_cols = ['path_cost', 'path_distance', 'path_hops',
                     'path_regenerators', 'path_osnr']

        with open(f"{output_dir}/label_columns.json", 'w') as f:
            json.dump(label_cols, f, indent=2)

        # Dataset statistics
        print("\nDataset Statistics:")
        print(f"  Networks: {complete_df['network_id'].nunique()}")
        print(f"  Routing problems: {len(complete_df)}")
        print(f"  Features: {len(feature_cols)}")
        print(f"  Labels: {len(label_cols)}")
        print(f"\nLabel distributions:")
        print(complete_df[label_cols].describe())

        return complete_df


if __name__ == '__main__':
    print("Building training dataset...")

    builder = DatasetBuilder(seed=42)

    # Build complete dataset
    df = builder.build_complete_dataset(
        n_synthetic=100,
        problems_per_network=100,
        output_dir='data/processed'
    )

    print("\nDataset generation complete!")
