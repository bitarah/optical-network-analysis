"""
Comprehensive Evaluation Framework for Routing Algorithms

Compares traditional and ML-based routing algorithms across multiple metrics:
- Path cost and quality
- Computation time
- Success rate
- Scalability

Provides statistical analysis and detailed performance reports.
"""

import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Optional
import json
import time
from dataclasses import dataclass, asdict
from collections import defaultdict
import pickle
import os
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from routing.traditional_routing import TraditionalRouter, GeneticAlgorithmRouter, RoutingResult
from routing.ml_routing import RandomForestRouter, XGBoostRouter, NeuralNetworkRouter
from data_generation.network_generator import OpticalNetworkGenerator


@dataclass
class AlgorithmMetrics:
    """Container for algorithm performance metrics."""
    algorithm_name: str
    avg_cost: float
    std_cost: float
    avg_time: float
    std_time: float
    avg_distance: float
    avg_regenerators: float
    success_rate: float
    total_problems: int
    successful_problems: int
    failed_problems: int
    cost_deviation_from_dijkstra: Optional[float] = None
    time_vs_dijkstra: Optional[float] = None


@dataclass
class NetworkMetrics:
    """Container for network-level metrics."""
    network_id: int
    n_nodes: int
    n_edges: int
    topology_type: str
    avg_edge_cost: float
    total_problems: int
    avg_computation_time: Dict[str, float] = None


class AlgorithmEvaluator:
    """Comprehensive evaluation framework for routing algorithms."""

    def __init__(self, verbose: bool = True):
        """
        Initialize the evaluator.

        Args:
            verbose: Print progress and results
        """
        self.verbose = verbose
        self.results = defaultdict(list)
        self.network_metrics = []
        self.global_metrics = {}

        # Initialize algorithm instances
        self.traditional_router = TraditionalRouter()
        self.ga_router = GeneticAlgorithmRouter(
            population_size=30,
            generations=50,
            mutation_rate=0.1,
            crossover_rate=0.7
        )

        # ML routers (to be loaded)
        self.rf_router = None
        self.xgb_router = None
        self.nn_router = None

        # Dijkstra baseline for comparison
        self.dijkstra_costs = {}

    def load_ml_models(
        self,
        rf_path: Optional[str] = None,
        xgb_path: Optional[str] = None,
        nn_path: Optional[str] = None
    ) -> None:
        """
        Load pre-trained ML models.

        Args:
            rf_path: Path to Random Forest model
            xgb_path: Path to XGBoost model
            nn_path: Path to Neural Network model
        """
        if self.verbose:
            print("Loading ML models...")

        # Load Random Forest
        if rf_path and os.path.exists(rf_path):
            try:
                with open(rf_path, 'rb') as f:
                    self.rf_router = pickle.load(f)
                if self.verbose:
                    print(f"  Random Forest loaded from {rf_path}")
            except Exception as e:
                print(f"  Warning: Failed to load Random Forest: {e}")
                self.rf_router = RandomForestRouter()
        else:
            self.rf_router = RandomForestRouter()
            if self.verbose:
                print("  Random Forest: Using untrained model (will use fallback)")

        # Load XGBoost
        if xgb_path and os.path.exists(xgb_path):
            try:
                with open(xgb_path, 'rb') as f:
                    self.xgb_router = pickle.load(f)
                if self.verbose:
                    print(f"  XGBoost loaded from {xgb_path}")
            except Exception as e:
                print(f"  Warning: Failed to load XGBoost: {e}")
                self.xgb_router = XGBoostRouter()
        else:
            self.xgb_router = XGBoostRouter()
            if self.verbose:
                print("  XGBoost: Using untrained model (will use fallback)")

        # Load Neural Network
        if nn_path and os.path.exists(nn_path):
            try:
                with open(nn_path, 'rb') as f:
                    self.nn_router = pickle.load(f)
                if self.verbose:
                    print(f"  Neural Network loaded from {nn_path}")
            except Exception as e:
                print(f"  Warning: Failed to load Neural Network: {e}")
                self.nn_router = NeuralNetworkRouter()
        else:
            self.nn_router = NeuralNetworkRouter()
            if self.verbose:
                print("  Neural Network: Using untrained model (will use fallback)")

    def evaluate_single_problem(
        self,
        G: nx.Graph,
        source: int,
        target: int,
        algorithms: List[str]
    ) -> Dict[str, RoutingResult]:
        """
        Evaluate all algorithms on a single routing problem.

        Args:
            G: NetworkX graph
            source: Source node
            target: Target node
            algorithms: List of algorithm names to evaluate

        Returns:
            Dictionary mapping algorithm name to RoutingResult
        """
        results = {}
        problem_key = f"{id(G)}_{source}_{target}"

        # Run each algorithm
        for algo in algorithms:
            try:
                if algo == 'Dijkstra':
                    result = self.traditional_router.dijkstra(G, source, target)
                    # Store Dijkstra cost as baseline
                    if result.success:
                        self.dijkstra_costs[problem_key] = result.cost

                elif algo == 'A*':
                    result = self.traditional_router.a_star(G, source, target)

                elif algo == 'Genetic':
                    result = self.ga_router.route(G, source, target)

                elif algo == 'Random Forest':
                    if self.rf_router is None:
                        continue
                    result = self.rf_router.route(G, source, target)

                elif algo == 'XGBoost':
                    if self.xgb_router is None:
                        continue
                    result = self.xgb_router.route(G, source, target)

                elif algo == 'Neural Network':
                    if self.nn_router is None:
                        continue
                    result = self.nn_router.route(G, source, target)

                else:
                    if self.verbose:
                        print(f"Unknown algorithm: {algo}")
                    continue

                results[algo] = result

            except Exception as e:
                if self.verbose:
                    print(f"Error evaluating {algo} on problem ({source}, {target}): {e}")
                # Create failed result
                results[algo] = RoutingResult(
                    algorithm=algo,
                    path=[],
                    cost=float('inf'),
                    distance=0,
                    n_regenerators=0,
                    computation_time=0,
                    success=False
                )

        return results

    def evaluate_network(
        self,
        G: nx.Graph,
        test_problems: List[Tuple[int, int]],
        algorithms: List[str],
        network_id: Optional[int] = None
    ) -> Dict[str, List[RoutingResult]]:
        """
        Evaluate all algorithms on multiple routing problems in one network.

        Args:
            G: NetworkX graph
            test_problems: List of (source, target) tuples
            algorithms: List of algorithm names
            network_id: Optional network identifier

        Returns:
            Dictionary mapping algorithm name to list of RoutingResults
        """
        if self.verbose:
            net_info = f"network {network_id}" if network_id is not None else "network"
            print(f"\nEvaluating {len(algorithms)} algorithms on {len(test_problems)} problems in {net_info}")
            print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

        network_results = defaultdict(list)

        # Evaluate each problem
        iterator = tqdm(test_problems, desc="Problems", disable=not self.verbose)
        for source, target in iterator:
            problem_results = self.evaluate_single_problem(G, source, target, algorithms)

            for algo, result in problem_results.items():
                network_results[algo].append(result)
                self.results[algo].append(result)

        # Store network-level metrics
        topology_type = G.graph.get('topology_type', 'unknown')
        avg_edge_cost = np.mean([data['total_cost'] for _, _, data in G.edges(data=True)])

        net_metrics = NetworkMetrics(
            network_id=network_id if network_id is not None else 0,
            n_nodes=G.number_of_nodes(),
            n_edges=G.number_of_edges(),
            topology_type=topology_type,
            avg_edge_cost=avg_edge_cost,
            total_problems=len(test_problems),
            avg_computation_time={}
        )

        # Calculate average computation time per algorithm
        for algo, results in network_results.items():
            times = [r.computation_time for r in results if r.success]
            net_metrics.avg_computation_time[algo] = np.mean(times) if times else 0

        self.network_metrics.append(net_metrics)

        return dict(network_results)

    def evaluate_multiple_networks(
        self,
        networks: List[Tuple[nx.Graph, List[Tuple[int, int]]]],
        algorithms: List[str]
    ) -> Dict[str, List[RoutingResult]]:
        """
        Run full benchmark across multiple networks.

        Args:
            networks: List of (graph, test_problems) tuples
            algorithms: List of algorithm names to evaluate

        Returns:
            Dictionary mapping algorithm name to all RoutingResults
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Starting comprehensive evaluation")
            print(f"  Networks: {len(networks)}")
            print(f"  Algorithms: {', '.join(algorithms)}")
            print(f"{'='*70}")

        # Clear previous results
        self.results = defaultdict(list)
        self.network_metrics = []
        self.dijkstra_costs = {}

        # Evaluate each network
        for i, (G, test_problems) in enumerate(networks):
            self.evaluate_network(G, test_problems, algorithms, network_id=i)

        if self.verbose:
            print(f"\n{'='*70}")
            print("Evaluation complete!")
            print(f"{'='*70}")

        return dict(self.results)

    def generate_comparison_report(self) -> Dict[str, AlgorithmMetrics]:
        """
        Generate statistical summary comparing all algorithms.

        Returns:
            Dictionary mapping algorithm name to AlgorithmMetrics
        """
        if not self.results:
            if self.verbose:
                print("No results to analyze. Run evaluation first.")
            return {}

        comparison = {}

        # Get Dijkstra baseline statistics
        dijkstra_results = self.results.get('Dijkstra', [])
        dijkstra_avg_cost = np.mean([r.cost for r in dijkstra_results if r.success]) if dijkstra_results else 0
        dijkstra_avg_time = np.mean([r.computation_time for r in dijkstra_results if r.success]) if dijkstra_results else 0

        # Calculate metrics for each algorithm
        for algo, results in self.results.items():
            if not results:
                continue

            # Filter successful results
            successful = [r for r in results if r.success]
            failed = [r for r in results if not r.success]

            if not successful:
                # All failed
                metrics = AlgorithmMetrics(
                    algorithm_name=algo,
                    avg_cost=float('inf'),
                    std_cost=0,
                    avg_time=0,
                    std_time=0,
                    avg_distance=0,
                    avg_regenerators=0,
                    success_rate=0,
                    total_problems=len(results),
                    successful_problems=0,
                    failed_problems=len(failed)
                )
            else:
                costs = [r.cost for r in successful]
                times = [r.computation_time for r in successful]
                distances = [r.distance for r in successful]
                regenerators = [r.n_regenerators for r in successful]

                avg_cost = np.mean(costs)

                # Calculate deviation from Dijkstra baseline
                cost_deviation = None
                if dijkstra_avg_cost > 0 and algo != 'Dijkstra':
                    cost_deviation = ((avg_cost - dijkstra_avg_cost) / dijkstra_avg_cost) * 100

                # Calculate time ratio vs Dijkstra
                time_vs_dijkstra = None
                if dijkstra_avg_time > 0 and algo != 'Dijkstra':
                    time_vs_dijkstra = np.mean(times) / dijkstra_avg_time

                metrics = AlgorithmMetrics(
                    algorithm_name=algo,
                    avg_cost=round(avg_cost, 2),
                    std_cost=round(np.std(costs), 2),
                    avg_time=round(np.mean(times) * 1000, 4),  # Convert to ms
                    std_time=round(np.std(times) * 1000, 4),  # Convert to ms
                    avg_distance=round(np.mean(distances), 2),
                    avg_regenerators=round(np.mean(regenerators), 2),
                    success_rate=round(len(successful) / len(results) * 100, 2),
                    total_problems=len(results),
                    successful_problems=len(successful),
                    failed_problems=len(failed),
                    cost_deviation_from_dijkstra=round(cost_deviation, 2) if cost_deviation is not None else None,
                    time_vs_dijkstra=round(time_vs_dijkstra, 2) if time_vs_dijkstra is not None else None
                )

            comparison[algo] = metrics

        self.global_metrics = comparison
        return comparison

    def print_comparison_report(self) -> None:
        """Print formatted comparison report to console."""
        if not self.global_metrics:
            self.generate_comparison_report()

        if not self.global_metrics:
            print("No metrics available.")
            return

        print(f"\n{'='*90}")
        print(f"{'ALGORITHM COMPARISON REPORT':^90}")
        print(f"{'='*90}\n")

        # Sort by average cost
        sorted_algos = sorted(
            self.global_metrics.items(),
            key=lambda x: x[1].avg_cost if x[1].avg_cost != float('inf') else 1e10
        )

        # Print header
        print(f"{'Algorithm':<20} {'Avg Cost':<12} {'Avg Time (ms)':<15} {'Success %':<12} {'Cost Dev %':<12}")
        print(f"{'-'*90}")

        # Print each algorithm
        for algo, metrics in sorted_algos:
            cost_str = f"{metrics.avg_cost:,.2f}" if metrics.avg_cost != float('inf') else "FAILED"
            time_str = f"{metrics.avg_time:.4f}"
            success_str = f"{metrics.success_rate:.1f}%"
            dev_str = f"{metrics.cost_deviation_from_dijkstra:+.2f}%" if metrics.cost_deviation_from_dijkstra is not None else "baseline"

            print(f"{algo:<20} {cost_str:<12} {time_str:<15} {success_str:<12} {dev_str:<12}")

        print(f"\n{'='*90}\n")

        # Detailed statistics
        print("DETAILED STATISTICS:\n")
        for algo, metrics in sorted_algos:
            print(f"\n{algo}:")
            print(f"  Average Cost: {metrics.avg_cost:,.2f} ± {metrics.std_cost:,.2f}")
            print(f"  Average Time: {metrics.avg_time:.4f} ± {metrics.std_time:.4f} ms")
            print(f"  Average Distance: {metrics.avg_distance:,.2f} km")
            print(f"  Average Regenerators: {metrics.avg_regenerators:.2f}")
            print(f"  Success Rate: {metrics.success_rate:.1f}% ({metrics.successful_problems}/{metrics.total_problems})")

            if metrics.cost_deviation_from_dijkstra is not None:
                print(f"  Cost Deviation from Dijkstra: {metrics.cost_deviation_from_dijkstra:+.2f}%")

            if metrics.time_vs_dijkstra is not None:
                print(f"  Time vs Dijkstra: {metrics.time_vs_dijkstra:.2f}x")

        # Network-level summary
        if self.network_metrics:
            print(f"\n{'='*90}")
            print("NETWORK-LEVEL SUMMARY:\n")
            print(f"Total Networks Evaluated: {len(self.network_metrics)}")

            topology_counts = defaultdict(int)
            for net in self.network_metrics:
                topology_counts[net.topology_type] += 1

            print(f"Topology Distribution:")
            for topo, count in topology_counts.items():
                print(f"  {topo}: {count}")

            avg_nodes = np.mean([net.n_nodes for net in self.network_metrics])
            avg_edges = np.mean([net.n_edges for net in self.network_metrics])
            print(f"\nAverage Network Size:")
            print(f"  Nodes: {avg_nodes:.1f}")
            print(f"  Edges: {avg_edges:.1f}")

        print(f"\n{'='*90}\n")

    def save_results(self, output_path: str) -> None:
        """
        Save evaluation results to JSON file.

        Args:
            output_path: Path to output JSON file
        """
        # Generate metrics if not already done
        if not self.global_metrics:
            self.generate_comparison_report()

        output_data = {
            'algorithms': {},
            'networks': [],
            'summary': {
                'total_problems': sum(m.total_problems for m in self.global_metrics.values()) // len(self.global_metrics) if self.global_metrics else 0,
                'total_networks': len(self.network_metrics),
                'algorithms_evaluated': list(self.global_metrics.keys())
            }
        }

        # Algorithm metrics
        for algo, metrics in self.global_metrics.items():
            output_data['algorithms'][algo] = asdict(metrics)

        # Network metrics
        for net_metrics in self.network_metrics:
            net_dict = asdict(net_metrics)
            output_data['networks'].append(net_dict)

        # Save to file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        if self.verbose:
            print(f"\nResults saved to: {output_path}")

    def analyze_scalability(self) -> Dict[str, Dict]:
        """
        Analyze how algorithms scale with network size.

        Returns:
            Dictionary with scalability analysis per algorithm
        """
        if not self.network_metrics or not self.results:
            return {}

        scalability = {}

        for algo in self.results.keys():
            # Group results by network
            network_performance = defaultdict(list)

            for i, net in enumerate(self.network_metrics):
                # Get results for this network
                net_results = [r for r in self.results[algo]
                             if hasattr(r, 'metrics') and r.success]

                if net_results:
                    avg_time = np.mean([r.computation_time for r in net_results])
                    avg_cost = np.mean([r.cost for r in net_results])

                    network_performance[net.n_nodes].append({
                        'time': avg_time,
                        'cost': avg_cost
                    })

            # Calculate scalability metrics
            size_bins = sorted(network_performance.keys())
            if len(size_bins) >= 2:
                times_by_size = [np.mean([p['time'] for p in network_performance[size]])
                               for size in size_bins]

                # Simple linear regression for time complexity
                log_sizes = np.log(size_bins)
                log_times = np.log(times_by_size)
                coeffs = np.polyfit(log_sizes, log_times, 1)
                time_complexity_exponent = coeffs[0]

                scalability[algo] = {
                    'time_complexity_exponent': round(time_complexity_exponent, 3),
                    'size_range': [int(min(size_bins)), int(max(size_bins))],
                    'time_range_ms': [round(min(times_by_size) * 1000, 4),
                                    round(max(times_by_size) * 1000, 4)]
                }

        return scalability

    def get_best_algorithm(self, metric: str = 'cost') -> Tuple[str, AlgorithmMetrics]:
        """
        Get the best performing algorithm by a specific metric.

        Args:
            metric: Metric to optimize ('cost', 'time', 'success_rate')

        Returns:
            Tuple of (algorithm_name, metrics)
        """
        if not self.global_metrics:
            self.generate_comparison_report()

        if not self.global_metrics:
            return None, None

        if metric == 'cost':
            best = min(self.global_metrics.items(),
                      key=lambda x: x[1].avg_cost if x[1].avg_cost != float('inf') else 1e10)
        elif metric == 'time':
            best = min(self.global_metrics.items(),
                      key=lambda x: x[1].avg_time)
        elif metric == 'success_rate':
            best = max(self.global_metrics.items(),
                      key=lambda x: x[1].success_rate)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        return best


def generate_test_problems(
    G: nx.Graph,
    n_problems: int = 50,
    seed: int = 42
) -> List[Tuple[int, int]]:
    """
    Generate random routing problems for a network.

    Args:
        G: NetworkX graph
        n_problems: Number of problems to generate
        seed: Random seed

    Returns:
        List of (source, target) tuples
    """
    np.random.seed(seed)
    nodes = list(G.nodes())
    problems = []

    for _ in range(n_problems):
        # Select random source and target
        source, target = np.random.choice(nodes, size=2, replace=False)
        problems.append((int(source), int(target)))

    return problems


if __name__ == '__main__':
    print("="*70)
    print("Comprehensive Routing Algorithm Evaluation Framework")
    print("="*70)

    # Generate test networks
    print("\n1. Generating test networks...")
    generator = OpticalNetworkGenerator(seed=42)

    networks = []

    # Small network
    G1 = generator.generate_random_geometric(30, radius=0.3, area_size=800)
    G1.graph['topology_type'] = 'geometric'
    problems1 = generate_test_problems(G1, n_problems=20, seed=42)
    networks.append((G1, problems1))

    # Medium network
    G2 = generator.generate_scale_free(50, m=3, area_size=1500)
    G2.graph['topology_type'] = 'scale_free'
    problems2 = generate_test_problems(G2, n_problems=20, seed=43)
    networks.append((G2, problems2))

    # Large network
    G3 = generator.generate_grid(8, 8, spacing=120)
    G3.graph['topology_type'] = 'grid'
    problems3 = generate_test_problems(G3, n_problems=20, seed=44)
    networks.append((G3, problems3))

    print(f"   Generated {len(networks)} test networks")

    # Initialize evaluator
    print("\n2. Initializing evaluator...")
    evaluator = AlgorithmEvaluator(verbose=True)

    # Load ML models (if available)
    # evaluator.load_ml_models(
    #     rf_path='models/random_forest.pkl',
    #     xgb_path='models/xgboost.pkl',
    #     nn_path='models/neural_network.pkl'
    # )

    # For demo, we'll use untrained models (they'll use Dijkstra fallback)
    evaluator.load_ml_models()

    # Define algorithms to evaluate
    algorithms = [
        'Dijkstra',
        'A*',
        'Genetic',
        'Random Forest',
        'XGBoost',
        'Neural Network'
    ]

    # Run evaluation
    print("\n3. Running evaluation...")
    results = evaluator.evaluate_multiple_networks(networks, algorithms)

    # Generate and print report
    print("\n4. Generating comparison report...")
    comparison = evaluator.generate_comparison_report()
    evaluator.print_comparison_report()

    # Analyze scalability
    print("\n5. Analyzing scalability...")
    scalability = evaluator.analyze_scalability()
    if scalability:
        print("\nScalability Analysis:")
        for algo, metrics in scalability.items():
            print(f"\n  {algo}:")
            print(f"    Time complexity exponent: O(n^{metrics['time_complexity_exponent']})")
            print(f"    Network size range: {metrics['size_range'][0]} - {metrics['size_range'][1]} nodes")
            print(f"    Time range: {metrics['time_range_ms'][0]:.4f} - {metrics['time_range_ms'][1]:.4f} ms")

    # Get best algorithms
    print("\n6. Best performing algorithms:")
    best_cost_algo, best_cost_metrics = evaluator.get_best_algorithm('cost')
    best_time_algo, best_time_metrics = evaluator.get_best_algorithm('time')
    best_success_algo, best_success_metrics = evaluator.get_best_algorithm('success_rate')

    print(f"   Best Cost: {best_cost_algo} ({best_cost_metrics.avg_cost:,.2f})")
    print(f"   Best Time: {best_time_algo} ({best_time_metrics.avg_time:.4f} ms)")
    print(f"   Best Success Rate: {best_success_algo} ({best_success_metrics.success_rate:.1f}%)")

    # Save results
    print("\n7. Saving results...")
    output_path = 'data/evaluation_results.json'
    evaluator.save_results(output_path)

    print("\n" + "="*70)
    print("Evaluation complete!")
    print("="*70)
