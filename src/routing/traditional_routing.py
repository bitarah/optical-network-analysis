"""
Traditional Routing Algorithms

Implements Dijkstra, A*, and Genetic Algorithm for optical network routing.
These serve as baselines for comparison with ML-based approaches.
"""

import networkx as nx
import numpy as np
from typing import List, Tuple, Dict, Callable
from dataclasses import dataclass
import time
import heapq


@dataclass
class RoutingResult:
    """Container for routing algorithm results."""
    algorithm: str
    path: List[int]
    cost: float
    distance: float
    n_regenerators: int
    computation_time: float
    success: bool
    metrics: Dict = None


class TraditionalRouter:
    """Implements traditional routing algorithms for optical networks."""

    def __init__(self, cost_weight: float = 1.0, distance_weight: float = 0.0):
        """
        Initialize router.

        Args:
            cost_weight: Weight for cost in objective function
            distance_weight: Weight for distance in objective function
        """
        self.cost_weight = cost_weight
        self.distance_weight = distance_weight

    def dijkstra(
        self,
        G: nx.Graph,
        source: int,
        target: int,
        weight: str = 'total_cost'
    ) -> RoutingResult:
        """
        Find shortest path using Dijkstra's algorithm.

        Args:
            G: NetworkX graph
            source: Source node
            target: Target node
            weight: Edge attribute to use as weight

        Returns:
            RoutingResult object
        """
        start_time = time.time()

        try:
            # NetworkX Dijkstra implementation
            path = nx.dijkstra_path(G, source, target, weight=weight)
            path_cost = nx.dijkstra_path_length(G, source, target, weight=weight)

            # Calculate additional metrics
            metrics = self._calculate_path_metrics(G, path)

            computation_time = time.time() - start_time

            return RoutingResult(
                algorithm='Dijkstra',
                path=path,
                cost=path_cost,
                distance=metrics['distance'],
                n_regenerators=metrics['n_regenerators'],
                computation_time=computation_time,
                success=True,
                metrics=metrics
            )

        except nx.NetworkXNoPath:
            computation_time = time.time() - start_time
            return RoutingResult(
                algorithm='Dijkstra',
                path=[],
                cost=float('inf'),
                distance=0,
                n_regenerators=0,
                computation_time=computation_time,
                success=False
            )

    def a_star(
        self,
        G: nx.Graph,
        source: int,
        target: int,
        weight: str = 'total_cost'
    ) -> RoutingResult:
        """
        Find path using A* algorithm with geographic heuristic.

        Args:
            G: NetworkX graph
            source: Source node
            target: Target node
            weight: Edge attribute to use as weight

        Returns:
            RoutingResult object
        """
        start_time = time.time()

        # Define heuristic function based on geographic distance
        def heuristic(u, v):
            """Euclidean distance heuristic."""
            pos = nx.get_node_attributes(G, 'pos')
            if u not in pos or v not in pos:
                return 0

            x1, y1 = pos[u]
            x2, y2 = pos[v]
            return np.sqrt((x2 - x1)**2 + (y2 - y1)**2) * 20000  # Scale to cost

        try:
            # NetworkX A* implementation
            path = nx.astar_path(G, source, target, heuristic=heuristic, weight=weight)
            path_length = nx.astar_path_length(G, source, target, heuristic=heuristic, weight=weight)

            # Calculate metrics
            metrics = self._calculate_path_metrics(G, path)

            computation_time = time.time() - start_time

            return RoutingResult(
                algorithm='A*',
                path=path,
                cost=path_length,
                distance=metrics['distance'],
                n_regenerators=metrics['n_regenerators'],
                computation_time=computation_time,
                success=True,
                metrics=metrics
            )

        except nx.NetworkXNoPath:
            computation_time = time.time() - start_time
            return RoutingResult(
                algorithm='A*',
                path=[],
                cost=float('inf'),
                distance=0,
                n_regenerators=0,
                computation_time=computation_time,
                success=False
            )

    def multi_objective_dijkstra(
        self,
        G: nx.Graph,
        source: int,
        target: int,
        cost_weight: float = 0.7,
        distance_weight: float = 0.3
    ) -> RoutingResult:
        """
        Dijkstra with multi-objective cost function.

        Args:
            G: NetworkX graph
            source: Source node
            target: Target node
            cost_weight: Weight for monetary cost
            distance_weight: Weight for distance

        Returns:
            RoutingResult object
        """
        start_time = time.time()

        # Create composite weight function
        def composite_weight(u, v, edge_attr):
            cost = edge_attr.get('total_cost', 0)
            distance = edge_attr.get('distance', 0)
            return cost_weight * cost + distance_weight * distance

        try:
            path = nx.dijkstra_path(G, source, target, weight=composite_weight)

            # Calculate actual cost
            total_cost = sum(G[path[i]][path[i+1]]['total_cost']
                           for i in range(len(path) - 1))

            metrics = self._calculate_path_metrics(G, path)

            computation_time = time.time() - start_time

            return RoutingResult(
                algorithm='Multi-Objective Dijkstra',
                path=path,
                cost=total_cost,
                distance=metrics['distance'],
                n_regenerators=metrics['n_regenerators'],
                computation_time=computation_time,
                success=True,
                metrics=metrics
            )

        except nx.NetworkXNoPath:
            computation_time = time.time() - start_time
            return RoutingResult(
                algorithm='Multi-Objective Dijkstra',
                path=[],
                cost=float('inf'),
                distance=0,
                n_regenerators=0,
                computation_time=computation_time,
                success=False
            )

    def k_shortest_paths(
        self,
        G: nx.Graph,
        source: int,
        target: int,
        k: int = 5,
        weight: str = 'total_cost'
    ) -> List[RoutingResult]:
        """
        Find k shortest paths.

        Args:
            G: NetworkX graph
            source: Source node
            target: Target node
            k: Number of paths to find
            weight: Edge attribute to use

        Returns:
            List of RoutingResult objects
        """
        start_time = time.time()

        results = []

        try:
            paths = list(nx.shortest_simple_paths(G, source, target, weight=weight))

            for i, path in enumerate(paths[:k]):
                path_cost = sum(G[path[j]][path[j+1]][weight]
                              for j in range(len(path) - 1))

                metrics = self._calculate_path_metrics(G, path)

                results.append(RoutingResult(
                    algorithm=f'{k}-Shortest Paths (rank {i+1})',
                    path=path,
                    cost=path_cost,
                    distance=metrics['distance'],
                    n_regenerators=metrics['n_regenerators'],
                    computation_time=(time.time() - start_time) / k,
                    success=True,
                    metrics=metrics
                ))

        except nx.NetworkXNoPath:
            pass

        return results

    def _calculate_path_metrics(self, G: nx.Graph, path: List[int]) -> Dict:
        """
        Calculate comprehensive metrics for a path.

        Args:
            G: NetworkX graph
            path: List of nodes in path

        Returns:
            Dictionary of metrics
        """
        if len(path) < 2:
            return {
                'distance': 0,
                'n_regenerators': 0,
                'n_hops': 0,
                'avg_utilization': 0
            }

        total_distance = 0
        total_regenerators = 0
        utilizations = []

        for i in range(len(path) - 1):
            edge = G[path[i]][path[i + 1]]
            total_distance += edge['distance']
            total_regenerators += edge['n_regenerators']
            utilizations.append(edge.get('utilization', 0))

        return {
            'distance': round(total_distance, 2),
            'n_regenerators': total_regenerators,
            'n_hops': len(path) - 1,
            'avg_utilization': round(np.mean(utilizations), 3) if utilizations else 0
        }


class GeneticAlgorithmRouter:
    """Genetic Algorithm for route optimization."""

    def __init__(
        self,
        population_size: int = 50,
        generations: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7
    ):
        """
        Initialize GA router.

        Args:
            population_size: Size of population
            generations: Number of generations
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
        """
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def route(
        self,
        G: nx.Graph,
        source: int,
        target: int
    ) -> RoutingResult:
        """
        Find route using genetic algorithm.

        Args:
            G: NetworkX graph
            source: Source node
            target: Target node

        Returns:
            RoutingResult object
        """
        start_time = time.time()

        # Initialize population with k-shortest paths
        try:
            initial_paths = list(nx.shortest_simple_paths(
                G, source, target, weight='total_cost'
            ))[:self.population_size]
        except nx.NetworkXNoPath:
            return RoutingResult(
                algorithm='Genetic Algorithm',
                path=[],
                cost=float('inf'),
                distance=0,
                n_regenerators=0,
                computation_time=time.time() - start_time,
                success=False
            )

        # Pad population if needed
        while len(initial_paths) < self.population_size:
            initial_paths.append(initial_paths[0])

        population = initial_paths

        # Evolve population
        best_path = population[0]
        best_fitness = self._fitness(G, best_path)

        for gen in range(self.generations):
            # Calculate fitness for all individuals
            fitness_scores = [self._fitness(G, path) for path in population]

            # Selection
            selected = self._selection(population, fitness_scores)

            # Crossover and mutation
            offspring = []
            for i in range(0, len(selected), 2):
                parent1 = selected[i]
                parent2 = selected[i + 1] if i + 1 < len(selected) else selected[0]

                if np.random.rand() < self.crossover_rate:
                    child = self._crossover(G, parent1, parent2, source, target)
                else:
                    child = parent1

                if np.random.rand() < self.mutation_rate:
                    child = self._mutate(G, child, source, target)

                offspring.append(child)

            population = offspring

            # Track best
            for path in population:
                fitness = self._fitness(G, path)
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_path = path

        # Calculate final metrics
        cost = sum(G[best_path[i]][best_path[i+1]]['total_cost']
                  for i in range(len(best_path) - 1))

        router = TraditionalRouter()
        metrics = router._calculate_path_metrics(G, best_path)

        computation_time = time.time() - start_time

        return RoutingResult(
            algorithm='Genetic Algorithm',
            path=best_path,
            cost=cost,
            distance=metrics['distance'],
            n_regenerators=metrics['n_regenerators'],
            computation_time=computation_time,
            success=True,
            metrics=metrics
        )

    def _fitness(self, G: nx.Graph, path: List[int]) -> float:
        """Calculate fitness (inverse of cost)."""
        if len(path) < 2:
            return 0.0

        try:
            cost = sum(G[path[i]][path[i+1]]['total_cost']
                      for i in range(len(path) - 1))
            return 1.0 / (1.0 + cost)
        except:
            return 0.0

    def _selection(
        self,
        population: List[List[int]],
        fitness_scores: List[float]
    ) -> List[List[int]]:
        """Tournament selection."""
        selected = []
        for _ in range(len(population)):
            tournament = np.random.choice(len(population), size=3, replace=False)
            winner = tournament[np.argmax([fitness_scores[i] for i in tournament])]
            selected.append(population[winner])
        return selected

    def _crossover(
        self,
        G: nx.Graph,
        parent1: List[int],
        parent2: List[int],
        source: int,
        target: int
    ) -> List[int]:
        """Single-point crossover."""
        # Find common nodes
        common = set(parent1) & set(parent2)
        common.discard(source)
        common.discard(target)

        if not common:
            return parent1

        # Pick random common node as crossover point
        crossover_node = np.random.choice(list(common))

        # Build child path
        idx1 = parent1.index(crossover_node)
        idx2 = parent2.index(crossover_node)

        child = parent1[:idx1] + parent2[idx2:]

        # Validate path
        if not self._is_valid_path(G, child):
            return parent1

        return child

    def _mutate(
        self,
        G: nx.Graph,
        path: List[int],
        source: int,
        target: int
    ) -> List[int]:
        """Mutate by replacing a segment with alternative path."""
        if len(path) <= 2:
            return path

        # Pick random segment
        idx1 = np.random.randint(0, len(path) - 1)
        idx2 = np.random.randint(idx1 + 1, len(path))

        node1, node2 = path[idx1], path[idx2]

        # Try to find alternative path between segment endpoints
        try:
            alt_segment = nx.shortest_path(G, node1, node2, weight='total_cost')
            mutated = path[:idx1] + alt_segment + path[idx2+1:]
            return mutated if self._is_valid_path(G, mutated) else path
        except:
            return path

    def _is_valid_path(self, G: nx.Graph, path: List[int]) -> bool:
        """Check if path is valid."""
        if len(path) < 2:
            return False

        for i in range(len(path) - 1):
            if not G.has_edge(path[i], path[i + 1]):
                return False

        return True


if __name__ == '__main__':
    print("Traditional routing algorithms ready!")
