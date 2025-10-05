# Evaluation Framework Usage Guide

## Overview

The `AlgorithmEvaluator` class provides a comprehensive framework for comparing routing algorithms in optical network optimization. It supports both traditional algorithms (Dijkstra, A*, Genetic Algorithm) and ML-based approaches (Random Forest, XGBoost, Neural Network).

## Quick Start

```python
from evaluation.evaluator import AlgorithmEvaluator, generate_test_problems
from data_generation.network_generator import OpticalNetworkGenerator

# 1. Generate or load test networks
generator = OpticalNetworkGenerator(seed=42)
G = generator.generate_random_geometric(30, radius=0.3, area_size=1000)

# 2. Generate test problems
problems = generate_test_problems(G, n_problems=50, seed=42)

# 3. Initialize evaluator
evaluator = AlgorithmEvaluator(verbose=True)

# 4. Load ML models (optional)
evaluator.load_ml_models(
    rf_path='models/random_forest.pkl',
    xgb_path='models/xgboost.pkl',
    nn_path='models/neural_network.pkl'
)

# 5. Evaluate algorithms
algorithms = ['Dijkstra', 'A*', 'Genetic', 'Random Forest', 'XGBoost', 'Neural Network']
networks = [(G, problems)]
results = evaluator.evaluate_multiple_networks(networks, algorithms)

# 6. Generate and print report
evaluator.generate_comparison_report()
evaluator.print_comparison_report()

# 7. Save results
evaluator.save_results('data/evaluation_results.json')
```

## Key Features

### 1. Single Problem Evaluation

Evaluate all algorithms on one routing problem:

```python
G = generator.generate_random_geometric(30, radius=0.3)
source, target = 0, 15

results = evaluator.evaluate_single_problem(
    G=G,
    source=source,
    target=target,
    algorithms=['Dijkstra', 'A*', 'Random Forest']
)

for algo, result in results.items():
    print(f"{algo}: Cost={result.cost}, Time={result.computation_time}s")
```

### 2. Network-Level Evaluation

Evaluate multiple problems on one network:

```python
problems = generate_test_problems(G, n_problems=50)

network_results = evaluator.evaluate_network(
    G=G,
    test_problems=problems,
    algorithms=['Dijkstra', 'A*', 'XGBoost'],
    network_id=0
)
```

### 3. Multi-Network Benchmark

Compare algorithms across different network topologies:

```python
# Generate diverse networks
networks = []

# Small geometric network
G1 = generator.generate_random_geometric(30, radius=0.3, area_size=800)
problems1 = generate_test_problems(G1, n_problems=20)
networks.append((G1, problems1))

# Medium scale-free network
G2 = generator.generate_scale_free(50, m=3, area_size=1500)
problems2 = generate_test_problems(G2, n_problems=20)
networks.append((G2, problems2))

# Large grid network
G3 = generator.generate_grid(10, 10, spacing=100)
problems3 = generate_test_problems(G3, n_problems=20)
networks.append((G3, problems3))

# Run comprehensive evaluation
all_results = evaluator.evaluate_multiple_networks(networks, algorithms)
```

### 4. Statistical Comparison

Generate detailed statistics comparing algorithms:

```python
comparison = evaluator.generate_comparison_report()

for algo, metrics in comparison.items():
    print(f"\n{algo}:")
    print(f"  Average Cost: {metrics.avg_cost:,.2f} ± {metrics.std_cost:,.2f}")
    print(f"  Average Time: {metrics.avg_time:.4f} ms")
    print(f"  Success Rate: {metrics.success_rate:.1f}%")
    print(f"  Cost Deviation from Dijkstra: {metrics.cost_deviation_from_dijkstra:+.2f}%")
```

### 5. Scalability Analysis

Analyze how algorithms scale with network size:

```python
scalability = evaluator.analyze_scalability()

for algo, metrics in scalability.items():
    print(f"{algo}:")
    print(f"  Time complexity: O(n^{metrics['time_complexity_exponent']})")
    print(f"  Network size range: {metrics['size_range']}")
    print(f"  Time range: {metrics['time_range_ms']} ms")
```

### 6. Find Best Algorithm

Get the best performing algorithm by metric:

```python
# Best by cost
best_cost_algo, metrics = evaluator.get_best_algorithm('cost')
print(f"Best cost: {best_cost_algo} ({metrics.avg_cost:,.2f})")

# Best by computation time
best_time_algo, metrics = evaluator.get_best_algorithm('time')
print(f"Fastest: {best_time_algo} ({metrics.avg_time:.4f} ms)")

# Best by success rate
best_success_algo, metrics = evaluator.get_best_algorithm('success_rate')
print(f"Most reliable: {best_success_algo} ({metrics.success_rate:.1f}%)")
```

## Metrics Tracked

The evaluator collects the following metrics for each algorithm:

### Performance Metrics
- **Average Cost**: Mean path cost across all problems
- **Standard Deviation**: Cost variance
- **Average Time**: Mean computation time (in milliseconds)
- **Time Standard Deviation**: Time variance

### Path Quality Metrics
- **Average Distance**: Mean physical path distance (km)
- **Average Regenerators**: Mean number of regenerators needed
- **Success Rate**: Percentage of problems solved successfully

### Comparative Metrics
- **Cost Deviation from Dijkstra**: Percentage difference from optimal (Dijkstra baseline)
- **Time vs Dijkstra**: Speedup or slowdown compared to Dijkstra

### Network Metrics
- **Network Size**: Number of nodes and edges
- **Topology Type**: Network architecture (geometric, scale-free, grid)
- **Average Edge Cost**: Mean cost of network links

## Output Format

Results are saved in JSON format:

```json
{
  "algorithms": {
    "Dijkstra": {
      "algorithm_name": "Dijkstra",
      "avg_cost": 8748905.09,
      "std_cost": 3565329.69,
      "avg_time": 0.0853,
      "std_time": 0.0672,
      "avg_distance": 431.34,
      "avg_regenerators": 2.44,
      "success_rate": 90.0,
      "total_problems": 10,
      "successful_problems": 9,
      "failed_problems": 1
    },
    "Random Forest": {
      "algorithm_name": "Random Forest",
      "avg_cost": 8748905.09,
      "avg_time": 0.0638,
      "success_rate": 90.0,
      "cost_deviation_from_dijkstra": 0.0,
      "time_vs_dijkstra": 0.75
    }
  },
  "networks": [
    {
      "network_id": 0,
      "n_nodes": 20,
      "n_edges": 45,
      "topology_type": "geometric",
      "avg_edge_cost": 2607525.91,
      "total_problems": 10
    }
  ],
  "summary": {
    "total_problems": 10,
    "total_networks": 1,
    "algorithms_evaluated": ["Dijkstra", "A*", "Random Forest", "XGBoost", "Neural Network"]
  }
}
```

## Advanced Usage

### Custom Algorithm Selection

Select specific algorithms to evaluate:

```python
# Only traditional algorithms
traditional = ['Dijkstra', 'A*', 'Genetic']
results = evaluator.evaluate_multiple_networks(networks, traditional)

# Only ML algorithms
ml_only = ['Random Forest', 'XGBoost', 'Neural Network']
results = evaluator.evaluate_multiple_networks(networks, ml_only)

# Custom combination
custom = ['Dijkstra', 'Random Forest', 'Neural Network']
results = evaluator.evaluate_multiple_networks(networks, custom)
```

### Loading Pre-trained Models

The evaluator supports loading pre-trained ML models:

```python
# Load all models
evaluator.load_ml_models(
    rf_path='models/random_forest.pkl',
    xgb_path='models/xgboost.pkl',
    nn_path='models/neural_network.pkl'
)

# Load only specific models
evaluator.load_ml_models(rf_path='models/random_forest.pkl')

# If models not provided or not found, untrained models will use Dijkstra fallback
```

### Progress Tracking

Enable verbose mode for progress tracking:

```python
evaluator = AlgorithmEvaluator(verbose=True)  # Shows progress bars and status
evaluator = AlgorithmEvaluator(verbose=False)  # Silent mode
```

### Error Handling

The evaluator includes comprehensive error handling:

- Failed routing attempts are recorded with `success=False`
- Exceptions during algorithm execution are caught and logged
- Missing ML models automatically fall back to Dijkstra
- Invalid problems are skipped with warnings

## Example Output

```
==========================================================================================
                               ALGORITHM COMPARISON REPORT
==========================================================================================

Algorithm            Avg Cost     Avg Time (ms)   Success %    Cost Dev %
------------------------------------------------------------------------------------------
Dijkstra             8,748,905.09 0.0853          90.0%        baseline
A*                   8,748,905.09 0.2109          90.0%        +0.00%
Random Forest        8,748,905.09 0.0638          90.0%        +0.00%
XGBoost              8,748,905.09 0.0608          90.0%        +0.00%
Neural Network       8,748,905.09 0.0601          90.0%        +0.00%

==========================================================================================

DETAILED STATISTICS:

Dijkstra:
  Average Cost: 8,748,905.09 ± 3,565,329.69
  Average Time: 0.0853 ± 0.0672 ms
  Average Distance: 431.34 km
  Average Regenerators: 2.44
  Success Rate: 90.0% (9/10)

Random Forest:
  Average Cost: 8,748,905.09 ± 3,565,329.69
  Average Time: 0.0638 ± 0.0157 ms
  Average Distance: 431.34 km
  Average Regenerators: 2.44
  Success Rate: 90.0% (9/10)
  Cost Deviation from Dijkstra: +0.00%
  Time vs Dijkstra: 0.75x
```

## Best Practices

1. **Always use Dijkstra as baseline**: Include 'Dijkstra' in your algorithm list to get cost deviation metrics

2. **Generate diverse test problems**: Use `generate_test_problems()` with different seeds for comprehensive testing

3. **Test on multiple network types**: Evaluate on geometric, scale-free, and grid topologies to assess generalization

4. **Skip Genetic Algorithm for large evaluations**: GA is slow; use it only for detailed analysis of specific problems

5. **Train ML models before evaluation**: Pre-train models on representative datasets for fair comparison

6. **Save results frequently**: Use `save_results()` to persist evaluation data

7. **Analyze scalability**: Use `analyze_scalability()` to understand performance vs network size

## Troubleshooting

### "Model not trained, using fallback"
- ML models haven't been trained yet
- They will use Dijkstra algorithm as fallback
- Train models first or load pre-trained models

### Slow evaluation with Genetic Algorithm
- GA is computationally expensive
- Reduce population_size and generations for faster evaluation
- Or skip GA for initial benchmarks

### All algorithms showing same cost
- Untrained ML models use Dijkstra fallback
- Train models on dataset before evaluation
- Or verify models are loaded correctly

## Integration

The evaluator integrates seamlessly with other project modules:

```python
# With dataset builder
from data_generation.dataset_builder import DatasetBuilder
builder = DatasetBuilder()
X_train, y_train = builder.generate_routing_dataset(num_networks=100)

# Train models
rf_router = RandomForestRouter()
rf_router.train(X_train, y_train, X_test, y_test)

# Save and load for evaluation
import pickle
with open('models/random_forest.pkl', 'wb') as f:
    pickle.dump(rf_router, f)

# Use in evaluator
evaluator.load_ml_models(rf_path='models/random_forest.pkl')
```
