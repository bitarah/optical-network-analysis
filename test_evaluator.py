"""
Quick test of the evaluation framework
"""

import sys
sys.path.append('src')

from evaluation.evaluator import AlgorithmEvaluator, generate_test_problems
from data_generation.network_generator import OpticalNetworkGenerator

print("="*70)
print("Quick Evaluation Framework Test")
print("="*70)

# Generate a small test network
print("\n1. Generating test network...")
generator = OpticalNetworkGenerator(seed=42)
G = generator.generate_random_geometric(20, radius=0.35, area_size=600)
G.graph['topology_type'] = 'geometric'

print(f"   Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Generate test problems
problems = generate_test_problems(G, n_problems=10, seed=42)
print(f"   Generated {len(problems)} routing problems")

# Initialize evaluator
print("\n2. Initializing evaluator...")
evaluator = AlgorithmEvaluator(verbose=True)
evaluator.load_ml_models()

# Evaluate only fast algorithms (skip Genetic Algorithm)
algorithms = [
    'Dijkstra',
    'A*',
    'Random Forest',
    'XGBoost',
    'Neural Network'
]

print("\n3. Running evaluation (fast algorithms only)...")
networks = [(G, problems)]
results = evaluator.evaluate_multiple_networks(networks, algorithms)

# Generate report
print("\n4. Generating comparison report...")
comparison = evaluator.generate_comparison_report()
evaluator.print_comparison_report()

# Save results
print("\n5. Saving results...")
evaluator.save_results('data/quick_evaluation_results.json')

print("\n" + "="*70)
print("Quick test complete!")
print("="*70)
