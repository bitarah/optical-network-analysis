"""
Comprehensive Visualization and Evaluation Script for Optical Network Routing

This script:
1. Loads trained ML models
2. Generates test networks
3. Runs all routing algorithms
4. Generates comprehensive visualizations
5. Saves detailed evaluation results
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
from pathlib import Path
import networkx as nx
from typing import Dict, List, Tuple

from src.evaluation.evaluator import AlgorithmEvaluator, generate_test_problems
from src.data_generation.network_generator import OpticalNetworkGenerator
from src.routing.ml_routing import RandomForestRouter, XGBoostRouter

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10


class ComprehensiveVisualizer:
    """Generate all visualizations for the optical network routing project."""

    def __init__(self, output_dir='results/plots'):
        """Initialize visualizer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_created = []

    def plot_algorithm_performance_comparison(
        self,
        metrics: Dict,
        output_name='01_algorithm_performance_comparison.png'
    ):
        """Create bar chart comparing algorithm performance."""
        print("\nGenerating algorithm performance comparison...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Algorithm Performance Comparison', fontsize=16, fontweight='bold')

        algorithms = list(metrics.keys())

        # 1. Average Cost
        ax = axes[0, 0]
        costs = [metrics[algo].avg_cost for algo in algorithms]
        colors = sns.color_palette("husl", len(algorithms))
        bars = ax.bar(algorithms, costs, color=colors, alpha=0.8, edgecolor='black')
        ax.set_ylabel('Average Path Cost (USD)', fontweight='bold')
        ax.set_title('Average Path Cost by Algorithm')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height < float('inf'):
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'${height:,.0f}',
                       ha='center', va='bottom', fontsize=9)

        # 2. Computation Time
        ax = axes[0, 1]
        times = [metrics[algo].avg_time for algo in algorithms]
        bars = ax.bar(algorithms, times, color=colors, alpha=0.8, edgecolor='black')
        ax.set_ylabel('Average Computation Time (ms)', fontweight='bold')
        ax.set_title('Computation Time by Algorithm')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=9)

        # 3. Success Rate
        ax = axes[1, 0]
        success_rates = [metrics[algo].success_rate for algo in algorithms]
        bars = ax.bar(algorithms, success_rates, color=colors, alpha=0.8, edgecolor='black')
        ax.set_ylabel('Success Rate (%)', fontweight='bold')
        ax.set_title('Success Rate by Algorithm')
        ax.set_ylim([0, 105])
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=9)

        # 4. Average Regenerators
        ax = axes[1, 1]
        regenerators = [metrics[algo].avg_regenerators for algo in algorithms]
        bars = ax.bar(algorithms, regenerators, color=colors, alpha=0.8, edgecolor='black')
        ax.set_ylabel('Average Regenerators', fontweight='bold')
        ax.set_title('Average Regenerators by Algorithm')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        output_path = self.output_dir / output_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.plots_created.append(output_name)
        print(f"  Saved: {output_path}")

    def plot_cost_vs_time_scatter(
        self,
        results: Dict,
        output_name='02_cost_vs_time_scatter.png'
    ):
        """Create scatter plot of cost vs computation time."""
        print("\nGenerating cost vs time scatter plot...")

        fig, ax = plt.subplots(figsize=(12, 8))

        colors = sns.color_palette("husl", len(results))

        for i, (algo, algo_results) in enumerate(results.items()):
            successful = [r for r in algo_results if r.success]
            if not successful:
                continue

            costs = [r.cost for r in successful]
            times = [r.computation_time * 1000 for r in successful]  # Convert to ms

            ax.scatter(times, costs, label=algo, alpha=0.6, s=50, color=colors[i], edgecolors='black', linewidth=0.5)

        ax.set_xlabel('Computation Time (ms)', fontweight='bold')
        ax.set_ylabel('Path Cost (USD)', fontweight='bold')
        ax.set_title('Cost vs Computation Time Trade-off', fontsize=14, fontweight='bold')
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)

        # Log scale if needed
        if ax.get_ylim()[1] / ax.get_ylim()[0] > 100:
            ax.set_yscale('log')

        plt.tight_layout()
        output_path = self.output_dir / output_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.plots_created.append(output_name)
        print(f"  Saved: {output_path}")

    def plot_feature_importance(
        self,
        rf_model,
        xgb_model,
        output_name='03_feature_importance.png'
    ):
        """Plot feature importance for RF and XGBoost."""
        print("\nGenerating feature importance plots...")

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('Feature Importance Analysis', fontsize=16, fontweight='bold')

        # Random Forest
        ax = axes[0]
        rf_importance = rf_model.get_feature_importance()
        if rf_importance:
            features = list(rf_importance.keys())
            importances = list(rf_importance.values())

            # Sort by importance
            sorted_idx = np.argsort(importances)[-15:]  # Top 15
            sorted_features = [features[i] for i in sorted_idx]
            sorted_importances = [importances[i] for i in sorted_idx]

            y_pos = np.arange(len(sorted_features))
            ax.barh(y_pos, sorted_importances, color=sns.color_palette("viridis", len(sorted_features)))
            ax.set_yticks(y_pos)
            ax.set_yticklabels(sorted_features)
            ax.set_xlabel('Importance', fontweight='bold')
            ax.set_title('Random Forest - Top 15 Features')
            ax.grid(axis='x', alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No feature importance available',
                   ha='center', va='center', transform=ax.transAxes)

        # XGBoost
        ax = axes[1]
        xgb_importance = xgb_model.get_feature_importance()
        if xgb_importance:
            features = list(xgb_importance.keys())
            importances = list(xgb_importance.values())

            # Sort by importance
            sorted_idx = np.argsort(importances)[-15:]  # Top 15
            sorted_features = [features[i] for i in sorted_idx]
            sorted_importances = [importances[i] for i in sorted_idx]

            y_pos = np.arange(len(sorted_features))
            ax.barh(y_pos, sorted_importances, color=sns.color_palette("plasma", len(sorted_features)))
            ax.set_yticks(y_pos)
            ax.set_yticklabels(sorted_features)
            ax.set_xlabel('Importance', fontweight='bold')
            ax.set_title('XGBoost - Top 15 Features')
            ax.grid(axis='x', alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No feature importance available',
                   ha='center', va='center', transform=ax.transAxes)

        plt.tight_layout()
        output_path = self.output_dir / output_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.plots_created.append(output_name)
        print(f"  Saved: {output_path}")

    def plot_network_topology_with_route(
        self,
        G: nx.Graph,
        path: List[int],
        source: int,
        target: int,
        output_name='04_network_topology_example.png'
    ):
        """Visualize network topology with example route."""
        print("\nGenerating network topology visualization...")

        fig, ax = plt.subplots(figsize=(14, 10))

        pos = nx.get_node_attributes(G, 'pos')

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
        from matplotlib.lines import Line2D
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

        ax.set_title(f'Network Topology with Route from Node {source} to Node {target}',
                    fontsize=14, fontweight='bold')
        ax.axis('off')

        plt.tight_layout()
        output_path = self.output_dir / output_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.plots_created.append(output_name)
        print(f"  Saved: {output_path}")

    def plot_training_results(
        self,
        training_results: Dict,
        output_name='05_training_results.png'
    ):
        """Visualize training results from models."""
        print("\nGenerating training results visualization...")

        if not training_results or 'models' not in training_results:
            print("  Skipping: No training results available")
            return

        models = training_results['models']

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('ML Model Training Results', fontsize=16, fontweight='bold')

        model_names = list(models.keys())
        colors = sns.color_palette("Set2", len(model_names))

        # R² Score
        ax = axes[0]
        r2_scores = [models[name]['test_r2'] for name in model_names]
        bars = ax.bar(model_names, r2_scores, color=colors, alpha=0.8, edgecolor='black')
        ax.set_ylabel('R² Score', fontweight='bold')
        ax.set_title('Model R² Score (Test Set)')
        ax.set_ylim([0, 1])
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

        # MAE
        ax = axes[1]
        mae_scores = [models[name]['test_mae'] for name in model_names]
        bars = ax.bar(model_names, mae_scores, color=colors, alpha=0.8, edgecolor='black')
        ax.set_ylabel('Mean Absolute Error (USD)', fontweight='bold')
        ax.set_title('Model MAE (Test Set)')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'${height:,.0f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Training Time
        ax = axes[2]
        training_times = [models[name]['training_time'] for name in model_names]
        bars = ax.bar(model_names, training_times, color=colors, alpha=0.8, edgecolor='black')
        ax.set_ylabel('Training Time (seconds)', fontweight='bold')
        ax.set_title('Model Training Time')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}s',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout()
        output_path = self.output_dir / output_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.plots_created.append(output_name)
        print(f"  Saved: {output_path}")

    def plot_algorithm_success_rates(
        self,
        metrics: Dict,
        output_name='06_algorithm_success_rates.png'
    ):
        """Create detailed success rate visualization."""
        print("\nGenerating algorithm success rates visualization...")

        fig, ax = plt.subplots(figsize=(12, 8))

        algorithms = list(metrics.keys())
        success_rates = [metrics[algo].success_rate for algo in algorithms]
        successful = [metrics[algo].successful_problems for algo in algorithms]
        failed = [metrics[algo].failed_problems for algo in algorithms]

        x = np.arange(len(algorithms))
        width = 0.35

        bars1 = ax.bar(x - width/2, successful, width, label='Successful',
                      color='green', alpha=0.7, edgecolor='black')
        bars2 = ax.bar(x + width/2, failed, width, label='Failed',
                      color='red', alpha=0.7, edgecolor='black')

        ax.set_ylabel('Number of Problems', fontweight='bold')
        ax.set_title('Algorithm Success and Failure Rates', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(algorithms, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # Add percentage labels
        for i, (bar1, bar2, rate) in enumerate(zip(bars1, bars2, success_rates)):
            total = successful[i] + failed[i]
            ax.text(i, max(successful[i], failed[i]) + total * 0.05,
                   f'{rate:.1f}%',
                   ha='center', va='bottom', fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

        plt.tight_layout()
        output_path = self.output_dir / output_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.plots_created.append(output_name)
        print(f"  Saved: {output_path}")

    def plot_cost_distribution(
        self,
        results: Dict,
        output_name='07_cost_distribution.png'
    ):
        """Plot distribution of path costs for each algorithm."""
        print("\nGenerating cost distribution plot...")

        fig, ax = plt.subplots(figsize=(12, 8))

        for algo, algo_results in results.items():
            successful = [r for r in algo_results if r.success]
            if not successful:
                continue

            costs = [r.cost for r in successful]
            ax.hist(costs, bins=30, alpha=0.5, label=algo, edgecolor='black')

        ax.set_xlabel('Path Cost (USD)', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title('Distribution of Path Costs by Algorithm', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / output_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.plots_created.append(output_name)
        print(f"  Saved: {output_path}")


def main():
    """Main execution pipeline."""
    print("="*80)
    print(" COMPREHENSIVE VISUALIZATION AND EVALUATION PIPELINE")
    print("="*80)

    # Create output directories
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    plots_dir = results_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    # Initialize components
    print("\n[1/6] Initializing components...")
    evaluator = AlgorithmEvaluator(verbose=True)
    visualizer = ComprehensiveVisualizer(output_dir=plots_dir)
    generator = OpticalNetworkGenerator(seed=42)

    # Load ML models
    print("\n[2/6] Loading trained ML models...")
    models_dir = Path('models')

    rf_path = models_dir / 'random_forest.pkl'
    xgb_path = models_dir / 'xgboost.pkl'

    evaluator.load_ml_models(
        rf_path=str(rf_path) if rf_path.exists() else None,
        xgb_path=str(xgb_path) if xgb_path.exists() else None,
        nn_path=None  # Neural network not included in this evaluation
    )

    # Generate test networks
    print("\n[3/6] Generating test networks...")
    networks = []

    # Generate 10 diverse test networks
    for i in range(10):
        if i < 3:
            # Small networks
            G = generator.generate_random_geometric(25, radius=0.3, area_size=800)
            G.graph['topology_type'] = 'geometric'
        elif i < 6:
            # Medium networks
            G = generator.generate_scale_free(40, m=3, area_size=1200)
            G.graph['topology_type'] = 'scale_free'
        else:
            # Large networks
            rows = np.random.randint(6, 9)
            cols = np.random.randint(6, 9)
            G = generator.generate_grid(rows, cols, spacing=100)
            G.graph['topology_type'] = 'grid'

        # Generate test problems
        n_problems = 10  # 10 problems per network
        problems = generate_test_problems(G, n_problems=n_problems, seed=42+i)
        networks.append((G, problems))

        print(f"  Network {i+1}: {G.number_of_nodes()} nodes, "
              f"{G.number_of_edges()} edges, {len(problems)} problems")

    print(f"\nGenerated {len(networks)} test networks with "
          f"{sum(len(p) for _, p in networks)} total routing problems")

    # Define algorithms to evaluate
    algorithms = ['Dijkstra', 'A*', 'Random Forest', 'XGBoost']

    # Run evaluation
    print("\n[4/6] Running comprehensive evaluation...")
    results = evaluator.evaluate_multiple_networks(networks, algorithms)

    # Generate comparison report
    print("\n[5/6] Generating comparison metrics...")
    metrics = evaluator.generate_comparison_report()
    evaluator.print_comparison_report()

    # Save evaluation results
    print("\n[6/6] Generating visualizations and saving results...")

    # Save results to JSON
    output_path = results_dir / 'model_results.json'
    evaluator.save_results(str(output_path))

    # Generate all visualizations
    print("\nCreating visualizations...")

    # 1. Algorithm performance comparison
    visualizer.plot_algorithm_performance_comparison(metrics)

    # 2. Cost vs time scatter
    visualizer.plot_cost_vs_time_scatter(results)

    # 3. Feature importance
    if evaluator.rf_router and evaluator.xgb_router:
        visualizer.plot_feature_importance(evaluator.rf_router, evaluator.xgb_router)

    # 4. Network topology with route (use first successful path from first network)
    G_example, problems_example = networks[0]
    source, target = problems_example[0]
    dijkstra_result = results['Dijkstra'][0]
    if dijkstra_result.success:
        visualizer.plot_network_topology_with_route(
            G_example, dijkstra_result.path, source, target
        )

    # 5. Training results
    training_results_path = results_dir / 'training_results.json'
    if training_results_path.exists():
        with open(training_results_path, 'r') as f:
            training_results = json.load(f)
        visualizer.plot_training_results(training_results)

    # 6. Success rates
    visualizer.plot_algorithm_success_rates(metrics)

    # 7. Cost distribution
    visualizer.plot_cost_distribution(results)

    # Print summary
    print("\n" + "="*80)
    print(" EXECUTION SUMMARY")
    print("="*80)

    print(f"\nVisualizations Created: {len(visualizer.plots_created)}")
    for i, plot_name in enumerate(visualizer.plots_created, 1):
        print(f"  {i}. {plot_name}")

    print(f"\nEvaluation Metrics:")
    print(f"  Total Networks Evaluated: {len(networks)}")
    print(f"  Total Routing Problems: {sum(len(p) for _, p in networks)}")
    print(f"  Algorithms Evaluated: {len(algorithms)}")

    print(f"\nBest Performing Algorithms:")
    best_cost_algo, best_cost_metrics = evaluator.get_best_algorithm('cost')
    best_time_algo, best_time_metrics = evaluator.get_best_algorithm('time')
    best_success_algo, best_success_metrics = evaluator.get_best_algorithm('success_rate')

    print(f"  Best Cost: {best_cost_algo} (${best_cost_metrics.avg_cost:,.2f})")
    print(f"  Fastest: {best_time_algo} ({best_time_metrics.avg_time:.4f} ms)")
    print(f"  Most Reliable: {best_success_algo} ({best_success_metrics.success_rate:.1f}%)")

    print(f"\nOutput Files:")
    print(f"  Results: {output_path}")
    print(f"  Plots: {plots_dir}/")

    print("\n" + "="*80)
    print(" VISUALIZATION AND EVALUATION COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
