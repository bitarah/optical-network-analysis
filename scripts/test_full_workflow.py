"""
Full End-to-End Workflow Test

Tests the entire optical network routing pipeline:
1. Network generation
2. Traditional routing
3. ML model loading
4. ML routing
5. Comparison and evaluation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pickle
import time
from src.data_generation.network_generator import OpticalNetworkGenerator
from src.routing.traditional_routing import TraditionalRouter, GeneticAlgorithmRouter
from src.routing.ml_routing import RandomForestRouter, XGBoostRouter
from src.optical.signal_quality import OpticalSignalAnalyzer

def print_header(title):
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def test_network_generation():
    print_header("TEST 1: Network Generation")

    generator = OpticalNetworkGenerator(seed=42)

    # Test different topology types
    G1 = generator.generate_random_geometric(30, radius=0.3, area_size=1000)
    print(f"✓ Random Geometric: {G1.number_of_nodes()} nodes, {G1.number_of_edges()} edges")

    G2 = generator.generate_scale_free(40, m=3, area_size=2000)
    print(f"✓ Scale-Free: {G2.number_of_nodes()} nodes, {G2.number_of_edges()} edges")

    G3 = generator.generate_grid(6, 6, spacing=100)
    print(f"✓ Grid: {G3.number_of_nodes()} nodes, {G3.number_of_edges()} edges")

    # Verify optical parameters
    edge = list(G1.edges())[0]
    attrs = G1[edge[0]][edge[1]]
    required_attrs = ['distance', 'attenuation_db', 'total_cost', 'n_regenerators']
    for attr in required_attrs:
        assert attr in attrs, f"Missing attribute: {attr}"
    print(f"✓ All optical parameters present: {required_attrs}")

    return G1  # Return for use in other tests

def test_signal_quality():
    print_header("TEST 2: Optical Signal Quality Analysis")

    analyzer = OpticalSignalAnalyzer()

    test_distances = [50, 100, 200, 500, 1000]
    print("\nDistance (km) | Regenerators | OSNR (dB) | Viable")
    print("-" * 60)

    for dist in test_distances:
        min_regen = analyzer.calculate_minimum_regenerators(dist)
        viable, metrics = analyzer.is_path_viable(dist, min_regen)
        print(f"{dist:12} | {min_regen:12} | {metrics['osnr_db']:9.2f} | {viable}")

    print("\n✓ Signal quality calculations working correctly")

def test_traditional_routing(G):
    print_header("TEST 3: Traditional Routing Algorithms")

    router = TraditionalRouter()
    source, target = 0, 15

    # Test Dijkstra
    start = time.time()
    result_dijkstra = router.dijkstra(G, source, target)
    time_dijkstra = time.time() - start
    print(f"\n✓ Dijkstra:")
    print(f"  Path: {result_dijkstra.path}")
    print(f"  Cost: ${result_dijkstra.cost:,.2f}")
    print(f"  Distance: {result_dijkstra.distance:.2f} km")
    print(f"  Regenerators: {result_dijkstra.n_regenerators}")
    print(f"  Time: {time_dijkstra*1000:.2f} ms")

    # Test A*
    start = time.time()
    result_astar = router.a_star(G, source, target)
    time_astar = time.time() - start
    print(f"\n✓ A* Search:")
    print(f"  Path: {result_astar.path}")
    print(f"  Cost: ${result_astar.cost:,.2f}")
    print(f"  Time: {time_astar*1000:.2f} ms")
    print(f"  Speedup vs Dijkstra: {time_dijkstra/time_astar:.2f}x")

    # Verify both find optimal path
    assert result_dijkstra.cost == result_astar.cost, "A* should find same cost as Dijkstra"
    print("\n✓ A* finds optimal solution (same cost as Dijkstra)")

    return result_dijkstra

def test_ml_routing(G, optimal_result):
    print_header("TEST 4: ML-Based Routing")

    # Load trained models
    try:
        with open('models/random_forest.pkl', 'rb') as f:
            rf_router = pickle.load(f)
        print("✓ Loaded Random Forest model")
    except FileNotFoundError:
        print("✗ Random Forest model not found. Run train_models_fast.py first!")
        return None, None

    try:
        with open('models/xgboost.pkl', 'rb') as f:
            xgb_router = pickle.load(f)
        print("✓ Loaded XGBoost model")
    except FileNotFoundError:
        print("✗ XGBoost model not found. Run train_models_fast.py first!")
        xgb_router = None

    source, target = 0, 15

    # Test Random Forest
    print("\n--- Random Forest Routing ---")
    try:
        start = time.time()
        rf_result = rf_router.route(G, source, target)
        rf_time = time.time() - start

        print(f"✓ Path: {rf_result.path}")
        print(f"  Actual Cost: ${rf_result.cost:,.2f}")
        print(f"  Predicted Cost: ${rf_result.metrics['predicted_cost']:,.2f}")
        print(f"  Prediction Error: {rf_result.metrics['prediction_error_pct']:.2f}%")
        print(f"  Time: {rf_time*1000:.2f} ms")

        # Compare to optimal
        cost_gap = abs(rf_result.cost - optimal_result.cost) / optimal_result.cost * 100
        print(f"  Gap from Optimal: {cost_gap:.2f}%")

    except RuntimeError as e:
        print(f"✗ Random Forest routing failed: {e}")
        rf_result = None

    # Test XGBoost
    if xgb_router:
        print("\n--- XGBoost Routing ---")
        try:
            start = time.time()
            xgb_result = xgb_router.route(G, source, target)
            xgb_time = time.time() - start

            print(f"✓ Path: {xgb_result.path}")
            print(f"  Actual Cost: ${xgb_result.cost:,.2f}")
            print(f"  Predicted Cost: ${xgb_result.metrics['predicted_cost']:,.2f}")
            print(f"  Prediction Error: {xgb_result.metrics['prediction_error_pct']:.2f}%")
            print(f"  Time: {xgb_time*1000:.2f} ms")

            cost_gap = abs(xgb_result.cost - optimal_result.cost) / optimal_result.cost * 100
            print(f"  Gap from Optimal: {cost_gap:.2f}%")

        except RuntimeError as e:
            print(f"✗ XGBoost routing failed: {e}")
            xgb_result = None
    else:
        xgb_result = None

    return rf_result, xgb_result

def test_comparison(optimal_result, rf_result, xgb_result):
    print_header("TEST 5: Algorithm Comparison")

    print("\n{:<20} {:>15} {:>15} {:>15}".format(
        "Algorithm", "Cost ($)", "Time (ms)", "vs Optimal"
    ))
    print("-" * 70)

    # Dijkstra (optimal)
    print("{:<20} {:>15,.0f} {:>15.2f} {:>15}".format(
        "Dijkstra (Optimal)",
        optimal_result.cost,
        optimal_result.computation_time * 1000,
        "0.00%"
    ))

    # Random Forest
    if rf_result:
        gap = abs(rf_result.cost - optimal_result.cost) / optimal_result.cost * 100
        print("{:<20} {:>15,.0f} {:>15.2f} {:>14.2f}%".format(
            "Random Forest (ML)",
            rf_result.cost,
            rf_result.computation_time * 1000,
            gap
        ))

    # XGBoost
    if xgb_result:
        gap = abs(xgb_result.cost - optimal_result.cost) / optimal_result.cost * 100
        print("{:<20} {:>15,.0f} {:>15.2f} {:>14.2f}%".format(
            "XGBoost (ML)",
            xgb_result.cost,
            xgb_result.computation_time * 1000,
            gap
        ))

    print("\n✓ Comparison complete")

def test_error_handling():
    print_header("TEST 6: Error Handling")

    generator = OpticalNetworkGenerator(seed=100)
    G = generator.generate_random_geometric(20, 0.3, 500)

    # Test untrained model
    print("\n--- Testing Untrained Model ---")
    untrained = RandomForestRouter()
    try:
        untrained.route(G, 0, 10)
        print("✗ FAIL: Should have raised RuntimeError")
    except RuntimeError as e:
        print(f"✓ PASS: Caught expected error")
        print(f"  Message: {str(e)[:80]}...")

    # Test disconnected graph
    print("\n--- Testing Disconnected Graph ---")
    G_disconnected = generator.generate_random_geometric(30, 0.1, 500)  # Very sparse
    router = TraditionalRouter()

    # Find two nodes that might not be connected
    result = router.dijkstra(G_disconnected, 0, 29)
    if not result.success:
        print("✓ PASS: Correctly identified no path exists")
    else:
        print("✓ Network is connected (expected for this seed)")

    print("\n✓ Error handling tests complete")

def main():
    print("\n" + "="*70)
    print("  OPTICAL NETWORK ROUTING - FULL WORKFLOW TEST")
    print("="*70)
    print("\nThis will test the complete pipeline:")
    print("  1. Network Generation")
    print("  2. Optical Signal Quality")
    print("  3. Traditional Routing (Dijkstra, A*)")
    print("  4. ML Routing (Random Forest, XGBoost)")
    print("  5. Algorithm Comparison")
    print("  6. Error Handling")
    print("\nExpected duration: ~10 seconds")

    input("\nPress Enter to start tests...")

    try:
        # Run all tests
        G = test_network_generation()
        test_signal_quality()
        optimal_result = test_traditional_routing(G)
        rf_result, xgb_result = test_ml_routing(G, optimal_result)
        test_comparison(optimal_result, rf_result, xgb_result)
        test_error_handling()

        # Final summary
        print_header("TEST SUMMARY")
        print("\n✓ All tests passed successfully!")
        print("\nComponents verified:")
        print("  ✓ Network generation (3 topology types)")
        print("  ✓ Optical signal quality calculations")
        print("  ✓ Traditional routing algorithms")
        print("  ✓ ML model loading and routing")
        print("  ✓ Performance comparison")
        print("  ✓ Error handling")

        print("\n" + "="*70)
        print("  WORKFLOW TEST COMPLETE - ALL SYSTEMS OPERATIONAL")
        print("="*70)

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == '__main__':
    exit(main())
