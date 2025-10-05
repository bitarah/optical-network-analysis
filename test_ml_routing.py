"""
Test ML routing with no fallbacks - should raise errors explicitly
"""

import sys
import os
sys.path.insert(0, 'src')

import pickle
from data_generation.network_generator import OpticalNetworkGenerator
from routing.ml_routing import RandomForestRouter, XGBoostRouter

print("="*60)
print("TEST: ML Routing with Explicit Error Handling")
print("="*60)

# Generate test network
print("\n1. Generating test network...")
generator = OpticalNetworkGenerator(seed=42)
G = generator.generate_random_geometric(20, radius=0.3, area_size=500)
print(f"✓ Network created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Test untrained model (should raise RuntimeError)
print("\n2. Testing UNTRAINED model (should raise RuntimeError)...")
try:
    untrained_router = RandomForestRouter()
    result = untrained_router.route(G, 0, 10)
    print("✗ FAIL: Should have raised RuntimeError!")
except RuntimeError as e:
    print(f"✓ PASS: Caught expected error")
    print(f"  Error message: {e}")

# Load trained model
print("\n3. Loading TRAINED model...")
try:
    with open('models/random_forest.pkl', 'rb') as f:
        trained_router = pickle.load(f)
    print(f"✓ Model loaded: {trained_router.model_name}")
    print(f"  is_trained flag: {trained_router.is_trained}")
except FileNotFoundError:
    print("✗ Model file not found. Run train_models_fast.py first!")
    sys.exit(1)

# Test trained model
print("\n4. Testing TRAINED model routing...")
try:
    result = trained_router.route(G, 0, 10)
    print(f"✓ Routing successful!")
    print(f"  Algorithm: {result.algorithm}")
    print(f"  Path length: {len(result.path)} hops")
    print(f"  Cost: ${result.cost:,.2f}")
    print(f"  Computation time: {result.computation_time*1000:.2f}ms")
    if result.metrics:
        print(f"  Predicted cost: ${result.metrics['predicted_cost']:,.2f}")
        print(f"  Prediction error: {result.metrics['prediction_error_pct']:.2f}%")
except RuntimeError as e:
    print(f"✗ FAIL: {e}")

print("\n5. Testing XGBoost model...")
try:
    with open('models/xgboost.pkl', 'rb') as f:
        xgb_router = pickle.load(f)
    result = xgb_router.route(G, 0, 15)
    print(f"✓ XGBoost routing successful!")
    print(f"  Path: {result.path[:5]}... (showing first 5 nodes)")
    print(f"  Cost: ${result.cost:,.2f}")
except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "="*60)
print("TEST COMPLETE")
print("="*60)
print("\nKey Changes:")
print("  • Removed fallback_router attribute")
print("  • Removed silent fallback to Dijkstra")
print("  • Added explicit RuntimeError when model not trained")
print("  • Added prediction_error metrics to results")
print("  • ML models now fail loudly with clear error messages")
