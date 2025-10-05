"""
Test True ML-Guided Pathfinding

Verifies that ML models make their own routing decisions and produce
different (potentially suboptimal) paths compared to Dijkstra.
"""

import sys
sys.path.insert(0, 'src')

import pickle
from src.data_generation.network_generator import OpticalNetworkGenerator
from src.routing.traditional_routing import TraditionalRouter

print("="*80)
print("  TESTING TRUE ML-GUIDED PATHFINDING")
print("="*80)

# Generate test network
print("\n1. Generating test network...")
generator = OpticalNetworkGenerator(seed=42)
G = generator.generate_random_geometric(50, radius=0.3, area_size=1000)
print(f"✓ Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Load ML models
print("\n2. Loading ML models...")
try:
    with open('models/random_forest.pkl', 'rb') as f:
        rf_router = pickle.load(f)
    print("✓ Random Forest loaded")
except FileNotFoundError:
    print("✗ Random Forest not found. Run: python scripts/train_models_fast.py")
    sys.exit(1)

try:
    with open('models/xgboost.pkl', 'rb') as f:
        xgb_router = pickle.load(f)
    print("✓ XGBoost loaded")
except FileNotFoundError:
    print("✗ XGBoost not found")
    xgb_router = None

# Test routing on multiple problems
print("\n3. Testing routing on 10 problems...")
print("\n{:<15} {:>10} {:>10} {:>10} {:>12}".format(
    "Problem", "Dijkstra", "RF Cost", "XGB Cost", "RF Match?"
))
print("-" * 80)

trad_router = TraditionalRouter()
matches = 0
total = 0

for i in range(10):
    source = i * 5
    target = source + 20

    if source >= G.number_of_nodes() or target >= G.number_of_nodes():
        continue

    # Dijkstra (optimal)
    dijkstra_result = trad_router.dijkstra(G, source, target)
    if not dijkstra_result.success:
        continue

    # Random Forest (ML)
    rf_result = rf_router.route(G, source, target)

    # XGBoost (ML)
    if xgb_router:
        xgb_result = xgb_router.route(G, source, target)
        xgb_cost = xgb_result.cost
    else:
        xgb_cost = 0

    # Check if paths match
    paths_match = (rf_result.path == dijkstra_result.path)
    if paths_match:
        matches += 1
    total += 1

    match_str = "SAME ✓" if paths_match else "DIFF ✗"

    print("{:<15} ${:>9,.0f} ${:>9,.0f} ${:>9,.0f} {:>12}".format(
        f"{source}→{target}",
        dijkstra_result.cost,
        rf_result.cost,
        xgb_cost if xgb_cost > 0 else 0,
        match_str
    ))

print("-" * 80)

# Summary
print(f"\n4. Results Summary:")
print(f"  Total problems: {total}")
print(f"  ML paths matching Dijkstra: {matches}/{total} ({matches/total*100:.1f}%)")
print(f"  ML paths different: {total-matches}/{total} ({(total-matches)/total*100:.1f}%)")

if matches == total:
    print("\n⚠️  WARNING: All ML paths match Dijkstra!")
    print("  This means ML is still using shortest path internally.")
    print("  Expected: ML should make different choices on some problems.")
else:
    print("\n✓ SUCCESS: ML is making independent routing decisions!")
    print("  ML paths differ from optimal, showing true ML-guided pathfinding.")

# Detailed comparison of one different path
print("\n5. Detailed Path Comparison (first difference):")
for i in range(10):
    source = i * 5
    target = source + 20

    if source >= G.number_of_nodes() or target >= G.number_of_nodes():
        continue

    dijkstra_result = trad_router.dijkstra(G, source, target)
    if not dijkstra_result.success:
        continue

    rf_result = rf_router.route(G, source, target)

    if rf_result.path != dijkstra_result.path:
        print(f"\nProblem: {source} → {target}")
        print(f"  Dijkstra path: {dijkstra_result.path}")
        print(f"  Dijkstra cost: ${dijkstra_result.cost:,.2f}")
        print(f"  Dijkstra hops: {len(dijkstra_result.path)-1}")
        print(f"\n  ML path: {rf_result.path}")
        print(f"  ML cost: ${rf_result.cost:,.2f}")
        print(f"  ML hops: {len(rf_result.path)-1}")
        print(f"\n  Cost difference: ${rf_result.cost - dijkstra_result.cost:,.2f}")
        print(f"  ML overhead: {(rf_result.cost/dijkstra_result.cost - 1)*100:.2f}%")
        print(f"  Hop difference: {len(rf_result.path) - len(dijkstra_result.path)}")
        break

print("\n" + "="*80)
print("  TEST COMPLETE")
print("="*80)
