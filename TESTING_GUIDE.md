# Testing Guide - Optical Network Route Optimizer

## Quick Test (5 minutes)

### 1. Test ML Routing with Trained Models
```bash
python test_ml_routing.py
```

**What it tests:**
- ✅ Untrained models raise explicit errors
- ✅ Trained models route successfully
- ✅ Prediction metrics are computed
- ✅ No silent fallbacks

---

## Full Test Suite (30 minutes)

### Step 1: Test Data Generation (5 min)
```bash
# Generate a small test dataset
python scripts/generate_dataset.py
```

**Expected output:**
- 2,569 training samples
- 50 synthetic networks
- 2 real networks (Abilene, GEANT)
- `data/processed/training_data.csv` created

---

### Step 2: Test Model Training (2 min)
```bash
# Train models quickly
python scripts/train_models_fast.py
```

**Expected output:**
- Random Forest: R² ~0.937, training time ~0.2s
- XGBoost: R² ~0.937, training time ~0.1s
- Models saved to `models/` directory

---

### Step 3: Test Traditional Algorithms (1 min)
```bash
python -c "
import sys
sys.path.insert(0, 'src')

from data_generation.network_generator import OpticalNetworkGenerator
from routing.traditional_routing import TraditionalRouter, GeneticAlgorithmRouter

# Generate test network
gen = OpticalNetworkGenerator(42)
G = gen.generate_random_geometric(30, 0.3, 1000)

# Test Dijkstra
router = TraditionalRouter()
result = router.dijkstra(G, 0, 20)
print(f'✓ Dijkstra: {len(result.path)} hops, cost: \${result.cost:,.2f}')

# Test A*
result = router.a_star(G, 0, 20)
print(f'✓ A*: {len(result.path)} hops, cost: \${result.cost:,.2f}')

# Test Genetic Algorithm (slow!)
ga = GeneticAlgorithmRouter(population_size=20, generations=10)
result = ga.route(G, 0, 20)
print(f'✓ GA: {len(result.path)} hops, cost: \${result.cost:,.2f}')
"
```

---

### Step 4: Test ML Algorithms (1 min)
```bash
python test_ml_routing.py
```

---

### Step 5: Test Evaluation Framework (5 min)
```bash
python test_evaluator.py
```

**Expected output:**
- Evaluation of 10 routing problems
- Performance comparison table
- Results saved to `data/quick_evaluation_results.json`

---

### Step 6: Test Network Generation (2 min)
```bash
python -c "
import sys
sys.path.insert(0, 'src')

from data_generation.network_generator import OpticalNetworkGenerator

gen = OpticalNetworkGenerator(42)

# Test all topology types
G1 = gen.generate_random_geometric(50, 0.3, 1000)
print(f'✓ Random Geometric: {G1.number_of_nodes()} nodes, {G1.number_of_edges()} edges')

G2 = gen.generate_scale_free(60, 3, 2000)
print(f'✓ Scale-Free: {G2.number_of_nodes()} nodes, {G2.number_of_edges()} edges')

G3 = gen.generate_grid(7, 7, 100)
print(f'✓ Grid: {G3.number_of_nodes()} nodes, {G3.number_of_edges()} edges')

# Check optical parameters
edge = list(G1.edges())[0]
attrs = G1[edge[0]][edge[1]]
print(f'✓ Edge attributes: {list(attrs.keys())}')
assert 'distance' in attrs
assert 'total_cost' in attrs
assert 'n_regenerators' in attrs
print('✓ All optical parameters present')
"
```

---

### Step 7: Test Signal Quality Calculations (1 min)
```bash
python -c "
import sys
sys.path.insert(0, 'src')

from optical.signal_quality import OpticalSignalAnalyzer

analyzer = OpticalSignalAnalyzer()

# Test distance calculations
for dist in [50, 100, 200, 500]:
    min_regen = analyzer.calculate_minimum_regenerators(dist)
    viable, metrics = analyzer.is_path_viable(dist, min_regen)
    print(f'Distance: {dist}km → Regenerators: {min_regen}, OSNR: {metrics[\"osnr_db\"]}dB, Viable: {viable}')

print('✓ Signal quality calculations working')
"
```

---

## Integration Tests

### Full End-to-End Test
Create and run this complete workflow test:

```bash
python scripts/test_full_workflow.py
```

Let me create that script for you:

