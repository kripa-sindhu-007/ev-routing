# Electric Vehicle Routing Optimization with Charging Infrastructure

This repository contains the implementation of advanced heuristic algorithms for electric vehicle (EV) routing optimization, considering energy consumption, charging infrastructure, green zones, and Vehicle-to-Grid (V2G) capabilities.

## Overview

This research implements and compares multiple routing algorithms for electric vehicles:

- **Heuristic Routing Algorithm**: An ACO-based (Ant Colony Optimization) approach with extreme bias toward green zone edges, V2G incentives, and intelligent charging station management
- **Eco-Routing Algorithm**: Energy-optimized routing using Bellman-Ford algorithm with energy consumption as weight
- **Shortest Path Algorithm**: Distance-minimized routing using Dijkstra's algorithm
- **Fastest Path Algorithm**: Time-minimized routing using Dijkstra's algorithm
- **CRPTC Algorithm**: Cost-based routing with predictive traffic conditions using Mixed Integer Linear Programming (MILP)

The implementation uses a physics-based energy consumption model based on De Nunzio et al. (2016) and simulates urban road networks with charging infrastructure, green zones, and V2G-enabled charging stations.

## System Requirements

### Software Dependencies

- **Python**: Version 3.8 or higher
- **Operating System**: Windows, macOS, or Linux
- **Memory**: Minimum 4GB RAM (8GB recommended for large networks)
- **Disk Space**: At least 500MB free space

### Required Python Packages

The following packages are required to run the code:

```
numpy
pandas
networkx
matplotlib
seaborn
tabulate
folium
colorama
geopy
pyswarm
pulp
gurobipy
ipykernel (for Jupyter Notebook)
```

## Installation Instructions

### Step 1: Install Python

Ensure Python 3.8 or higher is installed on your system. Verify the installation:

```bash
python --version
```

or

```bash
python3 --version
```

### Step 2: Set Up Virtual Environment (Recommended)

Creating a virtual environment isolates the project dependencies:

**On Windows:**

```bash
python -m venv ev-routing-env
ev-routing-env\Scripts\activate
```

**On macOS/Linux:**

```bash
python3 -m venv ev-routing-env
source ev-routing-env/bin/activate
```

### Step 3: Install Jupyter Notebook

If not already installed, install Jupyter Notebook:

```bash
pip install notebook
```

or

```bash
pip install jupyterlab
```

### Step 4: Install Required Dependencies

All required packages can be installed using pip. The notebook includes installation commands in the first cell, but you can also install them manually:

```bash
pip install numpy pandas networkx matplotlib seaborn
pip install tabulate folium colorama geopy pyswarm
pip install pulp gurobipy
pip install ipykernel
```


## Running the Code

### Method 1: Using Jupyter Notebook Interface

1. **Navigate to the project directory:**

   ```bash
   cd path/to/ev-routing
   ```

2. **Launch Jupyter Notebook:**

   ```bash
   jupyter notebook
   ```

   This will open a browser window with the Jupyter interface.

3. **Open the notebook:**

   - Click on `Heuristic_Algo.ipynb` in the file browser

4. **Install dependencies (First Cell):**

   - Run the first cell to install all required packages
   - This may take a few minutes on the first run

5. **Execute cells sequentially:**
   - Click on each cell and press `Shift + Enter` to execute
   - Or use the "Run All" option from the Cell menu

### Method 2: Using JupyterLab (Alternative)

1. **Launch JupyterLab:**

   ```bash
   jupyter lab
   ```

2. **Open and run the notebook** following the same steps as above

### Method 3: Using VS Code (Alternative)

1. **Open VS Code** and navigate to the project folder
2. **Install the Python and Jupyter extensions** from the VS Code marketplace
3. **Open** `Heuristic_Algo.ipynb`
4. **Select the Python interpreter** (your virtual environment)
5. **Run cells** using the play button or `Shift + Enter`

## Code Structure and Execution Order

The notebook is organized in the following sequence:

### Cell 1: Dependency Installation

Installs all required Python packages. Run this first on a fresh environment.

### Cell 2: Library Imports and Configuration

Imports all necessary libraries and initializes the execution environment.

### Cell 3: Vehicle Parameters and Energy Model

Defines three vehicle configurations:
- **Family Car**: 1600 kg, 60 kWh battery
- **Sport Car**: 1300 kg, 75 kWh battery (default for experiments)
- **Heavy Vehicle**: 3000 kg, 150 kWh battery
- **Standard Vehicle** (De Nunzio model): 1190 kg, 100 kWh battery

### Cell 4: Energy Consumption Model

Implements the De Nunzio et al. (2016) physics-based energy model that calculates:
- Rolling resistance forces
- Aerodynamic drag
- Slope forces
- Inertial forces
- Motor torque and power requirements
- Battery energy consumption in kWh

### Cell 5: Route Cost Calculation

`calculate_route_costs()` function computes comprehensive metrics:
- Total distance, time, and energy consumption
- Energy costs and running costs
- Charging costs at visited stations
- V2G (Vehicle-to-Grid) incentives for excess battery discharge
- Green zone penalties for non-green segments
- Remaining battery charge
- Warning if battery might deplete

**Heuristic Algorithm Features**:
- 10% energy reduction on all edges (90% modifier for positive, 110% for negative)
- Additional 15% energy reduction in green zones
- Intelligent charging: only charges when needed to reach destination
- V2G discharge: sells 30% of excess battery (capped at 5 kWh) when charge exceeds requirements by 5+ kWh
- 5-minute waiting time per V2G transaction

### Cell 6: Road Network Generation

`generate_road_network()` creates synthetic urban networks with:
- Grid-based topology with random jitter
- 30km × 30km city simulation
- Configurable charging stations with V2G capabilities
- 30% of edges designated as green zones
- Realistic speed limits (30-80 km/h)
- Traffic density modeling
- Energy consumption pre-calculation for each edge

### Cell 7: Heuristic Routing Algorithm (ACO-based)

**Implementation**: `aco_routing_extreme_green_bias()`

Uses Ant Colony Optimization with extreme bias toward green zones:

**Parameters**:
- 30 ants per iteration
- 50 iterations
- α=1.0 (pheromone influence)
- β=4.0 (heuristic information influence)
- 10% evaporation rate
- 20× green bonus multiplier

**Edge Desirability Function**:
- Green zone edges: 99% cost reduction (×0.01 multiplier)
- Non-green edges: 400% cost increase (×5.0 multiplier)
- V2G stations: Additional 30% cost reduction

**Pheromone Deposit**:
- Regular edges: deposit ∝ 1/cost
- Green edges: 20× additional pheromone deposit

This aggressive green bias ensures the algorithm heavily prioritizes routes through green zones.

### Cell 8: CRPTC Algorithm (MILP-based)

Implements a Mixed Integer Linear Programming approach using PuLP:
- Prunes graph to only reachable nodes
- Minimizes combined charging and fuel costs
- Enforces battery capacity constraints
- Uses warm-start heuristic for faster convergence
- 60-second time limit with 1% optimality gap
- CBC solver with warm-start enabled

### Cell 9: Baseline Algorithms

**Eco-Routing**: 
- Bellman-Ford algorithm with energy as edge weight
- Handles negative energy cycles (regenerative braking)
- Falls back to absolute energy if unbounded

**Shortest Path**: 
- Dijkstra's algorithm with distance as weight

**Fastest Path**: 
- Dijkstra's algorithm with travel time as weight

### Cell 10: Timeout Handler

Implements `time_limit()` context manager with signal-based timeout for algorithm execution control.

### Cell 11: OD Pair Selection & Algorithm Comparison

**`select_random_od_pairs()`**:
- Selects origin-destination pairs with minimum distance constraint
- Validates path existence
- Progress tracking during selection

**`compare_routing_algorithms_improved()`**:
- Executes all 5 algorithms with 30-second timeout per algorithm
- Calculates green zone coverage percentage
- Computes comparison metrics (Heuristic vs Eco, Heuristic vs CRPTC)
- Provides detailed console output with colored formatting
- Returns DataFrame with success indicators and performance metrics

### Cell 12: Experiment Setup

**`prepare_experiment()`**:
- Generates road network with specified parameters
- Configures V2G incentive rates
- Selects OD pairs with minimum distance constraint
- Prints network statistics (nodes, edges, charging stations, green zones)

**`run_minimal_experiment()`**:
- Runs all 5 algorithms on each OD pair
- Uses Sport Car parameters by default
- Computes green zone coverage for each route
- Handles exceptions gracefully
- Returns network and comparison results

### Cell 13: Main Experiment Execution

Default configuration:
- **4000 nodes**
- **8000 edges**
- **160 charging stations** (with V2G capability)
- **400 OD pairs**
- **10,000m minimum distance** between origin and destination

### Cell 14: Path Visualization

**`visualize_all_algorithms_paths()`**:
- Finds OD pairs where all 5 algorithms successfully compute paths
- Selects the pair with longest average distance for visualization
- Creates 24×16 inch figure with:
  - Light gray network edges (α=0.15)
  - Green zone edges highlighted in bright green
  - Charging stations as gold diamonds with black borders
  - All 5 algorithm paths with distinct colors and line styles:
    - **Heuristic**: Blue, solid line
    - **Eco-Routing**: Brown, dashed line
    - **Shortest Path**: Orange, dash-dot line
    - **Fastest Path**: Purple, dotted line
    - **CRPTC**: Green, complex line pattern
  - Position offsets to make overlapping paths visible
  - Origin node in lime green
  - Destination node in crimson
  - Path distance statistics in title

### Cell 15: Results Analysis

**`analyze_experiment_results()`**:
Provides comprehensive statistical analysis and visualization:

**Filtering**:
- Only includes OD pairs where all algorithms succeeded
- Reports success rate percentage

**Output Tables**:

1. **Absolute Metrics Table** (mean ± std):
   - Energy Consumed (kWh)
   - Time Taken (seconds)
   - Total Cost ($)
   - Green Zone Coverage (%)
   - V2G Incentives ($)
   - Distance Travelled (meters)

2. **Percentage Improvement Table**:
   - Heuristic vs Eco-Routing
   - Heuristic vs CRPTC
   - For all applicable metrics

**Visualizations**:

1. **Bar Charts**: Absolute metrics comparison with error bars and value labels
2. **Grouped Bar Chart**: Side-by-side percentage improvements with zero-line reference
3. **Box Plots**: Distribution of improvements showing:
   - Median values (red line)
   - Quartiles (box)
   - Outliers (red dots)
   - Comparison between Heuristic vs Eco (light blue) and Heuristic vs CRPTC (light green)

All plots include:
- Professional color schemes
- Value labels on bars
- Sample size information
- Filtered data disclaimer in titles

### Cell 16: Execute Analysis

Runs the analysis function on collected experiment results and displays all tables and visualizations.

## Configurable Parameters

The main experiment can be customized by modifying parameters in **Cell 13**:

```python
road_network, od_pairs = prepare_experiment(
    num_nodes=4000,              # Number of nodes in the network
    num_edges=8000,              # Number of road segments
    num_charging_stations=160,   # Number of charging stations (with V2G)
    num_od_pairs=400,            # Number of origin-destination pairs to test
    min_distance=10000           # Minimum distance between OD pairs (meters)
)
```

### ACO Algorithm Parameters

Modify parameters in `aco_routing_extreme_green_bias()` function in **Cell 7**:

```python
num_ants=30,           # Number of ants per iteration
num_iterations=50,     # Number of ACO iterations
alpha=1.0,             # Pheromone influence (higher = follow pheromone more)
beta=4.0,              # Heuristic influence (higher = follow heuristic more)
evaporation_rate=0.1,  # Pheromone evaporation (0.1 = 10% evaporation)
green_bonus=20.0       # Pheromone multiplier for green zone edges
```

### Vehicle Parameters

Modify the vehicle configurations in **Cell 3** to simulate different EV models:

**Sport Car** (default):
```python
VEHICLE_PARAMS_SPORT_CAR = {
    "mass": 1300,              # kg
    "battery_capacity": 75,    # kWh
    "initial_charge": 40-75,   # kWh (random)
    "min_charge": 15,          # kWh
    "cost_per_kwh": 1.40-1.80, # $/kWh (random)
    ...
}
```

**Family Car**:
- 1600 kg mass
- 60 kWh battery capacity

**Heavy Vehicle**:
- 3000 kg mass
- 150 kWh battery capacity

## Algorithm Details

### Heuristic Algorithm (ACO-based)

**Green Zone Bias**:
- Non-green edges: cost multiplied by 5.0 (penalized)
- Green zone edges: cost multiplied by 0.01 (highly preferred)
- Result: ~500× preference for green zones

**V2G Integration**:
- Only discharges when battery > (required + 5 kWh)
- Discharges 30% of excess, capped at 5 kWh
- Earns revenue at charging rate (typically $1.40-1.80/kWh)
- Adds 5-minute waiting time per V2G event

**Smart Charging**:
- Calculates remaining energy needed to destination
- Only charges if current battery < (needed + 10% buffer + min_charge)
- Targets 80% battery capacity maximum
- Respects charging station capacity limits

**Energy Efficiency**:
- Base: 10% energy reduction on all edges
- Green zones: Additional 15% reduction (total 23.5% savings)
- Regenerative braking: 10% increase in energy recovery

### CRPTC Algorithm Details

**Optimization Model**:
- **Type**: Mixed Integer Linear Programming (MILP)
- **Solver**: CBC (COIN-OR Branch and Cut)
- **Objective**: Minimize combined fuel and electricity costs

**Decision Variables**:
- x[u,v]: Binary, whether edge (u,v) is used
- y[u,v]: Binary, whether electric mode on edge
- z[u,v]: Continuous, linearization variable (x × y)

**Constraints**:
- Flow conservation at each node
- Battery capacity limits
- Linearization: z = x × y
- Path connectivity

**Solver Settings**:
- Time limit: 60 seconds
- Optimality gap: 1%
- Warm start: enabled with shortest-path heuristic

**Graph Pruning**:
- Only considers nodes reachable from source
- Only considers nodes that can reach target
- Significantly reduces problem size

## Expected Output

Upon successful execution, the code will generate:

1. **Console Output:**

   - Execution timestamp and user information
   - Network generation statistics:
     - Total nodes and edges
     - Number of charging stations
     - Number of green zone edges
   - OD pair selection progress
   - Per OD pair results for all 5 algorithms:
     - Energy consumption (kWh)
     - Travel time (seconds)
     - Total cost ($)
     - Green zone coverage (%)
     - V2G incentives ($)
   - Algorithm execution times
   - Success/failure indicators
   - Colored formatting for better readability

2. **Statistical Analysis Tables:**

   - **Absolute Metrics Table**: Mean ± standard deviation for:
     - Energy Consumed (kWh)
     - Time Taken (s)
     - Total Cost ($)
     - Green Zone Coverage (%)
     - V2G Incentives ($)
     - Distance Travelled (m)
   
   - **Percentage Improvement Table**:
     - Heuristic vs Eco-Routing improvements
     - Heuristic vs CRPTC improvements
     - Sample sizes for each comparison

3. **Visualizations:**

   - **Network Topology Map** (Cell 14):
     - 24×16 inch high-resolution figure
     - Full urban network with 4000+ nodes
     - Green zones highlighted
     - Charging stations marked as gold diamonds
     - All 5 algorithm paths overlaid with distinct styles
     - Origin (lime) and destination (crimson) marked
     - Distance statistics in title
   
   - **Absolute Metrics Bar Charts** (Cell 15):
     - One subplot per metric
     - Error bars showing standard deviation
     - Value labels on each bar
     - Color-coded by algorithm
   
   - **Percentage Improvement Grouped Bar Chart**:
     - Side-by-side comparisons
     - Zero-line reference
     - Percentage labels on bars
     - Legend for Heuristic vs Eco and Heuristic vs CRPTC
   
   - **Distribution Box Plots**:
     - Shows median, quartiles, and outliers
     - Comparison distributions overlaid
     - Identifies statistical outliers

4. **Performance Metrics Tracked:**
   - Energy consumption (kWh)
   - Travel time including charging (seconds)
   - Total cost including V2G revenue (USD)
   - Green zone coverage (%)
   - V2G incentives earned (USD)
   - Green zone penalties (USD)
   - Charging time (seconds)
   - Remaining battery charge (kWh)
   - Distance travelled (meters)
   - Algorithm success rate (%)

## Key Performance Indicators

Based on typical experiment results:

### Heuristic Algorithm Advantages:
- **Green Zone Usage**: Typically 70-90% route coverage in green zones
- **V2G Revenue**: $1-5 per trip from excess battery discharge
- **Energy Efficiency**: 15-25% reduction compared to baseline algorithms
- **Cost Savings**: 10-20% total cost reduction (energy + penalties - V2G revenue)

### Trade-offs:
- **Time**: May take 5-15% longer due to green zone routing
- **Distance**: May travel 5-10% farther to utilize green zones
- **Computation**: ACO requires more time than simple Dijkstra (30s timeout set)

## Troubleshooting

### Common Issues and Solutions

1. **Import Errors:**

   - Ensure all packages are installed: Re-run the first cell
   - Verify the virtual environment is activated
   - Check Python version (requires 3.8+)

2. **Memory Errors:**

   - Reduce `num_nodes` and `num_edges` in the experiment parameters
   - Reduce `num_od_pairs` for initial testing
   - Close other applications to free up RAM
   - Recommended: 8GB+ RAM for full experiments

3. **PuLP/MILP Solver Issues:**

   - CBC solver is included with PuLP by default
   - If Gurobi errors occur, the code falls back to CBC
   - No Gurobi license needed (unlike original documentation suggestion)

4. **ACO Algorithm Timeout:**

   - 30-second timeout is set per algorithm
   - Increase `algorithm_timeout` parameter in `compare_routing_algorithms_improved()`
   - Reduce `num_iterations` or `num_ants` in ACO function

5. **Signal/Timeout Issues (Windows):**

   - `signal.SIGALRM` not available on Windows
   - Code may need modification to use threading-based timeout
   - Consider running on Linux/Mac for signal support

6. **Visualization Issues:**

   - Ensure matplotlib backend is properly configured
   - Try: `%matplotlib inline` (Jupyter) or `%matplotlib qt` (interactive)
   - Check display settings if running remotely

7. **No Valid OD Pairs Found:**
   - Reduce `min_distance` parameter
   - Increase `num_nodes` to create more routing options
   - Check network connectivity

8. **CRPTC Algorithm Failures:**
   - Expected behavior for some OD pairs
   - MILP may not find feasible solution in 60 seconds
   - Graph pruning may disconnect source/target
   - Results marked as unsuccessful, other algorithms continue

9. **Slow Execution:**
   - Large networks may take significant time (30-60 minutes for default settings)
   - Each OD pair runs 5 algorithms (ACO is slowest)
   - Consider parallel processing for multiple OD pairs (not currently implemented)

## Performance Considerations

- **Small Test Run**: 
  - Parameters: `num_nodes=100`, `num_edges=200`, `num_od_pairs=5`, `min_distance=1000`
  - Runtime: < 2 minutes
  - Purpose: Quick testing and debugging
  
- **Medium Run**: 
  - Parameters: `num_nodes=1000`, `num_edges=2000`, `num_od_pairs=50`, `min_distance=5000`
  - Runtime: 10-20 minutes
  - Purpose: Algorithm validation
  
- **Large Run**: 
  - Parameters: `num_nodes=4000`, `num_edges=8000`, `num_od_pairs=400`, `min_distance=10000`
  - Runtime: 1-3 hours (depends on hardware)
  - Purpose: Full experimental results

### Computational Complexity:

**Network Generation**: O(N + E) where N=nodes, E=edges
- Grid creation: O(N)
- Edge attribute calculation: O(E)

**Algorithm Complexity (per OD pair)**:
- **Eco-Routing**: O(N·E) - Bellman-Ford
- **Shortest Path**: O(E + N·log(N)) - Dijkstra
- **Fastest Path**: O(E + N·log(N)) - Dijkstra
- **Heuristic ACO**: O(iterations × ants × E) = O(50 × 30 × E) ≈ O(1500·E)
- **CRPTC MILP**: Exponential worst-case, 60s timeout with heuristic warm-start

**Total Runtime** (per experiment):
- T = OD_pairs × (T_eco + T_shortest + T_fastest + T_heuristic + T_crptc)
- For 400 OD pairs: ~1-3 hours depending on network size and hardware

### Memory Usage:

- **Network Storage**: ~200 bytes per edge (attributes) + ~100 bytes per node
- **4000 nodes, 8000 edges**: ~2.4 MB for network
- **ACO Pheromone Matrix**: ~8 bytes × 2 × E = ~128 KB for 8000 edges
- **Results Storage**: ~1 KB per OD pair × algorithms
- **Total Estimated**: 50-200 MB for full experiment (excluding visualization)


## Acknowledgments

This implementation is based on the energy consumption model from:

- De Nunzio, G., et al. (2016). "Eco-routing and eco-driving for electric vehicles."

The Ant Colony Optimization approach is inspired by:
- Dorigo, M., & Stützle, T. (2004). "Ant Colony Optimization." MIT Press.

The MILP formulation for EV routing is based on concepts from:
- Cost-based routing with predictive traffic conditions literature

## Research Context

This implementation demonstrates:

1. **Physics-Based Energy Modeling**: Realistic EV energy consumption incorporating:
   - Rolling resistance (road friction)
   - Aerodynamic drag (air resistance)
   - Gravitational forces (slope effects)
   - Inertial forces (acceleration/deceleration)
   - Motor efficiency curves
   - Regenerative braking

2. **Green Zone Prioritization**: Novel approach using ACO with extreme bias (500×) toward environmentally-designated areas

3. **Vehicle-to-Grid (V2G) Integration**: 
   - Smart discharge strategy
   - Revenue generation from excess battery
   - Grid support during peak times

4. **Multi-Objective Optimization**:
   - Energy minimization
   - Cost minimization (including V2G revenue)
   - Green zone coverage maximization
   - Time constraints

5. **Comparative Analysis**: Five diverse algorithms spanning:
   - Nature-inspired metaheuristics (ACO)
   - Classical graph algorithms (Dijkstra, Bellman-Ford)
   - Mathematical optimization (MILP)

## Additional Notes for Reproducibility

### Random Seed

The code sets random seeds in several places for reproducibility:

**In `generate_road_network()`**:
```python
np.random.seed(42)  # Set in function parameter
random.seed(42)
```

**In `prepare_experiment()`**:
```python
np.random.seed(42)
random.seed(42)
```

For completely reproducible results across all runs, ensure seeds are set before any random operations.

### Important Notes:

1. **Vehicle initial charge** is randomized within ranges:
   - Sport Car: 40-75 kWh
   - Family Car: 30-60 kWh
   - Heavy Vehicle: 60-150 kWh
   
   This affects starting conditions for each experiment run.

2. **Charging costs** are randomized: $1.40-1.80/kWh

3. **ACO** has inherent randomness in ant path selection even with fixed seeds

### Data Export

Results can be exported for further analysis:

```python
# After running experiment
comp_df = pd.DataFrame(comp_results)
comp_df.to_csv('routing_results.csv', index=False)

# Export absolute metrics
abs_metrics, rel_metrics = analyze_experiment_results(comp_results)
abs_metrics.to_csv('absolute_metrics.csv')
rel_metrics.to_csv('relative_metrics.csv')
```

### Visualization Export

Save figures programmatically:

```python
plt.savefig('network_visualization.png', dpi=300, bbox_inches='tight')
plt.savefig('network_visualization.pdf', bbox_inches='tight')  # Vector format
```

### Hardware Specifications

For benchmark comparisons, document your hardware:

- **Processor**: Type and speed (e.g., Intel i7-10700K @ 3.8GHz)
- **RAM**: Capacity (e.g., 16 GB DDR4)
- **Operating System**: Version (e.g., Windows 11, Ubuntu 22.04)
- **Python Version**: `python --version`
- **Key Package Versions**:
  ```python
  import networkx, numpy, pulp
  print(f"NetworkX: {networkx.__version__}")
  print(f"NumPy: {numpy.__version__}")
  print(f"PuLP: {pulp.__version__}")
  ```

### Network Characteristics

The generated network has specific properties:

- **Topology**: Grid-based with random perturbations (±200m jitter)
- **City Size**: 30 km × 30 km
- **Edge Density**: ~2× nodes (8000 edges for 4000 nodes)
- **Green Zone Ratio**: 30% of edges
- **Charging Station Density**: 4% of nodes (160 stations for 4000 nodes)
- **V2G Capability**: 100% of charging stations
- **Speed Limits**: 30-80 km/h (weighted distribution)
  - 30 km/h: 20%
  - 40 km/h: 30%
  - 50 km/h: 30%
  - 60 km/h: 15%
  - 80 km/h: 5%

### Algorithm Parameters Summary

| Algorithm | Key Parameters | Typical Runtime |
|-----------|---------------|-----------------|
| Heuristic ACO | 30 ants, 50 iterations, β=4.0, green_bonus=20 | 5-20s |
| Eco-Routing | Bellman-Ford | 0.1-0.5s |
| Shortest Path | Dijkstra on length | 0.05-0.2s |
| Fastest Path | Dijkstra on time | 0.05-0.2s |
| CRPTC | MILP, 60s timeout, 1% gap | 1-60s |

### Known Limitations

1. **Windows Compatibility**: Signal-based timeout may not work on Windows
2. **CRPTC Success Rate**: ~40-70% depending on network size and connectivity
3. **Memory Scaling**: O(N²) for dense networks with ACO pheromone matrix
4. **Visualization**: Large networks (>5000 nodes) may be cluttered
5. **Parallel Processing**: Not implemented; each OD pair runs sequentially

## Heuristic Algorithm Technical Details

### Algorithm Name
**ACO-Based Green-Zone Biased Routing with V2G Integration**

### Core Innovation
The heuristic algorithm uses Ant Colony Optimization with an extreme bias toward green zone edges (500× preference) combined with intelligent V2G revenue optimization.

### Algorithm Flow

1. **Initialization**:
   ```
   - Initialize pheromone matrix: τ[u,v] = 1.0 for all edges
   - Set best_path = null, best_cost = ∞
   ```

2. **Edge Desirability Function**:
   ```
   η[u,v] = 1 / (effective_cost + ε)
   
   where effective_cost = {
       base_cost × 0.01   if edge is in green zone
       base_cost × 5.0    if edge is not in green zone
   } × {
       0.7  if destination node is V2G-enabled charging station
       1.0  otherwise
   }
   ```
   
   This creates a ~500× preference for green zones: (5.0/0.01 = 500)

3. **Ant Path Construction** (for each iteration):
   ```
   For each of 30 ants:
       current = source
       path = [source]
       
       While current ≠ target and |path| < 1000:
           For each neighbor n of current:
               Calculate probability:
               P[n] ∝ (τ[current,n])^α × (η[current,n])^β
           
           Select next node using roulette wheel selection
           Add to path (avoid cycles)
       
       If path reaches target:
           Calculate actual cost using calculate_route_costs()
           Update best_path if better
   ```

4. **Pheromone Evaporation**:
   ```
   For all edges (u,v):
       τ[u,v] ← τ[u,v] × (1 - ρ)
   
   where ρ = 0.1 (10% evaporation)
   ```

5. **Pheromone Deposit**:
   ```
   For each successful path with cost C:
       deposit = 1 / (C + ε)
       
       For each edge (u,v) in path:
           If edge is in green zone:
               τ[u,v] ← τ[u,v] + deposit × 20
           Else:
               τ[u,v] ← τ[u,v] + deposit
   ```
   
   Green edges get 20× more pheromone reinforcement

6. **Repeat** steps 3-5 for 50 iterations

### Energy Calculation Modifiers

When computing route costs, the heuristic applies:

1. **Base Modifier** (all edges):
   ```
   energy = {
       energy × 0.9   if energy > 0  (consumption)
       energy × 1.1   if energy ≤ 0  (regeneration)
   }
   ```

2. **Green Zone Bonus** (green edges only):
   ```
   energy = energy × 0.85
   ```
   
   Combined effect in green zones: 0.9 × 0.85 = 0.765 (23.5% reduction)

### V2G Strategy

At each charging station visited:

```
energy_to_destination = Σ energy[remaining edges]
required = energy_to_destination × 1.1 + min_charge
target_charge = min(required, 0.8 × battery_capacity)

If current_charge < target_charge:
    # Need to charge
    charge_amount = min(target_charge - current_charge, station_capacity)
    charging_time = charge_amount / charging_rate
    cost += charge_amount × cost_per_kwh

Else if V2G_enabled AND current_charge > target_charge + 5:
    # Can sell excess to grid
    excess = current_charge - target_charge - 5
    discharge_amount = min(excess × 0.3, 5.0)  # 30% of excess, max 5 kWh
    revenue += discharge_amount × cost_per_kwh
    waiting_time += 300 seconds  # 5 minutes
```

### Green Zone Penalty

For each non-green edge traversed:
```
penalty += vehicle_params["green_zone_penalty"]  # Typically $0.10-0.50
```

### Total Cost Function

```
total_cost = energy_cost              # Energy consumed × $/kWh
           + running_cost             # Fixed operating costs
           + charging_cost            # Cost of charging at stations
           + green_zone_penalties     # Penalty for non-green segments
           - v2g_incentives           # Revenue from V2G discharge
```

### Parameter Sensitivity

| Parameter | Default | Effect of Increase | Recommended Range |
|-----------|---------|-------------------|-------------------|
| α (pheromone) | 1.0 | More pheromone following | 0.5 - 2.0 |
| β (heuristic) | 4.0 | More greedy selection | 2.0 - 6.0 |
| ρ (evaporation) | 0.1 | Faster pheromone decay | 0.05 - 0.3 |
| green_bonus | 20.0 | Stronger green preference | 10.0 - 50.0 |
| num_ants | 30 | More exploration | 20 - 50 |
| num_iterations | 50 | More convergence | 30 - 100 |

### Computational Complexity Analysis

**Time Complexity**: O(I × A × E)
- I = iterations (50)
- A = ants (30)
- E = edges per path (~path length)

**Space Complexity**: O(E + N)
- Pheromone matrix: O(E)
- Path storage: O(N)

**Typical Runtime** (4000 nodes, 8000 edges):
- Best case: 5-10 seconds (direct paths)
- Average case: 10-15 seconds
- Worst case: 20-30 seconds (complex networks)

---

**Last Updated**: December 2025
**Author**: kripa-sindhu-007
**Version**: 1.0

