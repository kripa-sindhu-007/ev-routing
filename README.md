# Electric Vehicle Routing Optimization with Charging Infrastructure

This repository contains the implementation of heuristic algorithms for electric vehicle (EV) routing optimization, considering energy consumption, charging infrastructure, green zones, and Vehicle-to-Grid (V2G) capabilities.

## Overview

This research implements and compares multiple routing algorithms for electric vehicles:

- **Heuristic Routing Algorithm**: A novel approach incorporating green zones and V2G incentives
- **Eco-Routing Algorithm**: Energy-optimized routing
- **Shortest Path Algorithm**: Distance-minimized routing
- **Fastest Path Algorithm**: Time-minimized routing
- **CRPTC Algorithm**: Cost-based routing with predictive traffic conditions

The implementation uses a physics-based energy consumption model based on De Nunzio et al. (2016) and simulates urban road networks with charging infrastructure.

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

**Note**: Gurobi requires a license. For academic use, free academic licenses are available at [https://www.gurobi.com/academia/](https://www.gurobi.com/academia/). Alternatively, the code can run without Gurobi with minor modifications to the optimization components.

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

Defines the electric vehicle specifications and implements the De Nunzio energy consumption model.

### Cell 4-8: Routing Algorithms Implementation

Contains the core implementations of:

- Heuristic routing algorithm with green zones
- Eco-routing algorithm
- Shortest path algorithm
- Fastest path algorithm
- CRPTC algorithm

### Cell 9-10: Network Generation and Experiment Setup

Functions to generate synthetic urban road networks with:

- Configurable number of nodes and edges
- Charging station placement
- Green zone designation
- Origin-destination (OD) pair generation

### Cell 11: Experiment Execution

Runs the comparative experiment with all algorithms on multiple OD pairs.

### Cell 12-14: Visualization and Results

Generates visualizations and comparative analysis of the routing algorithms.

## Configurable Parameters

The main experiment can be customized by modifying parameters in **Cell 11**:

```python
road_network, od_pairs = prepare_experiment(
    num_nodes=4000,              # Number of nodes in the network
    num_edges=8000,              # Number of road segments
    num_charging_stations=160,   # Number of charging stations
    num_od_pairs=10,             # Number of origin-destination pairs to test
    min_distance=5000            # Minimum distance between OD pairs (meters)
)
```

### Vehicle Parameters

Modify the `VEHICLE_PARAMETERS` dictionary in **Cell 3** to simulate different EV models:

- Battery capacity
- Motor specifications
- Charging costs
- Initial charge level

## Expected Output

Upon successful execution, the code will generate:

1. **Console Output:**

   - Execution timestamp and user information
   - Progress updates for each OD pair
   - Energy consumption, travel time, cost, and green zone coverage for each algorithm
   - Comparative metrics

2. **Visualizations:**

   - Network topology with charging stations and green zones
   - Route comparisons on 2D maps
   - Performance comparison charts

3. **Performance Metrics:**
   - Energy consumption (kWh)
   - Travel time (seconds)
   - Total cost (USD)
   - Green zone coverage (%)
   - V2G incentives (USD)

## Troubleshooting

### Common Issues and Solutions

1. **Import Errors:**

   - Ensure all packages are installed: Re-run the first cell
   - Verify the virtual environment is activated

2. **Memory Errors:**

   - Reduce `num_nodes` and `num_edges` in the experiment parameters
   - Close other applications to free up RAM

3. **Gurobi License Error:**

   - Obtain and activate a Gurobi license
   - Or modify the code to use alternative solvers (e.g., GLPK, CBC)

4. **Visualization Issues:**

   - Ensure matplotlib backend is properly configured
   - Try running: `%matplotlib inline` before visualization cells

5. **Slow Execution:**
   - Large networks may take significant time
   - Consider reducing the number of OD pairs for initial testing
   - Use smaller network sizes during development

## Performance Considerations

- **Small Test Run**: Use `num_nodes=100`, `num_edges=200`, `num_od_pairs=5` for quick testing (< 1 minute)
- **Medium Run**: Use `num_nodes=1000`, `num_edges=2000`, `num_od_pairs=10` (5-10 minutes)
- **Full Experiment**: Default parameters `num_nodes=4000`, `num_edges=8000` (30-60 minutes)

## Citation

If you use this code in your research, please cite:

```
[Your Research Paper Citation]
Author(s), Title, Journal Name, Year
```

## License

[Specify your license here]

## Contact Information

For questions or issues regarding this implementation:

- **Author**: kripa-sindhu-007
- **GitHub**: [Repository URL]
- **Email**: [Your email address]

## Acknowledgments

This implementation is based on the energy consumption model from:

- De Nunzio, G., et al. (2016). "Eco-routing and eco-driving for electric vehicles."

## Additional Notes for Reproducibility

### Random Seed

The code uses random number generation for network creation and vehicle parameters. For reproducible results, you can set a random seed at the beginning of the notebook:

```python
import random
import numpy as np
random.seed(42)
np.random.seed(42)
```

### Data Export

Results can be exported for further analysis:

```python
comp_df = pd.DataFrame(comp_results)
comp_df.to_csv('routing_results.csv', index=False)
```

### Hardware Specifications

For benchmark comparisons, document your hardware specifications:

- Processor type and speed
- RAM capacity
- Operating system version

---

**Last Updated**: December 2025  
**Version**: 1.0
