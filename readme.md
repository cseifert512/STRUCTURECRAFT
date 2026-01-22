# STRUCTURECRAFT - Mini Branch

A lightweight finite element analysis (FEA) framework for 2D frame structures, built from scratch to learn structural engineering and computational mechanics.

## What is This?

This is a **structural analysis engine** that can analyze beams, frames, and other 2D structures. It uses the **Finite Element Method (FEM)** to calculate:

- **Displacements**: How much structures deform under loads
- **Reactions**: Forces at supports
- **Internal forces**: Stresses and moments within elements

Think of it as a mini version of commercial software like SAP2000, ETABS, or ANSYS - but built for learning and understanding the fundamentals.

## Features

✅ **2D Frame Elements**: Euler-Bernoulli beam elements with axial, shear, and bending  
✅ **Linear Static Analysis**: Solve `K × d = F` for displacements  
✅ **Boundary Conditions**: Support constraints (pinned, fixed, etc.)  
✅ **Uniformly Distributed Loads (UDL)**: Support for distributed loads on beam elements  
✅ **Portal Frames**: Multi-element frame structures with combined loading  
✅ **Post-processing**: Extract reactions, internal forces, moments, and engineering metrics  
✅ **Validated Results**: Tested against closed-form analytical solutions  
✅ **Visualization**: Matplotlib plots showing deformed shapes and force diagrams  
✅ **Design Space Exploration**: Generate and evaluate hundreds of design variants  
✅ **Pareto Frontier Analysis**: Identify optimal trade-offs between competing objectives  
✅ **Parametric Design**: Material/section catalogs and parametric model generation  
✅ **Batch Evaluation**: Automated evaluation pipeline for design optimization  

## Installation

### Prerequisites
- Python 3.10 or higher
- pip

### Setup

1. **Clone or navigate to the project directory:**
   ```powershell
   cd STRUCTURECRAFT
   ```

2. **Create a virtual environment (recommended):**
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1  # On Windows PowerShell
   ```

3. **Install the package in editable mode:**
   ```powershell
   pip install -e .
   ```

4. **Install development dependencies (for testing):**
   ```powershell
   pip install -e ".[dev]"
   ```

## Quick Start

### Run a Demo

**Cantilever Beam:**
```powershell
py demos/run_single_case.py
```

**Simply Supported Beam:**
```powershell
py demos/run_simply_supported.py
```

**Simply Supported Beam with UDL:**
```powershell
py demos/run_udl_beam.py
```

**UDL Validation:**
```powershell
py demos/run_udl_validation.py
```

**Portal Frame (Gravity + Lateral Loads):**
```powershell
py demos/run_portal_frame.py
```

**Post-processing Demo:**
```powershell
py demos/run_postprocessing.py
```

**Design Space Exploration (Day 3):**
```powershell
py demos/run_portal_search.py --n 500 --seed 42
```
This generates:
- `artifacts/results.csv` - All design results
- `artifacts/pareto.png` - Pareto frontier plot
- `artifacts/top10.md` - Top 10 optimal designs
- `artifacts/winner_frame.png` - Deformed shape visualization

### Run Tests

```powershell
pytest tests/
```

Or run specific tests:
```powershell
pytest tests/test_cantilever.py -v
pytest tests/test_simply_supported.py -v
pytest tests/test_simply_supported_udl.py -v
pytest tests/test_portal_frame.py -v
pytest tests/test_portal_frame_equilibrium.py -v
pytest tests/test_postprocessing.py -v
pytest tests/test_invariants.py -v
pytest tests/test_explore.py -v
pytest tests/test_search_pipeline_smoke.py -v
```

## Project Structure

```
STRUCTURECRAFT/
├── mini_branch/          # Main package
│   ├── model.py         # Data models (Node, Frame2D)
│   ├── elements.py      # Element stiffness matrices
│   ├── assembly.py      # Global stiffness matrix assembly
│   ├── loads.py         # Load assembly (point loads, UDL)
│   ├── solve.py         # Linear solver with boundary conditions
│   ├── post.py          # Post-processing (reactions, forces, moments)
│   ├── checks.py        # Engineering checks
│   ├── catalog.py       # Material and section catalogs (Day 3)
│   ├── explore.py       # Parametric design generation and evaluation (Day 3)
│   ├── pareto.py        # Pareto frontier analysis (Day 3)
│   └── viz.py           # Visualization functions (Day 3)
├── tests/               # Test suite
│   ├── test_cantilever.py
│   ├── test_simply_supported.py
│   ├── test_simply_supported_udl.py
│   ├── test_portal_frame.py
│   ├── test_portal_frame_equilibrium.py
│   ├── test_postprocessing.py
│   ├── test_invariants.py
│   ├── test_explore.py
│   └── test_search_pipeline_smoke.py
├── demos/               # Example scripts
│   ├── run_single_case.py
│   ├── run_simply_supported.py
│   ├── run_udl_beam.py
│   ├── run_udl_validation.py
│   ├── run_portal_frame.py
│   ├── run_postprocessing.py
│   └── run_portal_search.py  # Design space exploration (Day 3)
├── artifacts/            # Generated results (Day 3)
│   ├── results.csv
│   ├── pareto.png
│   ├── top10.md
│   └── winner_frame.png
└── pyproject.toml       # Package configuration
```

## Validation Status

### ✅ Day 1 Deliverables (Completed)

**Validated Beam Cases:**
1. **Cantilever Beam** - Tip point load
   - ✅ Deflection matches: `δ = PL³/(3EI)`
   - ✅ Rotation matches: `θ = PL²/(2EI)`
   - ✅ Reactions verified

2. **Simply Supported Beam** - Midspan point load
   - ✅ Reactions match: `R = P/2` at each support
   - ✅ Deflection matches: `δ = PL³/(48EI)`

**Engineering Invariants:**
- ✅ Stiffness matrix symmetry: `K = K^T`
- ✅ Force equilibrium: `ΣF_y = 0`
- ✅ Moment equilibrium: `ΣM = 0`

### ✅ Day 2 Deliverables (Completed)

**Uniformly Distributed Loads (UDL):**
- ✅ Simply supported beam with UDL validated against: `δ_max = 5wL⁴/(384EI)`
- ✅ Reactions match: `R = wL/2` at each support
- ✅ Moment distribution matches analytical solutions

**Portal Frames:**
- ✅ Stability checks (no mechanism errors)
- ✅ Equilibrium validation (forces and moments balance)
- ✅ Combined loading (gravity + lateral loads)
- ✅ Drift calculations for building code compliance

**Post-processing:**
- ✅ Reaction extraction at supports
- ✅ Internal force and moment diagrams
- ✅ Engineering metrics (drift, maximum moments, etc.)

### ✅ Day 3 Deliverables (Completed)

**Design Space Exploration:**
- ✅ Parametric portal frame generator (`make_portal`)
- ✅ Material and section catalog (12 timber sections)
- ✅ Variant sampling with fabrication constraints
- ✅ Batch evaluation pipeline (hundreds of designs)
- ✅ Metrics extraction (drift, moments, volume, carbon)

**Pareto Frontier Analysis:**
- ✅ Non-dominated sorting algorithm
- ✅ Pareto-optimal design identification
- ✅ Trade-off visualization (volume vs. drift)

**Visualization:**
- ✅ Pareto frontier plots
- ✅ Deformed shape visualization
- ✅ Design summaries (top 10 markdown report)

**Results Export:**
- ✅ CSV export for all design results
- ✅ Ready for Day 4 ML training

All tests pass and match closed-form analytical solutions within numerical tolerance.

## How It Works

### The Core Equation

The fundamental equation of structural analysis:

```
F = K × d
```

Where:
- **F** = Applied forces vector
- **K** = Global stiffness matrix (how stiff the structure is)
- **d** = Displacements vector (how much it moves) ← **What we solve for**

### The Process

1. **Model**: Define nodes (points) and elements (beam segments)
2. **Assemble**: Build the global stiffness matrix from element contributions
3. **Constrain**: Apply boundary conditions (supports)
4. **Solve**: Compute displacements: `d = K⁻¹ × F`
5. **Post-process**: Extract reactions, internal forces, etc.

### Example: Cantilever Beam

```python
from mini_branch.model import Node, Frame2D
from mini_branch.assembly import assemble_global_K, DOF_PER_NODE
from mini_branch.solve import solve_linear

# Define structure
nodes = {
    0: Node(0, 0.0, 0.0),  # Fixed end
    1: Node(1, 3.0, 0.0),  # Free end
}
elements = [Frame2D(0, 0, 1, E=210e9, A=0.01, I=8e-6)]

# Build stiffness matrix
K = assemble_global_K(nodes, elements)

# Apply load (1000 N downward at tip)
F = np.zeros(DOF_PER_NODE * len(nodes))
F[DOF_PER_NODE*1 + 1] = -1000.0

# Fix left end (cantilever)
fixed = [0, 1, 2]  # ux, uy, rz at node 0

# Solve!
d, R, _ = solve_linear(K, F, fixed)

# Results
tip_deflection = d[DOF_PER_NODE*1 + 1]  # -0.00536 m (downward)
reaction = R[1]  # 1000 N (upward)
```

## Dependencies

- **numpy**: Matrix operations and linear algebra
- **matplotlib**: Visualization and plotting
- **pandas**: Data analysis and CSV export (Day 3)
- **pytest**: Testing framework (dev dependency)

## What's Next?

Future enhancements planned:

- [x] Uniformly distributed loads (UDL) ✅
- [x] Portal frames ✅
- [x] Post-processing and visualization improvements ✅
- [x] Design space exploration and Pareto analysis ✅
- [ ] ML-guided design optimization (Day 4)
- [ ] Multi-story structures
- [ ] Dynamic analysis (modal, time-history)
- [ ] Nonlinear analysis (material/geometric nonlinearity)
- [ ] Code compliance (AISC/NDS)
- [ ] Export to CAD formats

## Learning Resources

This project implements concepts from:
- **Finite Element Method (FEM)**: Matrix structural analysis
- **Structural Mechanics**: Beam theory, Euler-Bernoulli beams
- **Linear Algebra**: Matrix operations, solving systems of equations
- **Multi-Objective Optimization**: Pareto optimality and trade-off analysis
- **Design Space Exploration**: Parametric design and batch evaluation

## License

Educational project - use freely for learning!

## Contributing

This is a learning project. Feel free to explore, modify, and experiment!

---

**Status**: ✅ Day 3 Complete - Design Space Exploration, Pareto Analysis, and Visualization implemented

See `planning_and_docs/day3.md` for detailed documentation of Day 3 implementation.

