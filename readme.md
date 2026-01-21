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
│   └── ...
├── tests/               # Test suite
│   ├── test_cantilever.py
│   ├── test_simply_supported.py
│   ├── test_simply_supported_udl.py
│   ├── test_portal_frame.py
│   ├── test_portal_frame_equilibrium.py
│   ├── test_postprocessing.py
│   └── test_invariants.py
├── demos/               # Example scripts
│   ├── run_single_case.py
│   ├── run_simply_supported.py
│   ├── run_udl_beam.py
│   ├── run_udl_validation.py
│   ├── run_portal_frame.py
│   └── run_postprocessing.py
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
- **pytest**: Testing framework (dev dependency)

## What's Next?

Future enhancements planned:

- [x] Uniformly distributed loads (UDL) ✅
- [x] Portal frames ✅
- [x] Post-processing and visualization improvements ✅
- [ ] Multi-story structures
- [ ] Dynamic analysis (modal, time-history)
- [ ] Nonlinear analysis (material/geometric nonlinearity)
- [ ] Optimization capabilities
- [ ] Export to CAD formats

## Learning Resources

This project implements concepts from:
- **Finite Element Method (FEM)**: Matrix structural analysis
- **Structural Mechanics**: Beam theory, Euler-Bernoulli beams
- **Linear Algebra**: Matrix operations, solving systems of equations

## License

Educational project - use freely for learning!

## Contributing

This is a learning project. Feel free to explore, modify, and experiment!

---

**Status**: ✅ Day 2 Complete - UDL, Portal Frames, and Post-processing validated

