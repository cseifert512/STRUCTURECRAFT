# StructureCraft - 3D Spaceframe Design Engine

A structural analysis and design exploration platform for 3D spaceframes and canopy structures. Built with a modern tech stack: **Next.js frontend → FastAPI backend → mini_branch FEM engine**.

## What is This?

StructureCraft is a **structural design tool** that enables rapid exploration of 3D canopy and spaceframe designs. It combines:

- **Finite Element Analysis (FEM)**: Linear elastic analysis for 2D frames and 3D trusses
- **Design Exploration**: Generate and evaluate hundreds of design variants
- **Engineering Code Checks**: Timber design checks per CSA O86 / NDS
- **Buckling Analysis**: Eigenvalue buckling for stability assessment
- **Interactive 3D Visualization**: Real-time WebGL rendering with Three.js

## Features

### Structural Analysis
✅ **2D Frame Elements**: Euler-Bernoulli beam elements with axial, shear, and bending  
✅ **3D Truss Elements**: Axial-only members with direction cosine transformations  
✅ **Linear Static Analysis**: Solve `K × d = F` for displacements  
✅ **Buckling Analysis**: Eigenvalue problem `(K - λKg)φ = 0` for critical load factors  
✅ **Timber Code Checks**: Axial and combined utilization per NDS 3.9.2  
✅ **Mechanism Detection**: Condition number checks to catch unstable structures  

### Design Exploration
✅ **Parametric Generation**: Canopy structures with configurable topology, shape, and supports  
✅ **Batch Evaluation**: Evaluate hundreds of design variants automatically  
✅ **Pareto Frontier**: Identify optimal trade-offs (volume vs displacement vs complexity)  
✅ **ML Surrogate Model**: RandomForest for accelerated design screening  
✅ **Guided Search**: ML-guided candidate selection with physics verification  

### Web Application
✅ **Interactive 3D Viewer**: Real-time canopy visualization with Three.js  
✅ **Parameter Controls**: Sliders and dropdowns for design parameters  
✅ **Engineering Metrics**: Displacement, forces, utilization, buckling factor  
✅ **Color-Coded Members**: Visual feedback by force or utilization status  
✅ **Export**: CSV cut lists and JSON model exports  

## Tech Stack

```
┌─────────────────────────────────────────────────────────┐
│                    Next.js Frontend                      │
│  React + TypeScript + Three.js + Zustand + Tailwind     │
└─────────────────────────┬───────────────────────────────┘
                          │ REST API
┌─────────────────────────▼───────────────────────────────┐
│                   FastAPI Backend                        │
│        Pydantic models + async endpoints                 │
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│                 mini_branch Engine                       │
│   FEM solver + buckling + timber checks + generation    │
└─────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites
- Python 3.10+
- Node.js 18+
- pip

### Backend Setup

```powershell
cd STRUCTURECRAFT

# Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install Python dependencies
pip install -e .
pip install -r api/requirements.txt
```

### Frontend Setup

```powershell
cd web
npm install
```

### Running the Application

**Terminal 1 - Backend:**
```powershell
cd STRUCTURECRAFT
uvicorn api.main:app --reload --port 8000
```

**Terminal 2 - Frontend:**
```powershell
cd STRUCTURECRAFT/web
npm run dev
```

Open http://localhost:3000 in your browser.

## Quick Start - CLI Demos

### Beam Analysis
```powershell
py demos/run_single_case.py          # Cantilever beam
py demos/run_simply_supported.py     # Simply supported beam
py demos/run_udl_beam.py             # Beam with distributed load
py demos/run_portal_frame.py         # Portal frame
```

### 3D Canopy Analysis
```powershell
py demos/run_canopy_single.py        # Single canopy design
py demos/run_canopy_search.py        # Batch canopy search
py demos/run_space_truss.py          # 3D truss tetrahedron
```

### Design Exploration
```powershell
py demos/run_portal_search.py --n 500 --seed 42
py demos/train_surrogate.py
py demos/guided_search.py
```

### Run Tests
```powershell
pytest tests/ -q
```

## Project Structure

```
STRUCTURECRAFT/
├── api/                      # FastAPI backend
│   ├── main.py              # REST endpoints
│   └── requirements.txt     # Backend dependencies
│
├── web/                      # Next.js frontend
│   ├── src/
│   │   ├── app/             # Next.js app router
│   │   ├── components/      # React components
│   │   │   ├── Canvas3D.tsx     # Three.js 3D viewer
│   │   │   ├── ControlPanel.tsx # Parameter controls
│   │   │   ├── MetricsCard.tsx  # Engineering metrics display
│   │   │   └── ExplorePanel.tsx # Batch exploration UI
│   │   ├── lib/             # API client and types
│   │   └── store/           # Zustand state management
│   └── package.json
│
├── mini_branch/              # FEM engine
│   ├── kernel/              # Core solver infrastructure
│   │   ├── dof.py           # DOF management
│   │   ├── assemble.py      # Matrix assembly
│   │   ├── solve.py         # Linear solver
│   │   └── buckling.py      # Eigenvalue buckling analysis
│   ├── checks/              # Design code checks
│   │   └── timber.py        # Timber capacity checks (NDS/CSA O86)
│   ├── v3d/                 # 3D truss elements
│   │   ├── model.py         # Node3D, Truss3D
│   │   └── elements.py      # 3D truss stiffness
│   ├── generative/          # Parametric geometry
│   │   └── canopy.py        # Canopy generator
│   ├── model.py             # 2D frame models
│   ├── elements.py          # 2D frame stiffness
│   ├── assembly.py          # 2D assembly
│   ├── loads.py             # Load vectors
│   ├── solve.py             # 2D solver
│   ├── post.py              # Post-processing
│   ├── catalog.py           # Materials and sections
│   ├── explore.py           # 2D design exploration
│   ├── pareto.py            # Pareto analysis
│   └── viz.py               # Matplotlib visualization
│
├── tests/                    # Test suite (51 tests)
├── demos/                    # Example scripts
├── artifacts/                # Generated outputs
└── pyproject.toml           # Package configuration
```

## Engineering Features

### Buckling Analysis

The engine performs eigenvalue buckling analysis to determine critical load factors:

```python
from mini_branch.kernel.buckling import critical_buckling_factor, build_global_Kg

# After linear solve, compute geometric stiffness
Kg = build_global_Kg(nodes, bars, forces, dof_manager, element_geometry_3d)

# Solve eigenvalue problem
lambda_cr = critical_buckling_factor(K, Kg, fixed_dofs)

# lambda_cr > 1.0 means structure is stable under current loads
```

### Timber Code Checks

Implements timber design checks per CSA O86 / NDS:

```python
from mini_branch.checks.timber import DOUGLAS_FIR_CAPACITY, axial_utilization

# Check member utilization
util = axial_utilization(axial_force, area, DOUGLAS_FIR_CAPACITY)
# util < 1.0 = PASS, util >= 1.0 = FAIL
```

### 3D Truss Analysis

Full 3D truss analysis with direction cosine transformations:

```python
from mini_branch.v3d.elements import truss3d_global_stiffness, truss3d_axial_force

# 6x6 element stiffness from direction cosines
ke = truss3d_global_stiffness(nodes, element)

# Extract axial force after solve
N = truss3d_axial_force(nodes, element, d_global)
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/generate` | POST | Generate and analyze a canopy design |
| `/api/explore` | POST | Batch exploration of design variants |
| `/api/export/csv` | POST | Export cut list as CSV |
| `/api/export/json` | POST | Export model as JSON |

### Response Fields

The `/api/generate` endpoint returns:

```typescript
{
  success: boolean
  nodes: NodeData[]
  bars: BarData[]      // includes force, utilization, status
  metrics: {
    max_displacement_mm: number
    max_tension_kn: number
    max_compression_kn: number
    volume: number
    n_length_bins: number
    // Engineering checks
    buckling_factor: number      // > 1.0 = stable
    max_utilization: number      // < 1.0 = all members pass
    n_failing_members: number    // count of overstressed members
  }
}
```

## Validation Status

### ✅ 2D Frame Analysis
- Cantilever beam: `δ = PL³/(3EI)` ✓
- Simply supported: `δ = PL³/(48EI)` ✓
- UDL beam: `δ = 5wL⁴/(384EI)` ✓
- Portal frames with combined loading ✓

### ✅ 3D Truss Analysis
- Tetrahedron equilibrium ✓
- Space truss symmetry ✓
- Canopy structures with various topologies ✓

### ✅ Engineering Checks
- Buckling factor calculation ✓
- Timber utilization (tension/compression) ✓
- Combined interaction checks ✓

### ✅ Full Stack Integration
- Next.js ↔ FastAPI communication ✓
- Real-time 3D visualization ✓
- Engineering metrics display ✓
- Utilization-based coloring ✓

## Dependencies

### Python
- **numpy**: Matrix operations
- **scipy**: Eigenvalue solver for buckling
- **pandas**: Data analysis
- **scikit-learn**: ML surrogate model
- **fastapi**: REST API
- **uvicorn**: ASGI server
- **pydantic**: Data validation

### Node.js
- **next**: React framework
- **three**: 3D graphics
- **@react-three/fiber**: React Three.js bindings
- **@react-three/drei**: Three.js helpers
- **zustand**: State management
- **tailwindcss**: Styling

## What's Next?

- [x] 2D frame analysis ✅
- [x] 3D truss analysis ✅
- [x] Buckling analysis ✅
- [x] Timber code checks ✅
- [x] Web application ✅
- [x] Design exploration ✅
- [x] ML surrogate model ✅
- [ ] P-Delta (second-order) analysis
- [ ] Modal/dynamic analysis
- [ ] Steel code checks (AISC)
- [ ] Connection modeling
- [ ] Export to IFC/CAD formats

## License

Educational project - use freely for learning!

---

**Status**: ✅ Engineering Depth Upgrade Complete - Buckling analysis, timber checks, full-stack integration
