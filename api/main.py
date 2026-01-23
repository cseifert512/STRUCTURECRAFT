# api/main.py
"""
FastAPI backend for StructureCraft - exposes mini_branch engine as REST API.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Tuple
import sys
from pathlib import Path
import io
import csv
import json

# Add project root to path to import mini_branch
sys.path.insert(0, str(Path(__file__).parent.parent))

from mini_branch.generative import generate_canopy, CanopyParams
from mini_branch.generative.canopy import compute_length_bins
from mini_branch.v3d.elements import truss3d_global_stiffness, truss3d_axial_force, element_geometry_3d
from mini_branch.kernel.dof import DOFManager
from mini_branch.kernel.assemble import assemble_global_K
from mini_branch.kernel.solve import solve_linear, MechanismError
import numpy as np
import pandas as pd


app = FastAPI(
    title="StructureCraft API",
    description="3D Spaceframe Design Engine",
    version="1.0.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Request/Response Models
# =============================================================================

class DesignParams(BaseModel):
    """Input parameters for canopy generation."""
    width: float = Field(10.0, ge=4.0, le=20.0, description="Footprint width (m)")
    depth: float = Field(8.0, ge=4.0, le=16.0, description="Footprint depth (m)")
    nx: int = Field(5, ge=2, le=10, description="Grid divisions X")
    ny: int = Field(4, ge=2, le=10, description="Grid divisions Y")
    min_height: float = Field(2.5, ge=1.5, le=6.0, description="Min height (m)")
    max_height: float = Field(4.0, ge=1.5, le=6.0, description="Max height (m)")
    heightfield: str = Field("paraboloid", description="Shape: flat, paraboloid, ridge, saddle")
    topology: str = Field("grid", description="Topology: grid, diagrid, triangulated")
    support_layout: str = Field("edges", description="Support: edges, corners, perimeter_4")
    A_cm2: float = Field(8.0, ge=3.0, le=20.0, description="Cross-section area (cm²)")
    gravity_kn: float = Field(50.0, ge=20.0, le=150.0, description="Gravity load (kN)")


class NodeData(BaseModel):
    """Node geometry data."""
    id: int
    x: float
    y: float
    z: float


class BarData(BaseModel):
    """Bar element data."""
    id: int
    ni: int
    nj: int
    length: float
    force: float


class MetricsData(BaseModel):
    """Performance metrics."""
    n_nodes: int
    n_bars: int
    n_supports: int
    max_displacement: float
    max_displacement_mm: float
    max_tension: float
    max_tension_kn: float
    max_compression: float
    max_compression_kn: float
    volume: float
    total_length: float
    n_length_bins: int
    max_member_length: float
    max_member_length_mm: float
    min_member_length: float


class DesignResult(BaseModel):
    """Complete design result."""
    success: bool
    error: Optional[str] = None
    nodes: Optional[List[NodeData]] = None
    bars: Optional[List[BarData]] = None
    support_nodes: Optional[List[int]] = None
    metrics: Optional[MetricsData] = None
    params: Optional[Dict[str, Any]] = None


# =============================================================================
# Exploration Models
# =============================================================================

class ExploreParams(BaseModel):
    """Parameters for batch exploration."""
    base_params: DesignParams = Field(..., description="Base design to vary around")
    n_designs: int = Field(100, ge=10, le=500, description="Number of variants")
    seed: int = Field(42, ge=1, description="Random seed")
    variation_pct: float = Field(0.3, ge=0.1, le=0.5, description="Variation percentage")


class ExploreDesignData(BaseModel):
    """Single design in exploration results."""
    index: int
    ok: bool
    reason: Optional[str] = None
    # Parameters
    width: float
    depth: float
    nx: int
    ny: int
    min_height: float
    max_height: float
    heightfield: str
    topology: str
    support_layout: str
    A_cm2: float
    gravity_kn: float
    # Metrics (optional if failed)
    volume: Optional[float] = None
    max_displacement: Optional[float] = None
    max_displacement_mm: Optional[float] = None
    n_length_bins: Optional[int] = None
    max_member_length_mm: Optional[float] = None
    max_tension_kn: Optional[float] = None
    max_compression_kn: Optional[float] = None
    n_nodes: Optional[int] = None
    n_bars: Optional[int] = None
    # Pareto flag
    is_pareto: bool = False


class ExploreResult(BaseModel):
    """Batch exploration result."""
    success: bool
    error: Optional[str] = None
    n_total: int = 0
    n_successful: int = 0
    n_pareto: int = 0
    designs: List[ExploreDesignData] = []


# =============================================================================
# Stability Check
# =============================================================================

def check_stability(topology: str, support_layout: str, nx: int, ny: int, heightfield: str) -> Optional[str]:
    """Check if parameter combination is likely stable."""
    
    # Flat, ridge, and saddle shapes need more supports than paraboloid
    # Paraboloid (dome) has inherent membrane stiffness from curvature
    if heightfield in ['flat', 'ridge', 'saddle']:
        if support_layout == 'corners':
            return f"Unstable: '{heightfield}' shape needs 'edges' support (not enough curvature for corner supports only)"
    
    if support_layout == 'corners':
        if topology == 'diagrid':
            return "Unstable: 'corners' support with 'diagrid' topology fails. Use 'edges' or 'grid'."
        if topology == 'triangulated':
            return "Risky: 'corners' with 'triangulated' may be unstable."
        if nx < 3 or ny < 3:
            return "Unstable: 'corners' needs at least 3x3 grid."
    
    if support_layout == 'perimeter_4':
        if nx < 4 or ny < 4:
            return "Too few supports: 'perimeter_4' needs at least 4x4 grid."
    
    return None


# =============================================================================
# Design Generation
# =============================================================================

def generate_and_solve(params: DesignParams) -> DesignResult:
    """Generate canopy structure and solve it. Returns geometry even if solve fails."""
    
    print(f"\n{'='*60}")
    print(f"[DEBUG] GENERATE REQUEST")
    print(f"[DEBUG] heightfield: {params.heightfield}")
    print(f"[DEBUG] topology: {params.topology}")
    print(f"[DEBUG] support_layout: {params.support_layout}")
    print(f"[DEBUG] grid: {params.nx}x{params.ny}")
    print(f"{'='*60}")
    
    # Convert params
    A = params.A_cm2 / 10000  # cm² to m²
    gravity_load = -params.gravity_kn * 1000  # kN to N, negative for down
    
    # Auto-adjust support layout for non-dome shapes
    # Flat, ridge, saddle shapes need edge supports for stability
    effective_support = params.support_layout
    if params.heightfield in ['flat', 'ridge', 'saddle'] and params.support_layout == 'corners':
        effective_support = 'edges'  # Auto-upgrade to edges for stability
        print(f"[INFO] Auto-adjusted support from 'corners' to 'edges' for '{params.heightfield}' shape")
    
    print(f"[DEBUG] effective_support: {effective_support}")
    
    try:
        canopy_params = CanopyParams(
            width=params.width,
            depth=params.depth,
            nx=params.nx,
            ny=params.ny,
            max_height=max(params.max_height, params.min_height + 0.1),
            min_height=params.min_height,
            heightfield=params.heightfield,
            topology=params.topology,
            support_layout=effective_support,  # Use auto-adjusted support
            E=210e9,
            A=A,
            gravity_load=gravity_load,
        )
        
        # Generate structure (geometry only)
        nodes, bars, fixed_dofs, F = generate_canopy(canopy_params)
        print(f"[DEBUG] Generated: {len(nodes)} nodes, {len(bars)} bars, {len(fixed_dofs)} fixed DOFs")
    except Exception as e:
        return DesignResult(success=False, error=f"Failed to generate geometry: {str(e)}")
    
    n_nodes = len(nodes)
    n_bars = len(bars)
    n_supports = len(fixed_dofs) // 3
    
    # Get support node IDs
    dof = DOFManager(dof_per_node=3)
    support_node_ids = []
    for i in range(n_nodes):
        if dof.idx(i, 0) in fixed_dofs:
            support_node_ids.append(i)
    
    # Build geometry response (always returned)
    nodes_list = [
        NodeData(id=n.id, x=round(n.x, 4), y=round(n.y, 4), z=round(n.z, 4))
        for n in nodes.values()
    ]
    
    # Compute lengths (geometry only, no solve needed)
    lengths = []
    total_length = 0.0
    volume = 0.0
    for bar in bars:
        L, _, _, _ = element_geometry_3d(nodes, bar)
        lengths.append((bar.id, L))
        total_length += L
        volume += bar.A * L
    
    # Try to solve
    solve_error = None
    forces = {}
    max_displacement = 0.0
    max_tension = 0.0
    max_compression = 0.0
    
    print(f"[DEBUG] Starting solve...")
    print(f"[DEBUG] n_supports: {n_supports}")
    print(f"[DEBUG] fixed_dofs count: {len(fixed_dofs)}")
    
    try:
        ndof = dof.ndof(n_nodes)
        print(f"[DEBUG] Total DOFs: {ndof}, Free DOFs: {ndof - len(fixed_dofs)}")
        
        contributions = []
        for bar in bars:
            dof_map = dof.element_dof_map([bar.ni, bar.nj])
            ke = truss3d_global_stiffness(nodes, bar)
            contributions.append((dof_map, ke))
        
        K = assemble_global_K(ndof, contributions)
        print(f"[DEBUG] Assembled K matrix: shape {K.shape}")
        # Check condition number before solve (for debugging)
        all_dofs = np.arange(K.shape[0])
        free_dofs = np.array([i for i in all_dofs if i not in set(fixed_dofs)])
        Kff = K[np.ix_(free_dofs, free_dofs)]
        cond = np.linalg.cond(Kff)
        print(f"[DEBUG] Condition number: {cond:.2e}")
        
        print(f"[DEBUG] Calling solve_linear...")
        # Use higher tolerance for space frames (1e14 instead of default 1e12)
        d, R, free = solve_linear(K, F, fixed_dofs, cond_limit=1e14)
        print(f"[DEBUG] Solve successful!")
        
        # Extract displacements
        for node_id in nodes:
            ux = d[dof.idx(node_id, 0)]
            uy = d[dof.idx(node_id, 1)]
            uz = d[dof.idx(node_id, 2)]
            total = np.sqrt(ux**2 + uy**2 + uz**2)
            max_displacement = max(max_displacement, total)
        
        # Extract forces
        tension_count = 0
        compression_count = 0
        for bar in bars:
            N = truss3d_axial_force(nodes, bar, d)
            forces[bar.id] = N
            if N > 0:
                tension_count += 1
            elif N < 0:
                compression_count += 1
            if N > max_tension:
                max_tension = N
            if N < max_compression:
                max_compression = N
        
        # Debug output
        print(f"[DEBUG] Force distribution: {tension_count} tension, {compression_count} compression")
        print(f"[DEBUG] Max tension: {max_tension/1000:.2f} kN, Max compression: {abs(max_compression)/1000:.2f} kN")
                
    except MechanismError as e:
        print(f"[ERROR] MechanismError: {str(e)}")
        solve_error = f"Structure unstable: {str(e)}"
    except Exception as e:
        print(f"[ERROR] Solve exception: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        solve_error = f"Solve failed: {str(e)}"
    
    # Build bars list (with forces if solved, zero otherwise)
    bars_list = []
    for bar in bars:
        L, _, _, _ = element_geometry_3d(nodes, bar)
        bars_list.append(BarData(
            id=bar.id,
            ni=bar.ni,
            nj=bar.nj,
            length=round(L, 4),
            force=round(forces.get(bar.id, 0), 2)
        ))
    
    # Length bins
    bins = compute_length_bins(lengths, tolerance=0.010)
    
    # Metrics (partial if solve failed)
    metrics = MetricsData(
        n_nodes=n_nodes,
        n_bars=n_bars,
        n_supports=n_supports,
        max_displacement=round(max_displacement, 6),
        max_displacement_mm=round(max_displacement * 1000, 2),
        max_tension=round(max_tension, 2),
        max_tension_kn=round(max_tension / 1000, 2),
        max_compression=round(abs(max_compression), 2),
        max_compression_kn=round(abs(max_compression) / 1000, 2),
        volume=round(volume, 6),
        total_length=round(total_length, 2),
        n_length_bins=len(bins),
        max_member_length=round(max(L for _, L in lengths), 4),
        max_member_length_mm=round(max(L for _, L in lengths) * 1000, 1),
        min_member_length=round(min(L for _, L in lengths), 4),
    )
    
    return DesignResult(
        success=solve_error is None,
        error=solve_error,
        nodes=nodes_list,
        bars=bars_list,
        support_nodes=support_node_ids,
        metrics=metrics,
        params=params.model_dump()
    )


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/")
async def root():
    """Health check."""
    return {"status": "ok", "service": "StructureCraft API"}


@app.post("/api/generate", response_model=DesignResult)
async def generate_design(params: DesignParams):
    """Generate and solve a canopy design."""
    return generate_and_solve(params)


@app.post("/api/export/csv")
async def export_csv(params: DesignParams):
    """Export cut list as CSV."""
    result = generate_and_solve(params)
    
    if not result.success:
        raise HTTPException(status_code=400, detail=result.error)
    
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['bar_id', 'node_i', 'node_j', 'length_m', 'length_mm', 'force_kN', 'force_type'])
    
    for bar in sorted(result.bars, key=lambda b: b.length):
        force_type = "T" if bar.force > 0 else "C"
        writer.writerow([
            bar.id, bar.ni, bar.nj,
            bar.length, round(bar.length * 1000, 1),
            round(bar.force / 1000, 2), force_type
        ])
    
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=canopy_cutlist.csv"}
    )


@app.post("/api/export/json")
async def export_json(params: DesignParams):
    """Export model as JSON."""
    result = generate_and_solve(params)
    
    if not result.success:
        raise HTTPException(status_code=400, detail=result.error)
    
    model = {
        "version": "1.0",
        "type": "canopy",
        "parameters": result.params,
        "metrics": result.metrics.model_dump(),
        "geometry": {
            "nodes": [n.model_dump() for n in result.nodes],
            "bars": [b.model_dump() for b in result.bars],
            "supports": result.support_nodes
        }
    }
    
    return StreamingResponse(
        iter([json.dumps(model, indent=2)]),
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=canopy_model.json"}
    )


# =============================================================================
# Exploration Functions
# =============================================================================

def sample_variants_around_base(
    base_params: DesignParams,
    n_designs: int,
    seed: int,
    variation_pct: float,
) -> List[DesignParams]:
    """Generate design variants by varying parameters around a base design."""
    rng = np.random.default_rng(seed)
    variants = []
    
    heightfields = ['flat', 'paraboloid', 'ridge', 'saddle']
    topologies = ['grid', 'diagrid', 'triangulated']
    support_layouts = ['edges', 'corners', 'perimeter_4']
    
    for _ in range(n_designs):
        # Vary continuous parameters
        width = rng.uniform(
            base_params.width * (1 - variation_pct),
            base_params.width * (1 + variation_pct)
        )
        width = max(4.0, min(20.0, width))
        
        depth = rng.uniform(
            base_params.depth * (1 - variation_pct),
            base_params.depth * (1 + variation_pct)
        )
        depth = max(4.0, min(16.0, depth))
        
        nx = int(rng.integers(max(2, base_params.nx - 2), min(10, base_params.nx + 2) + 1))
        ny = int(rng.integers(max(2, base_params.ny - 2), min(10, base_params.ny + 2) + 1))
        
        max_height = rng.uniform(
            base_params.max_height * (1 - variation_pct * 0.5),
            base_params.max_height * (1 + variation_pct * 0.5)
        )
        max_height = max(1.5, min(6.0, max_height))
        
        min_height = rng.uniform(
            base_params.min_height * (1 - variation_pct * 0.5),
            min(max_height - 0.25, base_params.min_height * (1 + variation_pct * 0.5))
        )
        min_height = max(1.5, min(max_height - 0.25, min_height))
        
        A_cm2 = rng.uniform(
            base_params.A_cm2 * (1 - variation_pct),
            base_params.A_cm2 * (1 + variation_pct)
        )
        A_cm2 = max(3.0, min(20.0, A_cm2))
        
        gravity_kn = rng.uniform(
            base_params.gravity_kn * (1 - variation_pct),
            base_params.gravity_kn * (1 + variation_pct)
        )
        gravity_kn = max(20.0, min(150.0, gravity_kn))
        
        # Sample categorical parameters
        heightfield = rng.choice(heightfields)
        topology = rng.choice(topologies)
        support_layout = rng.choice(support_layouts, p=[0.7, 0.15, 0.15])
        
        variants.append(DesignParams(
            width=round(width, 2),
            depth=round(depth, 2),
            nx=nx,
            ny=ny,
            min_height=round(min_height, 2),
            max_height=round(max_height, 2),
            heightfield=heightfield,
            topology=topology,
            support_layout=support_layout,
            A_cm2=round(A_cm2, 2),
            gravity_kn=round(gravity_kn, 1),
        ))
    
    return variants


def evaluate_single_design(params: DesignParams, index: int) -> ExploreDesignData:
    """Evaluate a single design for exploration."""
    result = generate_and_solve(params)
    
    data = ExploreDesignData(
        index=index,
        ok=result.success,
        reason=result.error,
        width=params.width,
        depth=params.depth,
        nx=params.nx,
        ny=params.ny,
        min_height=params.min_height,
        max_height=params.max_height,
        heightfield=params.heightfield,
        topology=params.topology,
        support_layout=params.support_layout,
        A_cm2=params.A_cm2,
        gravity_kn=params.gravity_kn,
    )
    
    if result.success and result.metrics:
        data.volume = result.metrics.volume
        data.max_displacement = result.metrics.max_displacement
        data.max_displacement_mm = result.metrics.max_displacement_mm
        data.n_length_bins = result.metrics.n_length_bins
        data.max_member_length_mm = result.metrics.max_member_length_mm
        data.max_tension_kn = result.metrics.max_tension_kn
        data.max_compression_kn = result.metrics.max_compression_kn
        data.n_nodes = result.metrics.n_nodes
        data.n_bars = result.metrics.n_bars
    
    return data


def compute_pareto_mask(designs: List[ExploreDesignData]) -> List[bool]:
    """Compute Pareto frontier for successful designs."""
    # Get successful designs with valid metrics
    successful = [d for d in designs if d.ok and d.volume is not None and d.max_displacement is not None]
    
    if len(successful) == 0:
        return [False] * len(designs)
    
    # Build objective matrix (minimize volume, displacement, length_bins)
    n = len(successful)
    obj_values = np.zeros((n, 3))
    
    for i, d in enumerate(successful):
        obj_values[i, 0] = d.volume or 0
        obj_values[i, 1] = d.max_displacement or 0
        obj_values[i, 2] = d.n_length_bins or 0
    
    # Find Pareto frontier
    is_pareto = np.ones(n, dtype=bool)
    
    for i in range(n):
        if not is_pareto[i]:
            continue
        
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            
            # Check if j dominates i
            better_or_equal = obj_values[j] <= obj_values[i]
            strictly_better = obj_values[j] < obj_values[i]
            
            if better_or_equal.all() and strictly_better.any():
                is_pareto[i] = False
                break
    
    # Map back to full list
    result = [False] * len(designs)
    successful_indices = [i for i, d in enumerate(designs) if d.ok and d.volume is not None]
    
    for idx, pareto_val in zip(successful_indices, is_pareto):
        result[idx] = bool(pareto_val)
    
    return result


@app.post("/api/explore", response_model=ExploreResult)
async def explore_designs(params: ExploreParams):
    """Run batch exploration of design variants."""
    print(f"\n{'='*60}")
    print(f"[DEBUG] EXPLORE REQUEST")
    print(f"[DEBUG] n_designs: {params.n_designs}")
    print(f"[DEBUG] seed: {params.seed}")
    print(f"[DEBUG] variation_pct: {params.variation_pct}")
    print(f"{'='*60}")
    
    try:
        # Generate variants
        variants = sample_variants_around_base(
            params.base_params,
            params.n_designs,
            params.seed,
            params.variation_pct,
        )
        
        print(f"[DEBUG] Generated {len(variants)} variants")
        
        # Evaluate each variant
        designs = []
        for i, variant in enumerate(variants):
            if (i + 1) % 20 == 0:
                print(f"[DEBUG] Evaluating {i + 1}/{len(variants)}...")
            design_data = evaluate_single_design(variant, i)
            designs.append(design_data)
        
        # Compute Pareto frontier
        pareto_mask = compute_pareto_mask(designs)
        for i, is_pareto in enumerate(pareto_mask):
            designs[i].is_pareto = is_pareto
        
        n_successful = sum(1 for d in designs if d.ok)
        n_pareto = sum(pareto_mask)
        
        print(f"[DEBUG] Results: {n_successful} successful, {n_pareto} Pareto-optimal")
        
        return ExploreResult(
            success=True,
            n_total=len(designs),
            n_successful=n_successful,
            n_pareto=n_pareto,
            designs=designs,
        )
        
    except Exception as e:
        print(f"[ERROR] Exploration failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return ExploreResult(
            success=False,
            error=str(e),
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

