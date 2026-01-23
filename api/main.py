# api/main.py
"""
FastAPI backend for StructureCraft - exposes mini_branch engine as REST API.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
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
# Stability Check
# =============================================================================

def check_stability(topology: str, support_layout: str, nx: int, ny: int) -> Optional[str]:
    """Check if parameter combination is likely stable."""
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
    
    # Convert params
    A = params.A_cm2 / 10000  # cm² to m²
    gravity_load = -params.gravity_kn * 1000  # kN to N, negative for down
    
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
            support_layout=params.support_layout,
            E=210e9,
            A=A,
            gravity_load=gravity_load,
        )
        
        # Generate structure (geometry only)
        nodes, bars, fixed_dofs, F = generate_canopy(canopy_params)
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
    
    try:
        ndof = dof.ndof(n_nodes)
        
        contributions = []
        for bar in bars:
            dof_map = dof.element_dof_map([bar.ni, bar.nj])
            ke = truss3d_global_stiffness(nodes, bar)
            contributions.append((dof_map, ke))
        
        K = assemble_global_K(ndof, contributions)
        d, R, free = solve_linear(K, F, fixed_dofs)
        
        # Extract displacements
        for node_id in nodes:
            ux = d[dof.idx(node_id, 0)]
            uy = d[dof.idx(node_id, 1)]
            uz = d[dof.idx(node_id, 2)]
            total = np.sqrt(ux**2 + uy**2 + uz**2)
            max_displacement = max(max_displacement, total)
        
        # Extract forces
        for bar in bars:
            N = truss3d_axial_force(nodes, bar, d)
            forces[bar.id] = N
            if N > max_tension:
                max_tension = N
            if N < max_compression:
                max_compression = N
                
    except MechanismError as e:
        solve_error = f"Structure unstable: {str(e)}"
    except Exception as e:
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

