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
from mini_branch.kernel.solve import solve_linear, solve_pdelta, MechanismError, ConvergenceError
from mini_branch.kernel.buckling import build_global_Kg, critical_buckling_factor, check_member_buckling
from mini_branch.kernel.modal import build_lumped_mass_matrix, natural_frequencies
from mini_branch.kernel.connections import ConnectionType, CONNECTION_STIFFNESS
from mini_branch.checks.timber import DOUGLAS_FIR_CAPACITY, axial_utilization
from mini_branch.checks.steel import STEEL_SECTIONS, check_steel_member
# 2D Frame imports
from mini_branch.model import Node, Frame2D
from mini_branch.assembly import assemble_global_K as assemble_2d_K, DOF_PER_NODE, dof_index
from mini_branch.loads import assemble_element_loads_global
from mini_branch.explore import PortalParams, make_portal
from mini_branch.catalog import Material, Section, DEFAULT_MATERIAL, TIMBER_SECTIONS
from mini_branch.diagrams import (
    DiagramPoint as DiagramPointData,
    DeflectedPoint as DeflectedPointData,
    ElementDiagramData as ElementDiagramDataClass,
    compute_frame_diagrams,
    get_frame_summary,
)
from mini_branch.post import (
    element_end_forces_local,
    compute_nodal_displacements,
    compute_drift,
    compute_reactions,
)
# Section extraction imports
from mini_branch.section import (
    extract_section,
    find_slice_coordinate,
    calculate_tributary_udl,
    get_section_span,
    SectionExtractionResult,
)
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
    # Load factor for strength/serviceability
    load_factor: float = Field(1.0, ge=1.0, le=2.0, description="Load factor: 1.0=service, 1.4/1.6=ultimate")
    # Material options
    material_type: str = Field("timber", description="Material: timber or steel")
    steel_section: str = Field("HSS4x4x1/4", description="Steel section if material=steel")
    # Connection type
    connection_type: str = Field("pinned", description="Connection type: pinned, semi_rigid, rigid")
    # Advanced analysis options
    pdelta_enabled: bool = Field(False, description="Enable P-Delta (second-order) analysis")
    modal_enabled: bool = Field(False, description="Enable modal (natural frequency) analysis")


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
    utilization: Optional[float] = None  # Stress utilization ratio (< 1.0 = pass)
    status: Optional[str] = None         # "PASS" or "FAIL"


class DeflectionLimitCheck(BaseModel):
    """Single deflection limit check result."""
    limit_mm: float
    actual_mm: float
    ratio: float
    status: str  # "PASS" or "FAIL"


class DeflectionCheck(BaseModel):
    """Deflection serviceability check results."""
    L_360: DeflectionLimitCheck
    L_240: DeflectionLimitCheck
    L_180: DeflectionLimitCheck
    governing: str = "L_240"
    overall_pass: bool


class DesignReportCard(BaseModel):
    """Summary of design compliance checks."""
    deflection_pass: bool
    strength_pass: bool
    buckling_pass: bool
    shipping_pass: bool  # Member length < shipping limit
    overall_pass: bool
    summary: str  # Human-readable summary


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
    # Engineering checks
    buckling_factor: Optional[float] = None       # Critical load factor (> 1.0 = safe)
    max_utilization: Optional[float] = None       # Worst member stress ratio
    n_failing_members: Optional[int] = None       # Count of members with util > 1.0
    critical_member_id: Optional[int] = None      # ID of worst member
    # P-Delta analysis
    pdelta_iterations: Optional[int] = None       # Iterations to converge (0 = linear)
    pdelta_amplification: Optional[float] = None  # Displacement amplification factor
    # Modal analysis
    natural_frequencies_hz: Optional[List[float]] = None  # First N natural frequencies in Hz
    # Material info
    material_type: Optional[str] = None
    steel_section_name: Optional[str] = None
    # Connection info
    connection_type: Optional[str] = None
    # Load factor applied
    load_factor: Optional[float] = None
    # Deflection check results
    deflection_check: Optional[DeflectionCheck] = None
    # Design report card
    design_report_card: Optional[DesignReportCard] = None


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
# 2D Frame Models
# =============================================================================

class Frame2DParams(BaseModel):
    """Input parameters for 2D portal frame generation."""
    span: float = Field(6.0, ge=2.0, le=20.0, description="Horizontal span (m)")
    height: float = Field(3.0, ge=2.0, le=10.0, description="Column height (m)")
    brace: int = Field(0, ge=0, le=1, description="Bracing: 0=none, 1=X-brace")
    # Section indices (from timber sections catalog)
    sec_col: int = Field(5, ge=0, le=10, description="Column section index")
    sec_beam: int = Field(5, ge=0, le=10, description="Beam section index")
    sec_brace: int = Field(3, ge=0, le=10, description="Brace section index")
    # Loads
    udl_kn_m: float = Field(2.0, ge=0.5, le=20.0, description="UDL on beam (kN/m)")
    lateral_kn: float = Field(5.0, ge=0.0, le=50.0, description="Lateral load at top (kN)")
    # Visualization
    deflection_scale: float = Field(50.0, ge=1.0, le=500.0, description="Deflection scale factor")
    n_diagram_points: int = Field(21, ge=11, le=51, description="Points per element for diagrams")


class Frame2DNodeData(BaseModel):
    """2D frame node data."""
    id: int
    x: float
    y: float
    ux: float = 0.0       # Displacement in X
    uy: float = 0.0       # Displacement in Y
    rz: float = 0.0       # Rotation about Z
    is_support: bool = False


class Frame2DElementData(BaseModel):
    """2D frame element data."""
    id: int
    ni: int
    nj: int
    element_type: str     # "column", "beam", "brace"
    length: float
    # Section info
    section_name: str
    A: float              # Area (m²)
    I: float              # Moment of inertia (m⁴)
    # End forces (local coordinates)
    N_i: float = 0.0      # Axial at node i
    N_j: float = 0.0      # Axial at node j
    V_i: float = 0.0      # Shear at node i
    V_j: float = 0.0      # Shear at node j
    M_i: float = 0.0      # Moment at node i
    M_j: float = 0.0      # Moment at node j


class DiagramPoint(BaseModel):
    """Single point on a force diagram."""
    x_local: float
    x_global: float
    y_global: float
    N: float
    V: float
    M: float


class DeflectedPoint(BaseModel):
    """Point on deflected shape curve."""
    x: float
    y: float


class ElementDiagramData(BaseModel):
    """Complete diagram data for one element."""
    element_id: int
    element_type: str
    ni: int
    nj: int
    length: float
    points: List[DiagramPoint]
    deflected_shape: List[DeflectedPoint]
    N_i: float
    N_j: float
    V_i: float
    V_j: float
    M_i: float
    M_j: float
    max_N: float
    max_V: float
    max_M: float


class ReactionData(BaseModel):
    """Reaction forces at a support."""
    node_id: int
    Rx: float
    Ry: float
    Mz: float = 0.0


class Frame2DMetrics(BaseModel):
    """Metrics for 2D frame analysis."""
    n_nodes: int
    n_elements: int
    n_supports: int
    # Displacement
    max_displacement_mm: float
    drift_mm: float
    drift_ratio: float
    drift_passes: bool
    # Forces
    max_axial_force: float
    max_shear_force: float
    max_moment: float
    critical_element_N: Optional[int] = None
    critical_element_V: Optional[int] = None
    critical_element_M: Optional[int] = None


class Frame2DResult(BaseModel):
    """Complete 2D frame analysis result."""
    success: bool
    error: Optional[str] = None
    # Geometry
    nodes: Optional[List[Frame2DNodeData]] = None
    elements: Optional[List[Frame2DElementData]] = None
    # Results
    reactions: Optional[List[ReactionData]] = None
    metrics: Optional[Frame2DMetrics] = None
    # Diagram data
    diagrams: Optional[List[ElementDiagramData]] = None
    # Parameters used
    params: Optional[Dict[str, Any]] = None


# =============================================================================
# Section Extraction Models
# =============================================================================

class SectionExtractParams(BaseModel):
    """Parameters for extracting a 2D section from 3D spaceframe."""
    # 3D design params (to regenerate structure)
    width: float = Field(10.0, ge=4.0, le=20.0)
    depth: float = Field(8.0, ge=4.0, le=16.0)
    nx: int = Field(5, ge=2, le=10)
    ny: int = Field(4, ge=2, le=10)
    min_height: float = Field(2.5, ge=1.5, le=6.0)
    max_height: float = Field(4.0, ge=1.5, le=6.0)
    heightfield: str = Field("paraboloid")
    topology: str = Field("grid")
    support_layout: str = Field("edges")
    A_cm2: float = Field(8.0, ge=3.0, le=20.0)
    gravity_kn: float = Field(50.0, ge=20.0, le=150.0)
    # Section settings
    slice_axis: str = Field("y", description="Axis to slice: 'x' or 'y'")
    slice_position: float = Field(0.5, ge=0.0, le=1.0, description="Normalized position (0-1)")
    # Load override (optional)
    udl_override_kn_m: Optional[float] = Field(None, ge=0.0, le=50.0)
    deflection_scale: float = Field(50.0, ge=1.0, le=500.0)
    n_diagram_points: int = Field(21, ge=11, le=51)


class SectionExtractResult(BaseModel):
    """Result of section extraction with 2D analysis."""
    success: bool
    error: Optional[str] = None
    # Section info
    slice_axis: Optional[str] = None
    slice_value: Optional[float] = None
    tributary_width: Optional[float] = None
    calculated_udl_kn_m: Optional[float] = None
    applied_udl_kn_m: Optional[float] = None
    section_span: Optional[float] = None
    n_nodes_extracted: Optional[int] = None
    n_elements_extracted: Optional[int] = None
    # 2D frame result
    frame_result: Optional[Frame2DResult] = None


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
# Deflection and Design Report Card Helpers
# =============================================================================

def compute_deflection_check(max_disp: float, span: float) -> DeflectionCheck:
    """
    Check deflection against common serviceability limits.
    
    Args:
        max_disp: Maximum displacement (m)
        span: Governing span (m) - typically min(width, depth)
    
    Returns:
        DeflectionCheck with limits, ratios, and pass/fail status
    """
    limits = {
        'L_360': span / 360,  # Floor live load
        'L_240': span / 240,  # Total load (typical for roofs/canopies)
        'L_180': span / 180,  # Roof with plaster ceiling
    }
    
    results = {}
    for name, limit in limits.items():
        ratio = max_disp / limit if limit > 0 else 0
        results[name] = DeflectionLimitCheck(
            limit_mm=round(limit * 1000, 2),
            actual_mm=round(max_disp * 1000, 2),
            ratio=round(ratio, 3),
            status='PASS' if ratio <= 1.0 else 'FAIL'
        )
    
    # Overall pass if meets L/240 (typical for roofs/canopies)
    governing_limit = 'L_240'
    overall_pass = results[governing_limit].status == 'PASS'
    
    return DeflectionCheck(
        L_360=results['L_360'],
        L_240=results['L_240'],
        L_180=results['L_180'],
        governing=governing_limit,
        overall_pass=overall_pass,
    )


def compute_design_report_card(
    deflection_check: Optional[DeflectionCheck],
    max_utilization: Optional[float],
    buckling_factor: Optional[float],
    max_member_length: float,
    shipping_limit: float = 6.0  # meters
) -> DesignReportCard:
    """
    Generate a summary report card for the design.
    
    Args:
        deflection_check: Deflection check results
        max_utilization: Worst member utilization ratio
        buckling_factor: Critical buckling load factor
        max_member_length: Maximum member length (m)
        shipping_limit: Maximum transportable length (m)
    
    Returns:
        DesignReportCard with pass/fail for each check
    """
    deflection_pass = deflection_check.overall_pass if deflection_check else True
    strength_pass = max_utilization is not None and max_utilization <= 1.0
    buckling_pass = buckling_factor is not None and buckling_factor >= 1.0
    shipping_pass = max_member_length <= shipping_limit
    
    overall_pass = deflection_pass and strength_pass and buckling_pass and shipping_pass
    
    # Build summary
    issues = []
    if not deflection_pass:
        issues.append("deflection exceeds limits")
    if not strength_pass:
        issues.append("overstressed members")
    if not buckling_pass:
        issues.append("buckling risk")
    if not shipping_pass:
        issues.append(f"members exceed {shipping_limit}m shipping limit")
    
    if overall_pass:
        summary = "All checks pass ✓"
    else:
        summary = f"Issues: {', '.join(issues)}"
    
    return DesignReportCard(
        deflection_pass=deflection_pass,
        strength_pass=strength_pass,
        buckling_pass=buckling_pass,
        shipping_pass=shipping_pass,
        overall_pass=overall_pass,
        summary=summary,
    )


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
    # Apply load factor to gravity load
    gravity_load = -params.gravity_kn * 1000 * params.load_factor  # kN to N, negative for down
    
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
    member_utils = {}
    buckling_factor = None
    pdelta_iterations = None
    pdelta_amplification = None
    natural_freqs = None
    
    print(f"[DEBUG] Starting solve...")
    print(f"[DEBUG] n_supports: {n_supports}")
    print(f"[DEBUG] fixed_dofs count: {len(fixed_dofs)}")
    print(f"[DEBUG] pdelta_enabled: {params.pdelta_enabled}")
    print(f"[DEBUG] modal_enabled: {params.modal_enabled}")
    
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
        
        print(f"[DEBUG] Calling solver...")
        # Use higher tolerance for space frames (1e14 instead of default 1e12)
        
        if params.pdelta_enabled:
            # P-Delta (second-order) analysis
            print(f"[DEBUG] Using P-Delta analysis...")
            d, R, pdelta_iterations, pdelta_amplification = solve_pdelta(
                K=K,
                F=F,
                fixed_dofs=fixed_dofs,
                nodes=nodes,
                bars=bars,
                dof_manager=dof,
                element_geometry_func=element_geometry_3d,
                axial_force_func=truss3d_axial_force,
                build_kg_func=build_global_Kg,
                max_iter=10,
                tol=1e-4,
                cond_limit=1e14
            )
            print(f"[DEBUG] P-Delta converged in {pdelta_iterations} iterations, amplification={pdelta_amplification:.3f}")
        else:
            # Linear analysis
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
        
        # =====================================================================
        # ENGINEERING CHECKS: Buckling and Utilization
        # =====================================================================
        
        # Compute member utilizations (material-dependent)
        member_utils = {}
        for bar in bars:
            N = forces.get(bar.id, 0.0)
            L, _, _, _ = element_geometry_3d(nodes, bar)
            
            if params.material_type == 'steel':
                # Steel check per AISC
                section = STEEL_SECTIONS.get(params.steel_section, STEEL_SECTIONS['HSS4x4x1/4'])
                result = check_steel_member(N, section, L, K=1.0)
                member_utils[bar.id] = {
                    'utilization': result['combined_util'],
                    'status': result['status']
                }
            else:
                # Timber check (default)
                util = axial_utilization(N, bar.A, DOUGLAS_FIR_CAPACITY)
                member_utils[bar.id] = {
                    'utilization': util,
                    'status': 'PASS' if util <= 1.0 else 'FAIL'
                }
        
        # Buckling analysis (only if compression members exist)
        buckling_factor = None
        if max_compression < 0:
            try:
                Kg = build_global_Kg(
                    nodes, bars, forces, dof, element_geometry_3d
                )
                buckling_factor = critical_buckling_factor(K, Kg, fixed_dofs)
                if buckling_factor == float('inf'):
                    buckling_factor = 999.0  # Cap for JSON serialization
                print(f"[DEBUG] Buckling factor: {buckling_factor:.2f}")
            except Exception as e:
                print(f"[DEBUG] Buckling analysis skipped: {e}")
                buckling_factor = None
        
        # Modal analysis (if enabled)
        if params.modal_enabled:
            try:
                # Use material-appropriate density
                if params.material_type == 'steel':
                    density = 7850.0  # kg/m³ for steel
                else:
                    density = 500.0   # kg/m³ for timber
                
                M = build_lumped_mass_matrix(
                    nodes, bars, density, dof, element_geometry_3d
                )
                freqs, _ = natural_frequencies(K, M, fixed_dofs, n_modes=5)
                natural_freqs = [round(f, 3) for f in freqs.tolist()]
                print(f"[DEBUG] Natural frequencies (Hz): {natural_freqs}")
            except Exception as e:
                print(f"[DEBUG] Modal analysis failed: {e}")
                natural_freqs = None
                
    except MechanismError as e:
        print(f"[ERROR] MechanismError: {str(e)}")
        solve_error = f"Structure unstable: {str(e)}"
    except ConvergenceError as e:
        print(f"[ERROR] ConvergenceError: {str(e)}")
        solve_error = f"P-Delta did not converge: {str(e)}"
    except Exception as e:
        print(f"[ERROR] Solve exception: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        solve_error = f"Solve failed: {str(e)}"
    
    # Build bars list (with forces and utilization if solved)
    bars_list = []
    for bar in bars:
        L, _, _, _ = element_geometry_3d(nodes, bar)
        util_data = member_utils.get(bar.id, {})
        bars_list.append(BarData(
            id=bar.id,
            ni=bar.ni,
            nj=bar.nj,
            length=round(L, 4),
            force=round(forces.get(bar.id, 0), 2),
            utilization=round(util_data.get('utilization', 0), 3) if util_data else None,
            status=util_data.get('status') if util_data else None,
        ))
    
    # Length bins
    bins = compute_length_bins(lengths, tolerance=0.010)
    
    # Compute utilization summary
    max_utilization = None
    n_failing_members = None
    critical_member_id = None
    if member_utils:
        utils = [(bid, data['utilization']) for bid, data in member_utils.items()]
        max_util_entry = max(utils, key=lambda x: x[1])
        max_utilization = round(max_util_entry[1], 3)
        critical_member_id = max_util_entry[0]
        n_failing_members = sum(1 for _, u in utils if u > 1.0)
    
    # Compute deflection check (use governing span = min(width, depth))
    max_member_len = max(L for _, L in lengths)
    deflection_check = None
    if max_displacement > 0:
        governing_span = min(params.width, params.depth)
        deflection_check = compute_deflection_check(max_displacement, governing_span)
    
    # Compute design report card
    design_report_card = compute_design_report_card(
        deflection_check=deflection_check,
        max_utilization=max_utilization,
        buckling_factor=buckling_factor,
        max_member_length=max_member_len,
        shipping_limit=6.0  # 6m typical shipping limit
    )
    
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
        max_member_length=round(max_member_len, 4),
        max_member_length_mm=round(max_member_len * 1000, 1),
        min_member_length=round(min(L for _, L in lengths), 4),
        # Engineering checks
        buckling_factor=round(buckling_factor, 2) if buckling_factor is not None else None,
        max_utilization=max_utilization,
        n_failing_members=n_failing_members,
        critical_member_id=critical_member_id,
        # P-Delta results
        pdelta_iterations=pdelta_iterations,
        pdelta_amplification=round(pdelta_amplification, 3) if pdelta_amplification is not None else None,
        # Modal results
        natural_frequencies_hz=natural_freqs,
        # Material info
        material_type=params.material_type,
        steel_section_name=params.steel_section if params.material_type == 'steel' else None,
        # Connection info
        connection_type=params.connection_type,
        # Load factor applied
        load_factor=params.load_factor,
        # Deflection check results
        deflection_check=deflection_check,
        # Design report card
        design_report_card=design_report_card,
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


@app.post("/api/export/pdf")
async def export_pdf(params: DesignParams):
    """Export design report as PDF."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib import colors
    except ImportError:
        raise HTTPException(status_code=500, detail="reportlab not installed - run: pip install reportlab")
    
    result = generate_and_solve(params)
    
    if not result.success:
        raise HTTPException(status_code=400, detail=result.error)
    
    # Create PDF in memory
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=12,
        textColor=colors.HexColor('#1e3a5f')
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=12,
        spaceBefore=12,
        spaceAfter=6,
        textColor=colors.HexColor('#1e3a5f')
    )
    
    story = []
    
    # Title
    story.append(Paragraph("StructureCraft Design Report", title_style))
    story.append(Spacer(1, 12))
    
    # Design Parameters
    story.append(Paragraph("Design Parameters", heading_style))
    param_data = [
        ['Parameter', 'Value'],
        ['Footprint', f"{params.width} × {params.depth} m"],
        ['Grid', f"{params.nx} × {params.ny}"],
        ['Height Range', f"{params.min_height} - {params.max_height} m"],
        ['Shape', params.heightfield.capitalize()],
        ['Topology', params.topology.capitalize()],
        ['Support Layout', params.support_layout.replace('_', ' ').title()],
        ['Material', params.material_type.capitalize()],
        ['Load Factor', str(params.load_factor)],
        ['Gravity Load', f"{params.gravity_kn} kN"],
    ]
    
    param_table = Table(param_data, colWidths=[2*inch, 2*inch])
    param_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3a5f')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f5f5f5')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(param_table)
    story.append(Spacer(1, 12))
    
    # Structural Metrics
    if result.metrics:
        story.append(Paragraph("Structural Analysis Results", heading_style))
        m = result.metrics
        metrics_data = [
            ['Metric', 'Value'],
            ['Nodes', str(m.n_nodes)],
            ['Members', str(m.n_bars)],
            ['Supports', str(m.n_supports)],
            ['Total Length', f"{m.total_length:.1f} m"],
            ['Volume', f"{m.volume:.4f} m³"],
            ['Max Displacement', f"{m.max_displacement_mm:.1f} mm"],
            ['Max Tension', f"{m.max_tension_kn:.1f} kN"],
            ['Max Compression', f"{m.max_compression_kn:.1f} kN"],
        ]
        
        if m.max_utilization is not None:
            metrics_data.append(['Max Utilization', f"{m.max_utilization:.2%}"])
        if m.buckling_factor is not None:
            metrics_data.append(['Buckling Factor', f"{m.buckling_factor:.2f}"])
        
        metrics_table = Table(metrics_data, colWidths=[2*inch, 2*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3a5f')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f5f5f5')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(metrics_table)
        story.append(Spacer(1, 12))
        
        # Design Report Card
        if m.design_report_card:
            story.append(Paragraph("Design Report Card", heading_style))
            rc = m.design_report_card
            
            def status_text(passed: bool) -> str:
                return "✓ PASS" if passed else "✗ FAIL"
            
            card_data = [
                ['Check', 'Status'],
                ['Deflection (L/240)', status_text(rc.deflection_pass)],
                ['Member Strength', status_text(rc.strength_pass)],
                ['Buckling Stability', status_text(rc.buckling_pass)],
                ['Shipping (<6m)', status_text(rc.shipping_pass)],
                ['OVERALL', status_text(rc.overall_pass)],
            ]
            
            card_table = Table(card_data, colWidths=[2*inch, 1.5*inch])
            card_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3a5f')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('BACKGROUND', (-1, -1), (-1, -1), 
                 colors.HexColor('#d4edda') if rc.overall_pass else colors.HexColor('#f8d7da')),
                ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
            ]))
            story.append(card_table)
            story.append(Spacer(1, 6))
            story.append(Paragraph(f"<i>{rc.summary}</i>", styles['Normal']))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    
    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=canopy_design_report.pdf"}
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


# =============================================================================
# 2D Frame Endpoints
# =============================================================================

def generate_and_solve_frame2d(params: Frame2DParams) -> Frame2DResult:
    """Generate and solve a 2D portal frame with full diagram data."""
    
    print(f"\n{'='*60}")
    print(f"[DEBUG] FRAME2D GENERATE REQUEST")
    print(f"[DEBUG] span: {params.span}, height: {params.height}")
    print(f"[DEBUG] brace: {params.brace}")
    print(f"[DEBUG] udl: {params.udl_kn_m} kN/m, lateral: {params.lateral_kn} kN")
    print(f"{'='*60}")
    
    try:
        # Use timber material and sections
        material = DEFAULT_MATERIAL  # Using timber (Douglas Fir, E = 12 GPa)
        sections = TIMBER_SECTIONS
        
        # Ensure section indices are valid
        n_sec = len(sections)
        sec_col = min(params.sec_col, n_sec - 1)
        sec_beam = min(params.sec_beam, n_sec - 1)
        sec_brace = min(params.sec_brace, n_sec - 1)
        
        # Create portal params
        portal_params = PortalParams(
            span=params.span,
            height=params.height,
            brace=params.brace,
            sec_col=sec_col,
            sec_beam=sec_beam,
            sec_brace=sec_brace,
            udl_w=-params.udl_kn_m * 1000,  # Convert kN/m to N/m, negative for down
            wind_P=params.lateral_kn * 1000,  # Convert kN to N
        )
        
        # Generate frame geometry
        nodes, elements, fixed_dofs, element_udls, nodal_loads = make_portal(
            portal_params, material, sections
        )
        
        n_nodes = len(nodes)
        n_elements = len(elements)
        
        print(f"[DEBUG] Generated: {n_nodes} nodes, {n_elements} elements")
        
    except Exception as e:
        print(f"[ERROR] Frame generation failed: {str(e)}")
        return Frame2DResult(success=False, error=f"Generation failed: {str(e)}")
    
    # Assemble and solve
    try:
        # Assemble stiffness matrix
        K = assemble_2d_K(nodes, elements)
        ndof = DOF_PER_NODE * n_nodes
        
        # Assemble load vector
        F = np.zeros(ndof)
        F += assemble_element_loads_global(nodes, elements, element_udls)
        
        # Add nodal loads
        for node_id, load in nodal_loads.items():
            F[dof_index(node_id, 0)] += load[0]  # Fx
            F[dof_index(node_id, 1)] += load[1]  # Fy
            F[dof_index(node_id, 2)] += load[2]  # Mz
        
        print(f"[DEBUG] Assembled K: {K.shape}, F sum: {np.sum(np.abs(F)):.2f}")
        
        # Solve
        from mini_branch.solve import solve_linear as solve_2d
        d, R, _ = solve_2d(K, F, fixed_dofs)
        
        print(f"[DEBUG] Solve successful! Max displacement: {np.max(np.abs(d)):.6f} m")
        
    except MechanismError as e:
        print(f"[ERROR] MechanismError: {str(e)}")
        return Frame2DResult(success=False, error=f"Structure unstable: {str(e)}")
    except Exception as e:
        print(f"[ERROR] Solve failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return Frame2DResult(success=False, error=f"Solve failed: {str(e)}")
    
    # Compute diagrams
    try:
        diagram_data = compute_frame_diagrams(
            nodes, elements, d, element_udls,
            scale=params.deflection_scale,
            n_points=params.n_diagram_points
        )
        summary = get_frame_summary(diagram_data)
        
        print(f"[DEBUG] Computed diagrams for {len(diagram_data)} elements")
        
    except Exception as e:
        print(f"[WARNING] Diagram computation failed: {str(e)}")
        diagram_data = []
        summary = {}
    
    # Build node data
    displacements = compute_nodal_displacements(nodes, d)
    support_node_ids = set(dof // DOF_PER_NODE for dof in fixed_dofs)
    
    nodes_list = []
    for node_id, node in nodes.items():
        disp = displacements.get(node_id, {'ux': 0, 'uy': 0, 'rz': 0})
        nodes_list.append(Frame2DNodeData(
            id=node_id,
            x=round(node.x, 4),
            y=round(node.y, 4),
            ux=round(disp['ux'], 8),
            uy=round(disp['uy'], 8),
            rz=round(disp['rz'], 8),
            is_support=node_id in support_node_ids,
        ))
    
    # Build element data
    elements_list = []
    for elem in elements:
        # Determine element type
        dx = nodes[elem.nj].x - nodes[elem.ni].x
        dy = nodes[elem.nj].y - nodes[elem.ni].y
        L = float(np.hypot(dx, dy))
        c = dx / L if L > 0 else 1.0
        
        if abs(c) < 0.1:
            elem_type = "column"
        elif abs(c) > 0.9:
            elem_type = "beam"
        else:
            elem_type = "brace"
        
        # Get end forces
        udl_w = element_udls.get(elem.id, None)
        f_local = element_end_forces_local(nodes, elem, d, udl_w)
        
        # Find section name
        sec_name = "Unknown"
        for sec in sections:
            if abs(sec.A - elem.A) < 1e-6 and abs(sec.I - elem.I) < 1e-10:
                sec_name = sec.name
                break
        
        elements_list.append(Frame2DElementData(
            id=elem.id,
            ni=elem.ni,
            nj=elem.nj,
            element_type=elem_type,
            length=round(L, 4),
            section_name=sec_name,
            A=elem.A,
            I=elem.I,
            N_i=round(float(f_local[0]), 2),
            N_j=round(float(f_local[3]), 2),
            V_i=round(float(f_local[1]), 2),
            V_j=round(float(f_local[4]), 2),
            M_i=round(float(f_local[2]), 2),
            M_j=round(float(f_local[5]), 2),
        ))
    
    # Build reaction data
    reactions_dict = compute_reactions(R, fixed_dofs, nodes)
    reactions_list = [
        ReactionData(
            node_id=nid,
            Rx=round(r['Rx'], 2),
            Ry=round(r['Ry'], 2),
            Mz=round(r['Mz'], 2),
        )
        for nid, r in reactions_dict.items()
    ]
    
    # Build diagram data for response
    diagrams_list = []
    for diag in diagram_data:
        diagrams_list.append(ElementDiagramData(
            element_id=diag.element_id,
            element_type=diag.element_type,
            ni=diag.ni,
            nj=diag.nj,
            length=diag.length,
            points=[DiagramPoint(
                x_local=p.x_local,
                x_global=p.x_global,
                y_global=p.y_global,
                N=p.N,
                V=p.V,
                M=p.M,
            ) for p in diag.points],
            deflected_shape=[DeflectedPoint(x=p.x, y=p.y) for p in diag.deflected_shape],
            N_i=diag.N_i,
            N_j=diag.N_j,
            V_i=diag.V_i,
            V_j=diag.V_j,
            M_i=diag.M_i,
            M_j=diag.M_j,
            max_N=diag.max_N,
            max_V=diag.max_V,
            max_M=diag.max_M,
        ))
    
    # Compute metrics
    drift = compute_drift(nodes, d, params.height)
    max_disp = max(disp['magnitude'] for disp in displacements.values())
    
    metrics = Frame2DMetrics(
        n_nodes=n_nodes,
        n_elements=n_elements,
        n_supports=len(support_node_ids),
        max_displacement_mm=round(max_disp * 1000, 2),
        drift_mm=round(drift['max_drift_mm'], 2),
        drift_ratio=round(drift['drift_ratio'], 6),
        drift_passes=drift['drift_passes'],
        max_axial_force=round(summary.get('max_axial_force', 0), 2),
        max_shear_force=round(summary.get('max_shear_force', 0), 2),
        max_moment=round(summary.get('max_moment', 0), 2),
        critical_element_N=summary.get('critical_element_N'),
        critical_element_V=summary.get('critical_element_V'),
        critical_element_M=summary.get('critical_element_M'),
    )
    
    return Frame2DResult(
        success=True,
        nodes=nodes_list,
        elements=elements_list,
        reactions=reactions_list,
        metrics=metrics,
        diagrams=diagrams_list,
        params=params.model_dump(),
    )


@app.post("/api/frame2d/generate", response_model=Frame2DResult)
async def generate_frame2d(params: Frame2DParams):
    """Generate and solve a 2D portal frame with force diagrams."""
    return generate_and_solve_frame2d(params)


@app.post("/api/frame2d/diagrams", response_model=Frame2DResult)
async def get_frame2d_diagrams(params: Frame2DParams):
    """Get force diagram data for a 2D portal frame (alias for generate)."""
    return generate_and_solve_frame2d(params)


# =============================================================================
# Section Extraction Endpoint
# =============================================================================

@app.post("/api/section/extract", response_model=SectionExtractResult)
async def extract_section_from_3d(params: SectionExtractParams):
    """
    Extract a 2D frame section from a 3D spaceframe at a given slice position.
    
    This endpoint:
    1. Generates the 3D spaceframe with given parameters
    2. Slices at the specified X or Y coordinate
    3. Extracts nodes and bars on that plane
    4. Projects to 2D and solves with force diagrams
    5. Returns complete 2D analysis with tributary loading
    """
    print(f"\n{'='*60}")
    print(f"[DEBUG] SECTION EXTRACT REQUEST")
    print(f"[DEBUG] slice_axis: {params.slice_axis}, position: {params.slice_position}")
    print(f"[DEBUG] 3D params: {params.width}x{params.depth}, {params.nx}x{params.ny}")
    print(f"{'='*60}")
    
    try:
        # Step 1: Generate 3D structure
        canopy_params = CanopyParams(
            width=params.width,
            depth=params.depth,
            nx=params.nx,
            ny=params.ny,
            min_height=params.min_height,
            max_height=params.max_height,
            heightfield=params.heightfield,
            topology=params.topology,
            support_layout=params.support_layout,
            E=12e9,  # Timber
            A=params.A_cm2 * 1e-4,  # cm² to m²
            gravity_load=params.gravity_kn * 1000,  # kN to N
        )
        
        nodes_3d, bars_3d, support_nodes_3d = generate_canopy(canopy_params)
        
        print(f"[DEBUG] Generated 3D: {len(nodes_3d)} nodes, {len(bars_3d)} bars")
        
    except Exception as e:
        print(f"[ERROR] 3D generation failed: {str(e)}")
        return SectionExtractResult(success=False, error=f"3D generation failed: {str(e)}")
    
    try:
        # Step 2: Find actual slice coordinate
        slice_axis = params.slice_axis.lower()
        if slice_axis not in ['x', 'y']:
            slice_axis = 'y'
        
        slice_value = find_slice_coordinate(nodes_3d, slice_axis, params.slice_position)
        
        print(f"[DEBUG] Slicing at {slice_axis}={slice_value}")
        
        # Step 3: Extract 2D section
        # Use timber properties
        section = TIMBER_SECTIONS[5]  # Mid-range section
        
        extraction = extract_section(
            nodes_3d=nodes_3d,
            bars_3d=bars_3d,
            support_nodes_3d=support_nodes_3d,
            slice_axis=slice_axis,
            slice_value=slice_value,
            tolerance=0.1,  # 10cm tolerance
            E=DEFAULT_MATERIAL.E,
            A=section.A,
            I=section.I,
        )
        
        print(f"[DEBUG] Extracted: {extraction.n_nodes_found} nodes, {extraction.n_bars_found} bars")
        print(f"[DEBUG] Tributary width: {extraction.tributary_width}m")
        
        if extraction.n_nodes_found < 2:
            return SectionExtractResult(
                success=False,
                error=f"Not enough nodes found at {slice_axis}={slice_value:.2f}m (found {extraction.n_nodes_found})"
            )
        
        if extraction.n_bars_found == 0:
            return SectionExtractResult(
                success=False,
                error=f"No bars found connecting nodes at {slice_axis}={slice_value:.2f}m"
            )
        
    except Exception as e:
        print(f"[ERROR] Section extraction failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return SectionExtractResult(success=False, error=f"Section extraction failed: {str(e)}")
    
    try:
        # Step 4: Calculate tributary UDL
        footprint_area = params.width * params.depth
        section_span = get_section_span(extraction.nodes_2d)
        
        calculated_udl = calculate_tributary_udl(
            total_gravity_load=params.gravity_kn * 1000,  # kN to N
            footprint_area=footprint_area,
            tributary_width=extraction.tributary_width,
            section_span=section_span,
        )
        
        # Use override if provided, otherwise use calculated
        if params.udl_override_kn_m is not None:
            applied_udl = params.udl_override_kn_m * 1000  # kN/m to N/m
        else:
            applied_udl = calculated_udl
        
        print(f"[DEBUG] Calculated UDL: {calculated_udl/1000:.2f} kN/m")
        print(f"[DEBUG] Applied UDL: {applied_udl/1000:.2f} kN/m")
        
    except Exception as e:
        print(f"[ERROR] UDL calculation failed: {str(e)}")
        return SectionExtractResult(success=False, error=f"UDL calculation failed: {str(e)}")
    
    try:
        # Step 5: Solve 2D frame
        nodes = extraction.nodes_2d
        elements = extraction.elements_2d
        fixed_dofs = extraction.fixed_dofs
        
        n_nodes = len(nodes)
        n_elements = len(elements)
        
        # Build element UDL map - apply UDL to horizontal elements only
        element_udls = {}
        for elem in elements:
            ni = nodes[elem.ni]
            nj = nodes[elem.nj]
            # Check if element is roughly horizontal (beam-like)
            dy = abs(nj.y - ni.y)
            dx = abs(nj.x - ni.x)
            if dx > 0.1 and dy / dx < 0.3:  # Mostly horizontal
                element_udls[elem.id] = -applied_udl  # Negative = downward
        
        # Assemble and solve
        K = assemble_2d_K(nodes, elements)
        ndof = DOF_PER_NODE * n_nodes
        
        F = np.zeros(ndof)
        F += assemble_element_loads_global(nodes, elements, element_udls)
        
        from mini_branch.solve import solve_linear as solve_2d
        d, R, _ = solve_2d(K, F, fixed_dofs)
        
        print(f"[DEBUG] Solve successful! Max displacement: {np.max(np.abs(d)):.6f} m")
        
    except MechanismError as e:
        print(f"[ERROR] MechanismError: {str(e)}")
        return SectionExtractResult(success=False, error=f"Structure unstable: {str(e)}")
    except Exception as e:
        print(f"[ERROR] Solve failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return SectionExtractResult(success=False, error=f"Solve failed: {str(e)}")
    
    try:
        # Step 6: Compute diagrams
        diagram_data = compute_frame_diagrams(
            nodes, elements, d, element_udls,
            scale=params.deflection_scale,
            n_points=params.n_diagram_points
        )
        summary = get_frame_summary(diagram_data)
        
        print(f"[DEBUG] Computed diagrams for {len(diagram_data)} elements")
        
    except Exception as e:
        print(f"[WARNING] Diagram computation failed: {str(e)}")
        diagram_data = []
        summary = {}
    
    # Build response
    displacements = compute_nodal_displacements(nodes, d)
    support_node_ids = set(dof // DOF_PER_NODE for dof in fixed_dofs)
    
    nodes_list = []
    for node_id, node in nodes.items():
        disp = displacements.get(node_id, {'ux': 0, 'uy': 0, 'rz': 0})
        nodes_list.append(Frame2DNodeData(
            id=node_id,
            x=round(node.x, 4),
            y=round(node.y, 4),
            ux=round(disp['ux'], 8),
            uy=round(disp['uy'], 8),
            rz=round(disp['rz'], 8),
            is_support=node_id in support_node_ids,
        ))
    
    elements_list = []
    for elem in elements:
        dx = nodes[elem.nj].x - nodes[elem.ni].x
        dy = nodes[elem.nj].y - nodes[elem.ni].y
        L = float(np.hypot(dx, dy))
        c = dx / L if L > 0 else 1.0
        
        if abs(c) < 0.3:
            elem_type = "column"
        elif abs(c) > 0.7:
            elem_type = "beam"
        else:
            elem_type = "brace"
        
        udl_w = element_udls.get(elem.id, None)
        f_local = element_end_forces_local(nodes, elem, d, udl_w)
        
        elements_list.append(Frame2DElementData(
            id=elem.id,
            ni=elem.ni,
            nj=elem.nj,
            element_type=elem_type,
            length=round(L, 4),
            section_name="Extracted",
            A=elem.A,
            I=elem.I,
            N_i=round(float(f_local[0]), 2),
            N_j=round(float(f_local[3]), 2),
            V_i=round(float(f_local[1]), 2),
            V_j=round(float(f_local[4]), 2),
            M_i=round(float(f_local[2]), 2),
            M_j=round(float(f_local[5]), 2),
        ))
    
    reactions_dict = compute_reactions(R, fixed_dofs, nodes)
    reactions_list = [
        ReactionData(
            node_id=nid,
            Rx=round(r['Rx'], 2),
            Ry=round(r['Ry'], 2),
            Mz=round(r['Mz'], 2),
        )
        for nid, r in reactions_dict.items()
    ]
    
    diagrams_list = []
    for diag in diagram_data:
        diagrams_list.append(ElementDiagramData(
            element_id=diag.element_id,
            element_type=diag.element_type,
            ni=diag.ni,
            nj=diag.nj,
            length=diag.length,
            points=[DiagramPoint(
                x_local=p.x_local,
                x_global=p.x_global,
                y_global=p.y_global,
                N=p.N,
                V=p.V,
                M=p.M,
            ) for p in diag.points],
            deflected_shape=[DeflectedPoint(x=p.x, y=p.y) for p in diag.deflected_shape],
            N_i=diag.N_i,
            N_j=diag.N_j,
            V_i=diag.V_i,
            V_j=diag.V_j,
            M_i=diag.M_i,
            M_j=diag.M_j,
            max_N=diag.max_N,
            max_V=diag.max_V,
            max_M=diag.max_M,
        ))
    
    # Compute metrics
    max_height = max(n.y for n in nodes.values()) if nodes else 1.0
    drift = compute_drift(nodes, d, max_height)
    max_disp = max(disp['magnitude'] for disp in displacements.values()) if displacements else 0
    
    metrics = Frame2DMetrics(
        n_nodes=n_nodes,
        n_elements=n_elements,
        n_supports=len(support_node_ids),
        max_displacement_mm=round(max_disp * 1000, 2),
        drift_mm=round(drift['max_drift_mm'], 2),
        drift_ratio=round(drift['drift_ratio'], 6) if drift['drift_ratio'] else 0,
        drift_passes=drift['drift_passes'],
        max_axial_force=round(summary.get('max_axial_force', 0), 2),
        max_shear_force=round(summary.get('max_shear_force', 0), 2),
        max_moment=round(summary.get('max_moment', 0), 2),
        critical_element_N=summary.get('critical_element_N'),
        critical_element_V=summary.get('critical_element_V'),
        critical_element_M=summary.get('critical_element_M'),
    )
    
    frame_result = Frame2DResult(
        success=True,
        nodes=nodes_list,
        elements=elements_list,
        reactions=reactions_list,
        metrics=metrics,
        diagrams=diagrams_list,
        params={"slice_axis": slice_axis, "slice_value": slice_value},
    )
    
    return SectionExtractResult(
        success=True,
        slice_axis=slice_axis,
        slice_value=round(slice_value, 3),
        tributary_width=round(extraction.tributary_width, 3),
        calculated_udl_kn_m=round(calculated_udl / 1000, 3),
        applied_udl_kn_m=round(applied_udl / 1000, 3),
        section_span=round(section_span, 3),
        n_nodes_extracted=extraction.n_nodes_found,
        n_elements_extracted=extraction.n_bars_found,
        frame_result=frame_result,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

