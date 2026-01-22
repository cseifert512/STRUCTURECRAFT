# Parametric portal frame generator and design exploration
"""
EXPLORE: PARAMETRIC PORTAL FRAME GENERATOR
==========================================

PURPOSE:
--------
This module turns a portal frame into a parameterized function. Instead of
hardcoding geometry, loads, and section properties, we define them as parameters
and generate the model programmatically.

WHY THIS MATTERS:
-----------------
1. **Design Exploration**: We can generate hundreds of variants by just changing
   parameters (span, height, section indices, etc.)

2. **Automation**: The evaluation loop (Day 3 Block D) can call make_portal()
   thousands of times without manual intervention.

3. **Reproducibility**: Same parameters always produce the same model.

4. **ML Features (Day 4)**: Parameters become features for machine learning.
   The ML model learns: "If span=8m, height=4m, sec_beam=5, what's the drift?"

ENGINEERING CONTEXT:
--------------------
Portal frames are THE fundamental building block of structural engineering.
Every building, warehouse, and frame structure uses this concept.

Key behaviors:
- Gravity loads (UDL on beam) cause vertical deflection and bending
- Lateral loads (wind) cause horizontal drift (sway)
- Larger sections = stiffer = less deflection but more material
- Bracing adds lateral stiffness (reduces drift) but adds material

The challenge: Find the optimal balance between:
- Stiffness (low drift) vs. Material (low volume/carbon)
- This is a multi-objective optimization problem (Pareto frontier)
"""

from dataclasses import dataclass
from typing import Tuple, Dict, List
import numpy as np
import pandas as pd

from .catalog import Material, Section
from .model import Node, Frame2D
from .assembly import DOF_PER_NODE, dof_index


@dataclass
class PortalParams:
    """
    Parameters that define a portal frame design variant.
    
    WHY A DATACLASS?
    ----------------
    - Groups related parameters together
    - Easy to pass around as a single object
    - Can be converted to dict for CSV export
    - Type hints help catch errors
    
    WHY THESE SPECIFIC PARAMETERS?
    ------------------------------
    - span, height: Geometry (affects stiffness, moments)
    - brace: Topology (affects lateral stiffness)
    - sec_col, sec_beam, sec_brace: Section choices (affect stiffness, volume)
    - udl_w, wind_P: Loads (affect response)
    - shipping_limit: Constraint (fabrication reality)
    
    Parameters:
    -----------
    span : float
        Horizontal distance between columns (meters)
        - Typical range: 4-12 m for timber frames
        - Larger span = more beam bending, higher moments
        - Affects: beam deflection, beam moment, column moments
    
    height : float
        Column height (meters)
        - Typical range: 2.5-6 m for single-story
        - Taller = more lateral drift, higher column moments
        - Affects: drift, column moments, stability
    
    brace : int
        Bracing configuration
        - 0: No bracing (simple portal)
        - 1: X-brace (diagonal members for lateral stiffness)
        - Bracing reduces drift but adds material and complexity
    
    sec_col : int
        Section index for columns (from TIMBER_SECTIONS catalog)
        - Index into TIMBER_SECTIONS list
        - Larger index = larger section = stiffer columns
        - Affects: column strength, volume, cost
    
    sec_beam : int
        Section index for beam (from TIMBER_SECTIONS catalog)
        - Index into TIMBER_SECTIONS list
        - Larger index = larger section = stiffer beam
        - Affects: beam deflection, beam moment, volume
    
    sec_brace : int
        Section index for braces (from TIMBER_SECTIONS catalog)
        - Only used if brace=1
        - Typically smaller than columns/beam (braces are tension/compression)
        - Affects: lateral stiffness, volume
    
    udl_w : float
        Uniform distributed load on beam (N/m)
        - Negative = downward (gravity: snow, weight)
        - Typical: -1000 to -5000 N/m
        - Affects: vertical deflection, beam moments
    
    wind_P : float
        Lateral point load at top (N)
        - Positive = push to the right
        - Typical: 2000-10000 N
        - Affects: horizontal drift, column moments
    
    shipping_limit : float
        Maximum member length for shipping (meters)
        - Typical: 12 m (40 ft) in North America
        - Constraint: max(element_lengths) <= shipping_limit
        - If violated, design is rejected (fabrication constraint)
    """
    span: float
    height: float
    brace: int  # 0 = none, 1 = X-brace
    sec_col: int
    sec_beam: int
    sec_brace: int
    udl_w: float
    wind_P: float
    shipping_limit: float = 12.0


def make_portal(
    params: PortalParams,
    material: Material,
    sections: List[Section]
) -> Tuple[Dict[int, Node], List[Frame2D], List[int], Dict[int, float], Dict[int, np.ndarray]]:


    """
    Generate a complete portal frame model from parameters.
    
    WHY THIS FUNCTION?
    ------------------
    This is the core of Day 3: turn parameters into an analysis-ready model.
    The Day 2 demo hardcoded everything. This function makes it parametric.
    
    ENGINEERING PROCESS:
    -------------------
    1. Create nodes at key points (base, top, corners)
    2. Create elements connecting nodes (columns, beam, optional braces)
    3. Assign section properties from catalog
    4. Define boundary conditions (pinned bases)
    5. Define loads (UDL on beam, lateral at top)
    
    WHY THIS RETURN FORMAT?
    -----------------------
    Returns everything needed for analysis:
    - nodes, elements: For assemble_global_K()
    - fixed_dofs: For solve_linear()
    - element_udl_map: For assemble_element_loads_global()
    - nodal_loads: For adding point loads to F vector
    
    This separation keeps make_portal() focused on model generation,
    not solving. The solver code stays separate (good architecture).
    
    Parameters:
    -----------
    params : PortalParams
        Design parameters (geometry, sections, loads)
    
    material : Material
        Material properties (E, density, carbon_factor)
        - Used for all elements (same material)
    
    sections : List[Section]
        Section catalog (TIMBER_SECTIONS)
        - params.sec_col, sec_beam, sec_brace index into this
    
    Returns:
    --------
    nodes : Dict[int, Node]
        Node dictionary: {node_id: Node(x, y)}
        - Node 0: Left base (0, 0)
        - Node 1: Left top (0, H)
        - Node 2: Right top (L, H)
        - Node 3: Right base (L, 0)
        - If braced: Nodes 4, 5 for brace connections (optional)
    
    elements : List[Frame2D]
        List of frame elements
        - Element 0: Left column (0 → 1)
        - Element 1: Beam (1 → 2)
        - Element 2: Right column (2 → 3)
        - If braced: Elements 3, 4 for X-brace
    
    fixed_dofs : List[int]
        List of fixed DOF indices (boundary conditions)
        - Pinned bases: ux, uy at nodes 0 and 3
    
    element_udl_map : Dict[int, float]
        UDL values by element ID
        - {element_id: w_value}
        - Only beam (element 1) has UDL
    
    nodal_loads : Dict[int, np.ndarray]
        Point loads by node ID
        - {node_id: [Fx, Fy, Mz]} in global coordinates
        - Node 1 (left top) gets lateral load: [wind_P, 0, 0]
    
    Example:
    --------
    >>> from mini_branch.catalog import TIMBER_SECTIONS, DEFAULT_MATERIAL
    >>> params = PortalParams(
    ...     span=6.0, height=3.0, brace=0,
    ...     sec_col=3, sec_beam=4, sec_brace=1,
    ...     udl_w=-2000.0, wind_P=5000.0
    ... )
    >>> nodes, elements, fixed, udls, loads = make_portal(
    ...     params, DEFAULT_MATERIAL, TIMBER_SECTIONS
    ... )
    >>> # Now ready for analysis:
    >>> # K = assemble_global_K(nodes, elements)
    >>> # F = assemble_element_loads_global(nodes, elements, udls)
    >>> # ... add nodal loads to F ...
    >>> # d, R = solve_linear(K, F, fixed)
    """
    
    # ========================================================================
    # STEP 1: VALIDATE INPUTS
    # ========================================================================
    """
    WHY VALIDATE?
    -------------
    - Catch errors early (before building model)
    - Prevent invalid designs (negative span, out-of-range indices)
    - Better error messages than cryptic solver failures
    """
    
    if params.span <= 0:
        raise ValueError(f"span must be positive, got {params.span}")
    if params.height <= 0:
        raise ValueError(f"height must be positive, got {params.height}")
    if params.brace not in [0, 1]:
        raise ValueError(f"brace must be 0 or 1, got {params.brace}")
    
    # Check section indices are valid
    if params.sec_col < 0 or params.sec_col >= len(sections):
        raise ValueError(f"sec_col index {params.sec_col} out of range [0, {len(sections)})")
    if params.sec_beam < 0 or params.sec_beam >= len(sections):
        raise ValueError(f"sec_beam index {params.sec_beam} out of range [0, {len(sections)})")
    if params.sec_brace < 0 or params.sec_brace >= len(sections):
        raise ValueError(f"sec_brace index {params.sec_brace} out of range [0, {len(sections)})")
    
    # ========================================================================
    # STEP 2: GET SECTION PROPERTIES FROM CATALOG
    # ========================================================================
    """
    WHY INDEX INTO CATALOG?
    ----------------------
    - params.sec_col is just an integer (e.g., 3)
    - We look up the actual Section object from the catalog
    - This gives us A, I, S properties for that element
    
    ENGINEERING NOTE:
    -----------------
    Different elements can have different sections:
    - Columns might be larger (more axial load)
    - Beam might be larger (more bending)
    - Braces might be smaller (just tension/compression)
    
    This flexibility is key to optimization: we can try different combinations.
    """
    
    sec_col = sections[params.sec_col]
    sec_beam = sections[params.sec_beam]
    sec_brace = sections[params.sec_brace]
    
    # ========================================================================
    # STEP 3: CREATE NODES
    # ========================================================================
    """
    NODE LAYOUT:
    -----------
    Standard portal frame: 4 nodes forming a rectangle
    
        1 ──────────── 2
        │               │
        │               │
        0 ──────────── 3
    
    Node 0: Left base (0, 0)
    Node 1: Left top (0, H)
    Node 2: Right top (L, H)
    Node 3: Right base (L, 0)
    
    WHY THIS LAYOUT?
    ----------------
    - Standard topology for portal frames
    - Nodes at key points (supports, connections)
    - Simple, symmetric, easy to understand
    
    COORDINATE SYSTEM:
    ------------------
    - Origin (0, 0) at left base
    - X-axis: horizontal (right = positive)
    - Y-axis: vertical (up = positive)
    - This matches typical engineering conventions
    """
    
    L = params.span
    H = params.height
    
    nodes = {
        0: Node(0, 0.0, 0.0),      # Left base
        1: Node(1, 0.0, H),        # Left top
        2: Node(2, L, H),          # Right top
        3: Node(3, L, 0.0),        # Right base
    }
    
    # ========================================================================
    # STEP 4: CREATE ELEMENTS
    # ========================================================================
    """
    ELEMENT LAYOUT:
    --------------
    Basic portal: 3 elements
    
    Element 0: Left column (Node 0 → 1)
    Element 1: Beam (Node 1 → 2)
    Element 2: Right column (Node 2 → 3)
    
    If braced: Add 2 diagonal elements for X-brace
    
    WHY THIS ORDER?
    --------------
    - Columns first (0, 2), then beam (1)
    - If braced, braces come last (3, 4)
    - Consistent ordering makes debugging easier
    
    ELEMENT PROPERTIES:
    -------------------
    Each element gets properties from its assigned section:
    - E: From material (same for all)
    - A: From section catalog (different per element type)
    - I: From section catalog (different per element type)
    
    WHY DIFFERENT SECTIONS?
    -----------------------
    - Columns: Need to resist axial load + bending (from lateral load)
    - Beam: Needs to resist bending (from UDL)
    - Braces: Need to resist tension/compression (from lateral load)
    
    This is realistic: engineers size each member for its role.
    """
    
    elements = []
    element_id = 0
    
    # Left column (Node 0 → 1)
    elements.append(Frame2D(
        id=element_id,
        ni=0,
        nj=1,
        E=material.E,
        A=sec_col.A,
        I=sec_col.I,
    ))
    element_id += 1
    
    # Beam (Node 1 → 2)
    elements.append(Frame2D(
        id=element_id,
        ni=1,
        nj=2,
        E=material.E,
        A=sec_beam.A,
        I=sec_beam.I,
    ))
    element_id += 1
    
    # Right column (Node 2 → 3)
    elements.append(Frame2D(
        id=element_id,
        ni=2,
        nj=3,
        E=material.E,
        A=sec_col.A,
        I=sec_col.I,
    ))
    element_id += 1
    
    # ========================================================================
    # STEP 5: ADD BRACING (IF REQUESTED)
    # ========================================================================
    """
    X-BRACE LAYOUT:
    --------------
    If brace=1, add two diagonal members:
    
        1 ──────────── 2
        │  ╲         ╱  │
        │    ╲     ╱    │
        │      ╲ ╱      │
        0 ──────────── 3
    
    Element 3: Diagonal from Node 1 → 3 (top-left to bottom-right)
    Element 4: Diagonal from Node 0 → 2 (bottom-left to top-right)
    
    WHY X-BRACE?
    ------------
    - Adds lateral stiffness (reduces drift)
    - Works in both directions (wind from left or right)
    - One brace in tension, one in compression (depending on load direction)
    
    ENGINEERING PRINCIPLE:
    ---------------------
    Braces form a "truss" that resists lateral loads through axial forces.
    This is more efficient than relying only on column bending.
    
    TRADE-OFF:
    ----------
    - Pro: Reduces drift (better performance)
    - Con: Adds material (more volume/carbon)
    - Con: Adds complexity (more connections)
    
    This is exactly the kind of trade-off we want to explore in optimization!
    """
    
    if params.brace == 1:
        # Diagonal from top-left to bottom-right (Node 1 → 3)
        elements.append(Frame2D(
            id=element_id,
            ni=1,
            nj=3,
            E=material.E,
            A=sec_brace.A,
            I=sec_brace.I,  # Braces can have I too (though often just axial)
        ))
        element_id += 1
        
        # Diagonal from bottom-left to top-right (Node 0 → 2)
        elements.append(Frame2D(
            id=element_id,
            ni=0,
            nj=2,
            E=material.E,
            A=sec_brace.A,
            I=sec_brace.I,
        ))
        element_id += 1
    
    # ========================================================================
    # STEP 6: DEFINE BOUNDARY CONDITIONS
    # ========================================================================
    """
    PINNED BASES:
    -------------
    Standard portal frame: pinned supports at both bases
    
    Node 0 (left base):  fix ux=0, uy=0  (translation fixed, rotation free)
    Node 3 (right base): fix ux=0, uy=0  (translation fixed, rotation free)
    
    WHY PINNED?
    -----------
    - Prevents rigid body motion (translation)
    - Allows rotation (reduces moment at base)
    - Easier to construct (no moment connection needed)
    - Standard for portal frames
    
    DOF INDEXING:
    -------------
    For node i:
    - DOF 0: ux (horizontal displacement)
    - DOF 1: uy (vertical displacement)
    - DOF 2: rz (rotation)
    
    Global DOF index = DOF_PER_NODE * node_id + local_dof
    
    Example:
    - Node 0, ux: DOF 0*3 + 0 = 0
    - Node 0, uy: DOF 0*3 + 1 = 1
    - Node 3, ux: DOF 3*3 + 0 = 9
    - Node 3, uy: DOF 3*3 + 1 = 10
    """
    
    fixed_dofs = [
        dof_index(0, 0),  # Node 0, ux
        dof_index(0, 1),  # Node 0, uy
        dof_index(3, 0),  # Node 3, ux
        dof_index(3, 1),  # Node 3, uy
    ]
    
    # ========================================================================
    # STEP 7: DEFINE LOADS
    # ========================================================================
    """
    LOAD TYPES:
    ----------
    1. UDL on beam (distributed load)
       - Applied to Element 1 (beam)
       - Converted to equivalent nodal loads by assemble_element_loads_global()
    
    2. Lateral point load (concentrated)
       - Applied at Node 1 (left top)
       - Represents wind/seismic load
       - Applied directly to F vector
    
    WHY THESE LOADS?
    ---------------
    - UDL: Represents gravity (snow, equipment, self-weight)
    - Lateral: Represents wind or seismic (horizontal push)
    - This is a standard load combination for building codes
    
    LOAD DIRECTION:
    ---------------
    - UDL: Negative = downward (gravity)
    - Lateral: Positive = push to the right
    
    This matches typical engineering sign conventions.
    """
    
    # UDL on beam (Element 1)
    element_udl_map = {
        1: params.udl_w,  # Only beam has UDL
    }
    
    # Lateral point load at Node 1 (left top)
    # Format: {node_id: [Fx, Fy, Mz]} in global coordinates
    nodal_loads = {
        1: np.array([params.wind_P, 0.0, 0.0], dtype=float),  # Horizontal force only
    }
    
    # ========================================================================
    # STEP 8: RETURN COMPLETE MODEL
    # ========================================================================
    """
    RETURN FORMAT:
    -------------
    We return everything needed for analysis:
    
    1. nodes, elements: For assemble_global_K()
    2. fixed_dofs: For solve_linear()
    3. element_udl_map: For assemble_element_loads_global()
    4. nodal_loads: For adding to F vector
    
    WHY THIS FORMAT?
    ----------------
    - Separates model generation from solving
    - make_portal() doesn't need to know about solver
    - Easy to test: can check nodes/elements without solving
    - Flexible: can modify loads before solving if needed
    
    USAGE PATTERN:
    -------------
    nodes, elements, fixed, udls, loads = make_portal(params, material, sections)
    
    # Build stiffness matrix
    K = assemble_global_K(nodes, elements)
    
    # Build load vector
    F = assemble_element_loads_global(nodes, elements, udls)
    for node_id, load_vec in loads.items():
        F[dof_index(node_id, 0)] += load_vec[0]  # Fx
        F[dof_index(node_id, 1)] += load_vec[1]  # Fy
        F[dof_index(node_id, 2)] += load_vec[2]  # Mz
    
    # Solve
    d, R, free = solve_linear(K, F, fixed)
    """
    
    return nodes, elements, fixed_dofs, element_udl_map, nodal_loads

# ============================================================================
#VARIANT SAMPLING
# ============================================================================

def sample_params(
    rng: np.random.Generator,
    sections: List[Section],
    n: int,
    span_range: Tuple[float, float] = (4.0, 12.0),
    height_range: Tuple[float, float] = (2.5, 6.0),
    udl_range: Tuple[float, float] = (-5000.0, -1000.0),
    wind_range: Tuple[float, float] = (2000.0, 10000.0),
    shipping_limit: float = 12.0,
    min_brace_angle: float = 25.0,
) -> List[PortalParams]:
    """
    Generate random portal frame design variants.
    
    WHY THIS FUNCTION?
    ------------------
    This is the heart of design exploration: we need to generate many
    different design variants to explore the design space. Instead of
    manually creating each variant, we randomly sample parameters.
    
    ENGINEERING PROCESS:
    -------------------
    1. Sample random parameters (span, height, sections, loads)
    2. Check constraints (shipping, section variety, brace angle)
    3. Reject invalid designs or apply penalties
    4. Return list of valid variants
    
    WHY RANDOM SAMPLING?
    --------------------
    - Explores design space broadly (not just local optima)
    - Easy to implement (no complex algorithms)
    - Good for initial exploration (Day 3)
    - Day 4 will use smarter sampling (ML-guided)
    
    PARAMETER RANGES:
    -----------------
    - span: 4-12 m (typical building bay widths)
    - height: 2.5-6 m (single-story heights)
    - UDL: -5000 to -1000 N/m (realistic gravity loads)
    - Wind: 2000-10000 N (typical wind loads)
    
    These ranges represent realistic design scenarios.
    
    Parameters:
    -----------
    rng : np.random.Generator
        NumPy random number generator (for reproducibility)
        - Use np.random.default_rng(seed) to create
        - Same seed = same sequence of random numbers
    
    sections : List[Section]
        Section catalog (TIMBER_SECTIONS)
        - Used to pick random section indices
    
    n : int
        Number of variants to generate
        - May return fewer if many are rejected by constraints
    
    span_range : Tuple[float, float]
        (min, max) span in meters
        - Default: (4.0, 12.0) m
    
    height_range : Tuple[float, float]
        (min, max) height in meters
        - Default: (2.5, 6.0) m
    
    udl_range : Tuple[float, float]
        (min, max) UDL in N/m (both should be negative for downward)
        - Default: (-5000.0, -1000.0) N/m
    
    wind_range : Tuple[float, float]
        (min, max) lateral load in N
        - Default: (2000.0, 10000.0) N
    
    shipping_limit : float
        Maximum member length for shipping (meters)
        - Default: 12.0 m (40 ft)
        - Designs with longer members are rejected
    
    min_brace_angle : float
        Minimum brace angle in degrees
        - Default: 25.0°
        - Braces with angle < min_brace_angle are rejected
        - Too shallow = ineffective and hard to construct
    
    Returns:
    --------
    List[PortalParams]
        List of valid design variants
        - May be shorter than n if some designs are rejected
        - All returned designs satisfy constraints
    
    Example:
    --------
    >>> import numpy as np
    >>> from mini_branch.catalog import TIMBER_SECTIONS
    >>> rng = np.random.default_rng(seed=42)
    >>> variants = sample_params(rng, TIMBER_SECTIONS, n=100)
    >>> print(f"Generated {len(variants)} valid variants")
    """
    
    variants = []
    max_attempts = n * 3  # Try up to 3x to account for rejections
    attempts = 0
    
    while len(variants) < n and attempts < max_attempts:
        attempts += 1
        
        # ====================================================================
        # STEP 1: SAMPLE RANDOM PARAMETERS
        # ====================================================================
        """
        WHY UNIFORM RANDOM SAMPLING?
        -----------------------------
        - Simple and fast
        - Explores entire design space uniformly
        - No bias toward any region
        - Good baseline for comparison (Day 4 will use smarter sampling)
        
        ALTERNATIVES (Day 4):
        - Latin Hypercube Sampling (better coverage)
        - ML-guided sampling (focus on promising regions)
        - Multi-objective optimization (Pareto-optimal)
        """
        
        # Sample geometry
        span = rng.uniform(span_range[0], span_range[1])
        height = rng.uniform(height_range[0], height_range[1])
        
        # Sample bracing (0 = none, 1 = X-brace)
        brace = rng.integers(0, 2)  # 0 or 1
        
        # Sample section indices (random from catalog)
        n_sections = len(sections)
        sec_col = rng.integers(0, n_sections)
        sec_beam = rng.integers(0, n_sections)
        sec_brace = rng.integers(0, n_sections)
        
        # Sample loads
        udl_w = rng.uniform(udl_range[0], udl_range[1])
        wind_P = rng.uniform(wind_range[0], wind_range[1])
        
        # ====================================================================
        # STEP 2: CHECK CONSTRAINTS
        # ====================================================================
        """
        CONSTRAINT CHECKING:
        --------------------
        We check three fabrication-aware constraints:
        
        1. Shipping limit: Long members can't be shipped
        2. Section variety: Too many unique sections = expensive
        3. Brace angle: Shallow braces are ineffective
        
        WHY REJECT INSTEAD OF PENALIZE?
        --------------------------------
        - Some constraints are hard limits (can't ship 20m member)
        - Rejection is simpler than penalty functions
        - Day 4 might use penalties for soft constraints
        """
        
        # Constraint 1: Shipping limit
        # Check if any member exceeds shipping limit
        max_member_length = max(span, height)
        if brace == 1:
            # Diagonal brace length = sqrt(span² + height²)
            brace_length = np.hypot(span, height)
            max_member_length = max(max_member_length, brace_length)
        
        if max_member_length > shipping_limit:
            continue  # Reject this design
        
        # Constraint 2: Section variety
        # Count unique sections used
        unique_sections = set([sec_col, sec_beam])
        if brace == 1:
            unique_sections.add(sec_brace)
        
        if len(unique_sections) > 3:
            # Too many unique sections = expensive fabrication
            # For now, we'll allow it but could reject
            # (Day 4 might use this as a penalty instead)
            pass
        
        # Constraint 3: Brace angle (if braced)
        if brace == 1:
            # Calculate brace angle from horizontal
            # Diagonal from (0, H) to (L, 0) or (0, 0) to (L, H)
            # Angle = arctan(height / span) or arctan(span / height)
            # We want the smaller angle (more critical)
            angle1 = np.degrees(np.arctan(height / span))
            angle2 = np.degrees(np.arctan(span / height))
            min_angle = min(angle1, angle2)
            
            if min_angle < min_brace_angle:
                continue  # Reject: brace angle too shallow
        
        # ====================================================================
        # STEP 3: CREATE PARAMS AND ADD TO LIST
        # ====================================================================
        """
        WHY CREATE PortalParams HERE?
        ------------------------------
        - Validates parameters (catches errors early)
        - Makes code cleaner (single object instead of many variables)
        - Easy to return and use later
        """
        
        params = PortalParams(
            span=span,
            height=height,
            brace=brace,
            sec_col=sec_col,
            sec_beam=sec_beam,
            sec_brace=sec_brace,
            udl_w=udl_w,
            wind_P=wind_P,
            shipping_limit=shipping_limit,
        )
        
        variants.append(params)
    
    # ========================================================================
    # STEP 4: RETURN RESULTS
    # ========================================================================
    """
    WHY RETURN LIST?
    ----------------
    - Easy to iterate over variants
    - Can check length to see how many were generated
    - Can filter/sort before evaluation
    """
    
    if len(variants) < n:
        print(f"Warning: Generated {len(variants)} variants (requested {n})")
        print(f"  Some designs were rejected by constraints")
    
    return variants


def check_constraints(
    params: PortalParams,
    sections: List[Section],
    shipping_limit: float = 12.0,
    min_brace_angle: float = 25.0,
    max_unique_sections: int = 3,
) -> Tuple[bool, str]:
    """
    Check if a design variant satisfies fabrication constraints.
    
    WHY A SEPARATE FUNCTION?
    -------------------------
    - Can be used independently (not just in sampling)
    - Easy to test constraint logic
    - Can be called from evaluate_variant() to flag issues
    
    Parameters:
    -----------
    params : PortalParams
        Design parameters to check
    
    sections : List[Section]
        Section catalog (for reference, not used here)
    
    shipping_limit : float
        Maximum member length (meters)
    
    min_brace_angle : float
        Minimum brace angle (degrees)
    
    max_unique_sections : int
        Maximum number of unique sections allowed
    
    Returns:
    --------
    Tuple[bool, str]
        (is_valid, reason)
        - is_valid: True if design satisfies all constraints
        - reason: Empty string if valid, error message if invalid
    """
    
    # Check shipping limit
    max_member_length = max(params.span, params.height)
    if params.brace == 1:
        brace_length = np.hypot(params.span, params.height)
        max_member_length = max(max_member_length, brace_length)
    
    if max_member_length > shipping_limit:
        return False, f"Member length {max_member_length:.2f}m exceeds shipping limit {shipping_limit:.1f}m"
    
    # Check section variety
    unique_sections = set([params.sec_col, params.sec_beam])
    if params.brace == 1:
        unique_sections.add(params.sec_brace)
    
    if len(unique_sections) > max_unique_sections:
        return False, f"Too many unique sections: {len(unique_sections)} > {max_unique_sections}"
    
    # Check brace angle
    if params.brace == 1:
        angle1 = np.degrees(np.arctan(params.height / params.span))
        angle2 = np.degrees(np.arctan(params.span / params.height))
        min_angle = min(angle1, angle2)
        
        if min_angle < min_brace_angle:
            return False, f"Brace angle {min_angle:.1f}° < minimum {min_brace_angle:.1f}°"
    
    return True, ""


# ============================================================================
# VARIANT EVALUATION
# ============================================================================

def evaluate_variant(
    params: PortalParams,
    material: Material,
    sections: List[Section],
) -> Dict:
    """
    Evaluate a single portal frame design variant.
    
    This function takes design parameters, builds the model, solves it,
    and extracts key performance metrics (drift, moments, volume, etc.).
    
    WHY THIS FUNCTION?
    ------------------
    - Encapsulates the full evaluation pipeline (model → solve → metrics)
    - Handles failures gracefully (returns ok=False with reason)
    - Returns a clean dict that can be converted to DataFrame row
    - Reusable in both batch search and single-case analysis
    
    Parameters:
    -----------
    params : PortalParams
        Design parameters to evaluate
    
    material : Material
        Material properties (E, density, carbon_factor)
    
    sections : List[Section]
        Section catalog (TIMBER_SECTIONS)
    
    Returns:
    --------
    Dict
        Dictionary with all params fields + metrics:
        - All PortalParams fields (span, height, brace, etc.)
        - drift: float (max horizontal displacement at top, m)
        - max_abs_M: float (maximum absolute moment, N·m)
        - volume: float (total material volume, m³)
        - carbon: float (embodied carbon, kg CO₂)
        - ok: bool (True if solved successfully)
        - reason: str (empty if ok, error message if failed)
    """
    from .assembly import assemble_global_K
    from .loads import assemble_element_loads_global
    from .solve import solve_linear, MechanismError
    from .post import element_end_forces_local
    from .elements import element_geometry
    
    # Initialize result dict with all params
    result = {
        'span': params.span,
        'height': params.height,
        'brace': params.brace,
        'sec_col': params.sec_col,
        'sec_beam': params.sec_beam,
        'sec_brace': params.sec_brace,
        'udl_w': params.udl_w,
        'wind_P': params.wind_P,
        'shipping_limit': params.shipping_limit,
    }
    
    try:
        # ====================================================================
        # STEP 1: BUILD MODEL
        # ====================================================================
        nodes, elements, fixed_dofs, element_udl_map, nodal_loads = make_portal(
            params, material, sections
        )
        
        # ====================================================================
        # STEP 2: ASSEMBLE STIFFNESS MATRIX
        # ====================================================================
        K = assemble_global_K(nodes, elements)
        ndof = DOF_PER_NODE * len(nodes)
        
        # ====================================================================
        # STEP 3: ASSEMBLE LOAD VECTOR
        # ====================================================================
        F = np.zeros(ndof)
        
        # Add UDL loads
        F += assemble_element_loads_global(nodes, elements, element_udl_map)
        
        # Add nodal loads
        for node_id, load_vec in nodal_loads.items():
            F[dof_index(node_id, 0)] += load_vec[0]  # Fx
            F[dof_index(node_id, 1)] += load_vec[1]  # Fy
            F[dof_index(node_id, 2)] += load_vec[2]  # Mz
        
        # ====================================================================
        # STEP 4: SOLVE
        # ====================================================================
        d, R, free = solve_linear(K, F, fixed_dofs)
        
        # ====================================================================
        # STEP 5: EXTRACT METRICS
        # ====================================================================
        
        # Drift: max horizontal displacement at top nodes (1 and 2)
        ux_node1 = d[dof_index(1, 0)]  # Left top, horizontal
        ux_node2 = d[dof_index(2, 0)]  # Right top, horizontal
        drift = max(abs(ux_node1), abs(ux_node2))
        
        # Max moment: check all elements
        max_abs_M = 0.0
        for element in elements:
            udl_w = element_udl_map.get(element.id, None)
            f_local = element_end_forces_local(nodes, element, d, udl_w=udl_w)
            
            Mi = f_local[2]  # Moment at node i
            Mj = f_local[5]  # Moment at node j
            max_abs_M = max(max_abs_M, abs(Mi), abs(Mj))
        
        # Volume: sum of (A * L) for all elements
        volume = 0.0
        for element in elements:
            L, _, _ = element_geometry(nodes, element)
            volume += element.A * L
        
        # Carbon: volume * carbon_factor
        carbon = volume * material.carbon_factor
        
        # ====================================================================
        # STEP 6: RETURN SUCCESS RESULT
        # ====================================================================
        result.update({
            'drift': drift,
            'max_abs_M': max_abs_M,
            'volume': volume,
            'carbon': carbon,
            'ok': True,
            'reason': '',
        })
        
    except MechanismError as e:
        # Structure is unstable (mechanism)
        result.update({
            'drift': np.nan,
            'max_abs_M': np.nan,
            'volume': np.nan,
            'carbon': np.nan,
            'ok': False,
            'reason': f'unstable: {str(e)}',
        })
    except Exception as e:
        # Other errors (shouldn't happen, but be safe)
        result.update({
            'drift': np.nan,
            'max_abs_M': np.nan,
            'volume': np.nan,
            'carbon': np.nan,
            'ok': False,
            'reason': f'error: {str(e)}',
        })
    
    return result


def run_search(
    n: int,
    seed: int = 42,
    material: Material = None,
    sections: List[Section] = None,
) -> pd.DataFrame:
    """
    Run a batch search over portal frame design variants.
    
    This function generates random design variants, evaluates each one,
    and returns a DataFrame with all results. This is the core of Day 3's
    design exploration workflow.
    
    WHY THIS FUNCTION?
    ------------------
    - Encapsulates the entire search loop
    - Handles failures gracefully (continues even if some variants fail)
    - Returns clean DataFrame for analysis/plotting
    - Reproducible via seed
    
    Parameters:
    -----------
    n : int
        Number of variants to generate and evaluate
        - May return fewer if many are rejected by constraints
    
    seed : int
        Random seed for reproducibility
        - Same seed = same sequence of variants
        - Default: 42
    
    material : Material, optional
        Material properties (defaults to DEFAULT_MATERIAL)
    
    sections : List[Section], optional
        Section catalog (defaults to TIMBER_SECTIONS)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with one row per variant, columns:
        - All PortalParams fields (span, height, brace, sec_col, etc.)
        - drift: float (m)
        - max_abs_M: float (N·m)
        - volume: float (m³)
        - carbon: float (kg CO₂)
        - ok: bool
        - reason: str
    """
    from .catalog import DEFAULT_MATERIAL, TIMBER_SECTIONS
    
    # Use defaults if not provided
    if material is None:
        material = DEFAULT_MATERIAL
    if sections is None:
        sections = TIMBER_SECTIONS
    
    # Initialize random number generator
    rng = np.random.default_rng(seed)
    
    # Generate variants
    print(f"Generating {n} design variants...")
    variants = sample_params(rng, sections, n)
    print(f"Generated {len(variants)} valid variants (some may have been rejected by constraints)")
    
    # Evaluate each variant
    print(f"Evaluating {len(variants)} variants...")
    results = []
    
    for i, params in enumerate(variants):
        if (i + 1) % 50 == 0 or (i + 1) == len(variants):
            print(f"  Progress: {i + 1}/{len(variants)}")
        
        result = evaluate_variant(params, material, sections)
        results.append(result)
    
    print(f"Evaluation complete: {sum(r['ok'] for r in results)} successful, {sum(not r['ok'] for r in results)} failed")
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    return df