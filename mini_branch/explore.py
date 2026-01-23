# mini_branch/explore.py
"""Parametric portal frame generator and design exploration."""

from dataclasses import dataclass
from typing import Tuple, Dict, List
import numpy as np
import pandas as pd

from .catalog import Material, Section
from .model import Node, Frame2D
from .assembly import DOF_PER_NODE, dof_index


@dataclass
class PortalParams:
    """Portal frame design parameters."""
    span: float           # Horizontal distance between columns (m)
    height: float         # Column height (m)
    brace: int            # 0=none, 1=X-brace
    sec_col: int          # Section index for columns
    sec_beam: int         # Section index for beam
    sec_brace: int        # Section index for braces
    udl_w: float          # UDL on beam (N/m, negative=down)
    wind_P: float         # Lateral load at top (N)
    shipping_limit: float = 12.0  # Max member length (m)


def make_portal(
    params: PortalParams,
    material: Material,
    sections: List[Section]
) -> Tuple[Dict[int, Node], List[Frame2D], List[int], Dict[int, float], Dict[int, np.ndarray]]:
    """
    Generate portal frame model from parameters.
    
    Args:
        params: Design parameters
        material: Material properties
        sections: Section catalog
    
    Returns:
        nodes: {id: Node}
        elements: List[Frame2D]
        fixed_dofs: Constrained DOF indices
        element_udl_map: {element_id: w}
        nodal_loads: {node_id: [Fx, Fy, Mz]}
    """
    # Validate inputs
    if params.span <= 0:
        raise ValueError(f"span must be positive, got {params.span}")
    if params.height <= 0:
        raise ValueError(f"height must be positive, got {params.height}")
    if params.brace not in [0, 1]:
        raise ValueError(f"brace must be 0 or 1, got {params.brace}")
    
    n_sec = len(sections)
    if not (0 <= params.sec_col < n_sec):
        raise ValueError(f"sec_col {params.sec_col} out of range [0, {n_sec})")
    if not (0 <= params.sec_beam < n_sec):
        raise ValueError(f"sec_beam {params.sec_beam} out of range [0, {n_sec})")
    if not (0 <= params.sec_brace < n_sec):
        raise ValueError(f"sec_brace {params.sec_brace} out of range [0, {n_sec})")
    
    # Get sections
    sec_col = sections[params.sec_col]
    sec_beam = sections[params.sec_beam]
    sec_brace = sections[params.sec_brace]
    
    L, H = params.span, params.height
    
    # Create nodes: 0=left base, 1=left top, 2=right top, 3=right base
    nodes = {
        0: Node(0, 0.0, 0.0),
        1: Node(1, 0.0, H),
        2: Node(2, L, H),
        3: Node(3, L, 0.0),
    }
    
    # Create elements
    elements = []
    eid = 0
    
    # Left column (0→1)
    elements.append(Frame2D(id=eid, ni=0, nj=1, E=material.E, A=sec_col.A, I=sec_col.I))
    eid += 1
    
    # Beam (1→2)
    elements.append(Frame2D(id=eid, ni=1, nj=2, E=material.E, A=sec_beam.A, I=sec_beam.I))
    eid += 1
    
    # Right column (2→3)
    elements.append(Frame2D(id=eid, ni=2, nj=3, E=material.E, A=sec_col.A, I=sec_col.I))
    eid += 1
    
    # Optional X-brace
    if params.brace == 1:
        elements.append(Frame2D(id=eid, ni=1, nj=3, E=material.E, A=sec_brace.A, I=sec_brace.I))
        eid += 1
        elements.append(Frame2D(id=eid, ni=0, nj=2, E=material.E, A=sec_brace.A, I=sec_brace.I))
        eid += 1
    
    # Pinned supports at bases (fix ux, uy)
    fixed_dofs = [
        dof_index(0, 0), dof_index(0, 1),  # Node 0
        dof_index(3, 0), dof_index(3, 1),  # Node 3
    ]
    
    # Loads
    element_udl_map = {1: params.udl_w}  # UDL on beam only
    nodal_loads = {1: np.array([params.wind_P, 0.0, 0.0], dtype=float)}  # Lateral at node 1
    
    return nodes, elements, fixed_dofs, element_udl_map, nodal_loads


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
    
    Args:
        rng: NumPy random generator
        sections: Section catalog
        n: Number of variants to generate
        span_range: (min, max) span in meters
        height_range: (min, max) height in meters
        udl_range: (min, max) UDL in N/m
        wind_range: (min, max) lateral load in N
        shipping_limit: Max member length (m)
        min_brace_angle: Min brace angle (degrees)
    
    Returns:
        List of valid PortalParams (may be < n if many rejected)
    """
    variants = []
    max_attempts = n * 3
    attempts = 0
    n_sections = len(sections)
    
    while len(variants) < n and attempts < max_attempts:
        attempts += 1
        
        # Sample parameters
        span = rng.uniform(*span_range)
        height = rng.uniform(*height_range)
        brace = int(rng.integers(0, 2))
        sec_col = int(rng.integers(0, n_sections))
        sec_beam = int(rng.integers(0, n_sections))
        sec_brace = int(rng.integers(0, n_sections))
        udl_w = rng.uniform(*udl_range)
        wind_P = rng.uniform(*wind_range)
        
        # Check shipping limit
        max_len = max(span, height)
        if brace == 1:
            max_len = max(max_len, np.hypot(span, height))
        if max_len > shipping_limit:
            continue
        
        # Check brace angle
        if brace == 1:
            angle = min(np.degrees(np.arctan(height / span)),
                       np.degrees(np.arctan(span / height)))
            if angle < min_brace_angle:
                continue
        
        variants.append(PortalParams(
            span=span, height=height, brace=brace,
            sec_col=sec_col, sec_beam=sec_beam, sec_brace=sec_brace,
            udl_w=udl_w, wind_P=wind_P, shipping_limit=shipping_limit,
        ))
    
    if len(variants) < n:
        print(f"Warning: Generated {len(variants)}/{n} variants (some rejected by constraints)")
    
    return variants


def check_constraints(
    params: PortalParams,
    sections: List[Section],
    shipping_limit: float = 12.0,
    min_brace_angle: float = 25.0,
    max_unique_sections: int = 3,
) -> Tuple[bool, str]:
    """
    Check if design satisfies fabrication constraints.
    
    Returns:
        (is_valid, reason): True if valid, else (False, error_message)
    """
    # Shipping limit
    max_len = max(params.span, params.height)
    if params.brace == 1:
        max_len = max(max_len, np.hypot(params.span, params.height))
    if max_len > shipping_limit:
        return False, f"Member {max_len:.2f}m > limit {shipping_limit:.1f}m"
    
    # Section variety
    unique = {params.sec_col, params.sec_beam}
    if params.brace == 1:
        unique.add(params.sec_brace)
    if len(unique) > max_unique_sections:
        return False, f"Too many sections: {len(unique)} > {max_unique_sections}"
    
    # Brace angle
    if params.brace == 1:
        angle = min(np.degrees(np.arctan(params.height / params.span)),
                   np.degrees(np.arctan(params.span / params.height)))
        if angle < min_brace_angle:
            return False, f"Brace angle {angle:.1f}° < min {min_brace_angle:.1f}°"
    
    return True, ""


def evaluate_variant(
    params: PortalParams,
    material: Material,
    sections: List[Section],
) -> Dict:
    """
    Evaluate a portal frame design: build, solve, extract metrics.
    
    Returns:
        Dict with params + metrics (drift, max_abs_M, volume, carbon, ok, reason)
    """
    from .assembly import assemble_global_K
    from .loads import assemble_element_loads_global
    from .solve import solve_linear, MechanismError
    from .post import element_end_forces_local
    from .elements import element_geometry
    
    result = {
        'span': params.span, 'height': params.height, 'brace': params.brace,
        'sec_col': params.sec_col, 'sec_beam': params.sec_beam, 'sec_brace': params.sec_brace,
        'udl_w': params.udl_w, 'wind_P': params.wind_P, 'shipping_limit': params.shipping_limit,
    }
    
    try:
        nodes, elements, fixed_dofs, element_udl_map, nodal_loads = make_portal(params, material, sections)
        
        K = assemble_global_K(nodes, elements)
        ndof = DOF_PER_NODE * len(nodes)
        F = np.zeros(ndof)
        F += assemble_element_loads_global(nodes, elements, element_udl_map)
        
        for node_id, load_vec in nodal_loads.items():
            F[dof_index(node_id, 0)] += load_vec[0]
            F[dof_index(node_id, 1)] += load_vec[1]
            F[dof_index(node_id, 2)] += load_vec[2]
        
        d, R, free = solve_linear(K, F, fixed_dofs)
        
        # Extract metrics
        drift = max(abs(d[dof_index(1, 0)]), abs(d[dof_index(2, 0)]))
        
        max_abs_M = 0.0
        for elem in elements:
            udl_w = element_udl_map.get(elem.id, None)
            f_local = element_end_forces_local(nodes, elem, d, udl_w=udl_w)
            max_abs_M = max(max_abs_M, abs(f_local[2]), abs(f_local[5]))
        
        volume = sum(elem.A * element_geometry(nodes, elem)[0] for elem in elements)
        carbon = volume * material.carbon_factor
        
        result.update({'drift': drift, 'max_abs_M': max_abs_M, 'volume': volume,
                      'carbon': carbon, 'ok': True, 'reason': ''})
        
    except MechanismError as e:
        result.update({'drift': np.nan, 'max_abs_M': np.nan, 'volume': np.nan,
                      'carbon': np.nan, 'ok': False, 'reason': f'unstable: {e}'})
    except Exception as e:
        result.update({'drift': np.nan, 'max_abs_M': np.nan, 'volume': np.nan,
                      'carbon': np.nan, 'ok': False, 'reason': f'error: {e}'})
    
    return result


def run_search(
    n: int,
    seed: int = 42,
    material: Material = None,
    sections: List[Section] = None,
) -> pd.DataFrame:
    """
    Batch search over portal frame design variants.
    
    Args:
        n: Number of variants
        seed: Random seed
        material: Material (default: DEFAULT_MATERIAL)
        sections: Section catalog (default: TIMBER_SECTIONS)
    
    Returns:
        DataFrame with params + metrics for each variant
    """
    from .catalog import DEFAULT_MATERIAL, TIMBER_SECTIONS
    
    if material is None:
        material = DEFAULT_MATERIAL
    if sections is None:
        sections = TIMBER_SECTIONS
    
    rng = np.random.default_rng(seed)
    
    print(f"Generating {n} design variants...")
    variants = sample_params(rng, sections, n)
    print(f"Generated {len(variants)} valid variants")
    
    print(f"Evaluating {len(variants)} variants...")
    results = []
    for i, params in enumerate(variants):
        if (i + 1) % 50 == 0 or (i + 1) == len(variants):
            print(f"  Progress: {i + 1}/{len(variants)}")
        results.append(evaluate_variant(params, material, sections))
    
    n_ok = sum(r['ok'] for r in results)
    print(f"Complete: {n_ok} successful, {len(results) - n_ok} failed")
    
    return pd.DataFrame(results)
