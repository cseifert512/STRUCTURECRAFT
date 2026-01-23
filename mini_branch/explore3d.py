# mini_branch/explore3d.py
"""
EXPLORE3D: Batch Design Exploration for 3D Spaceframes
=======================================================

PURPOSE:
--------
Generate and evaluate many canopy design variants, then identify
the Pareto-optimal designs (best trade-offs between objectives).

This is where we go from "analyze one structure" to "explore the design space."

WORKFLOW:
---------
1. Sample random design parameters within constraints
2. Generate each canopy structure
3. Solve and extract metrics (displacement, volume, forces)
4. Compute fabrication metrics (length bins, unique parts)
5. Filter invalid designs (unstable, exceeds limits)
6. Identify Pareto frontier
7. Export results for analysis

OBJECTIVES (what we optimize):
------------------------------
- MINIMIZE: Material volume (cost proxy)
- MINIMIZE: Max displacement (stiffness)
- MINIMIZE: Unique length bins (fabrication simplicity)
- MAXIMIZE: Headroom/openness (architectural quality)

These objectives often conflict: stiffer = more material, simpler = less optimal.
The Pareto frontier shows the best possible trade-offs.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

from .generative import generate_canopy, CanopyParams
from .generative.canopy import compute_member_lengths, compute_length_bins
from .v3d.elements import truss3d_global_stiffness, truss3d_axial_force, element_geometry_3d
from .kernel.dof import DOFManager
from .kernel.assemble import assemble_global_K
from .kernel.solve import solve_linear, MechanismError


@dataclass
class CanopyMetrics:
    """
    Metrics extracted from a solved canopy structure.
    
    Structural:
    -----------
    max_displacement : float
        Maximum nodal displacement magnitude (m)
    max_vertical_disp : float
        Maximum vertical displacement (m)
    max_tension : float
        Maximum tensile force in any member (N)
    max_compression : float
        Maximum compressive force (absolute value) (N)
    
    Material:
    ---------
    total_length : float
        Sum of all member lengths (m)
    volume : float
        Total material volume = sum(A * L) (m^3)
    
    Fabrication:
    ------------
    n_length_bins : int
        Number of unique length categories (fewer = easier)
    max_member_length : float
        Longest member (shipping constraint)
    min_member_length : float
        Shortest member
    
    Geometry:
    ---------
    n_nodes : int
        Number of nodes
    n_bars : int
        Number of bars
    n_supports : int
        Number of supported nodes
    """
    # Structural
    max_displacement: float
    max_vertical_disp: float
    max_tension: float
    max_compression: float
    
    # Material
    total_length: float
    volume: float
    
    # Fabrication
    n_length_bins: int
    max_member_length: float
    min_member_length: float
    
    # Geometry
    n_nodes: int
    n_bars: int
    n_supports: int


def evaluate_canopy(params: CanopyParams) -> Tuple[bool, Optional[CanopyMetrics], str]:
    """
    Evaluate a single canopy design.
    
    Parameters:
    -----------
    params : CanopyParams
        Design parameters
    
    Returns:
    --------
    success : bool
        True if analysis completed successfully
    metrics : Optional[CanopyMetrics]
        Extracted metrics (None if failed)
    reason : str
        Empty if success, error message if failed
    """
    try:
        # Generate structure
        nodes, bars, fixed_dofs, F = generate_canopy(params)
        
        n_nodes = len(nodes)
        n_bars = len(bars)
        n_supports = len(fixed_dofs) // 3
        
        # Assemble
        dof = DOFManager(dof_per_node=3)
        ndof = dof.ndof(n_nodes)
        
        contributions = []
        for bar in bars:
            dof_map = dof.element_dof_map([bar.ni, bar.nj])
            ke = truss3d_global_stiffness(nodes, bar)
            contributions.append((dof_map, ke))
        
        K = assemble_global_K(ndof, contributions)
        
        # Solve
        d, R, free = solve_linear(K, F, fixed_dofs)
        
        # Extract displacements
        displacements = []
        vertical_disps = []
        for node_id in nodes:
            ux = d[dof.idx(node_id, 0)]
            uy = d[dof.idx(node_id, 1)]
            uz = d[dof.idx(node_id, 2)]
            total = np.sqrt(ux**2 + uy**2 + uz**2)
            displacements.append(total)
            vertical_disps.append(abs(uz))
        
        max_displacement = max(displacements)
        max_vertical_disp = max(vertical_disps)
        
        # Extract member forces
        max_tension = 0.0
        max_compression = 0.0
        
        for bar in bars:
            N = truss3d_axial_force(nodes, bar, d)
            if N > max_tension:
                max_tension = N
            if N < max_compression:
                max_compression = N
        
        max_compression = abs(max_compression)
        
        # Compute lengths and volume
        lengths = []
        total_length = 0.0
        volume = 0.0
        
        for bar in bars:
            L, _, _, _ = element_geometry_3d(nodes, bar)
            lengths.append(L)
            total_length += L
            volume += bar.A * L
        
        # Length bins for fabrication
        lengths_list = [(bar.id, L) for bar, L in zip(bars, lengths)]
        bins = compute_length_bins(lengths_list, tolerance=0.010)  # 10mm
        
        metrics = CanopyMetrics(
            max_displacement=max_displacement,
            max_vertical_disp=max_vertical_disp,
            max_tension=max_tension,
            max_compression=max_compression,
            total_length=total_length,
            volume=volume,
            n_length_bins=len(bins),
            max_member_length=max(lengths),
            min_member_length=min(lengths),
            n_nodes=n_nodes,
            n_bars=n_bars,
            n_supports=n_supports,
        )
        
        return True, metrics, ""
        
    except MechanismError as e:
        return False, None, f"unstable: {str(e)}"
    except Exception as e:
        return False, None, f"error: {str(e)}"


def sample_canopy_params(
    rng: np.random.Generator,
    n: int,
    width_range: Tuple[float, float] = (6.0, 15.0),
    depth_range: Tuple[float, float] = (5.0, 12.0),
    nx_range: Tuple[int, int] = (3, 8),
    ny_range: Tuple[int, int] = (3, 7),
    height_range: Tuple[float, float] = (2.0, 5.0),
    heightfields: List[str] = ['flat', 'paraboloid', 'ridge'],
    topologies: List[str] = ['grid', 'diagrid', 'triangulated'],
    support_layouts: List[str] = ['edges'],  # corners often unstable
    A_range: Tuple[float, float] = (0.0005, 0.002),
    gravity_range: Tuple[float, float] = (-100000, -30000),
) -> List[CanopyParams]:
    """
    Sample random canopy design variants.
    
    Returns a list of CanopyParams with randomly sampled values.
    """
    variants = []
    
    for _ in range(n):
        # Sample geometry
        width = rng.uniform(width_range[0], width_range[1])
        depth = rng.uniform(depth_range[0], depth_range[1])
        nx = rng.integers(nx_range[0], nx_range[1] + 1)
        ny = rng.integers(ny_range[0], ny_range[1] + 1)
        
        # Sample heights
        max_height = rng.uniform(height_range[0], height_range[1])
        min_height = rng.uniform(height_range[0] * 0.5, max_height * 0.9)
        
        # Sample categorical
        heightfield = rng.choice(heightfields)
        topology = rng.choice(topologies)
        support_layout = rng.choice(support_layouts)
        
        # Sample section
        A = rng.uniform(A_range[0], A_range[1])
        
        # Sample load
        gravity_load = rng.uniform(gravity_range[0], gravity_range[1])
        
        params = CanopyParams(
            width=width,
            depth=depth,
            nx=nx,
            ny=ny,
            max_height=max_height,
            min_height=min_height,
            heightfield=heightfield,
            topology=topology,
            support_layout=support_layout,
            E=210e9,  # Fixed: steel
            A=A,
            gravity_load=gravity_load,
        )
        
        variants.append(params)
    
    return variants


def run_batch_exploration(
    n: int = 100,
    seed: int = 42,
    show_progress: bool = True,
    **sample_kwargs
) -> pd.DataFrame:
    """
    Run batch exploration of canopy designs.
    
    Generates n random variants, evaluates each, and returns results as DataFrame.
    
    Parameters:
    -----------
    n : int
        Number of variants to generate and evaluate
    seed : int
        Random seed for reproducibility
    show_progress : bool
        Whether to show progress bar
    **sample_kwargs
        Additional arguments passed to sample_canopy_params()
    
    Returns:
    --------
    pd.DataFrame
        Results with columns for all parameters and metrics
        Includes 'ok' (success) and 'reason' (failure message)
    """
    rng = np.random.default_rng(seed)
    
    # Sample variants
    variants = sample_canopy_params(rng, n, **sample_kwargs)
    
    # Evaluate each
    results = []
    iterator = tqdm(variants, desc="Evaluating") if show_progress else variants
    
    for params in iterator:
        success, metrics, reason = evaluate_canopy(params)
        
        # Build result row
        row = {
            # Parameters
            'width': params.width,
            'depth': params.depth,
            'nx': params.nx,
            'ny': params.ny,
            'max_height': params.max_height,
            'min_height': params.min_height,
            'heightfield': params.heightfield,
            'topology': params.topology,
            'support_layout': params.support_layout,
            'A': params.A,
            'gravity_load': params.gravity_load,
            
            # Status
            'ok': success,
            'reason': reason,
        }
        
        # Add metrics if successful
        if success and metrics:
            row.update(asdict(metrics))
        else:
            # Fill with NaN for failed designs
            row.update({
                'max_displacement': np.nan,
                'max_vertical_disp': np.nan,
                'max_tension': np.nan,
                'max_compression': np.nan,
                'total_length': np.nan,
                'volume': np.nan,
                'n_length_bins': np.nan,
                'max_member_length': np.nan,
                'min_member_length': np.nan,
                'n_nodes': np.nan,
                'n_bars': np.nan,
                'n_supports': np.nan,
            })
        
        results.append(row)
    
    return pd.DataFrame(results)


def pareto_mask(
    df: pd.DataFrame,
    objectives: List[Tuple[str, str]] = None,
) -> pd.Series:
    """
    Compute Pareto frontier mask for multi-objective optimization.
    
    A design is Pareto-optimal if no other design is better in ALL objectives.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Results dataframe (must have 'ok' column and objective columns)
    objectives : List[Tuple[str, str]]
        List of (column_name, 'min' or 'max') tuples
        Default: minimize volume, minimize displacement, minimize length_bins
    
    Returns:
    --------
    pd.Series
        Boolean mask: True for Pareto-optimal designs
    """
    if objectives is None:
        objectives = [
            ('volume', 'min'),
            ('max_displacement', 'min'),
            ('n_length_bins', 'min'),
        ]
    
    # Filter to successful designs only
    successful = df[df['ok'] == True].copy()
    
    if len(successful) == 0:
        return pd.Series(False, index=df.index)
    
    # Extract objective values (negate for max objectives)
    obj_values = np.zeros((len(successful), len(objectives)))
    
    for j, (col, direction) in enumerate(objectives):
        values = successful[col].values
        if direction == 'max':
            values = -values  # Negate so we always minimize
        obj_values[:, j] = values
    
    # Find Pareto frontier
    n = len(successful)
    is_pareto = np.ones(n, dtype=bool)
    
    for i in range(n):
        if not is_pareto[i]:
            continue
        
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            
            # Check if j dominates i (j is better or equal in all objectives, strictly better in at least one)
            better_or_equal = obj_values[j] <= obj_values[i]
            strictly_better = obj_values[j] < obj_values[i]
            
            if better_or_equal.all() and strictly_better.any():
                is_pareto[i] = False
                break
    
    # Map back to original index
    result = pd.Series(False, index=df.index)
    result.loc[successful.index] = is_pareto
    
    return result


def rank_designs(
    df: pd.DataFrame,
    weights: Dict[str, float] = None,
) -> pd.Series:
    """
    Rank designs by weighted score.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Results dataframe
    weights : Dict[str, float]
        Weights for each objective (positive = minimize, negative = maximize)
        Default: equal weight to volume, displacement, length_bins
    
    Returns:
    --------
    pd.Series
        Score for each design (lower is better)
    """
    if weights is None:
        weights = {
            'volume': 1.0,
            'max_displacement': 1.0,
            'n_length_bins': 0.5,
        }
    
    successful = df[df['ok'] == True].copy()
    
    if len(successful) == 0:
        return pd.Series(np.nan, index=df.index)
    
    # Normalize each metric to [0, 1] range
    scores = pd.Series(0.0, index=df.index)
    
    for col, weight in weights.items():
        if col not in successful.columns:
            continue
        
        values = successful[col].values
        
        # Handle NaN
        valid = ~np.isnan(values)
        if not valid.any():
            continue
        
        # Normalize
        vmin = values[valid].min()
        vmax = values[valid].max()
        
        if vmax > vmin:
            normalized = (values - vmin) / (vmax - vmin)
        else:
            normalized = np.zeros_like(values)
        
        # Add weighted score
        scores.loc[successful.index] += weight * normalized
    
    # Set failed designs to infinity
    scores.loc[df['ok'] == False] = np.inf
    
    return scores

