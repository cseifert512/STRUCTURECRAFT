# app/services/exploration_service.py
"""
Exploration service: handles batch design generation and Pareto analysis.

This service wraps the explore3d module to provide batch exploration
functionality for the Streamlit interface, with parameter variation
centered around a base design.
"""

import sys
from pathlib import Path
from typing import Dict, Tuple, Any, List, Optional
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mini_branch.generative import generate_canopy, CanopyParams
from mini_branch.generative.canopy import compute_member_lengths, compute_length_bins
from mini_branch.v3d.elements import truss3d_global_stiffness, truss3d_axial_force, element_geometry_3d
from mini_branch.kernel.dof import DOFManager
from mini_branch.kernel.assemble import assemble_global_K
from mini_branch.kernel.solve import solve_linear, MechanismError


class ExplorationService:
    """Service for batch exploration of canopy design variants."""
    
    @staticmethod
    def sample_variants_around_base(
        base_params: Dict[str, Any],
        n_designs: int,
        seed: int = 42,
        variation_pct: float = 0.3,
    ) -> List[CanopyParams]:
        """
        Generate design variants by varying parameters around a base design.
        
        Parameters:
        -----------
        base_params : Dict
            Base design parameters (from current sidebar state)
        n_designs : int
            Number of variants to generate
        seed : int
            Random seed for reproducibility
        variation_pct : float
            Percentage variation for continuous parameters (0.3 = +/- 30%)
        
        Returns:
        --------
        List[CanopyParams]
            List of design variants
        """
        rng = np.random.default_rng(seed)
        variants = []
        
        # Define parameter ranges based on base + variation
        base_width = base_params.get('width', 10.0)
        base_depth = base_params.get('depth', 8.0)
        base_nx = base_params.get('nx', 5)
        base_ny = base_params.get('ny', 4)
        base_min_h = base_params.get('min_height', 2.5)
        base_max_h = base_params.get('max_height', 4.0)
        base_A = base_params.get('A', 0.0008)
        base_gravity = base_params.get('gravity_load', -50000.0)
        
        # All categorical options
        heightfields = ['flat', 'paraboloid', 'ridge', 'saddle']
        topologies = ['grid', 'diagrid', 'triangulated']
        support_layouts = ['edges', 'corners', 'perimeter_4']
        
        for _ in range(n_designs):
            # Vary continuous parameters by +/- variation_pct
            width = rng.uniform(
                base_width * (1 - variation_pct),
                base_width * (1 + variation_pct)
            )
            width = max(4.0, min(20.0, width))  # Clamp to valid range
            
            depth = rng.uniform(
                base_depth * (1 - variation_pct),
                base_depth * (1 + variation_pct)
            )
            depth = max(4.0, min(16.0, depth))
            
            # Vary grid dimensions by +/- 2
            nx = rng.integers(max(2, base_nx - 2), min(10, base_nx + 2) + 1)
            ny = rng.integers(max(2, base_ny - 2), min(10, base_ny + 2) + 1)
            
            # Heights
            max_height = rng.uniform(
                base_max_h * (1 - variation_pct * 0.5),
                base_max_h * (1 + variation_pct * 0.5)
            )
            max_height = max(1.5, min(6.0, max_height))
            
            min_height = rng.uniform(
                base_min_h * (1 - variation_pct * 0.5),
                min(max_height - 0.25, base_min_h * (1 + variation_pct * 0.5))
            )
            min_height = max(1.5, min(max_height - 0.25, min_height))
            
            # Section area
            A = rng.uniform(
                base_A * (1 - variation_pct),
                base_A * (1 + variation_pct)
            )
            A = max(0.0003, min(0.002, A))
            
            # Gravity load
            gravity_load = rng.uniform(
                base_gravity * (1 + variation_pct),  # More negative
                base_gravity * (1 - variation_pct)   # Less negative
            )
            gravity_load = max(-150000, min(-20000, gravity_load))
            
            # Sample categorical parameters (include all options)
            heightfield = rng.choice(heightfields)
            topology = rng.choice(topologies)
            # Prefer 'edges' for stability but include others
            support_layout = rng.choice(support_layouts, p=[0.7, 0.15, 0.15])
            
            params = CanopyParams(
                width=width,
                depth=depth,
                nx=int(nx),
                ny=int(ny),
                max_height=max_height,
                min_height=min_height,
                heightfield=heightfield,
                topology=topology,
                support_layout=support_layout,
                E=210e9,
                A=A,
                gravity_load=gravity_load,
            )
            
            variants.append(params)
        
        return variants
    
    @staticmethod
    def evaluate_single(params: CanopyParams) -> Dict[str, Any]:
        """
        Evaluate a single canopy design variant.
        
        Returns a dict with all parameters and metrics.
        """
        result = {
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
        }
        
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
            max_displacement = 0.0
            max_vertical_disp = 0.0
            for node_id in nodes:
                ux = d[dof.idx(node_id, 0)]
                uy = d[dof.idx(node_id, 1)]
                uz = d[dof.idx(node_id, 2)]
                total = np.sqrt(ux**2 + uy**2 + uz**2)
                max_displacement = max(max_displacement, total)
                max_vertical_disp = max(max_vertical_disp, abs(uz))
            
            # Extract forces
            max_tension = 0.0
            max_compression = 0.0
            
            for bar in bars:
                N = truss3d_axial_force(nodes, bar, d)
                if N > max_tension:
                    max_tension = N
                if N < max_compression:
                    max_compression = N
            
            # Compute lengths and volume
            lengths = []
            total_length = 0.0
            volume = 0.0
            
            for bar in bars:
                L, _, _, _ = element_geometry_3d(nodes, bar)
                lengths.append((bar.id, L))
                total_length += L
                volume += bar.A * L
            
            # Length bins
            bins = compute_length_bins(lengths, tolerance=0.010)
            
            result.update({
                'ok': True,
                'reason': '',
                'max_displacement': max_displacement,
                'max_vertical_disp': max_vertical_disp,
                'max_tension': max_tension,
                'max_compression': abs(max_compression),
                'total_length': total_length,
                'volume': volume,
                'n_length_bins': len(bins),
                'max_member_length': max(L for _, L in lengths),
                'min_member_length': min(L for _, L in lengths),
                'n_nodes': n_nodes,
                'n_bars': n_bars,
                'n_supports': n_supports,
            })
            
        except MechanismError as e:
            result.update({
                'ok': False,
                'reason': f'unstable: {str(e)}',
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
        except Exception as e:
            result.update({
                'ok': False,
                'reason': f'error: {str(e)}',
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
        
        return result
    
    @staticmethod
    def run_exploration(
        base_params: Dict[str, Any],
        n_designs: int = 100,
        seed: int = 42,
        variation_pct: float = 0.3,
        progress_callback=None,
    ) -> pd.DataFrame:
        """
        Run batch exploration of design variants.
        
        Parameters:
        -----------
        base_params : Dict
            Base design parameters from current sidebar state
        n_designs : int
            Number of variants to generate and evaluate
        seed : int
            Random seed for reproducibility
        variation_pct : float
            Percentage variation for parameters
        progress_callback : callable, optional
            Function to call with (current, total) for progress updates
        
        Returns:
        --------
        pd.DataFrame
            Results with all parameters and metrics
        """
        # Sample variants
        variants = ExplorationService.sample_variants_around_base(
            base_params, n_designs, seed, variation_pct
        )
        
        # Evaluate each
        results = []
        for i, params in enumerate(variants):
            result = ExplorationService.evaluate_single(params)
            results.append(result)
            
            if progress_callback:
                progress_callback(i + 1, n_designs)
        
        return pd.DataFrame(results)
    
    @staticmethod
    def get_pareto_mask(
        df: pd.DataFrame,
        objectives: List[Tuple[str, str]] = None,
    ) -> pd.Series:
        """
        Compute Pareto frontier mask for multi-objective optimization.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Results dataframe
        objectives : List[Tuple[str, str]]
            List of (column_name, 'min' or 'max') tuples
            Default: minimize volume, displacement, length_bins
        
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
                values = -values
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
                
                # Check if j dominates i
                better_or_equal = obj_values[j] <= obj_values[i]
                strictly_better = obj_values[j] < obj_values[i]
                
                if better_or_equal.all() and strictly_better.any():
                    is_pareto[i] = False
                    break
        
        # Map back to original index
        result = pd.Series(False, index=df.index)
        result.loc[successful.index] = is_pareto
        
        return result
    
    @staticmethod
    def rank_designs(
        df: pd.DataFrame,
        weights: Dict[str, float] = None,
    ) -> pd.Series:
        """
        Rank designs by weighted normalized score.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Results dataframe
        weights : Dict[str, float]
            Weights for each metric (higher weight = more important)
        
        Returns:
        --------
        pd.Series
            Score for each design (lower is better)
        """
        if weights is None:
            weights = {
                'volume': 1.0,
                'max_displacement': 1.5,
                'n_length_bins': 0.5,
            }
        
        successful = df[df['ok'] == True].copy()
        
        if len(successful) == 0:
            return pd.Series(np.nan, index=df.index)
        
        scores = pd.Series(0.0, index=df.index)
        
        for col, weight in weights.items():
            if col not in successful.columns:
                continue
            
            values = successful[col].values
            valid = ~np.isnan(values)
            if not valid.any():
                continue
            
            vmin = values[valid].min()
            vmax = values[valid].max()
            
            if vmax > vmin:
                normalized = (values - vmin) / (vmax - vmin)
            else:
                normalized = np.zeros_like(values)
            
            scores.loc[successful.index] += weight * normalized
        
        scores.loc[df['ok'] == False] = np.inf
        
        return scores




