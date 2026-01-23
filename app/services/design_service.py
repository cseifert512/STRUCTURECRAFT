# app/services/design_service.py
"""
Design service: handles single design generation and analysis.
"""

import sys
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mini_branch.generative import generate_canopy, CanopyParams
from mini_branch.generative.canopy import compute_member_lengths, compute_length_bins
from mini_branch.v3d.elements import truss3d_global_stiffness, truss3d_axial_force, element_geometry_3d
from mini_branch.kernel.dof import DOFManager
from mini_branch.kernel.assemble import assemble_global_K
from mini_branch.kernel.solve import solve_linear, MechanismError
from mini_branch.checks.steel import SteelSection, check_steel_member


def compute_deflection_check(max_disp: float, span: float) -> Dict[str, Any]:
    """
    Check deflection against common serviceability limits.
    
    Args:
        max_disp: Maximum displacement (m)
        span: Governing span (m) - typically min(width, depth)
    
    Returns:
        dict with limits, ratios, and pass/fail status
    """
    limits = {
        'L/360': span / 360,  # Floor live load
        'L/240': span / 240,  # Total load (typical for roofs/canopies)
        'L/180': span / 180,  # Roof with plaster ceiling
    }
    
    results = {}
    for name, limit in limits.items():
        ratio = max_disp / limit if limit > 0 else 0
        results[name] = {
            'limit_mm': limit * 1000,
            'actual_mm': max_disp * 1000,
            'ratio': ratio,
            'status': 'PASS' if ratio <= 1.0 else 'FAIL'
        }
    
    # Overall pass if meets L/240 (typical for roofs/canopies)
    governing_limit = 'L/240'
    overall_pass = results[governing_limit]['status'] == 'PASS'
    
    return {
        'checks': results,
        'governing': governing_limit,
        'overall_pass': overall_pass,
    }


def compute_utilization(nodes: Dict, bars: List, forces: Dict, A: float, E: float = 210e9) -> Dict[str, Any]:
    """
    Compute member utilization ratios using AISC-style checks.
    
    Args:
        nodes: Node dictionary {id: Node3D}
        bars: List of bar elements
        forces: Dictionary {bar_id: axial_force}
        A: Cross-sectional area (m²)
        E: Young's modulus (Pa)
    
    Returns:
        dict with worst_util, n_pass, n_fail, utilizations
    """
    # Create generic section from area
    # Approximate as solid round for radius of gyration
    r = np.sqrt(A / np.pi)  # Approximate radius of gyration for solid round
    section = SteelSection(
        name='Custom',
        A=A,
        I=A * r**2 / 4,  # Approximate I for solid round
        r=r,
        S=A * r / 2,  # Approximate S
        Fy=250e6,  # 250 MPa yield (conservative steel)
        E=E
    )
    
    utilizations = {}
    for bar in bars:
        # Get member length
        ni, nj = nodes[bar.ni], nodes[bar.nj]
        L = np.sqrt((nj.x - ni.x)**2 + (nj.y - ni.y)**2 + (nj.z - ni.z)**2)
        
        N = forces.get(bar.id, 0.0)
        result = check_steel_member(N, section, L, K=1.0)
        utilizations[bar.id] = {
            'N_kN': N / 1000,
            'util': result['axial_util'],
            'combined_util': result['combined_util'],
            'status': result['status'],
            'governing': result['governing'],
        }
    
    if not utilizations:
        return {
            'worst_bar': None,
            'worst_util': 0.0,
            'n_pass': 0,
            'n_fail': 0,
            'all_pass': True,
            'utilizations': {},
        }
    
    worst_id = max(utilizations, key=lambda k: utilizations[k]['util'])
    worst_util = utilizations[worst_id]['util']
    n_pass = sum(1 for u in utilizations.values() if u['status'] == 'PASS')
    n_fail = len(utilizations) - n_pass
    
    return {
        'worst_bar': worst_id,
        'worst_util': worst_util,
        'n_pass': n_pass,
        'n_fail': n_fail,
        'all_pass': n_fail == 0,
        'utilizations': utilizations,
    }


class DesignService:
    """Service for generating and analyzing single canopy designs."""
    
    @staticmethod
    def create_params_from_brief(brief) -> CanopyParams:
        """Convert a DesignBrief to CanopyParams."""
        return CanopyParams(
            width=brief.width,
            depth=brief.depth,
            nx=brief.nx,
            ny=brief.ny,
            max_height=brief.max_height,
            min_height=brief.min_height,
            heightfield=brief.heightfield,
            topology=brief.topology,
            support_layout=brief.support_layout,
            E=brief.E,
            A=brief.A,
            gravity_load=brief.gravity_load,
        )
    
    @staticmethod
    def generate_and_solve(params: CanopyParams) -> Tuple[bool, Dict[str, Any], str]:
        """
        Generate a canopy and solve it.
        
        Returns:
            success: bool
            result: dict with nodes, bars, displacements, forces, metrics
            error: str (empty if success)
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
            forces = {}
            max_tension = 0.0
            max_compression = 0.0
            
            for bar in bars:
                N = truss3d_axial_force(nodes, bar, d)
                forces[bar.id] = N
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
            
            # Get support node IDs
            support_nodes = []
            for i in range(n_nodes):
                if dof.idx(i, 0) in fixed_dofs:
                    support_nodes.append(i)
            
            result = {
                'nodes': nodes,
                'bars': bars,
                'displacements': d,
                'forces': forces,
                'reactions': R,
                'support_nodes': support_nodes,
                'metrics': {
                    'n_nodes': n_nodes,
                    'n_bars': n_bars,
                    'n_supports': n_supports,
                    'max_displacement': max_displacement,
                    'max_vertical_disp': max_vertical_disp,
                    'max_tension': max_tension,
                    'max_compression': abs(max_compression),
                    'total_length': total_length,
                    'volume': volume,
                    'n_length_bins': len(bins),
                    'max_member_length': max(L for _, L in lengths),
                    'min_member_length': min(L for _, L in lengths),
                },
                'length_bins': bins,
            }
            
            return True, result, ""
            
        except MechanismError as e:
            return False, {}, f"Structure is unstable: {str(e)}"
        except Exception as e:
            return False, {}, f"Error: {str(e)}"
    
    @staticmethod
    def generate_preview(brief) -> Tuple[bool, Dict[str, Any], str]:
        """Generate and solve a design from a brief."""
        params = DesignService.create_params_from_brief(brief)
        return DesignService.generate_and_solve(params)

