# mini_branch/checks/timber.py
"""Timber design checks per CSA O86 / NDS style codes."""

from dataclasses import dataclass
from typing import Dict, List, Any
import numpy as np


@dataclass(frozen=True)
class TimberCapacity:
    """Allowable stresses for timber member (Pa)."""
    f_c: float      # Compression parallel to grain
    f_t: float      # Tension parallel to grain
    f_b: float      # Bending
    f_v: float      # Shear
    E_min: float    # Modulus for stability calculations


# Douglas Fir - Select Structural grade (typical values)
DOUGLAS_FIR_CAPACITY = TimberCapacity(
    f_c=11.0e6,     # 11 MPa compression
    f_t=8.3e6,      # 8.3 MPa tension
    f_b=14.5e6,     # 14.5 MPa bending
    f_v=1.4e6,      # 1.4 MPa shear
    E_min=9.0e9,    # 9 GPa for stability
)

# SPF (Spruce-Pine-Fir) - No.2 grade
SPF_CAPACITY = TimberCapacity(
    f_c=7.5e6,
    f_t=5.5e6,
    f_b=9.0e6,
    f_v=1.0e6,
    E_min=7.0e9,
)


def axial_utilization(
    N: float,
    A: float,
    capacity: TimberCapacity
) -> float:
    """
    Compute axial stress utilization ratio.
    
    Args:
        N: Axial force (positive=tension, negative=compression)
        A: Cross-sectional area (m²)
        capacity: Timber capacity values
    
    Returns:
        Utilization ratio (< 1.0 = pass)
    """
    if A <= 0:
        return float('inf')
    
    stress = abs(N) / A
    allowable = capacity.f_c if N < 0 else capacity.f_t
    
    return stress / allowable


def bending_utilization(
    M: float,
    S: float,
    capacity: TimberCapacity
) -> float:
    """
    Compute bending stress utilization ratio.
    
    Args:
        M: Bending moment (N·m)
        S: Section modulus (m³)
        capacity: Timber capacity values
    
    Returns:
        Utilization ratio (< 1.0 = pass)
    """
    if S <= 0:
        return float('inf')
    
    stress = abs(M) / S
    return stress / capacity.f_b


def combined_utilization(
    N: float,
    M: float,
    A: float,
    S: float,
    capacity: TimberCapacity
) -> float:
    """
    Combined axial + bending interaction check per NDS 3.9.2.
    
    For compression: (f_a/F_c)² + f_b/F_b ≤ 1.0
    For tension: f_a/F_t + f_b/F_b ≤ 1.0
    
    Args:
        N: Axial force (positive=tension, negative=compression)
        M: Bending moment (N·m)
        A: Cross-sectional area (m²)
        S: Section modulus (m³)
        capacity: Timber capacity values
    
    Returns:
        Utilization ratio (< 1.0 = pass)
    """
    if A <= 0 or S <= 0:
        return float('inf')
    
    f_a = abs(N) / A
    f_b = abs(M) / S
    
    if N < 0:  # Compression + bending
        # Squared term for compression interaction
        axial_ratio = (f_a / capacity.f_c) ** 2
        bending_ratio = f_b / capacity.f_b
    else:  # Tension + bending
        # Linear interaction for tension
        axial_ratio = f_a / capacity.f_t
        bending_ratio = f_b / capacity.f_b
    
    return axial_ratio + bending_ratio


def check_member(
    N: float,
    A: float,
    capacity: TimberCapacity,
    M: float = 0.0,
    S: float = 0.0,
) -> Dict[str, Any]:
    """
    Run all applicable checks for a single member.
    
    Args:
        N: Axial force
        A: Area
        capacity: Timber capacities
        M: Bending moment (optional, for frame members)
        S: Section modulus (optional, for frame members)
    
    Returns:
        Dict with utilization, status, and breakdown
    """
    axial_util = axial_utilization(N, A, capacity)
    
    if M != 0.0 and S > 0:
        # Combined check for members with bending
        comb_util = combined_utilization(N, M, A, S, capacity)
        util = max(axial_util, comb_util)
        check_type = 'combined'
    else:
        # Axial-only check for truss members
        util = axial_util
        comb_util = 0.0
        check_type = 'axial'
    
    return {
        'utilization': util,
        'axial_utilization': axial_util,
        'combined_utilization': comb_util,
        'check_type': check_type,
        'status': 'PASS' if util <= 1.0 else 'FAIL',
        'axial_force_kN': N / 1000,
        'stress_type': 'compression' if N < 0 else 'tension',
    }


def check_all_members(
    nodes: dict,
    bars: list,
    d_global: np.ndarray,
    capacity: TimberCapacity,
    axial_force_func,
    dof_per_node: int = 3,
) -> Dict[int, Dict[str, Any]]:
    """
    Run design checks on all members.
    
    Args:
        nodes: Node dictionary
        bars: List of bar elements
        d_global: Global displacement vector from solver
        capacity: Timber capacity values
        axial_force_func: Function(nodes, bar, d, dof_per_node) -> N
        dof_per_node: DOFs per node
    
    Returns:
        {bar_id: check_result} for all bars
    """
    results = {}
    
    for bar in bars:
        N = axial_force_func(nodes, bar, d_global, dof_per_node)
        results[bar.id] = check_member(N, bar.A, capacity)
    
    return results


def get_worst_member(checks: Dict[int, Dict[str, Any]]) -> tuple:
    """
    Find the member with highest utilization.
    
    Returns:
        (bar_id, utilization, status)
    """
    if not checks:
        return None, 0.0, 'N/A'
    
    worst_id = max(checks.keys(), key=lambda k: checks[k]['utilization'])
    worst = checks[worst_id]
    return worst_id, worst['utilization'], worst['status']


def count_failing_members(checks: Dict[int, Dict[str, Any]]) -> int:
    """Count members with utilization > 1.0."""
    return sum(1 for c in checks.values() if c['utilization'] > 1.0)




