# mini_branch/checks/steel.py
"""Steel design checks per AISC 360 (American Institute of Steel Construction)."""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional


@dataclass(frozen=True)
class SteelSection:
    """Steel section properties."""
    name: str       # Section designation (W12x26, HSS6x6x1/4, etc.)
    A: float        # Cross-sectional area (m²)
    I: float        # Moment of inertia (m⁴)
    r: float        # Radius of gyration (m)
    S: float        # Section modulus (m³)
    Fy: float       # Yield stress (Pa), default 345 MPa for A992
    E: float = 200e9  # Elastic modulus (Pa)


# Common steel sections (A992 Grade 50, Fy = 345 MPa)
# Values converted from imperial to metric
STEEL_SECTIONS = {
    # Wide Flange Sections
    'W12x26': SteelSection(
        name='W12x26', A=0.00495, I=0.000077, r=0.125, S=0.000514, Fy=345e6
    ),
    'W10x49': SteelSection(
        name='W10x49', A=0.00932, I=0.000113, r=0.110, S=0.000920, Fy=345e6
    ),
    'W8x31': SteelSection(
        name='W8x31', A=0.00590, I=0.000044, r=0.087, S=0.000432, Fy=345e6
    ),
    'W6x25': SteelSection(
        name='W6x25', A=0.00474, I=0.000022, r=0.068, S=0.000279, Fy=345e6
    ),
    # HSS (Hollow Structural Sections)
    'HSS6x6x1/4': SteelSection(
        name='HSS6x6x1/4', A=0.00335, I=0.0000235, r=0.0838, S=0.000309, Fy=317e6
    ),
    'HSS4x4x1/4': SteelSection(
        name='HSS4x4x1/4', A=0.00216, I=0.0000064, r=0.0544, S=0.000126, Fy=317e6
    ),
    'HSS3x3x3/16': SteelSection(
        name='HSS3x3x3/16', A=0.00119, I=0.0000019, r=0.0401, S=0.0000504, Fy=317e6
    ),
    # Round HSS
    'HSS4.000x0.250': SteelSection(
        name='HSS4.000x0.250', A=0.00192, I=0.0000042, r=0.0469, S=0.0000828, Fy=317e6
    ),
    # Angles (single)
    'L4x4x1/4': SteelSection(
        name='L4x4x1/4', A=0.00123, I=0.0000021, r=0.0413, S=0.0000328, Fy=250e6
    ),
    # Generic small section for demonstration
    'TUBE50x3': SteelSection(
        name='TUBE50x3', A=0.000564, I=0.000000177, r=0.0177, S=0.00000708, Fy=235e6
    ),
}


def euler_critical_stress(E: float, r: float, KL: float) -> float:
    """
    Euler buckling critical stress.
    
    Fe = π²E / (KL/r)²
    
    Args:
        E: Elastic modulus (Pa)
        r: Radius of gyration (m)
        KL: Effective length (m)
    
    Returns:
        Fe: Euler buckling stress (Pa)
    """
    slenderness = KL / r
    if slenderness < 1e-6:
        return float('inf')
    return np.pi**2 * E / slenderness**2


def compression_capacity(section: SteelSection, L: float, K: float = 1.0) -> float:
    """
    AISC 360-22 Chapter E: Compression capacity.
    
    Uses the unified column formula from AISC E3.
    
    Args:
        section: Steel section properties
        L: Unbraced length (m)
        K: Effective length factor (1.0 = pinned-pinned)
    
    Returns:
        Pn: Nominal compression capacity (N)
    """
    KL = K * L
    Fe = euler_critical_stress(section.E, section.r, KL)
    Fy = section.Fy
    
    # AISC E3 - Flexural buckling
    ratio = Fy / Fe
    
    if ratio <= 2.25:
        # Inelastic buckling (short/intermediate columns)
        Fcr = (0.658 ** ratio) * Fy
    else:
        # Elastic buckling (long columns)
        Fcr = 0.877 * Fe
    
    return Fcr * section.A


def tension_capacity(section: SteelSection) -> float:
    """
    AISC 360-22 Chapter D: Tension capacity.
    
    For yielding on gross section (D2-1): Pn = Fy * Ag
    (Rupture on net section not considered for truss members)
    
    Args:
        section: Steel section properties
    
    Returns:
        Pn: Nominal tension capacity (N)
    """
    return section.Fy * section.A


def bending_capacity(section: SteelSection) -> float:
    """
    AISC 360-22 Chapter F: Bending capacity.
    
    For compact sections (plastic yielding): Mn = Mp = Fy * Z
    Simplified to use S (elastic section modulus): Mn = Fy * S
    
    Args:
        section: Steel section properties
    
    Returns:
        Mn: Nominal moment capacity (N·m)
    """
    return section.Fy * section.S


def check_steel_member(
    N: float,
    section: SteelSection,
    L: float,
    K: float = 1.0,
    M: float = 0.0,
    phi_c: float = 0.9,
    phi_t: float = 0.9,
    phi_b: float = 0.9
) -> Dict[str, Any]:
    """
    Check steel member per AISC 360.
    
    Args:
        N: Axial force (N), positive = tension, negative = compression
        section: Steel section properties
        L: Unbraced length (m)
        K: Effective length factor
        M: Bending moment (N·m), optional
        phi_c: Resistance factor for compression (default 0.9)
        phi_t: Resistance factor for tension (default 0.9)
        phi_b: Resistance factor for bending (default 0.9)
    
    Returns:
        Dictionary with:
            - axial_util: Axial utilization ratio
            - bending_util: Bending utilization ratio  
            - combined_util: Combined utilization (AISC H1)
            - status: 'PASS' or 'FAIL'
            - governing: 'tension', 'compression', 'combined'
    """
    result = {
        'axial_util': 0.0,
        'bending_util': 0.0,
        'combined_util': 0.0,
        'status': 'PASS',
        'governing': 'none',
        'Pn': 0.0,
        'Mn': 0.0,
    }
    
    # Axial capacity
    if N < 0:
        # Compression
        Pn = compression_capacity(section, L, K)
        phi_P = phi_c * Pn
        result['governing'] = 'compression'
    else:
        # Tension
        Pn = tension_capacity(section)
        phi_P = phi_t * Pn
        result['governing'] = 'tension'
    
    result['Pn'] = Pn
    result['axial_util'] = abs(N) / phi_P if phi_P > 0 else 0.0
    
    # Bending capacity (if moment is applied)
    Mn = bending_capacity(section)
    phi_M = phi_b * Mn
    result['Mn'] = Mn
    result['bending_util'] = abs(M) / phi_M if phi_M > 0 else 0.0
    
    # Combined loading (AISC H1.1)
    Pr = abs(N)
    Pc = phi_P
    Mr = abs(M)
    Mc = phi_M
    
    if Pr / Pc >= 0.2:
        # H1-1a: Pr/Pc + (8/9)(Mr/Mc) <= 1.0
        combined = Pr / Pc + (8.0 / 9.0) * (Mr / Mc)
    else:
        # H1-1b: Pr/(2*Pc) + Mr/Mc <= 1.0
        combined = Pr / (2.0 * Pc) + Mr / Mc
    
    result['combined_util'] = combined
    
    # Determine overall status
    if combined > 1.0 or result['axial_util'] > 1.0:
        result['status'] = 'FAIL'
        if M > 0 and result['bending_util'] > result['axial_util']:
            result['governing'] = 'combined'
    
    return result


def check_all_steel_members(
    nodes: dict,
    bars: list,
    d_global: np.ndarray,
    section: SteelSection,
    axial_force_func,
    K: float = 1.0,
    dof_per_node: int = 3
) -> Dict[int, Dict[str, Any]]:
    """
    Check all members in a steel structure.
    
    Args:
        nodes: Node dictionary
        bars: List of bar elements
        d_global: Global displacement vector
        section: Steel section (same for all members)
        axial_force_func: Function to compute axial force
        K: Effective length factor (same for all members)
        dof_per_node: DOFs per node
    
    Returns:
        Dictionary mapping bar ID to check results
    """
    results = {}
    
    for bar in bars:
        # Get member length
        ni = nodes[bar.ni]
        nj = nodes[bar.nj]
        dx = nj.x - ni.x
        dy = nj.y - ni.y
        dz = nj.z - ni.z
        L = np.sqrt(dx**2 + dy**2 + dz**2)
        
        # Get axial force
        N = axial_force_func(nodes, bar, d_global, dof_per_node)
        
        # Check member
        results[bar.id] = check_steel_member(N, section, L, K)
    
    return results


def get_worst_steel_member(checks: Dict[int, Dict[str, Any]]) -> Tuple[int, Dict[str, Any]]:
    """Get the member with highest combined utilization."""
    worst_id = None
    worst_util = 0.0
    worst_result = None
    
    for bar_id, result in checks.items():
        if result['combined_util'] > worst_util:
            worst_util = result['combined_util']
            worst_id = bar_id
            worst_result = result
    
    return worst_id, worst_result


def slenderness_check(section: SteelSection, L: float, K: float = 1.0) -> Tuple[float, str]:
    """
    Check slenderness ratio against AISC limits.
    
    AISC recommends KL/r <= 200 for compression members
    and L/r <= 300 for tension members.
    
    Args:
        section: Steel section
        L: Unbraced length (m)
        K: Effective length factor
    
    Returns:
        slenderness: KL/r ratio
        status: 'PASS', 'WARNING', or 'FAIL'
    """
    KL = K * L
    slenderness = KL / section.r
    
    if slenderness <= 200:
        return slenderness, 'PASS'
    elif slenderness <= 300:
        return slenderness, 'WARNING'
    else:
        return slenderness, 'FAIL'

