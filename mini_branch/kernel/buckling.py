# mini_branch/kernel/buckling.py
"""Buckling analysis: geometric stiffness and eigenvalue problem."""

import numpy as np
import scipy.linalg
from typing import Dict, List, Tuple


def geometric_stiffness_truss3d(
    L: float,
    l: float, m: float, n: float,
    axial_force: float
) -> np.ndarray:
    """
    Build 6x6 geometric stiffness matrix for a 3D truss element.
    
    The geometric stiffness accounts for the P-delta effect where axial
    force affects transverse stiffness.
    
    Args:
        L: Element length
        l, m, n: Direction cosines (x, y, z)
        axial_force: Axial force N (positive=tension, negative=compression)
    
    Returns:
        6x6 geometric stiffness matrix Kg
    """
    N = axial_force
    coeff = N / L
    
    # For truss elements, Kg captures lateral stiffness reduction under compression
    # Kg = (N/L) * [G, -G; -G, G] where G = I - nn^T (perpendicular projection)
    # This is the consistent geometric stiffness for a 3D bar element
    
    # Identity minus outer product of direction vector
    dir_vec = np.array([l, m, n])
    G = np.eye(3) - np.outer(dir_vec, dir_vec)
    
    Kg = np.zeros((6, 6), dtype=float)
    Kg[0:3, 0:3] = G
    Kg[0:3, 3:6] = -G
    Kg[3:6, 0:3] = -G
    Kg[3:6, 3:6] = G
    Kg *= coeff
    
    return Kg


def build_global_Kg(
    nodes: dict,
    bars: list,
    forces: Dict[int, float],
    dof_manager,
    element_geometry_func
) -> np.ndarray:
    """
    Assemble global geometric stiffness matrix.
    
    Args:
        nodes: Node dictionary {id: Node3D}
        bars: List of bar elements
        forces: {bar_id: axial_force} from linear analysis
        dof_manager: DOFManager instance
        element_geometry_func: Function to get (L, l, m, n) from nodes and element
    
    Returns:
        Global geometric stiffness matrix Kg
    """
    ndof = dof_manager.ndof(len(nodes))
    Kg = np.zeros((ndof, ndof), dtype=float)
    
    for bar in bars:
        N = forces.get(bar.id, 0.0)
        if abs(N) < 1e-10:
            continue  # No geometric stiffness for zero-force members
        
        L, l, m, n = element_geometry_func(nodes, bar)
        kg = geometric_stiffness_truss3d(L, l, m, n, N)
        
        dof_map = dof_manager.element_dof_map([bar.ni, bar.nj])
        
        for a in range(6):
            for b in range(6):
                Kg[dof_map[a], dof_map[b]] += kg[a, b]
    
    return Kg


def critical_buckling_factor(
    K: np.ndarray,
    Kg: np.ndarray,
    fixed_dofs: List[int]
) -> float:
    """
    Solve eigenvalue buckling problem: (K - λ·Kg)·φ = 0
    
    The critical buckling factor λ_cr is the smallest positive eigenvalue.
    If λ_cr > 1.0, the structure is stable under current loads.
    If λ_cr < 1.0, the structure will buckle before reaching full load.
    
    Args:
        K: Global elastic stiffness matrix
        Kg: Global geometric stiffness matrix
        fixed_dofs: List of constrained DOF indices
    
    Returns:
        Critical buckling factor (λ_cr). Returns inf if no buckling modes found.
    """
    ndof = K.shape[0]
    
    # Partition to free DOFs
    fixed_set = set(fixed_dofs)
    free = np.array([i for i in range(ndof) if i not in fixed_set], dtype=int)
    
    if len(free) == 0:
        return float('inf')
    
    Kff = K[np.ix_(free, free)]
    Kgff = Kg[np.ix_(free, free)]
    
    # Check if Kg has any non-zero entries
    if np.allclose(Kgff, 0):
        return float('inf')  # No geometric effects, no buckling
    
    try:
        # Solve generalized eigenvalue problem: K·φ = λ·Kg·φ
        # Note: scipy.linalg.eig returns eigenvalues λ where K·φ = λ·Kg·φ
        eigenvalues, _ = scipy.linalg.eig(Kff, Kgff)
        
        # Filter to positive real eigenvalues
        real_positive = []
        for ev in eigenvalues:
            if np.isreal(ev) or abs(ev.imag) < 1e-10:
                val = ev.real
                if val > 1e-6:  # Positive and meaningful
                    real_positive.append(val)
        
        if not real_positive:
            return float('inf')
        
        return min(real_positive)
        
    except (np.linalg.LinAlgError, ValueError):
        return float('inf')


def member_slenderness(L: float, A: float, I: float) -> float:
    """
    Compute member slenderness ratio λ = L/r where r = sqrt(I/A).
    
    Args:
        L: Member length
        A: Cross-sectional area
        I: Moment of inertia
    
    Returns:
        Slenderness ratio
    """
    if A <= 0 or I <= 0:
        return float('inf')
    r = np.sqrt(I / A)  # Radius of gyration
    return L / r


def euler_buckling_load(E: float, I: float, L: float, k: float = 1.0) -> float:
    """
    Euler critical buckling load for a column.
    
    P_cr = π²EI / (kL)²
    
    Args:
        E: Young's modulus
        I: Moment of inertia
        L: Member length
        k: Effective length factor (1.0 for pinned-pinned)
    
    Returns:
        Critical buckling load P_cr
    """
    Le = k * L  # Effective length
    return (np.pi ** 2 * E * I) / (Le ** 2)


def euler_buckling_stress(E: float, slenderness: float) -> float:
    """
    Euler critical buckling stress.
    
    σ_cr = π²E / λ²
    
    Args:
        E: Young's modulus
        slenderness: Slenderness ratio L/r
    
    Returns:
        Critical buckling stress
    """
    if slenderness <= 0:
        return float('inf')
    return (np.pi ** 2 * E) / (slenderness ** 2)


def check_member_buckling(
    axial_force: float,
    E: float,
    A: float,
    I: float,
    L: float,
    k: float = 1.0
) -> Tuple[float, bool]:
    """
    Check if a compression member is buckling.
    
    Args:
        axial_force: Axial force (negative = compression)
        E: Young's modulus
        A: Cross-sectional area
        I: Moment of inertia
        L: Member length
        k: Effective length factor
    
    Returns:
        (utilization, is_safe): Buckling utilization ratio and safety flag
    """
    if axial_force >= 0:
        return 0.0, True  # Tension member, no buckling
    
    P_cr = euler_buckling_load(E, I, L, k)
    P_applied = abs(axial_force)
    
    utilization = P_applied / P_cr
    is_safe = utilization < 1.0
    
    return utilization, is_safe





