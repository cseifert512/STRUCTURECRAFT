# mini_branch/kernel/solve.py
"""Linear system solver with boundary conditions, mechanism detection, and P-Delta analysis."""

import numpy as np
from typing import Callable, Dict, List, Any


class MechanismError(RuntimeError):
    """Raised when structure is unstable or ill-conditioned."""
    pass


class ConvergenceError(RuntimeError):
    """Raised when iterative solution does not converge."""
    pass


def solve_linear(
    K: np.ndarray, 
    F: np.ndarray, 
    fixed_dofs: list[int], 
    cond_limit: float = 1e12
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve K·d = F with fixed boundary conditions via partitioning.
    
    Args:
        K: Global stiffness matrix (ndof x ndof)
        F: Global load vector (ndof,)
        fixed_dofs: List of constrained DOF indices (displacement = 0)
        cond_limit: Max condition number before raising MechanismError
    
    Returns:
        d: Displacement vector (ndof,)
        R: Reaction vector (ndof,)
        free: Array of free DOF indices
    
    Raises:
        MechanismError: If structure is unstable (cond > cond_limit)
    """
    ndof = K.shape[0]
    
    # Partition DOFs
    fixed = np.array(sorted(set(fixed_dofs)), dtype=int)
    free = np.array([i for i in range(ndof) if i not in set(fixed)], dtype=int)
    
    # Extract reduced system
    Kff = K[np.ix_(free, free)]
    Ff = F[free]
    
    # Check conditioning
    cond = np.linalg.cond(Kff)
    if not np.isfinite(cond) or cond > cond_limit:
        raise MechanismError(
            f"Unstable system (cond={cond:.2e}). Check supports. Need cond < {cond_limit:.0e}."
        )
    
    # Solve reduced system
    df = np.linalg.solve(Kff, Ff)
    
    # Assemble full displacement
    d = np.zeros(ndof, dtype=float)
    d[free] = df
    
    # Compute reactions: R = K·d - F
    R = K @ d - F
    
    return d, R, free


def solve_pdelta(
    K: np.ndarray,
    F: np.ndarray,
    fixed_dofs: List[int],
    nodes: dict,
    bars: list,
    dof_manager: Any,
    element_geometry_func: Callable,
    axial_force_func: Callable,
    build_kg_func: Callable,
    max_iter: int = 10,
    tol: float = 1e-4,
    cond_limit: float = 1e14
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """
    Iterative P-Delta (second-order) analysis.
    
    Captures geometric nonlinearity where axial forces affect lateral stiffness.
    Uses iterative approach: solve, compute Kg, solve with K+Kg, repeat until converged.
    
    Args:
        K: Global elastic stiffness matrix
        F: Global load vector
        fixed_dofs: Constrained DOF indices
        nodes: Node dictionary
        bars: List of bar elements
        dof_manager: DOFManager instance
        element_geometry_func: Function(nodes, bar) -> (L, l, m, n)
        axial_force_func: Function(nodes, bar, d, dof_per_node) -> N
        build_kg_func: Function(nodes, bars, forces, dof, geom_func) -> Kg
        max_iter: Maximum iterations (default 10)
        tol: Convergence tolerance (default 1e-4)
        cond_limit: Condition number limit
    
    Returns:
        d: Final displacement vector
        R: Reaction vector
        iterations: Number of iterations to converge
        amplification: Ratio of P-Delta to linear displacement (amplification factor)
    
    Raises:
        ConvergenceError: If solution doesn't converge within max_iter
        MechanismError: If structure becomes unstable during iteration
    """
    ndof = K.shape[0]
    dof_per_node = dof_manager.dof_per_node
    
    # Step 1: Linear solve to get initial displacement and forces
    d_linear, R_linear, free = solve_linear(K, F, fixed_dofs, cond_limit)
    
    # Compute initial axial forces
    forces = {}
    for bar in bars:
        forces[bar.id] = axial_force_func(nodes, bar, d_linear, dof_per_node)
    
    # Check if any compression - if not, P-Delta has no effect
    has_compression = any(f < 0 for f in forces.values())
    if not has_compression:
        return d_linear, R_linear, 0, 1.0
    
    # Step 2: Iterate with geometric stiffness
    d_prev = d_linear.copy()
    
    for iteration in range(max_iter):
        # Build geometric stiffness from current forces
        Kg = build_kg_func(nodes, bars, forces, dof_manager, element_geometry_func)
        
        # Effective stiffness: K + Kg (Kg is negative for compression)
        K_eff = K + Kg
        
        # Solve with effective stiffness
        try:
            d_new, R_new, _ = solve_linear(K_eff, F, fixed_dofs, cond_limit)
        except MechanismError as e:
            raise MechanismError(
                f"P-Delta diverged at iteration {iteration+1}: structure buckled. {e}"
            )
        
        # Check convergence: relative change in displacement norm
        d_change = np.linalg.norm(d_new - d_prev)
        d_norm = np.linalg.norm(d_new)
        
        if d_norm > 0 and d_change / d_norm < tol:
            # Converged - compute amplification factor
            linear_max = np.max(np.abs(d_linear))
            pdelta_max = np.max(np.abs(d_new))
            amplification = pdelta_max / linear_max if linear_max > 1e-12 else 1.0
            
            return d_new, R_new, iteration + 1, amplification
        
        # Update for next iteration
        d_prev = d_new.copy()
        
        # Recompute forces with new displacements
        for bar in bars:
            forces[bar.id] = axial_force_func(nodes, bar, d_new, dof_per_node)
    
    raise ConvergenceError(
        f"P-Delta did not converge after {max_iter} iterations. "
        f"Final change: {d_change/d_norm:.2e}, tolerance: {tol:.2e}"
    )
