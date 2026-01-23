# mini_branch/kernel/solve.py
"""Linear system solver with boundary conditions and mechanism detection."""

import numpy as np


class MechanismError(RuntimeError):
    """Raised when structure is unstable or ill-conditioned."""
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
