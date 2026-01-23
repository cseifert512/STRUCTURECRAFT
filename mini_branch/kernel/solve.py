# mini_branch/kernel/solve.py
"""
SOLVER: Dimension-Agnostic Linear System Solver
===============================================

PURPOSE:
--------
This module solves the structural equation K·d = F with boundary conditions.
It's ALREADY dimension-agnostic — it just needs:
- K: stiffness matrix (any size)
- F: load vector (matching size)
- fixed_dofs: list of constrained DOF indices

The solver doesn't care if K is 12×12 (4-node 2D frame) or 300×300
(100-node 3D truss). It just does linear algebra.

KEY FEATURES:
-------------
1. Partitioned solve: separates free DOFs from fixed DOFs
2. Mechanism detection: catches ill-conditioned systems BEFORE they explode
3. Reaction recovery: computes support reactions from equilibrium

This is the exact same algorithm from your existing solve.py,
promoted to the kernel for reuse by both 2D and 3D code.
"""

import numpy as np


class MechanismError(RuntimeError):
    """
    Raised when the structure is unstable (mechanism) or ill-conditioned.
    
    This happens when:
    - Not enough supports (rigid body motion possible)
    - Supports in wrong configuration (rotation not prevented)
    - Singular stiffness (e.g., all bars in a line)
    
    The error message includes the condition number to help diagnose.
    """
    pass


def solve_linear(
    K: np.ndarray, 
    F: np.ndarray, 
    fixed_dofs: list[int], 
    cond_limit: float = 1e12
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve K·d = F with fixed DOFs enforced by partitioning.
    
    This is THE core solver for all linear structural analysis.
    Works for any DOF count — 2D frames, 3D trusses, 3D frames.
    
    ALGORITHM (Partitioned Approach):
    ---------------------------------
    1. Partition DOFs into "free" (unknown) and "fixed" (known = 0)
    2. Extract Kff = K[free, free] (the reduced stiffness matrix)
    3. Solve Kff · df = Ff for free displacements
    4. Assemble full displacement vector (free DOFs filled, fixed DOFs = 0)
    5. Compute reactions: R = K·d - F (equilibrium at fixed DOFs)
    
    MECHANISM DETECTION:
    --------------------
    Before solving, we check the condition number of Kff.
    If cond(Kff) > cond_limit, the system is nearly singular → MechanismError.
    
    This catches:
    - Unstable structures (not enough supports)
    - Poorly conditioned systems (numerical issues)
    
    Parameters:
    -----------
    K : np.ndarray
        Global stiffness matrix, shape (ndof, ndof)
        Must be symmetric positive semi-definite
        
    F : np.ndarray
        Global load vector, shape (ndof,)
        Contains all applied loads (nodal + equivalent from distributed)
        
    fixed_dofs : list[int]
        List of DOF indices that are constrained (displacement = 0)
        Example: [0, 1, 9, 10] for pinned supports at nodes 0 and 3 in 2D
        
    cond_limit : float
        Maximum allowable condition number for Kff (default: 1e12)
        Higher = more tolerant of ill-conditioning
        Lower = stricter stability check
    
    Returns:
    --------
    d : np.ndarray
        Global displacement vector, shape (ndof,)
        d[i] = displacement at DOF i (zero for fixed DOFs)
        
    R : np.ndarray
        Reaction vector, shape (ndof,)
        R[i] = reaction force at DOF i
        Non-zero only at fixed DOFs (support reactions)
        
    free : np.ndarray
        Array of free DOF indices (for reference/debugging)
    
    Raises:
    -------
    MechanismError
        If the structure is unstable or ill-conditioned
        (condition number exceeds cond_limit)
    
    Example:
    --------
    >>> K = assemble_global_K(nodes, elements)
    >>> F = np.zeros(ndof)
    >>> F[5] = -1000  # Vertical load at node 1
    >>> fixed_dofs = [0, 1, 9, 10]  # Pin supports
    >>> d, R, free = solve_linear(K, F, fixed_dofs)
    >>> print(f"Max displacement: {np.max(np.abs(d)):.6f} m")
    """
    ndof = K.shape[0]
    
    # ========================================================================
    # STEP 1: PARTITION DOFs INTO FREE AND FIXED
    # ========================================================================
    # Fixed DOFs: known displacement = 0 (boundary conditions)
    # Free DOFs: unknown displacements (what we solve for)
    
    fixed = np.array(sorted(set(fixed_dofs)), dtype=int)
    all_dofs = np.arange(ndof, dtype=int)
    free = np.array([i for i in all_dofs if i not in set(fixed)], dtype=int)
    
    # ========================================================================
    # STEP 2: EXTRACT REDUCED SYSTEM (FREE DOFs ONLY)
    # ========================================================================
    # Kff = submatrix of K for free DOFs
    # Ff = subvector of F for free DOFs
    
    Kff = K[np.ix_(free, free)]
    Ff = F[free]
    
    # ========================================================================
    # STEP 3: MECHANISM / ILL-CONDITIONING CHECK
    # ========================================================================
    # Condition number = ratio of largest to smallest singular value
    # Large condition number → nearly singular → unstable or mechanism
    
    cond = np.linalg.cond(Kff)
    if not np.isfinite(cond) or cond > cond_limit:
        raise MechanismError(
            f"Unstable or ill-conditioned system (cond={cond:.2e}). "
            f"Check supports/bracing. Need cond < {cond_limit:.0e}."
        )
    
    # ========================================================================
    # STEP 4: SOLVE REDUCED SYSTEM
    # ========================================================================
    # Kff · df = Ff → df = Kff^(-1) · Ff
    # Using numpy's optimized solver (not explicit inverse)
    
    df = np.linalg.solve(Kff, Ff)
    
    # ========================================================================
    # STEP 5: ASSEMBLE FULL DISPLACEMENT VECTOR
    # ========================================================================
    # d[free] = df (solved values)
    # d[fixed] = 0 (boundary conditions)
    
    d = np.zeros(ndof, dtype=float)
    d[free] = df
    
    # ========================================================================
    # STEP 6: COMPUTE REACTIONS
    # ========================================================================
    # From equilibrium: K·d = F + R (where R is reactions at fixed DOFs)
    # Rearranging: R = K·d - F
    # R is non-zero only at fixed DOFs (at free DOFs, K·d = F exactly)
    
    R = K @ d - F
    
    return d, R, free

