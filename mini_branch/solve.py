# apply BCs, solve, reactions, mechanism detection

import numpy as np

class MechanismError(RuntimeError):
    pass

def solve_linear(K: np.ndarray, F: np.ndarray, fixed_dofs: list[int], cond_limit: float = 1e12):
    """
    Solve K d = F with fixed DOFs enforced by partitioning.
    Returns: d, reactions R (full vector), free_dofs
    """
    ndof = K.shape[0]
    fixed = np.array(sorted(set(fixed_dofs)), dtype=int)
    all_dofs = np.arange(ndof, dtype=int)
    free = np.array([i for i in all_dofs if i not in set(fixed)], dtype=int)

    Kff = K[np.ix_(free, free)]
    Ff = F[free]

    # Mechanism / ill-conditioning check (cheap and effective)
    cond = np.linalg.cond(Kff)
    if not np.isfinite(cond) or cond > cond_limit:
        raise MechanismError(f"Unstable or ill-conditioned system (cond={cond:.2e}). Check supports/bracing.")

    df = np.linalg.solve(Kff, Ff)

    d = np.zeros(ndof, dtype=float)
    d[free] = df
    # reactions at all DOFs
    R = K @ d - F
    return d, R, free
