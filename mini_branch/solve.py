# apply BCs, solve, reactions, mechanism detection
"""
BACKWARD COMPATIBILITY MODULE
=============================

This module re-exports the solver from the kernel.
Existing code that does `from mini_branch.solve import solve_linear`
will continue to work unchanged.

For new code, prefer importing directly from the kernel:
    from mini_branch.kernel import solve_linear, MechanismError
"""

# Re-export from kernel for backward compatibility
from .kernel.solve import solve_linear, MechanismError

__all__ = ['solve_linear', 'MechanismError']
