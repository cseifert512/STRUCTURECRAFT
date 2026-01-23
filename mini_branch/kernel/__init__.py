# mini_branch/kernel - Dimension-agnostic structural analysis core
"""Core abstractions for structural analysis: DOF management, assembly, solving, buckling, modal, connections."""

from .dof import DOFManager
from .assemble import assemble_global_K, assemble_global_F, add_nodal_load
from .solve import solve_linear, solve_pdelta, MechanismError, ConvergenceError
from .buckling import (
    geometric_stiffness_truss3d,
    build_global_Kg,
    critical_buckling_factor,
    member_slenderness,
    euler_buckling_load,
    euler_buckling_stress,
    check_member_buckling,
)
from .modal import (
    build_lumped_mass_matrix,
    build_consistent_mass_matrix,
    natural_frequencies,
    modal_participation_factors,
    effective_modal_mass,
)
from .connections import (
    ConnectionType,
    CONNECTION_STIFFNESS,
    Connection,
    fixity_factor,
    modify_frame_stiffness_for_connections,
    add_rotational_spring,
    compute_connection_moments,
    create_connection_matrix,
    default_rigid_connections,
    default_pinned_connections,
)

__all__ = [
    'DOFManager',
    'assemble_global_K', 'assemble_global_F', 'add_nodal_load',
    'solve_linear', 'solve_pdelta', 'MechanismError', 'ConvergenceError',
    'geometric_stiffness_truss3d', 'build_global_Kg', 'critical_buckling_factor',
    'member_slenderness', 'euler_buckling_load', 'euler_buckling_stress', 'check_member_buckling',
    'build_lumped_mass_matrix', 'build_consistent_mass_matrix', 'natural_frequencies',
    'modal_participation_factors', 'effective_modal_mass',
    'ConnectionType', 'CONNECTION_STIFFNESS', 'Connection', 'fixity_factor',
    'modify_frame_stiffness_for_connections', 'add_rotational_spring',
    'compute_connection_moments', 'create_connection_matrix',
    'default_rigid_connections', 'default_pinned_connections',
]
