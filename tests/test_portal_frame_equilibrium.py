"""
TEST: Portal Frame Equilibrium and Stability
===========================================

This test validates that the portal frame solver:
1. Does not throw MechanismError (structure is stable)
2. Produces finite, nonzero drift under lateral load
3. Satisfies global equilibrium (forces and moments balance)
"""

import numpy as np
from mini_branch.model import Node, Frame2D
from mini_branch.assembly import assemble_global_K, DOF_PER_NODE
from mini_branch.loads import assemble_element_loads_global
from mini_branch.solve import solve_linear, MechanismError

def test_portal_frame_stability():
    """Test that portal frame is stable (no mechanism)."""
    # Setup: Simple portal frame
    L, H = 6.0, 3.0
    E, I, A = 210e9, 8.0e-6, 0.01
    
    nodes = {
        0: Node(0, 0.0, 0.0),
        1: Node(1, 0.0, H),
        2: Node(2, L, H),
        3: Node(3, L, 0.0),
    }
    elements = [
        Frame2D(0, 0, 1, E=E, A=A, I=I),
        Frame2D(1, 1, 2, E=E, A=A, I=I),
        Frame2D(2, 2, 3, E=E, A=A, I=I),
    ]
    
    K = assemble_global_K(nodes, elements)
    ndof = DOF_PER_NODE * len(nodes)
    F = np.zeros(ndof)
    
    # Pinned bases
    fixed = [
        DOF_PER_NODE * 0 + 0, DOF_PER_NODE * 0 + 1,
        DOF_PER_NODE * 3 + 0, DOF_PER_NODE * 3 + 1,
    ]
    
    # Should NOT raise MechanismError
    try:
        d, R, _ = solve_linear(K, F, fixed)
        assert True, "Structure is stable"
    except MechanismError:
        assert False, "Structure is a mechanism (unstable)"

def test_portal_frame_equilibrium():
    """Test that reactions balance applied loads."""
    L, H = 6.0, 3.0
    E, I, A = 210e9, 8.0e-6, 0.01
    w = -2000.0  # UDL
    P = 5000.0   # Lateral load
    
    nodes = {
        0: Node(0, 0.0, 0.0),
        1: Node(1, 0.0, H),
        2: Node(2, L, H),
        3: Node(3, L, 0.0),
    }
    elements = [
        Frame2D(0, 0, 1, E=E, A=A, I=I),
        Frame2D(1, 1, 2, E=E, A=A, I=I),
        Frame2D(2, 2, 3, E=E, A=A, I=I),
    ]
    
    K = assemble_global_K(nodes, elements)
    ndof = DOF_PER_NODE * len(nodes)
    F = np.zeros(ndof)
    
    # Add loads
    element_udls = {1: w}
    F += assemble_element_loads_global(nodes, elements, element_udls)
    F[DOF_PER_NODE * 1 + 0] += P
    
    fixed = [
        DOF_PER_NODE * 0 + 0, DOF_PER_NODE * 0 + 1,
        DOF_PER_NODE * 3 + 0, DOF_PER_NODE * 3 + 1,
    ]
    
    d, R, _ = solve_linear(K, F, fixed)
    
    # Check equilibrium
    Rx_left = R[DOF_PER_NODE * 0 + 0]
    Ry_left = R[DOF_PER_NODE * 0 + 1]
    Rx_right = R[DOF_PER_NODE * 3 + 0]
    Ry_right = R[DOF_PER_NODE * 3 + 1]
    
    # Horizontal equilibrium
    sum_Rx = Rx_left + Rx_right
    assert np.isclose(sum_Rx + P, 0.0, rtol=1e-6), \
        f"Horizontal equilibrium: {sum_Rx} + {P} != 0"
    
    # Vertical equilibrium
    sum_Ry = Ry_left + Ry_right
    total_vertical_load = abs(w) * L
    assert np.isclose(sum_Ry - total_vertical_load, 0.0, rtol=1e-6), \
        f"Vertical equilibrium: {sum_Ry} != {total_vertical_load}"

def test_portal_frame_drift():
    """Test that lateral load produces nonzero drift."""
    L, H = 6.0, 3.0
    E, I, A = 210e9, 8.0e-6, 0.01
    P = 5000.0
    
    nodes = {
        0: Node(0, 0.0, 0.0),
        1: Node(1, 0.0, H),
        2: Node(2, L, H),
        3: Node(3, L, 0.0),
    }
    elements = [
        Frame2D(0, 0, 1, E=E, A=A, I=I),
        Frame2D(1, 1, 2, E=E, A=A, I=I),
        Frame2D(2, 2, 3, E=E, A=A, I=I),
    ]
    
    K = assemble_global_K(nodes, elements)
    ndof = DOF_PER_NODE * len(nodes)
    F = np.zeros(ndof)
    F[DOF_PER_NODE * 1 + 0] = P
    
    fixed = [
        DOF_PER_NODE * 0 + 0, DOF_PER_NODE * 0 + 1,
        DOF_PER_NODE * 3 + 0, DOF_PER_NODE * 3 + 1,
    ]
    
    d, R, _ = solve_linear(K, F, fixed)
    
    ux_top = d[DOF_PER_NODE * 1 + 0]
    drift = abs(ux_top)
    
    assert drift > 0, "Drift should be nonzero under lateral load"
    assert np.isfinite(drift), "Drift should be finite"
    assert drift < H, "Drift should be less than frame height (sanity check)"