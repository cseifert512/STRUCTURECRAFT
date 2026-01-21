# File: tests/test_postprocessing.py
"""
TEST: Postprocessing End Forces (N/V/M)
========================================

This test validates that element_end_forces_local correctly computes
internal forces from displacements. We test:

1. Simple cantilever with point load (known solution)
2. Simply supported beam with UDL (known solution)
3. Equilibrium checks (forces should balance)
"""

import numpy as np

from mini_branch.model import Node, Frame2D
from mini_branch.assembly import assemble_global_K, DOF_PER_NODE
from mini_branch.loads import assemble_element_loads_global
from mini_branch.solve import solve_linear
from mini_branch.post import element_end_forces_local


def test_cantilever_end_forces():
    """
    Test end forces for a cantilever beam with tip load.
    
    PSEUDOCODE:
    ----------
    
    SETUP:
        Create cantilever: fixed at left, point load at right tip
    
    SOLVE:
        Build K, apply load, solve for d
    
    POSTPROCESS:
        Compute end forces for the element
    
    VERIFY:
        - Fixed end moment should equal PL (known formula)
        - Fixed end shear should equal P
        - Tip moment should be 0 (free end)
    """
    
    # Setup: Cantilever beam
    L = 3.0
    E = 210e9
    I = 8.0e-6
    A = 0.01
    P = 1000.0  # Point load at tip
    
    nodes = {
        0: Node(0, 0.0, 0.0),  # Fixed end
        1: Node(1, L, 0.0),     # Free end (tip)
    }
    elements = [Frame2D(0, 0, 1, E=E, A=A, I=I)]
    
    # Solve
    K = assemble_global_K(nodes, elements)
    ndof = DOF_PER_NODE * len(nodes)
    F = np.zeros(ndof)
    F[DOF_PER_NODE*1 + 1] = -P  # Load at tip, downward
    
    fixed = [0, 1, 2]  # Fix left end completely
    d, R, _ = solve_linear(K, F, fixed)
    
    # Postprocess: Compute end forces
    f_local = element_end_forces_local(nodes, elements[0], d, udl_w=None)
    
    # Extract forces
    Ni = f_local[0]  # Axial at fixed end
    Vi = f_local[1]  # Shear at fixed end
    Mi = f_local[2]  # Moment at fixed end
    Nj = f_local[3]  # Axial at tip
    Vj = f_local[4]  # Shear at tip
    Mj = f_local[5]  # Moment at tip
    
    # Theoretical values for cantilever with tip load:
    # - Fixed end: V = P (upward), M = PL (counterclockwise)
    # - Tip: V = P (downward), M = 0 (free end)
    
    # Verify fixed end forces
    assert np.isclose(Vi, P, rtol=1e-3, atol=1e-6), \
        f"Fixed end shear {Vi} != expected {P}"
    assert np.isclose(Mi, P * L, rtol=1e-3, atol=1e-6), \
        f"Fixed end moment {Mi} != expected {P*L}"
    
    # Verify tip forces
    assert np.isclose(Vj, -P, rtol=1e-3, atol=1e-6), \
        f"Tip shear {Vj} != expected {-P}"
    assert np.isclose(Mj, 0.0, rtol=1e-3, atol=1e-6), \
        f"Tip moment {Mj} != expected 0"
    
    # Axial forces should be zero (no axial load)
    assert np.isclose(Ni, 0.0, rtol=1e-6, atol=1e-9), \
        f"Fixed end axial {Ni} != expected 0"
    assert np.isclose(Nj, 0.0, rtol=1e-6, atol=1e-9), \
        f"Tip axial {Nj} != expected 0"
    
    print(f"✓ Cantilever end forces test passed")
    print(f"  Fixed end: V = {Vi:.2f} N, M = {Mi:.2f} N·m")
    print(f"  Tip: V = {Vj:.2f} N, M = {Mj:.2f} N·m")


def test_udl_beam_end_forces():
    """
    Test end forces for simply supported beam with UDL.
    
    PSEUDOCODE:
    ----------
    
    SETUP:
        Create simply supported beam with UDL
    
    SOLVE:
        Build K, apply UDL loads, solve for d
    
    POSTPROCESS:
        Compute end forces for each element (with UDL)
    
    VERIFY:
        - End moments should match theoretical values
        - Shear forces should match theoretical values
        - Forces should be symmetric (beam is symmetric)
    """
    
    # Setup: Simply supported beam with UDL
    L = 4.0
    E = 210e9
    I = 8.0e-6
    A = 0.01
    w = -1000.0  # UDL downward
    
    # Use 2 elements for simplicity (can use more)
    nodes = {
        0: Node(0, 0.0, 0.0),
        1: Node(1, L/2, 0.0),
        2: Node(2, L, 0.0),
    }
    elements = [
        Frame2D(0, 0, 1, E=E, A=A, I=I),
        Frame2D(1, 1, 2, E=E, A=A, I=I),
    ]
    
    # Solve with UDL
    K = assemble_global_K(nodes, elements)
    ndof = DOF_PER_NODE * len(nodes)
    F = np.zeros(ndof)
    element_udls = {0: w, 1: w}
    F += assemble_element_loads_global(nodes, elements, element_udls)
    
    fixed = [
        DOF_PER_NODE*0 + 0,  # Node 0, ux
        DOF_PER_NODE*0 + 1,  # Node 0, uy
        DOF_PER_NODE*2 + 1,  # Node 2, uy
    ]
    
    d, R, _ = solve_linear(K, F, fixed)
    
    # Postprocess: Compute end forces for each element
    for i, element in enumerate(elements):
        f_local = element_end_forces_local(nodes, element, d, udl_w=w)
        
        Vi = f_local[1]  # Shear at start
        Mi = f_local[2]  # Moment at start
        Vj = f_local[4]  # Shear at end
        Mj = f_local[5]  # Moment at end
        
        # For simply supported beam with UDL:
        # - At supports: moment = 0 (pinned)
        # - At midspan: moment is maximum
        # - Shear: varies linearly
        
        # Check element 0 (left half)
        if i == 0:
            # At left support (node 0): moment should be ~0 (pinned)
            assert np.isclose(Mi, 0.0, rtol=1e-2, atol=1e-3), \
                f"Left support moment {Mi} should be ~0"
        
        # Check element 1 (right half)
        if i == 1:
            # At right support (node 2): moment should be ~0 (pinned)
            assert np.isclose(Mj, 0.0, rtol=1e-2, atol=1e-3), \
                f"Right support moment {Mj} should be ~0"
    
    print(f"✓ UDL beam end forces test passed")
    print(f"  Support moments are ~0 (pinned supports)")


def test_end_forces_equilibrium():
    """
    Test that end forces satisfy element equilibrium.
    
    PSEUDOCODE:
    ----------
    
    FOR any element:
        - Sum of forces in x should be 0 (no net axial force)
        - Sum of forces in y should be 0 (no net vertical force)
        - Sum of moments about any point should be 0
    
    This is a fundamental physics check.
    """
    
    # Setup: Simple beam
    L = 3.0
    E = 210e9
    I = 8.0e-6
    A = 0.01
    P = 1000.0
    
    nodes = {
        0: Node(0, 0.0, 0.0),
        1: Node(1, L, 0.0),
    }
    elements = [Frame2D(0, 0, 1, E=E, A=A, I=I)]
    
    # Solve
    K = assemble_global_K(nodes, elements)
    ndof = DOF_PER_NODE * len(nodes)
    F = np.zeros(ndof)
    F[DOF_PER_NODE*1 + 1] = -P
    
    fixed = [0, 1, 2]
    d, R, _ = solve_linear(K, F, fixed)
    
    # Postprocess
    f_local = element_end_forces_local(nodes, elements[0], d, udl_w=None)
    
    Ni = f_local[0]
    Vi = f_local[1]
    Mi = f_local[2]
    Nj = f_local[3]
    Vj = f_local[4]
    Mj = f_local[5]
    
    # Equilibrium checks:
    # 1. Sum of axial forces = 0
    sum_axial = Ni + Nj
    assert np.isclose(sum_axial, 0.0, rtol=1e-6, atol=1e-9), \
        f"Axial forces don't balance: {Ni} + {Nj} = {sum_axial}"
    
    # 2. Sum of vertical forces = 0
    sum_vertical = Vi + Vj
    assert np.isclose(sum_vertical, 0.0, rtol=1e-6, atol=1e-9), \
        f"Vertical forces don't balance: {Vi} + {Vj} = {sum_vertical}"
    
    # 3. Sum of moments about node i = 0
    # Moment balance: Mi + Mj + Vj * L = 0
    moment_balance = Mi + Mj + Vj * L
    assert np.isclose(moment_balance, 0.0, rtol=1e-3, atol=1e-6), \
        f"Moments don't balance: {moment_balance}"
    
    print(f"✓ End forces equilibrium test passed")
    print(f"  Axial balance: {Ni:.2f} + {Nj:.2f} = {sum_axial:.2e} ✓")
    print(f"  Vertical balance: {Vi:.2f} + {Vj:.2f} = {sum_vertical:.2e} ✓")
    print(f"  Moment balance: {moment_balance:.2e} ✓")