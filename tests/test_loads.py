# File: tests/test_loads.py
"""
Test the loads.py module to verify UDL equivalent nodal loads work correctly.
"""

import numpy as np
from mini_branch.model import Node, Frame2D
from mini_branch.loads import frame2d_equiv_nodal_load_udl, assemble_element_loads_global
from mini_branch.assembly import assemble_global_K, DOF_PER_NODE
from mini_branch.solve import solve_linear

def test_udl_local_load_vector():
    """
    Test that the local UDL load vector has the correct structure.
    We'll check a simple case where we know the answer.
    """
    # Simple case: 4 meter beam with -1000 N/m downward
    L = 4.0
    w = -1000.0
    
    # Calculate equivalent nodal loads
    f_local = frame2d_equiv_nodal_load_udl(L, w)
    
    # Check shape (should be 6-DOF vector)
    assert f_local.shape == (6,), "Should return 6-DOF vector"
    
    # Check values
    force_per_node = w * L / 2.0  # -2000 N
    moment_magnitude = w * L * L / 12.0  # -1333.33... N·m
    
    # Expected vector: [0, -2000, -1333.33..., 0, -2000, +1333.33...]
    expected = np.array([
        0.0,
        force_per_node,
        moment_magnitude,
        0.0,
        force_per_node,
        -moment_magnitude  # Opposite sign at end
    ], dtype=float)
    
    # Verify they match
    np.testing.assert_allclose(f_local, expected, rtol=1e-10, atol=1e-12)
    print("✓ UDL local load vector is correct")


def test_udl_assembly_simple_beam():
    """
    Test assembling UDL loads into global force vector for a simple beam.
    """
    # Create a 2-element beam
    L = 4.0
    E = 210e9
    I = 8.0e-6
    A = 0.01
    
    nodes = {
        0: Node(0, 0.0, 0.0),
        1: Node(1, L/2, 0.0),
        2: Node(2, L, 0.0),
    }
    
    elements = [
        Frame2D(0, 0, 1, E=E, A=A, I=I),
        Frame2D(1, 1, 2, E=E, A=A, I=I),
    ]
    
    # Apply UDL to both elements
    element_udls = {
        0: -1000.0,  # -1000 N/m on element 0
        1: -1000.0,  # -1000 N/m on element 1
    }
    
    # Assemble UDL loads into global force vector
    F_udl = assemble_element_loads_global(nodes, elements, element_udls)
    
    # Check shape (should match total DOFs: 3 nodes × 3 DOF/node = 9)
    assert F_udl.shape == (9,), "Should match total DOFs (3 nodes × 3 DOF/node = 9)"
    
    # For a horizontal beam with UDL, we expect:
    # - No horizontal forces (axial forces are 0)
    # - Vertical forces at each node
    # - Moments at each node
    
    # Node 0: should have vertical force from element 0
    # Element 0: L=2.0, w=-1000, so Fy = wL/2 = -1000 N at node 0
    Fy_node0 = F_udl[DOF_PER_NODE*0 + 1]  # Vertical force at node 0
    expected_Fy0 = -1000.0 * 2.0 / 2.0  # = -1000 N
    assert np.isclose(Fy_node0, expected_Fy0, rtol=1e-10), \
        f"Node 0 vertical force: {Fy_node0} != expected {expected_Fy0}"
    
    # Node 1: should have forces from BOTH elements (connected to both)
    Fy_node1 = F_udl[DOF_PER_NODE*1 + 1]  # Vertical force at node 1
    # From element 0: wL/2 = -1000 N (at end node)
    # From element 1: wL/2 = -1000 N (at start node)
    # Total: -2000 N
    expected_Fy1 = -1000.0 * 2.0 / 2.0 + -1000.0 * 2.0 / 2.0  # = -2000 N
    assert np.isclose(Fy_node1, expected_Fy1, rtol=1e-10), \
        f"Node 1 vertical force: {Fy_node1} != expected {expected_Fy1}"
    
    print("✓ UDL assembly works correctly")
    print(f"  Node 0 force: {Fy_node0:.2f} N")
    print(f"  Node 1 force: {Fy_node1:.2f} N")


def test_udl_beam_solve():
    """
    Test that we can actually solve a beam with UDL loads.
    This is a sanity check that everything integrates correctly.
    """
    # Create a meshed beam (10 elements for accuracy)
    L = 4.0  # Total beam length
    E = 210e9
    I = 8.0e-6
    A = 0.01
    w0 = -1000.0  # -1000 N/m downward
    
    num_elements = 10
    num_nodes = num_elements + 1
    
    # Create nodes evenly spaced
    nodes = {}
    for i in range(num_nodes):
        x = L * i / num_elements
        nodes[i] = Node(i, x, 0.0)
    
    # Create elements
    elements = []
    for i in range(num_elements):
        elements.append(Frame2D(i, i, i+1, E=E, A=A, I=I))
    
    # Apply UDL to all elements
    element_udls = {i: w0 for i in range(num_elements)}
    
    # Build stiffness matrix
    K = assemble_global_K(nodes, elements)
    
    # Assemble UDL loads
    F = assemble_element_loads_global(nodes, elements, element_udls)
    
    # Define supports (simply supported: pinned at both ends)
    fixed = [
        DOF_PER_NODE*0 + 0,  # Node 0, ux
        DOF_PER_NODE*0 + 1,  # Node 0, uy
        DOF_PER_NODE*(num_nodes-1) + 1,  # Last node, uy
    ]
    
    # Solve
    d, R, _ = solve_linear(K, F, fixed)
    
    # Check that we got a solution (not all zeros)
    max_deflection = np.max(np.abs(d))
    assert max_deflection > 0, "Beam should deflect under UDL"
    
    # Check that reactions balance the applied load
    # Total applied load = w0 × L = -1000 × 4 = -4000 N
    total_applied = w0 * L
    total_reactions = R[DOF_PER_NODE*0 + 1] + R[DOF_PER_NODE*(num_nodes-1) + 1]
    
    # Reactions should balance applied load (within tolerance)
    equilibrium_error = total_reactions + total_applied
    assert np.isclose(equilibrium_error, 0.0, rtol=1e-3, atol=1e-6), \
        f"Forces don't balance: reactions {total_reactions} + applied {total_applied} = {equilibrium_error}"
    
    print("✓ Beam with UDL solves correctly")
    print(f"  Total applied load: {total_applied:.2f} N")
    print(f"  Total reactions: {total_reactions:.2f} N")
    print(f"  Equilibrium error: {equilibrium_error:.2e} N (should be ≈ 0)")
    print(f"  Max deflection: {max_deflection:.6f} m")