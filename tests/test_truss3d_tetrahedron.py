# tests/test_truss3d_tetrahedron.py
"""
TETRAHEDRON TEST: Validation of 3D Truss Analysis
=================================================

This is THE classic validation test for 3D structural analysis.
A regular tetrahedron with:
- 4 nodes (3 at base, 1 at apex)
- 6 bars (connecting all nodes)
- Fixed base (all 3 base nodes pinned)
- Vertical load at apex

Expected behavior:
1. SYMMETRY: Base reactions should be symmetric (equal horizontal components)
2. EQUILIBRIUM: ΣReactions = ΣApplied loads
3. FORCES: All bars should have the same magnitude of force (by symmetry)

If this test passes, we can be confident the 3D truss solver is correct.
"""

import pytest
import numpy as np

from mini_branch.v3d.model import Node3D, Truss3D
from mini_branch.v3d.elements import truss3d_global_stiffness, truss3d_axial_force
from mini_branch.kernel.dof import DOFManager
from mini_branch.kernel.assemble import assemble_global_K
from mini_branch.kernel.solve import solve_linear


def make_regular_tetrahedron(base_radius: float = 1.0, height: float = 1.0):
    """
    Create a regular tetrahedron with base in xy-plane and apex above center.
    
    Layout:
        - Node 0, 1, 2: Base triangle at z=0, equally spaced on a circle
        - Node 3: Apex at (0, 0, height)
        - 6 bars: 3 base edges + 3 legs to apex
    
    Parameters:
    -----------
    base_radius : float
        Radius of circumscribed circle for base triangle
    height : float
        Height of apex above base plane
    
    Returns:
    --------
    nodes : dict
        {id: Node3D}
    bars : list
        List of Truss3D elements
    """
    # Base triangle nodes (120° apart on circle)
    angles = [0, 2*np.pi/3, 4*np.pi/3]
    
    nodes = {}
    for i, angle in enumerate(angles):
        x = base_radius * np.cos(angle)
        y = base_radius * np.sin(angle)
        nodes[i] = Node3D(id=i, x=x, y=y, z=0.0)
    
    # Apex node at center, elevated
    nodes[3] = Node3D(id=3, x=0.0, y=0.0, z=height)
    
    # Material properties (steel)
    E = 210e9  # Pa
    A = 0.001  # m² (10 cm²)
    
    bars = []
    bar_id = 0
    
    # Base triangle edges: 0-1, 1-2, 2-0
    for i in range(3):
        j = (i + 1) % 3
        bars.append(Truss3D(id=bar_id, ni=i, nj=j, E=E, A=A))
        bar_id += 1
    
    # Legs from base to apex: 0-3, 1-3, 2-3
    for i in range(3):
        bars.append(Truss3D(id=bar_id, ni=i, nj=3, E=E, A=A))
        bar_id += 1
    
    return nodes, bars


class TestTetrahedronEquilibrium:
    """Test that reactions balance applied loads."""
    
    def test_vertical_load_equilibrium(self):
        """
        Apply vertical load at apex, check that vertical reactions sum to load.
        """
        nodes, bars = make_regular_tetrahedron()
        
        # Setup DOF manager (3 DOF per node for 3D truss)
        dof = DOFManager(dof_per_node=3)
        n_nodes = len(nodes)
        ndof = dof.ndof(n_nodes)
        
        # Assemble stiffness matrix
        contributions = []
        for bar in bars:
            dof_map = dof.element_dof_map([bar.ni, bar.nj])
            ke = truss3d_global_stiffness(nodes, bar)
            contributions.append((dof_map, ke))
        
        K = assemble_global_K(ndof, contributions)
        
        # Apply vertical load at apex (node 3)
        P = -10000.0  # N (downward)
        F = np.zeros(ndof)
        F[dof.idx(3, 2)] = P  # Fz at node 3
        
        # Fix all base nodes (nodes 0, 1, 2)
        fixed_dofs = []
        for node_id in [0, 1, 2]:
            fixed_dofs.extend(dof.node_dofs(node_id))
        
        # Solve
        d, R, free = solve_linear(K, F, fixed_dofs)
        
        # Check equilibrium: sum of vertical reactions = -applied load
        Rz_total = sum(R[dof.idx(i, 2)] for i in [0, 1, 2])
        
        # Reactions should balance applied load
        assert np.isclose(Rz_total, -P, rtol=1e-10), \
            f"Vertical equilibrium failed: ΣRz={Rz_total:.2f} N, applied={P:.2f} N"
        
        # Horizontal reactions should sum to zero (no horizontal load)
        Rx_total = sum(R[dof.idx(i, 0)] for i in [0, 1, 2])
        Ry_total = sum(R[dof.idx(i, 1)] for i in [0, 1, 2])
        
        assert np.isclose(Rx_total, 0.0, atol=1e-6), \
            f"Horizontal equilibrium failed: ΣRx={Rx_total:.6f} N"
        assert np.isclose(Ry_total, 0.0, atol=1e-6), \
            f"Horizontal equilibrium failed: ΣRy={Ry_total:.6f} N"


class TestTetrahedronSymmetry:
    """Test that symmetric structure produces symmetric response."""
    
    def test_symmetric_vertical_reactions(self):
        """
        With vertical load at apex center, base vertical reactions should be equal.
        """
        nodes, bars = make_regular_tetrahedron()
        
        dof = DOFManager(dof_per_node=3)
        ndof = dof.ndof(len(nodes))
        
        # Assemble
        contributions = [(dof.element_dof_map([b.ni, b.nj]), 
                         truss3d_global_stiffness(nodes, b)) for b in bars]
        K = assemble_global_K(ndof, contributions)
        
        # Vertical load at apex
        P = -10000.0
        F = np.zeros(ndof)
        F[dof.idx(3, 2)] = P
        
        # Fix base nodes
        fixed_dofs = []
        for node_id in [0, 1, 2]:
            fixed_dofs.extend(dof.node_dofs(node_id))
        
        # Solve
        d, R, free = solve_linear(K, F, fixed_dofs)
        
        # Extract vertical reactions at base nodes
        Rz = [R[dof.idx(i, 2)] for i in [0, 1, 2]]
        
        # All three should be equal (by symmetry)
        expected_Rz = -P / 3  # Load splits equally to 3 supports
        
        for i, rz in enumerate(Rz):
            assert np.isclose(rz, expected_Rz, rtol=1e-6), \
                f"Node {i}: Rz={rz:.2f} N, expected={expected_Rz:.2f} N"
    
    def test_symmetric_leg_forces(self):
        """
        The three legs (bars connecting base to apex) should have equal force.
        """
        nodes, bars = make_regular_tetrahedron()
        
        dof = DOFManager(dof_per_node=3)
        ndof = dof.ndof(len(nodes))
        
        # Assemble
        contributions = [(dof.element_dof_map([b.ni, b.nj]), 
                         truss3d_global_stiffness(nodes, b)) for b in bars]
        K = assemble_global_K(ndof, contributions)
        
        # Vertical load at apex
        P = -10000.0
        F = np.zeros(ndof)
        F[dof.idx(3, 2)] = P
        
        # Fix base
        fixed_dofs = []
        for node_id in [0, 1, 2]:
            fixed_dofs.extend(dof.node_dofs(node_id))
        
        # Solve
        d, R, free = solve_linear(K, F, fixed_dofs)
        
        # Compute axial forces in leg bars (bars 3, 4, 5 are legs to apex)
        leg_bars = bars[3:6]
        leg_forces = [truss3d_axial_force(nodes, bar, d) for bar in leg_bars]
        
        # All three legs should have the same force
        assert np.allclose(leg_forces, leg_forces[0], rtol=1e-6), \
            f"Leg forces not symmetric: {leg_forces}"
        
        # Under downward load, legs should be in compression (negative)
        assert leg_forces[0] < 0, f"Legs should be in compression, got {leg_forces[0]:.2f} N"


class TestTetrahedronDisplacement:
    """Test displacement behavior."""
    
    def test_apex_moves_downward(self):
        """Under downward load, apex should move down (negative z)."""
        nodes, bars = make_regular_tetrahedron()
        
        dof = DOFManager(dof_per_node=3)
        ndof = dof.ndof(len(nodes))
        
        contributions = [(dof.element_dof_map([b.ni, b.nj]), 
                         truss3d_global_stiffness(nodes, b)) for b in bars]
        K = assemble_global_K(ndof, contributions)
        
        P = -10000.0
        F = np.zeros(ndof)
        F[dof.idx(3, 2)] = P
        
        fixed_dofs = []
        for node_id in [0, 1, 2]:
            fixed_dofs.extend(dof.node_dofs(node_id))
        
        d, R, free = solve_linear(K, F, fixed_dofs)
        
        # Apex vertical displacement
        uz_apex = d[dof.idx(3, 2)]
        
        # Should be negative (downward)
        assert uz_apex < 0, f"Apex should move down, got uz={uz_apex:.6f} m"
        
        # Horizontal displacements should be zero (symmetric loading)
        ux_apex = d[dof.idx(3, 0)]
        uy_apex = d[dof.idx(3, 1)]
        
        assert np.isclose(ux_apex, 0.0, atol=1e-10), f"ux_apex should be 0, got {ux_apex}"
        assert np.isclose(uy_apex, 0.0, atol=1e-10), f"uy_apex should be 0, got {uy_apex}"
    
    def test_displacement_order_of_magnitude(self):
        """
        Sanity check: displacement should be small but not zero.
        
        For steel bars (E=210 GPa, A=10 cm²) of ~1m length under 10 kN:
        Expected: ~0.05 mm (very stiff structure)
        """
        nodes, bars = make_regular_tetrahedron()
        
        dof = DOFManager(dof_per_node=3)
        ndof = dof.ndof(len(nodes))
        
        contributions = [(dof.element_dof_map([b.ni, b.nj]), 
                         truss3d_global_stiffness(nodes, b)) for b in bars]
        K = assemble_global_K(ndof, contributions)
        
        P = -10000.0
        F = np.zeros(ndof)
        F[dof.idx(3, 2)] = P
        
        fixed_dofs = []
        for node_id in [0, 1, 2]:
            fixed_dofs.extend(dof.node_dofs(node_id))
        
        d, R, free = solve_linear(K, F, fixed_dofs)
        
        uz_apex = abs(d[dof.idx(3, 2)])
        
        # Should be in the range of 0.01 mm to 1 mm for this setup
        assert 1e-6 < uz_apex < 1e-3, \
            f"Displacement {uz_apex*1000:.4f} mm seems wrong (expected 0.01-1 mm)"


class TestTetrahedronMechanism:
    """Test that insufficiently supported structures are detected."""
    
    def test_insufficient_supports_detected(self):
        """
        If we only fix 2 base nodes, structure is a mechanism (can rotate).
        """
        nodes, bars = make_regular_tetrahedron()
        
        dof = DOFManager(dof_per_node=3)
        ndof = dof.ndof(len(nodes))
        
        contributions = [(dof.element_dof_map([b.ni, b.nj]), 
                         truss3d_global_stiffness(nodes, b)) for b in bars]
        K = assemble_global_K(ndof, contributions)
        
        P = -10000.0
        F = np.zeros(ndof)
        F[dof.idx(3, 2)] = P
        
        # Only fix 2 nodes - this should cause a mechanism
        fixed_dofs = []
        for node_id in [0, 1]:  # Missing node 2!
            fixed_dofs.extend(dof.node_dofs(node_id))
        
        from mini_branch.kernel.solve import MechanismError
        
        with pytest.raises(MechanismError):
            solve_linear(K, F, fixed_dofs)


