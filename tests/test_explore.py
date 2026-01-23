# File: tests/test_explore.py
"""
Test the explore.py module (parametric portal frame generator).

WHY THESE TESTS?
---------------
1. Verify make_portal() generates valid models
2. Catch regressions when we refactor
3. Ensure edge cases work (bracing, different parameters)
4. Fast feedback during development

TEST PHILOSOPHY:
---------------
- Each test focuses on one specific aspect
- Tests should be fast (no heavy computation)
- Tests should be deterministic (same input = same output)
- Tests verify structure, not just that code runs
"""

import numpy as np
import pytest

from mini_branch.explore import PortalParams, make_portal
from mini_branch.catalog import TIMBER_SECTIONS, DEFAULT_MATERIAL
from mini_branch.assembly import assemble_global_K, DOF_PER_NODE, dof_index
from mini_branch.loads import assemble_element_loads_global
from mini_branch.solve import solve_linear


def test_portal_params_creation():
    """
    Test that PortalParams can be created with valid values.
    """
    params = PortalParams(
        span=6.0,
        height=3.0,
        brace=0,
        sec_col=3,
        sec_beam=4,
        sec_brace=1,
        udl_w=-2000.0,
        wind_P=5000.0,
    )
    
    assert params.span == 6.0
    assert params.height == 3.0
    assert params.brace == 0
    assert params.sec_col == 3
    assert params.sec_beam == 4
    assert params.udl_w == -2000.0
    assert params.wind_P == 5000.0
    assert params.shipping_limit == 12.0  # Default value
    
    print("✓ PortalParams creation works")


def test_portal_params_default_shipping_limit():
    """
    Test that shipping_limit has a default value.
    """
    params = PortalParams(
        span=6.0,
        height=3.0,
        brace=0,
        sec_col=3,
        sec_beam=4,
        sec_brace=1,
        udl_w=-2000.0,
        wind_P=5000.0,
        # shipping_limit not specified
    )
    
    assert params.shipping_limit == 12.0
    print("✓ Default shipping_limit works")


def test_make_portal_basic_structure():
    """
    Test that make_portal() generates a basic portal frame with correct structure.
    """
    params = PortalParams(
        span=6.0,
        height=3.0,
        brace=0,
        sec_col=2,
        sec_beam=3,
        sec_brace=1,
        udl_w=-2000.0,
        wind_P=5000.0,
    )
    
    nodes, elements, fixed_dofs, element_udl_map, nodal_loads = make_portal(
        params, DEFAULT_MATERIAL, TIMBER_SECTIONS
    )
    
    # Check nodes: should have 4 nodes
    assert len(nodes) == 4, f"Expected 4 nodes, got {len(nodes)}"
    assert 0 in nodes and 1 in nodes and 2 in nodes and 3 in nodes
    
    # Check node positions
    assert nodes[0].x == 0.0 and nodes[0].y == 0.0, "Node 0 should be at (0, 0)"
    assert nodes[1].x == 0.0 and nodes[1].y == 3.0, "Node 1 should be at (0, H)"
    assert nodes[2].x == 6.0 and nodes[2].y == 3.0, "Node 2 should be at (L, H)"
    assert nodes[3].x == 6.0 and nodes[3].y == 0.0, "Node 3 should be at (L, 0)"
    
    # Check elements: should have 3 elements (2 columns + 1 beam)
    assert len(elements) == 3, f"Expected 3 elements, got {len(elements)}"
    
    # Check element connectivity
    elem_ids = [e.id for e in elements]
    assert set(elem_ids) == {0, 1, 2}, f"Element IDs should be 0, 1, 2, got {elem_ids}"
    
    # Check left column (Element 0: Node 0 → 1)
    col_left = elements[0]
    assert col_left.ni == 0 and col_left.nj == 1, "Left column should connect 0→1"
    
    # Check beam (Element 1: Node 1 → 2)
    beam = elements[1]
    assert beam.ni == 1 and beam.nj == 2, "Beam should connect 1→2"
    
    # Check right column (Element 2: Node 2 → 3)
    col_right = elements[2]
    assert col_right.ni == 2 and col_right.nj == 3, "Right column should connect 2→3"
    
    print("✓ Basic portal structure is correct")


def test_make_portal_section_properties():
    """
    Test that elements get correct section properties from catalog.
    """
    params = PortalParams(
        span=6.0,
        height=3.0,
        brace=0,
        sec_col=3,  # Use section index 3
        sec_beam=4,  # Use section index 4
        sec_brace=1,
        udl_w=-2000.0,
        wind_P=5000.0,
    )
    
    nodes, elements, fixed_dofs, element_udl_map, nodal_loads = make_portal(
        params, DEFAULT_MATERIAL, TIMBER_SECTIONS
    )
    
    # Get expected sections
    sec_col = TIMBER_SECTIONS[3]
    sec_beam = TIMBER_SECTIONS[4]
    
    # Check column properties (elements 0 and 2)
    for col_elem in [elements[0], elements[2]]:
        assert col_elem.E == DEFAULT_MATERIAL.E, "Column E should match material"
        assert col_elem.A == sec_col.A, "Column A should match section"
        assert col_elem.I == sec_col.I, "Column I should match section"
    
    # Check beam properties (element 1)
    beam = elements[1]
    assert beam.E == DEFAULT_MATERIAL.E, "Beam E should match material"
    assert beam.A == sec_beam.A, "Beam A should match section"
    assert beam.I == sec_beam.I, "Beam I should match section"
    
    print("✓ Section properties are assigned correctly")


def test_make_portal_boundary_conditions():
    """
    Test that boundary conditions are set correctly (pinned bases).
    """
    params = PortalParams(
        span=6.0,
        height=3.0,
        brace=0,
        sec_col=2,
        sec_beam=3,
        sec_brace=1,
        udl_w=-2000.0,
        wind_P=5000.0,
    )
    
    nodes, elements, fixed_dofs, element_udl_map, nodal_loads = make_portal(
        params, DEFAULT_MATERIAL, TIMBER_SECTIONS
    )
    
    # Should have 4 fixed DOFs (2 per pinned base)
    assert len(fixed_dofs) == 4, f"Expected 4 fixed DOFs, got {len(fixed_dofs)}"
    
    # Check Node 0 is fixed (ux, uy)
    assert dof_index(0, 0) in fixed_dofs, "Node 0 ux should be fixed"
    assert dof_index(0, 1) in fixed_dofs, "Node 0 uy should be fixed"
    assert dof_index(0, 2) not in fixed_dofs, "Node 0 rz should be free (pinned)"
    
    # Check Node 3 is fixed (ux, uy)
    assert dof_index(3, 0) in fixed_dofs, "Node 3 ux should be fixed"
    assert dof_index(3, 1) in fixed_dofs, "Node 3 uy should be fixed"
    assert dof_index(3, 2) not in fixed_dofs, "Node 3 rz should be free (pinned)"
    
    print("✓ Boundary conditions are correct (pinned bases)")


def test_make_portal_loads():
    """
    Test that loads are defined correctly.
    """
    params = PortalParams(
        span=6.0,
        height=3.0,
        brace=0,
        sec_col=2,
        sec_beam=3,
        sec_brace=1,
        udl_w=-2000.0,
        wind_P=5000.0,
    )
    
    nodes, elements, fixed_dofs, element_udl_map, nodal_loads = make_portal(
        params, DEFAULT_MATERIAL, TIMBER_SECTIONS
    )
    
    # Check UDL on beam (Element 1)
    assert 1 in element_udl_map, "Beam (element 1) should have UDL"
    assert element_udl_map[1] == -2000.0, "UDL should match params.udl_w"
    assert len(element_udl_map) == 1, "Only beam should have UDL"
    
    # Check lateral load at Node 1
    assert 1 in nodal_loads, "Node 1 should have lateral load"
    load_vec = nodal_loads[1]
    assert load_vec[0] == 5000.0, "Fx should match params.wind_P"
    assert load_vec[1] == 0.0, "Fy should be zero"
    assert load_vec[2] == 0.0, "Mz should be zero"
    
    print("✓ Loads are defined correctly")


def test_make_portal_braced():
    """
    Test that bracing adds diagonal elements correctly.
    """
    params = PortalParams(
        span=6.0,
        height=3.0,
        brace=1,  # WITH bracing
        sec_col=2,
        sec_beam=3,
        sec_brace=1,
        udl_w=-2000.0,
        wind_P=5000.0,
    )
    
    nodes, elements, fixed_dofs, element_udl_map, nodal_loads = make_portal(
        params, DEFAULT_MATERIAL, TIMBER_SECTIONS
    )
    
    # Should have 5 elements (2 columns + 1 beam + 2 braces)
    assert len(elements) == 5, f"Expected 5 elements with bracing, got {len(elements)}"
    
    # Check brace elements exist (elements 3 and 4)
    elem_ids = [e.id for e in elements]
    assert 3 in elem_ids and 4 in elem_ids, "Should have brace elements 3 and 4"
    
    # Find brace elements
    brace_elems = [e for e in elements if e.id >= 3]
    assert len(brace_elems) == 2, "Should have 2 brace elements"
    
    # Check brace connectivity (X-brace)
    # One brace: Node 1 → 3 (top-left to bottom-right)
    # Other brace: Node 0 → 2 (bottom-left to top-right)
    brace_connections = [(e.ni, e.nj) for e in brace_elems]
    assert (1, 3) in brace_connections or (3, 1) in brace_connections, "Should have brace 1↔3"
    assert (0, 2) in brace_connections or (2, 0) in brace_connections, "Should have brace 0↔2"
    
    # Check brace properties
    sec_brace = TIMBER_SECTIONS[1]
    for brace_elem in brace_elems:
        assert brace_elem.A == sec_brace.A, "Brace A should match section"
        assert brace_elem.I == sec_brace.I, "Brace I should match section"
    
    print("✓ Bracing adds diagonal elements correctly")


def test_make_portal_solvable():
    """
    Test that the generated model can actually be solved (is valid).
    """
    params = PortalParams(
        span=6.0,
        height=3.0,
        brace=0,
        sec_col=2,
        sec_beam=3,
        sec_brace=1,
        udl_w=-2000.0,
        wind_P=5000.0,
    )
    
    nodes, elements, fixed_dofs, element_udl_map, nodal_loads = make_portal(
        params, DEFAULT_MATERIAL, TIMBER_SECTIONS
    )
    
    # Build stiffness matrix
    K = assemble_global_K(nodes, elements)
    ndof = DOF_PER_NODE * len(nodes)
    
    # Build load vector
    F = np.zeros(ndof)
    F += assemble_element_loads_global(nodes, elements, element_udl_map)
    
    # Add nodal loads
    for node_id, load_vec in nodal_loads.items():
        F[dof_index(node_id, 0)] += load_vec[0]
        F[dof_index(node_id, 1)] += load_vec[1]
        F[dof_index(node_id, 2)] += load_vec[2]
    
    # Solve (should not raise exception)
    d, R, free = solve_linear(K, F, fixed_dofs)
    
    # Check solution is reasonable
    assert d.shape == (ndof,), "Displacement vector should match DOF count"
    assert R.shape == (ndof,), "Reaction vector should match DOF count"
    
    # Check that fixed DOFs are actually zero (within numerical tolerance)
    for dof in fixed_dofs:
        assert abs(d[dof]) < 1e-10, f"Fixed DOF {dof} should be zero, got {d[dof]}"
    
    print("✓ Generated model is solvable (valid structure)")


def test_make_portal_braced_solvable():
    """
    Test that braced model can be solved.
    """
    params = PortalParams(
        span=6.0,
        height=3.0,
        brace=1,
        sec_col=2,
        sec_beam=3,
        sec_brace=1,
        udl_w=-2000.0,
        wind_P=5000.0,
    )
    
    nodes, elements, fixed_dofs, element_udl_map, nodal_loads = make_portal(
        params, DEFAULT_MATERIAL, TIMBER_SECTIONS
    )
    
    # Build and solve
    K = assemble_global_K(nodes, elements)
    ndof = DOF_PER_NODE * len(nodes)
    F = np.zeros(ndof)
    F += assemble_element_loads_global(nodes, elements, element_udl_map)
    for node_id, load_vec in nodal_loads.items():
        F[dof_index(node_id, 0)] += load_vec[0]
        F[dof_index(node_id, 1)] += load_vec[1]
        F[dof_index(node_id, 2)] += load_vec[2]
    
    d, R, free = solve_linear(K, F, fixed_dofs)
    
    # Should solve successfully
    assert d.shape == (ndof,)
    print("✓ Braced model is solvable")


def test_make_portal_validation():
    """
    Test that make_portal() validates inputs correctly.
    """
    # Test negative span
    with pytest.raises(ValueError, match="span must be positive"):
        params = PortalParams(
            span=-1.0,  # Invalid
            height=3.0,
            brace=0,
            sec_col=2,
            sec_beam=3,
            sec_brace=1,
            udl_w=-2000.0,
            wind_P=5000.0,
        )
        make_portal(params, DEFAULT_MATERIAL, TIMBER_SECTIONS)
    
    # Test negative height
    with pytest.raises(ValueError, match="height must be positive"):
        params = PortalParams(
            span=6.0,
            height=-1.0,  # Invalid
            brace=0,
            sec_col=2,
            sec_beam=3,
            sec_brace=1,
            udl_w=-2000.0,
            wind_P=5000.0,
        )
        make_portal(params, DEFAULT_MATERIAL, TIMBER_SECTIONS)
    
    # Test invalid brace value
    with pytest.raises(ValueError, match="brace must be 0 or 1"):
        params = PortalParams(
            span=6.0,
            height=3.0,
            brace=2,  # Invalid
            sec_col=2,
            sec_beam=3,
            sec_brace=1,
            udl_w=-2000.0,
            wind_P=5000.0,
        )
        make_portal(params, DEFAULT_MATERIAL, TIMBER_SECTIONS)
    
    # Test invalid section index
    with pytest.raises(ValueError, match="sec_col.*out of range"):
        params = PortalParams(
            span=6.0,
            height=3.0,
            brace=0,
            sec_col=999,  # Invalid (out of range)
            sec_beam=3,
            sec_brace=1,
            udl_w=-2000.0,
            wind_P=5000.0,
        )
        make_portal(params, DEFAULT_MATERIAL, TIMBER_SECTIONS)
    
    print("✓ Input validation works correctly")


def test_make_portal_different_parameters():
    """
    Test that different parameters produce different models.
    """
    # Small frame
    params_small = PortalParams(
        span=4.0,
        height=2.5,
        brace=0,
        sec_col=1,
        sec_beam=2,
        sec_brace=1,
        udl_w=-1000.0,
        wind_P=3000.0,
    )
    
    # Large frame
    params_large = PortalParams(
        span=10.0,
        height=5.0,
        brace=1,
        sec_col=7,
        sec_beam=8,
        sec_brace=3,
        udl_w=-5000.0,
        wind_P=10000.0,
    )
    
    nodes_small, elements_small, _, _, _ = make_portal(
        params_small, DEFAULT_MATERIAL, TIMBER_SECTIONS
    )
    nodes_large, elements_large, _, _, _ = make_portal(
        params_large, DEFAULT_MATERIAL, TIMBER_SECTIONS
    )
    
    # Check they're different
    assert nodes_small[2].x == 4.0, "Small frame span should be 4.0"
    assert nodes_large[2].x == 10.0, "Large frame span should be 10.0"
    assert nodes_small[1].y == 2.5, "Small frame height should be 2.5"
    assert nodes_large[1].y == 5.0, "Large frame height should be 5.0"
    assert len(elements_small) == 3, "Small frame should have 3 elements"
    assert len(elements_large) == 5, "Large frame should have 5 elements (braced)"
    
    print("✓ Different parameters produce different models")


if __name__ == "__main__":
    # Run tests manually (or use pytest)
    test_portal_params_creation()
    test_portal_params_default_shipping_limit()
    test_make_portal_basic_structure()
    test_make_portal_section_properties()
    test_make_portal_boundary_conditions()
    test_make_portal_loads()
    test_make_portal_braced()
    test_make_portal_solvable()
    test_make_portal_braced_solvable()
    test_make_portal_validation()
    test_make_portal_different_parameters()
    print("\n✅ All explore.py tests passed!")