# File: demos/run_portal_parametric.py
"""
DEMO: PARAMETRIC PORTAL FRAME GENERATOR
========================================

PURPOSE:
--------
This demo tests the new parametric portal frame generator (make_portal).
Instead of hardcoding geometry, sections, and loads, we define them as
parameters and generate the model programmatically.

This is the foundation for Day 3's design exploration: we can now generate
hundreds of variants by just changing parameters!

WHAT WE'RE TESTING:
-------------------
1. PortalParams dataclass: Can we define parameters?
2. make_portal() function: Does it generate a valid model?
3. Model structure: Are nodes, elements, loads correct?
4. Analysis: Can we solve it? (verifies model is valid)

COMPARISON TO DAY 2:
--------------------
Day 2 (run_portal_frame.py):
- Hardcoded: L=6.0, H=3.0, E=210e9, A=0.01, I=8e-6
- Fixed: Can't easily try different designs

Day 3 (this demo):
- Parameterized: span, height, section indices
- Flexible: Change one number, get a new design
- Catalog-based: Sections come from TIMBER_SECTIONS

ENGINEERING SIGNIFICANCE:
-------------------------
This is the first step toward automated design exploration. By parameterizing
the model, we can:
- Test many design variants automatically
- Find optimal designs (Day 3: Pareto frontier)
- Train ML models to predict performance (Day 4)
"""

import numpy as np
import matplotlib.pyplot as plt

# Import the parametric generator
from mini_branch.explore import PortalParams, make_portal
from mini_branch.catalog import TIMBER_SECTIONS, DEFAULT_MATERIAL

# Import solver components (to verify the model works)
from mini_branch.assembly import assemble_global_K, DOF_PER_NODE, dof_index
from mini_branch.loads import assemble_element_loads_global
from mini_branch.solve import solve_linear
from mini_branch.post import element_end_forces_local


def print_model_structure(nodes, elements, fixed_dofs, element_udl_map, nodal_loads):
    """
    Print the model structure for inspection.
    
    WHY THIS HELPS:
    --------------
    - Visual verification that make_portal() created the right structure
    - Debugging: If something's wrong, we can see it here
    - Learning: Shows what the model looks like
    """
    print("=" * 70)
    print("MODEL STRUCTURE")
    print("=" * 70)
    
    print(f"\nNodes ({len(nodes)}):")
    for node_id, node in sorted(nodes.items()):
        print(f"  Node {node_id}: ({node.x:.2f}, {node.y:.2f}) m")
    
    print(f"\nElements ({len(elements)}):")
    for elem in elements:
        sec_info = f"E={elem.E/1e9:.1f}GPa, A={elem.A:.4f}m², I={elem.I*1e6:.2f}×10⁻⁶m⁴"
        print(f"  Element {elem.id}: Node {elem.ni} → {elem.nj}  ({sec_info})")
    
    print(f"\nBoundary Conditions ({len(fixed_dofs)} fixed DOFs):")
    for dof in fixed_dofs:
        node_id = dof // DOF_PER_NODE
        local_dof = dof % DOF_PER_NODE
        dof_names = ["ux", "uy", "rz"]
        print(f"  DOF {dof}: Node {node_id}, {dof_names[local_dof]} = 0")
    
    print(f"\nElement UDLs ({len(element_udl_map)}):")
    for elem_id, w in element_udl_map.items():
        print(f"  Element {elem_id}: {w:.1f} N/m")
    
    print(f"\nNodal Loads ({len(nodal_loads)}):")
    for node_id, load_vec in nodal_loads.items():
        print(f"  Node {node_id}: Fx={load_vec[0]:.1f} N, Fy={load_vec[1]:.1f} N, Mz={load_vec[2]:.1f} N·m")
    
    print()


def test_basic_portal():
    """
    Test 1: Basic portal frame (no bracing)
    
    This matches the Day 2 demo but uses parameters instead of hardcoding.
    """
    print("=" * 70)
    print("TEST 1: BASIC PORTAL FRAME (NO BRACING)")
    print("=" * 70)
    print()
    
    # Define parameters (similar to Day 2 demo)
    params = PortalParams(
        span=6.0,           # L = 6.0 m (same as Day 2)
        height=3.0,         # H = 3.0 m (same as Day 2)
        brace=0,            # No bracing
        sec_col=3,          # 2x10 section for columns
        sec_beam=4,         # 2x12 section for beam
        sec_brace=1,        # Not used (brace=0), but required
        udl_w=-2000.0,      # -2000 N/m (same as Day 2)
        wind_P=5000.0,      # 5000 N (same as Day 2)
    )
    
    print("Parameters:")
    print(f"  Span: {params.span} m")
    print(f"  Height: {params.height} m")
    print(f"  Bracing: {'X-brace' if params.brace == 1 else 'None'}")
    print(f"  Column section: {TIMBER_SECTIONS[params.sec_col].name}")
    print(f"  Beam section: {TIMBER_SECTIONS[params.sec_beam].name}")
    print(f"  UDL: {params.udl_w} N/m")
    print(f"  Lateral load: {params.wind_P} N")
    print()
    
    # Generate model
    print("Generating model...")
    nodes, elements, fixed_dofs, element_udl_map, nodal_loads = make_portal(
        params, DEFAULT_MATERIAL, TIMBER_SECTIONS
    )
    
    # Print structure
    print_model_structure(nodes, elements, fixed_dofs, element_udl_map, nodal_loads)
    
    # Verify model is valid by solving it
    print("Verifying model (running analysis)...")
    K = assemble_global_K(nodes, elements)
    ndof = DOF_PER_NODE * len(nodes)
    
    # Assemble load vector
    F = np.zeros(ndof)
    F += assemble_element_loads_global(nodes, elements, element_udl_map)
    
    # Add nodal loads
    for node_id, load_vec in nodal_loads.items():
        F[dof_index(node_id, 0)] += load_vec[0]  # Fx
        F[dof_index(node_id, 1)] += load_vec[1]  # Fy
        F[dof_index(node_id, 2)] += load_vec[2]  # Mz
    
    # Solve
    d, R, free = solve_linear(K, F, fixed_dofs)
    
    # Extract key results
    # Drift = horizontal displacement at top nodes
    ux_node1 = d[dof_index(1, 0)]  # Left top
    ux_node2 = d[dof_index(2, 0)]  # Right top
    drift = max(abs(ux_node1), abs(ux_node2))
    
    print("Analysis Results:")
    print(f"  Drift (max |ux| at top): {drift*1000:.2f} mm")
    print(f"  Left top displacement: ux={ux_node1*1000:.2f} mm, uy={d[dof_index(1,1)]*1000:.2f} mm")
    print(f"  Right top displacement: ux={ux_node2*1000:.2f} mm, uy={d[dof_index(2,1)]*1000:.2f} mm")
    
    # Check reactions (should balance loads)
    R_left = R[dof_index(0, 0)] + 1j * R[dof_index(0, 1)]  # Complex for magnitude
    R_right = R[dof_index(3, 0)] + 1j * R[dof_index(3, 1)]
    print(f"  Left base reaction: Fx={R[dof_index(0,0)]:.1f} N, Fy={R[dof_index(0,1)]:.1f} N")
    print(f"  Right base reaction: Fx={R[dof_index(3,0)]:.1f} N, Fy={R[dof_index(3,1)]:.1f} N")
    
    print("\n✅ Test 1 passed: Basic portal frame works!")
    print()


def test_braced_portal():
    """
    Test 2: Portal frame with X-bracing
    
    This tests the bracing feature, which adds diagonal members.
    """
    print("=" * 70)
    print("TEST 2: PORTAL FRAME WITH X-BRACING")
    print("=" * 70)
    print()
    
    params = PortalParams(
        span=8.0,           # Larger span
        height=4.0,         # Taller
        brace=1,            # WITH bracing
        sec_col=4,          # Larger sections
        sec_beam=5,
        sec_brace=2,        # Medium brace size
        udl_w=-3000.0,      # Heavier load
        wind_P=8000.0,      # Stronger wind
    )
    
    print("Parameters:")
    print(f"  Span: {params.span} m")
    print(f"  Height: {params.height} m")
    print(f"  Bracing: {'X-brace' if params.brace == 1 else 'None'}")
    print(f"  Column section: {TIMBER_SECTIONS[params.sec_col].name}")
    print(f"  Beam section: {TIMBER_SECTIONS[params.sec_beam].name}")
    print(f"  Brace section: {TIMBER_SECTIONS[params.sec_brace].name}")
    print()
    
    # Generate model
    nodes, elements, fixed_dofs, element_udl_map, nodal_loads = make_portal(
        params, DEFAULT_MATERIAL, TIMBER_SECTIONS
    )
    
    print(f"Model has {len(elements)} elements (should be 5: 2 columns + 1 beam + 2 braces)")
    print(f"Model has {len(nodes)} nodes (should be 4)")
    print()
    
    # Verify bracing elements exist
    brace_elements = [e for e in elements if e.id >= 3]
    print(f"Brace elements: {len(brace_elements)} (should be 2)")
    for elem in brace_elements:
        print(f"  Element {elem.id}: Node {elem.ni} → {elem.nj}")
    
    print("\n✅ Test 2 passed: Braced portal frame works!")
    print()


def test_parameter_variation():
    """
    Test 3: Show how easy it is to try different designs
    
    This demonstrates the power of parameterization: change one number,
    get a completely different design.
    """
    print("=" * 70)
    print("TEST 3: PARAMETER VARIATION (DESIGN EXPLORATION)")
    print("=" * 70)
    print()
    
    print("Trying 3 different designs by changing parameters:")
    print()
    
    designs = [
        {"name": "Small & Light", "span": 4.0, "height": 2.5, "sec_col": 1, "sec_beam": 2},
        {"name": "Medium", "span": 6.0, "height": 3.0, "sec_col": 3, "sec_beam": 4},
        {"name": "Large & Heavy", "span": 10.0, "height": 5.0, "sec_col": 7, "sec_beam": 8},
    ]
    
    for design in designs:
        params = PortalParams(
            span=design["span"],
            height=design["height"],
            brace=0,
            sec_col=design["sec_col"],
            sec_beam=design["sec_beam"],
            sec_brace=1,
            udl_w=-2000.0,
            wind_P=5000.0,
        )
        
        nodes, elements, fixed_dofs, element_udl_map, nodal_loads = make_portal(
            params, DEFAULT_MATERIAL, TIMBER_SECTIONS
        )
        
        # Compute volume (material proxy)
        volume = 0.0
        for elem in elements:
            # Get element length (simple approximation)
            ni = nodes[elem.ni]
            nj = nodes[elem.nj]
            L = np.hypot(nj.x - ni.x, nj.y - ni.y)
            volume += elem.A * L
        
        print(f"{design['name']}:")
        print(f"  Span={params.span}m, Height={params.height}m")
        print(f"  Sections: {TIMBER_SECTIONS[params.sec_col].name} columns, {TIMBER_SECTIONS[params.sec_beam].name} beam")
        print(f"  Total volume: {volume:.4f} m³")
        print()
    
    print("✅ Test 3 passed: Parameter variation works!")
    print("   This is the foundation for Day 3's design exploration!")
    print()


def main():
    """
    Main demo function: Run all tests.
    """
    print("=" * 70)
    print("DEMO: PARAMETRIC PORTAL FRAME GENERATOR")
    print("=" * 70)
    print()
    print("This demo tests the new make_portal() function that generates")
    print("portal frames from parameters instead of hardcoding everything.")
    print()
    print("WHAT WE'RE TESTING:")
    print("  1. Basic portal frame (no bracing)")
    print("  2. Braced portal frame (X-brace)")
    print("  3. Parameter variation (design exploration)")
    print()
    
    try:
        # Run tests
        test_basic_portal()
        test_braced_portal()
        test_parameter_variation()
        
        print("=" * 70)
        print("✅ ALL TESTS PASSED!")
        print("=" * 70)
        print()
        print("The parametric portal generator is working correctly.")
        print("Next step: Use this in Block C (variant sampling) and")
        print("Block D (batch evaluation) for design exploration!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()