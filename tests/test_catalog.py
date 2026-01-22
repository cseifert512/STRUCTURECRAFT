# File: tests/test_catalog.py
"""
Test the catalog.py module to verify material and section definitions work correctly.
"""

import pytest
from mini_branch.catalog import Material, Section, TIMBER_SECTIONS, DEFAULT_MATERIAL


def test_material_creation():
    """
    Test that we can create a Material and access its properties.
    """
    mat = Material(
        name="Test Material",
        E=10e9,
        density=500.0,
        carbon_factor=200.0
    )
    
    # Check properties are accessible
    assert mat.name == "Test Material"
    assert mat.E == 10e9
    assert mat.density == 500.0
    assert mat.carbon_factor == 200.0
    
    # Check it's frozen (immutable)
    with pytest.raises(Exception):  # dataclass.FrozenInstanceError
        mat.E = 20e9  # Should fail!
    
    print("✓ Material creation works")


def test_section_creation():
    """
    Test that we can create a Section and access its properties.
    """
    sec = Section(
        name="2x4",
        A=0.0034,
        I=2.24e-6,
        S=0.000050
    )
    
    assert sec.name == "2x4"
    assert sec.A == 0.0034
    assert sec.I == 2.24e-6
    assert sec.S == 0.000050
    
    print("✓ Section creation works")


def test_default_material():
    """
    Test that DEFAULT_MATERIAL has reasonable values.
    """
    # Check it's a Material instance
    assert isinstance(DEFAULT_MATERIAL, Material)
    
    # Check properties are positive and reasonable
    assert DEFAULT_MATERIAL.E > 0
    assert DEFAULT_MATERIAL.density > 0
    assert DEFAULT_MATERIAL.carbon_factor > 0
    
    # Check E is in reasonable range for timber (10-15 GPa)
    assert 10e9 <= DEFAULT_MATERIAL.E <= 15e9, \
        f"E should be 10-15 GPa for timber, got {DEFAULT_MATERIAL.E/1e9} GPa"
    
    print(f"✓ DEFAULT_MATERIAL: {DEFAULT_MATERIAL.name}, E={DEFAULT_MATERIAL.E/1e9:.1f} GPa")


def test_timber_sections_list():
    """
    Test that TIMBER_SECTIONS is a list with reasonable sections.
    """
    # Check it's a list
    assert isinstance(TIMBER_SECTIONS, list)
    
    # Check it has sections (at least 6, as per plan)
    assert len(TIMBER_SECTIONS) >= 6, \
        f"Should have at least 6 sections, got {len(TIMBER_SECTIONS)}"
    
    # Check each item is a Section
    for i, sec in enumerate(TIMBER_SECTIONS):
        assert isinstance(sec, Section), \
            f"Item {i} is not a Section: {type(sec)}"
        
        # Check properties are positive
        assert sec.A > 0, f"Section {i} ({sec.name}) has non-positive A"
        assert sec.I > 0, f"Section {i} ({sec.name}) has non-positive I"
        assert sec.S > 0, f"Section {i} ({sec.name}) has non-positive S"
    
    print(f"✓ TIMBER_SECTIONS has {len(TIMBER_SECTIONS)} valid sections")


def test_section_indexing():
    """
    Test that we can index into TIMBER_SECTIONS and use the sections.
    """
    # Get a section by index
    sec_0 = TIMBER_SECTIONS[0]
    sec_3 = TIMBER_SECTIONS[3]
    
    # Check they're different
    assert sec_0.name != sec_3.name
    
    # Check properties increase (generally, larger indices = larger sections)
    # (This might not always be true, but for our catalog it should be)
    assert sec_3.A >= sec_0.A, "Larger index should generally have larger area"
    
    print(f"✓ Section indexing works: {sec_0.name} (A={sec_0.A:.4f}) vs {sec_3.name} (A={sec_3.A:.4f})")


def test_section_ordering():
    """
    Test that sections are roughly ordered by size (for sanity).
    This isn't required, but it's nice if they're ordered.
    """
    areas = [sec.A for sec in TIMBER_SECTIONS]
    
    # Check that areas generally increase (allow some variation)
    # We'll check that the last section is larger than the first
    assert areas[-1] > areas[0], \
        f"Last section should be larger than first: {areas[-1]} vs {areas[0]}"
    
    print("✓ Sections are reasonably ordered")


def test_volume_calculation():
    """
    Test that we can use sections to calculate volume (Day 3 requirement).
    """
    # Simulate an element with length 5.0 m using section index 2
    sec = TIMBER_SECTIONS[2]
    length = 5.0
    
    # Volume = A × L
    volume = sec.A * length
    
    # Check it's positive and reasonable
    assert volume > 0
    assert volume == pytest.approx(0.0070 * 5.0, rel=1e-6)
    
    print(f"✓ Volume calculation works: {volume:.4f} m³ for {sec.name} × {length} m")


def test_carbon_calculation():
    """
    Test that we can use material and section to calculate carbon (Day 3 requirement).
    """
    sec = TIMBER_SECTIONS[3]
    length = 6.0
    
    # Volume = A × L
    volume = sec.A * length
    
    # Carbon = volume × carbon_factor
    carbon = volume * DEFAULT_MATERIAL.carbon_factor
    
    # Check it's positive and reasonable
    assert carbon > 0
    
    print(f"✓ Carbon calculation works: {carbon:.2f} kg CO₂ for {sec.name} × {length} m")


if __name__ == "__main__":
    # Run tests manually (or use pytest)
    test_material_creation()
    test_section_creation()
    test_default_material()
    test_timber_sections_list()
    test_section_indexing()
    test_section_ordering()
    test_volume_calculation()
    test_carbon_calculation()
    print("\n✅ All catalog tests passed!")