"""
CATALOG: MATERIAL AND SECTION PROPERTIES
=========================================

PURPOSE:
--------
This module defines a catalog of material and section properties that we can
reference by index. Instead of hardcoding E=210e9, A=0.01, I=8e-6 in every demo,
we create a library of standard sections and materials.

WHY THIS MATTERS:
-----------------
1. **Design Exploration**: We can try different section combinations easily
   by just changing indices (sec_col=3, sec_beam=5) instead of recalculating
   properties every time.

2. **Volume/Carbon Calculation**: To compute material volume, we need A for each
   element. Having sections in a list makes this trivial: volume = Σ(sections[i].A * L_i)

3. **ML Features (Day 4)**: Section indices become features for machine learning.
   Instead of using raw A, I values (which have different scales), we use discrete
   indices (0, 1, 2, ...) which are easier for ML models to learn from.

4. **Real-World Practice**: Engineers use standard section catalogs (AISC, NDS)
   in real design. This mimics that workflow.

ENGINEERING CONTEXT:
--------------------
- **Material**: Defines the substance (steel, timber, concrete)
  - E: How stiff the material is (affects deflection)
  - Density: How heavy it is (affects self-weight)
  - Carbon factor: Environmental impact (for sustainability metrics)

- **Section**: Defines the cross-sectional shape and size
  - A: Cross-sectional area (affects axial strength, volume)
  - I: Moment of inertia (affects bending stiffness)
  - S: Section modulus (affects bending stress capacity)

For timber, typical sections might be:
- 2x4, 2x6, 2x8, 2x10, 2x12 (nominal sizes)
- Or metric: 38x89, 38x140, 38x184, 38x235, 38x286 mm

We'll use simplified properties that represent realistic timber sections.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    """
    Material properties for structural analysis.
    
    WHY FROZEN DATACLASS?
    ---------------------
    - frozen=True makes it immutable (can't change after creation)
    - This prevents bugs: once you create a Material, it stays the same
    - Also makes it hashable (can use as dict key if needed)
    
    Parameters:
    -----------
    name : str
        Human-readable name (e.g., "Douglas Fir", "Steel")
    
    E : float
        Young's modulus (Pa = N/m²)
        - Steel: ~210 GPa = 210e9 Pa
        - Timber: ~10-15 GPa = 10e9-15e9 Pa
        - This is the "stiffness" - how much force per unit strain
    
    density : float
        Material density (kg/m³)
        - Steel: ~7850 kg/m³
        - Timber: ~500-700 kg/m³
        - Used to compute self-weight: weight = density × volume
    
    carbon_factor : float
        Embodied carbon per unit volume (kg CO₂/m³)
        - This is a sustainability metric
        - Represents the carbon footprint of producing 1 m³ of material
        - Timber: ~200-400 kg CO₂/m³ (lower is better)
        - Steel: ~2000-3000 kg CO₂/m³ (much higher)
        - For Day 3, we'll use placeholder values, but this sets up Day 4's
          carbon optimization objectives
    """
    name: str
    E: float  # Young's modulus (Pa)
    density: float  # kg/m³
    carbon_factor: float  # kg CO₂/m³


@dataclass(frozen=True)
class Section:
    """
    Cross-sectional properties for a structural member.
    
    WHY THESE THREE PROPERTIES?
    ---------------------------
    - A (Area): Needed for axial force calculations and volume computation
    - I (Moment of Inertia): Needed for bending stiffness (deflection)
    - S (Section Modulus): Needed for stress checks (σ = M/S)
    
    For a rectangular section (typical timber):
    - If width = b, depth = d
    - A = b × d
    - I = b × d³ / 12  (about the weak axis)
    - S = I / (d/2) = b × d² / 6
    
    We store S directly because:
    1. It's commonly used in design codes
    2. We don't need to know the exact depth to compute it
    3. It's more convenient for stress calculations
    
    Parameters:
    -----------
    name : str
        Human-readable name (e.g., "2x8", "38x184")
    
    A : float
        Cross-sectional area (m²)
        - For a 2x8 (actual ~38x184 mm): A ≈ 0.007 m²
    
    I : float
        Moment of inertia (m⁴)
        - For bending about the weak axis (horizontal beam)
        - Larger I = stiffer in bending
    
    S : float
        Section modulus (m³)
        - S = I / c, where c = distance to extreme fiber
        - For rectangular: S = b × d² / 6
        - Used in stress checks: σ_max = M_max / S
    """
    name: str
    A: float  # Cross-sectional area (m²)
    I: float  # Moment of inertia (m⁴)
    S: float  # Section modulus (m³)


# ============================================================================
# MATERIAL DEFINITIONS
# ============================================================================

# Default material: Douglas Fir (common structural timber)
# WHY DOUGLAS FIR?
# - Common in North American construction
# - Good strength-to-weight ratio
# - Realistic E values for timber structures
DEFAULT_MATERIAL = Material(
    name="Douglas Fir",
    E=12e9,  # 12 GPa (typical for structural timber)
    density=550.0,  # kg/m³ (typical for Douglas Fir)
    carbon_factor=300.0,  # kg CO₂/m³ (placeholder, but realistic for timber)
)


# ============================================================================
# SECTION DEFINITIONS (Timber Sections)
# ============================================================================

# WHY 6-12 OPTIONS?
# -----------------
# - Enough variety to explore design space
# - Not so many that it's overwhelming
# - Represents realistic design choices
# - Indices 0-11 become ML features in Day 4

# We'll create sections representing common timber sizes
# Format: nominal size (2x4, 2x6, etc.) with realistic properties
# 
# ENGINEERING NOTE:
# - Nominal sizes (2x4, 2x6) are "before planing"
# - Actual sizes are smaller (e.g., 2x4 → 38x89 mm actual)
# - We'll use properties based on actual dimensions

TIMBER_SECTIONS = [
    # Smallest section (lightest, cheapest, but weaker)
    Section(
        name="2x4",
        A=0.0034,  # 38x89 mm = 0.003382 m²
        I=2.24e-6,  # I = b×d³/12 = 0.038 × 0.089³ / 12 ≈ 2.24e-6 m⁴
        S=0.000050,  # S = b×d²/6 = 0.038 × 0.089² / 6 ≈ 0.000050 m³
    ),
    
    # Slightly larger
    Section(
        name="2x6",
        A=0.0052,  # 38x140 mm
        I=8.68e-6,  # I = 0.038 × 0.140³ / 12
        S=0.000124,  # S = 0.038 × 0.140² / 6
    ),
    
    # Medium size (common for beams)
    Section(
        name="2x8",
        A=0.0070,  # 38x184 mm
        I=19.4e-6,  # I = 0.038 × 0.184³ / 12
        S=0.000211,  # S = 0.038 × 0.184² / 6
    ),
    
    # Larger beam
    Section(
        name="2x10",
        A=0.0089,  # 38x235 mm
        I=40.8e-6,  # I = 0.038 × 0.235³ / 12
        S=0.000347,  # S = 0.038 × 0.235² / 6
    ),
    
    # Even larger
    Section(
        name="2x12",
        A=0.0108,  # 38x286 mm
        I=73.6e-6,  # I = 0.038 × 0.286³ / 12
        S=0.000515,  # S = 0.038 × 0.286² / 6
    ),
    
    # Extra large (for heavy loads)
    Section(
        name="4x8",
        A=0.0140,  # 89x184 mm (double width)
        I=44.4e-6,  # I = 0.089 × 0.184³ / 12
        S=0.000483,  # S = 0.089 × 0.184² / 6
    ),
    
    # Very large
    Section(
        name="4x10",
        A=0.0178,  # 89x235 mm
        I=91.6e-6,  # I = 0.089 × 0.235³ / 12
        S=0.000779,  # S = 0.089 × 0.235² / 6
    ),
    
    # Extra large
    Section(
        name="4x12",
        A=0.0216,  # 89x286 mm
        I=165.0e-6,  # I = 0.089 × 0.286³ / 12
        S=0.001154,  # S = 0.089 × 0.286² / 6
    ),
    
    # Glulam/engineered sections (stronger, stiffer)
    Section(
        name="GL6x8",
        A=0.0140,  # Similar area to 4x8
        I=60.0e-6,  # But higher I (engineered)
        S=0.000652,  # Better section modulus
    ),
    
    Section(
        name="GL6x10",
        A=0.0178,
        I=120.0e-6,
        S=0.001021,
    ),
    
    Section(
        name="GL6x12",
        A=0.0216,
        I=220.0e-6,
        S=0.001538,
    ),
]


# ============================================================================
# USAGE EXAMPLES (for reference, not executed)
# ============================================================================
"""
HOW TO USE THIS CATALOG:
------------------------

# Import the catalog
from mini_branch.catalog import TIMBER_SECTIONS, DEFAULT_MATERIAL

# Get a section by index
sec_col = TIMBER_SECTIONS[3]  # 2x10 section
print(sec_col.A)  # 0.0089 m²
print(sec_col.I)  # 40.8e-6 m⁴

# Use in element creation
element = Frame2D(
    id=0,
    ni=0,
    nj=1,
    E=DEFAULT_MATERIAL.E,
    A=sec_col.A,
    I=sec_col.I,
)

# Compute volume
volume = sec_col.A * element_length  # m³

# Compute carbon
carbon = volume * DEFAULT_MATERIAL.carbon_factor  # kg CO₂

# In Day 3's explore.py, we'll do:
# sec_col_idx = 3  # Just an integer!
# sec_col = TIMBER_SECTIONS[sec_col_idx]
# This makes it easy to try different combinations.
"""