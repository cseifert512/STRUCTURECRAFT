# mini_branch/catalog.py
"""Material and section property catalog for structural analysis."""

from dataclasses import dataclass

# Re-export timber capacities for convenience
from .checks.timber import TimberCapacity, DOUGLAS_FIR_CAPACITY, SPF_CAPACITY


@dataclass(frozen=True)
class Material:
    """Material properties."""
    name: str
    E: float           # Young's modulus (Pa)
    density: float     # kg/m³
    carbon_factor: float  # kg CO₂/m³


@dataclass(frozen=True)
class Section:
    """Cross-sectional properties."""
    name: str
    A: float   # Area (m²)
    I: float   # Moment of inertia (m⁴)
    S: float   # Section modulus (m³)


# Default material: Douglas Fir
DEFAULT_MATERIAL = Material(
    name="Douglas Fir",
    E=12e9,
    density=550.0,
    carbon_factor=300.0,
)

# Timber section catalog (nominal sizes with actual dimensions)
TIMBER_SECTIONS = [
    Section(name="2x4",   A=0.0034,  I=2.24e-6,  S=0.000050),   # 38x89 mm
    Section(name="2x6",   A=0.0052,  I=8.68e-6,  S=0.000124),   # 38x140 mm
    Section(name="2x8",   A=0.0070,  I=19.4e-6,  S=0.000211),   # 38x184 mm
    Section(name="2x10",  A=0.0089,  I=40.8e-6,  S=0.000347),   # 38x235 mm
    Section(name="2x12",  A=0.0108,  I=73.6e-6,  S=0.000515),   # 38x286 mm
    Section(name="4x8",   A=0.0140,  I=44.4e-6,  S=0.000483),   # 89x184 mm
    Section(name="4x10",  A=0.0178,  I=91.6e-6,  S=0.000779),   # 89x235 mm
    Section(name="4x12",  A=0.0216,  I=165.0e-6, S=0.001154),   # 89x286 mm
    Section(name="GL6x8", A=0.0140,  I=60.0e-6,  S=0.000652),   # Glulam
    Section(name="GL6x10",A=0.0178,  I=120.0e-6, S=0.001021),
    Section(name="GL6x12",A=0.0216,  I=220.0e-6, S=0.001538),
]
