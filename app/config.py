# app/config.py
"""
Application configuration and defaults.
"""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class AppConfig:
    """Global application configuration."""
    
    # App metadata
    app_name: str = "StructureCraft"
    app_subtitle: str = "3D Spaceframe Design Explorer"
    version: str = "0.1.0"
    
    # Default parameter ranges for design brief
    width_range: Tuple[float, float] = (4.0, 20.0)
    depth_range: Tuple[float, float] = (4.0, 16.0)
    height_range: Tuple[float, float] = (1.5, 6.0)
    
    nx_range: Tuple[int, int] = (2, 10)
    ny_range: Tuple[int, int] = (2, 10)
    
    # Default values
    default_width: float = 10.0
    default_depth: float = 8.0
    default_min_height: float = 2.5
    default_max_height: float = 4.0
    default_nx: int = 5
    default_ny: int = 4
    
    # Exploration settings
    default_n_designs: int = 200
    max_n_designs: int = 500
    
    # Material defaults (steel)
    default_E: float = 210e9  # Pa
    default_A: float = 0.0008  # m^2
    default_gravity_load: float = -50000.0  # N
    
    # Available options
    heightfields: List[str] = None
    topologies: List[str] = None
    support_layouts: List[str] = None
    
    def __post_init__(self):
        if self.heightfields is None:
            self.heightfields = ['flat', 'paraboloid', 'ridge', 'saddle']
        if self.topologies is None:
            self.topologies = ['grid', 'diagrid', 'triangulated']
        if self.support_layouts is None:
            self.support_layouts = ['edges', 'corners', 'perimeter_4']


# Global config instance
CONFIG = AppConfig()




