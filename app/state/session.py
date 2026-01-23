# app/state/session.py
"""
Session state management for Streamlit.

Provides typed accessors for session state to avoid
scattered st.session_state['key'] calls throughout the app.
"""

import streamlit as st
import pandas as pd
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict


@dataclass
class DesignBrief:
    """User-defined design constraints and preferences."""
    # Geometry
    width: float = 10.0
    depth: float = 8.0
    nx: int = 5
    ny: int = 4
    min_height: float = 2.5
    max_height: float = 4.0
    
    # Shape and structure
    heightfield: str = 'paraboloid'
    topology: str = 'grid'
    support_layout: str = 'edges'
    
    # Material
    E: float = 210e9
    A: float = 0.0008
    
    # Loading
    gravity_load: float = -50000.0
    
    # Exploration settings
    n_designs: int = 200
    seed: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExplorationResults:
    """Results from batch exploration."""
    results_df: pd.DataFrame
    pareto_mask: pd.Series
    n_successful: int
    n_failed: int
    n_pareto: int


@dataclass
class SelectedDesign:
    """Currently selected design for detailed view/export."""
    index: int
    params: Dict[str, Any]
    metrics: Dict[str, Any]
    nodes: Dict = None
    bars: list = None
    forces: Dict = None


# ============================================================================
# Brief State
# ============================================================================

def get_brief() -> DesignBrief:
    """Get the current design brief from session state."""
    if 'brief' not in st.session_state:
        st.session_state.brief = DesignBrief()
    return st.session_state.brief


def set_brief(brief: DesignBrief) -> None:
    """Update the design brief in session state."""
    st.session_state.brief = brief
    # Clear exploration results when brief changes
    clear_exploration()


def update_brief(**kwargs) -> DesignBrief:
    """Update specific fields of the brief."""
    brief = get_brief()
    for key, value in kwargs.items():
        if hasattr(brief, key):
            setattr(brief, key, value)
    st.session_state.brief = brief
    return brief


# ============================================================================
# Exploration Results State
# ============================================================================

def get_exploration_results() -> Optional[ExplorationResults]:
    """Get exploration results from session state."""
    return st.session_state.get('exploration_results', None)


def set_exploration_results(results: ExplorationResults) -> None:
    """Store exploration results in session state."""
    st.session_state.exploration_results = results


def clear_exploration() -> None:
    """Clear exploration results (e.g., when brief changes)."""
    if 'exploration_results' in st.session_state:
        del st.session_state.exploration_results
    if 'selected_design' in st.session_state:
        del st.session_state.selected_design


# ============================================================================
# Selected Design State
# ============================================================================

def get_selected_design() -> Optional[SelectedDesign]:
    """Get currently selected design."""
    return st.session_state.get('selected_design', None)


def set_selected_design(design: SelectedDesign) -> None:
    """Set the selected design."""
    st.session_state.selected_design = design


def clear_selected_design() -> None:
    """Clear selected design."""
    if 'selected_design' in st.session_state:
        del st.session_state.selected_design


# ============================================================================
# Utility
# ============================================================================

def clear_all() -> None:
    """Clear all session state."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]


