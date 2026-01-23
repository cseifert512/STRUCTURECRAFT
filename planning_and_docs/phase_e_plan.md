# Phase E: Streamlit Product Interface - Design Specification

## Executive Summary

Phase E is not just "add a Streamlit app." It's building the **product surface** that transforms our structural engine into a tool designers actually want to use. This document defines the architecture, UX philosophy, and implementation roadmap.

---

## 1. Product Vision

### What We're Building
A **Design Explorer** that lets architects and engineers:
- Define a design brief (constraints + preferences)
- Explore the solution space visually
- Understand trade-offs between objectives
- Select and export production-ready designs

### What We're NOT Building (Yet)
- Full CAD integration (Rhino, Revit)
- Team collaboration features
- Cloud compute for large batches
- Design history / versioning

### The One-Liner
> "Give me a light, open canopy that fits this footprint, and show me my options."

---

## 2. Architecture Principles

### 2.1 Separation of Concerns

```
┌─────────────────────────────────────────────────────────────┐
│                    STREAMLIT UI LAYER                       │
│  (pages, components, session state, layout)                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    SERVICE LAYER                            │
│  (DesignService, ExplorationService, ExportService)         │
│  - Orchestrates workflows                                   │
│  - Manages caching                                          │
│  - Handles async operations                                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    ENGINE LAYER                             │
│  (mini_branch core: generative, v3d, kernel, explore3d)     │
│  - Pure computation, no UI knowledge                        │
│  - Stateless functions                                      │
└─────────────────────────────────────────────────────────────┘
```

**Why this matters:**
- Engine can be used headless (CLI, API, notebooks)
- Services can be tested without UI
- UI can be reskinned without touching logic

### 2.2 File Structure

```
app/
├── __init__.py
├── main.py                    # Entry point: streamlit run app/main.py
├── config.py                  # App configuration, defaults
│
├── pages/                     # Multi-page app structure
│   ├── 1_🏠_Home.py          # Landing / project overview
│   ├── 2_📐_Design_Brief.py  # Define constraints & preferences
│   ├── 3_🔍_Explore.py       # Batch exploration & Pareto
│   ├── 4_🎯_Select.py        # Compare & select designs
│   └── 5_📦_Export.py        # Export selected design
│
├── components/               # Reusable UI components
│   ├── __init__.py
│   ├── sidebar.py            # Shared sidebar (project info, nav)
│   ├── design_card.py        # Single design summary card
│   ├── pareto_plot.py        # Interactive Pareto visualization
│   ├── model_viewer.py       # 3D model viewer (Plotly)
│   ├── metrics_panel.py      # Metrics display with status indicators
│   └── parameter_inputs.py   # Consistent parameter input widgets
│
├── services/                 # Business logic layer
│   ├── __init__.py
│   ├── design_service.py     # Single design generation & analysis
│   ├── exploration_service.py # Batch exploration management
│   ├── export_service.py     # File export (CSV, JSON, HTML)
│   └── cache_service.py      # Result caching
│
└── state/                    # Session state management
    ├── __init__.py
    └── session.py            # Typed session state accessors
```

### 2.3 State Management

Streamlit reruns on every interaction. We need disciplined state management:

```python
# state/session.py
from dataclasses import dataclass
from typing import Optional
import streamlit as st
import pandas as pd

@dataclass
class ProjectState:
    """Current project configuration."""
    name: str = "Untitled Project"
    brief: Optional[dict] = None  # Design brief parameters
    
@dataclass  
class ExplorationState:
    """Exploration results."""
    results_df: Optional[pd.DataFrame] = None
    pareto_mask: Optional[pd.Series] = None
    selected_indices: list = None
    
@dataclass
class SelectionState:
    """Selected design for export."""
    design_index: Optional[int] = None
    design_params: Optional[dict] = None
    design_metrics: Optional[dict] = None

def get_project() -> ProjectState:
    if 'project' not in st.session_state:
        st.session_state.project = ProjectState()
    return st.session_state.project

def get_exploration() -> ExplorationState:
    if 'exploration' not in st.session_state:
        st.session_state.exploration = ExplorationState()
    return st.session_state.exploration
```

---

## 3. User Experience Design

### 3.1 User Journey

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   BRIEF      │ ──► │   EXPLORE    │ ──► │   SELECT     │ ──► │   EXPORT     │
│              │     │              │     │              │     │              │
│ Define what  │     │ See options  │     │ Compare &    │     │ Get files    │
│ you want     │     │ & trade-offs │     │ choose       │     │ for fab      │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

### 3.2 Page Specifications

#### Page 1: Home
**Purpose:** Project context and quick start
**Elements:**
- Project name (editable)
- Quick stats if exploration exists (N designs, N Pareto)
- "Start New Brief" button
- Recent/example projects (future)

#### Page 2: Design Brief
**Purpose:** Define constraints and preferences
**Sections:**

```
┌─────────────────────────────────────────────────────────────┐
│ FOOTPRINT                                                   │
│ ┌─────────────────┐ ┌─────────────────┐                    │
│ │ Width: [6-15] m │ │ Depth: [5-12] m │                    │
│ └─────────────────┘ └─────────────────┘                    │
│ □ Lock aspect ratio                                         │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ HEIGHT & SHAPE                                              │
│ Min height: [__] m    Max height: [__] m                   │
│ Heightfield: [flat ▼] [paraboloid] [ridge] [saddle]        │
│ Preview: [small shape diagram]                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ STRUCTURE                                                   │
│ Topology: [grid ▼] [diagrid] [triangulated]                │
│ Grid density: [3-8] x [3-7]                                │
│ Support layout: [edges ▼] [corners] [perimeter]            │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ CONSTRAINTS                                                 │
│ Max displacement: [__] mm                                   │
│ Max member length: [__] m (shipping)                       │
│ Max unique lengths: [__] bins                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ PRIORITIES (drag to rank)                                   │
│ 1. ████████ Minimize material                              │
│ 2. ██████   Maximize stiffness                             │
│ 3. ████     Simplify fabrication                           │
└─────────────────────────────────────────────────────────────┘

[Preview Single Design]  [Run Exploration (200 variants)]
```

#### Page 3: Explore
**Purpose:** Visualize solution space and Pareto frontier
**Layout:**

```
┌────────────────────────────┬────────────────────────────────┐
│ PARETO PLOT               │ SELECTED DESIGN                │
│                           │                                │
│ [Interactive scatter]     │ [3D Model Viewer]              │
│ - Click to select         │                                │
│ - Hover for details       │ Metrics:                       │
│                           │ - Volume: 0.053 m³             │
│ Axes: [Volume ▼] vs       │ - Disp: 1.85 mm               │
│       [Displacement ▼]    │ - Bins: 5                      │
│                           │                                │
│ Show: ○ All ● Pareto only │ [Add to Comparison]            │
└────────────────────────────┴────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ FILTER & SORT                                               │
│ Volume: [0.05 ───●─── 0.50]  Disp: [0 ───●─── 100] mm     │
│ Topology: [x] grid [x] diagrid [x] triangulated            │
│ Sort by: [Score ▼]                                          │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ RESULTS TABLE (filtered)                      [Export CSV]  │
│ ┌────┬────────┬──────┬──────┬─────────────┬─────────┐      │
│ │ #  │ Volume │ Disp │ Bins │ Topology    │ Pareto? │      │
│ ├────┼────────┼──────┼──────┼─────────────┼─────────┤      │
│ │ 1  │ 0.053  │ 1.85 │ 5    │ triangulated│ ✓       │      │
│ │ 2  │ 0.064  │ 0.79 │ 7    │ grid        │ ✓       │      │
│ └────┴────────┴──────┴──────┴─────────────┴─────────┘      │
└─────────────────────────────────────────────────────────────┘
```

#### Page 4: Select
**Purpose:** Side-by-side comparison of shortlisted designs
**Features:**
- Compare up to 4 designs side-by-side
- 3D views synchronized (same rotation)
- Metrics comparison table
- "Winner" selection

#### Page 5: Export
**Purpose:** Generate production-ready outputs
**Exports:**
- Cut list (CSV)
- Model data (JSON)
- 3D visualization (HTML)
- Summary report (PDF - future)
- DXF/STEP (future)

---

## 4. Implementation Roadmap

### Phase E.1: Foundation (MVP)
**Goal:** Working app with core flow
**Scope:**
- [ ] App skeleton with multi-page structure
- [ ] Design Brief page with basic inputs
- [ ] Single design preview (not batch)
- [ ] Basic 3D viewer
- [ ] Export cut list

**Timeline:** 2-3 hours

### Phase E.2: Exploration Integration
**Goal:** Batch exploration in UI
**Scope:**
- [ ] "Run Exploration" with progress indicator
- [ ] Results caching (don't recompute on page switch)
- [ ] Pareto plot (interactive Plotly)
- [ ] Click-to-select design
- [ ] Results table with filtering

**Timeline:** 2-3 hours

### Phase E.3: Selection & Comparison
**Goal:** Design comparison workflow
**Scope:**
- [ ] Shortlist management
- [ ] Side-by-side comparison
- [ ] Final selection
- [ ] Export all formats

**Timeline:** 1-2 hours

### Phase E.4: Polish & Delight
**Goal:** Production-ready UX
**Scope:**
- [ ] Loading states and progress
- [ ] Error handling with helpful messages
- [ ] Tooltips and help text
- [ ] Responsive layout
- [ ] Keyboard shortcuts
- [ ] Save/load project state (JSON)

**Timeline:** 2 hours

---

## 5. Future Extensibility

### 5.1 Structure Types (Post-MVP)
The UI should be designed to support multiple structure types:
- Canopy/spaceframe (current)
- Portal frames (existing 2D engine)
- Trusses
- Cable structures

**Architecture implication:** Abstract "DesignBrief" and "DesignResult" interfaces.

### 5.2 Custom Objectives (Post-MVP)
Users may want different optimization targets:
- Carbon footprint
- Cost estimate
- Assembly time
- Visual openness

**Architecture implication:** Pluggable metrics system.

### 5.3 Collaboration (Future)
- Share designs via URL
- Team projects
- Comment/annotation

**Architecture implication:** Consider export formats that support round-tripping.

### 5.4 Compute Scaling (Future)
For large explorations (1000+ designs):
- Background job queue
- Progress streaming
- Incremental results display

**Architecture implication:** Async service layer.

---

## 6. Technical Decisions

### 6.1 Streamlit Version
Use Streamlit >= 1.28 for:
- `st.fragment` (partial reruns)
- Better caching (`st.cache_data`, `st.cache_resource`)
- Multi-page app improvements

### 6.2 Plotting Library
- **Plotly** for 3D and interactive plots (already using)
- Consistent with existing viz code

### 6.3 Styling
- Use Streamlit's native theming (config.toml)
- Custom CSS only for specific components
- Dark mode support

### 6.4 Testing
- Services layer: unit tests with pytest
- UI: manual testing (Streamlit test framework immature)
- Smoke test: `streamlit run app/main.py --server.headless true`

---

## 7. Success Criteria

### MVP (Phase E.1-E.2)
- [ ] User can define a brief and see 200 designs explored
- [ ] Pareto frontier is visible and interactive
- [ ] User can select a design and export cut list
- [ ] App runs without errors for 10 minutes of use

### Production-Ready (Phase E.3-E.4)
- [ ] Complete workflow in < 5 clicks
- [ ] No confusing states or dead ends
- [ ] All exports work correctly
- [ ] Loads in < 3 seconds
- [ ] Works on 1920x1080 and 1366x768 screens

---

## 8. Getting Started

### Prerequisites
```bash
pip install streamlit>=1.28 plotly pandas numpy
```

### Run Development
```bash
streamlit run app/main.py --server.runOnSave true
```

### Project Structure Creation
```bash
mkdir -p app/pages app/components app/services app/state
touch app/__init__.py app/main.py app/config.py
touch app/pages/__init__.py
touch app/components/__init__.py
touch app/services/__init__.py
touch app/state/__init__.py
```

---

## 9. Open Questions

1. **Project persistence:** Save to local JSON file? SQLite? Just session state?
2. **Exploration size:** Default to 200? Let user choose? Auto-scale based on brief complexity?
3. **Real-time preview:** Should brief changes trigger instant single-design preview?
4. **Mobile:** Support mobile browsers or desktop-only?

---

## Appendix: Component API Sketches

### design_card.py
```python
def design_card(
    design_id: int,
    params: dict,
    metrics: dict,
    is_selected: bool = False,
    on_select: callable = None,
) -> None:
    """Render a compact design summary card."""
```

### pareto_plot.py
```python
def pareto_plot(
    df: pd.DataFrame,
    pareto_mask: pd.Series,
    x_col: str = 'volume',
    y_col: str = 'max_displacement',
    selected_index: int = None,
    on_select: callable = None,
) -> go.Figure:
    """Create interactive Pareto scatter plot."""
```

### model_viewer.py
```python
def model_viewer(
    nodes: dict,
    bars: list,
    forces: dict = None,
    height: int = 400,
) -> go.Figure:
    """Render 3D structure viewer."""
```

