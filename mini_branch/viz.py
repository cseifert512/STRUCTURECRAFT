"""
VISUALIZATION: PLOTTING DESIGN RESULTS
=======================================

PURPOSE:
--------
This module creates visualizations of design exploration results. Visualizations
are critical for understanding trade-offs, identifying optimal designs, and
communicating results to stakeholders (engineers, clients, managers).

WHY THIS MATTERS:
-----------------
1. **Human Understanding**: Numbers in a CSV are hard to interpret. A plot
   instantly shows patterns, trends, and outliers.

2. **Pareto Frontier Visualization**: The Pareto plot shows the "efficient
   frontier" - the boundary of optimal designs. Engineers can see the trade-off
   curve and choose based on priorities.

3. **Deformed Shape Visualization**: Shows how the structure actually moves
   under load. This helps engineers understand structural behavior and verify
   that results make physical sense.

4. **Communication**: Plots are essential for reports, presentations, and
   design reviews. A good plot tells a story.

ENGINEERING CONTEXT:
--------------------
- **Pareto Plot**: Shows volume vs drift trade-off. Lower-left corner = best
  (low volume AND low drift). The frontier shows the best possible trade-offs.

- **Deformed Shape**: Shows structural response. Engineers can verify:
  - Does it bend in the expected direction?
  - Are deflections reasonable?
  - Does the structure look stable?
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, Polygon, Circle
from matplotlib.lines import Line2D
import matplotlib.patheffects as path_effects
from typing import Dict, List, Optional, Tuple
import os

from .model import Node, Frame2D
from .elements import element_geometry
from .assembly import DOF_PER_NODE, dof_index

# =============================================================================
# ARCHITECT-FRIENDLY COLOR PALETTE
# =============================================================================
# A sophisticated color scheme inspired by architectural presentation drawings

COLORS = {
    # Structure colors
    'structure_primary': '#2C3E50',      # Dark blue-gray (undeformed)
    'structure_secondary': '#E74C3C',    # Coral red (deformed)
    'structure_accent': '#3498DB',       # Sky blue (highlights)
    
    # Background and grid
    'background': '#FAFAFA',             # Off-white
    'grid': '#E0E0E0',                   # Light gray
    'ground': '#8B7355',                 # Earth brown
    
    # Loads and forces
    'load_gravity': '#9B59B6',           # Purple (gravity/UDL)
    'load_lateral': '#E67E22',           # Orange (wind/lateral)
    'reaction': '#27AE60',               # Green (reactions)
    
    # Annotations
    'dimension': '#7F8C8D',              # Gray
    'text': '#2C3E50',                   # Dark blue-gray
    
    # Pareto chart
    'pareto_optimal': '#E74C3C',         # Coral red
    'pareto_dominated': '#BDC3C7',       # Silver gray
    'pareto_highlight': '#F39C12',       # Golden yellow
}

# Font settings for architectural drawings
FONT_TITLE = {'family': 'sans-serif', 'weight': 'bold', 'size': 14}
FONT_LABEL = {'family': 'sans-serif', 'weight': 'normal', 'size': 11}
FONT_ANNOTATION = {'family': 'sans-serif', 'weight': 'normal', 'size': 9}


def plot_pareto(
    df,
    mask,
    outpath: str,
    x: str = "volume",
    y: str = "drift",
    title: str = "Pareto Frontier: Volume vs Drift",
) -> None:
    """
    Create a scatter plot showing the Pareto frontier of designs.
    
    WHY THIS FUNCTION?
    ------------------
    After running a design search, we have hundreds of designs. We need to
    visualize them to understand the trade-offs. This function creates a
    scatter plot where:
    - Each point is one design
    - X-axis = volume (material)
    - Y-axis = drift (stiffness)
    - Pareto-optimal designs are highlighted
    - The frontier shows the best possible trade-offs
    
    HOW IT WORKS (PSEUDOCODE):
    --------------------------
    
    STEP 1: UNDERSTAND THE DATA
    ---------------------------
    We have:
    - df: DataFrame with all designs (columns: volume, drift, ok, etc.)
    - mask: Boolean Series (True for Pareto-optimal designs)
    - x, y: Column names for the two objectives
    
    STEP 2: FILTER DATA
    -------------------
    - Only plot successful designs (ok=True)
    - Separate Pareto designs from dominated designs
    - This makes the plot clearer
    
    STEP 3: CREATE THE PLOT
    -----------------------
    - Scatter plot: all designs (gray, small)
    - Highlight Pareto designs (red, larger, with edge)
    - Add labels, grid, legend
    - Save to file
    
    STEP 4: INTERPRETATION
    ----------------------
    The plot shows:
    - Lower-left = best (low volume AND low drift)
    - Upper-right = worst (high volume AND high drift)
    - Pareto frontier = boundary of optimal designs
    - Moving along frontier = trade-off (improve one, worsen other)
    
    ENGINEERING INTERPRETATION:
    ---------------------------
    - **Ideal design**: Bottom-left corner (low volume, low drift)
    - **Pareto frontier**: The curve connecting optimal designs
    - **Trade-off**: Can't improve both objectives simultaneously
    - **Design choice**: Engineer picks based on priorities
      - "I care more about cost" → choose low volume
      - "I care more about stiffness" → choose low drift
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with design results
        Must have columns: x, y, ok (at minimum)
    
    mask : pd.Series
        Boolean mask: True for Pareto-optimal designs
        Same length as df, can be used as: df[mask]
    
    outpath : str
        File path to save the plot (e.g., "artifacts/pareto.png")
        - Will create directory if needed
        - Supports .png, .pdf, .svg formats
    
    x : str
        Column name for x-axis (default: "volume")
        - Represents first objective (material/cost)
    
    y : str
        Column name for y-axis (default: "drift")
        - Represents second objective (stiffness/performance)
    
    title : str
        Plot title (default: "Pareto Frontier: Volume vs Drift")
    
    Returns:
    --------
    None
        Saves plot to file (doesn't display)
    
    Example:
    --------
    >>> from mini_branch.pareto import pareto_mask
    >>> mask = pareto_mask(df)
    >>> plot_pareto(df, mask, "artifacts/pareto.png")
    """
    
    # ========================================================================
    # STEP 1: FILTER TO SUCCESSFUL DESIGNS ONLY
    # ========================================================================
    """
    WHY FILTER?
    -----------
    - Failed designs (ok=False) have NaN for metrics
    - We can't plot NaN values (they're not numbers)
    - Only successful designs are meaningful for comparison
    """
    
    # Only plot designs that solved successfully
    successful = df[df['ok'] == True].copy()
    
    if len(successful) == 0:
        raise ValueError("No successful designs to plot (all ok=False)")
    
    # Separate Pareto designs from dominated designs
    pareto_designs = successful[mask[successful.index]]
    dominated_designs = successful[~mask[successful.index]]
    
    # ========================================================================
    # STEP 2: EXTRACT DATA FOR PLOTTING
    # ========================================================================
    """
    WHY EXTRACT TO ARRAYS?
    ----------------------
    - Faster plotting with NumPy arrays
    - Easier to handle missing values
    - Cleaner code
    """
    
    # Pareto designs
    x_pareto = pareto_designs[x].values
    y_pareto = pareto_designs[y].values
    
    # Dominated designs
    x_dominated = dominated_designs[x].values
    y_dominated = dominated_designs[y].values
    
    # ========================================================================
    # STEP 3: CREATE THE FIGURE
    # ========================================================================
    """
    WHY MATPLOTLIB?
    ---------------
    - Standard Python plotting library
    - Good control over appearance
    - Easy to save to file
    - Works well with pandas DataFrames
    """
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # ========================================================================
    # STEP 4: PLOT DOMINATED DESIGNS (BACKGROUND)
    # ========================================================================
    """
    WHY PLOT DOMINATED DESIGNS?
    ----------------------------
    - Shows the full design space (not just optimal)
    - Helps understand where Pareto designs sit
    - Shows how many designs were explored
    - Makes Pareto designs stand out by contrast
    """
    
    if len(dominated_designs) > 0:
        ax.scatter(
            x_dominated, y_dominated,
            c='lightgray',
            s=30,  # Small points
            alpha=0.5,
            label=f'Dominated designs ({len(dominated_designs)})',
            edgecolors='none',
            zorder=1,  # Behind Pareto points
        )
    
    # ========================================================================
    # STEP 5: PLOT PARETO DESIGNS (HIGHLIGHTED)
    # ========================================================================
    """
    WHY HIGHLIGHT PARETO DESIGNS?
    ------------------------------
    - These are the designs worth considering
    - Make them visually prominent
    - Use color, size, and edge to stand out
    - These are the "winners" of the optimization
    """
    
    if len(pareto_designs) > 0:
        ax.scatter(
            x_pareto, y_pareto,
            c='red',
            s=100,  # Larger points
            alpha=0.8,
            label=f'Pareto-optimal ({len(pareto_designs)})',
            edgecolors='darkred',
            linewidths=2,
            zorder=2,  # In front of dominated points
        )
    
    # ========================================================================
    # STEP 6: ADD FORMATTING AND LABELS
    # ========================================================================
    """
    WHY FORMATTING MATTERS?
    ------------------------
    - Clear labels help interpretation
    - Grid makes it easier to read values
    - Title explains what the plot shows
    - Legend identifies what each color means
    """
    
    ax.set_xlabel(x.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    ax.set_ylabel(y.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    
    # ========================================================================
    # STEP 7: ADD ANNOTATIONS (OPTIONAL)
    # ========================================================================
    """
    WHY ANNOTATIONS?
    ----------------
    - Help identify key designs
    - Show design counts
    - Add context (e.g., "Lower is better")
    """
    
    # Add text box with summary
    textstr = f'Total designs: {len(successful)}\n'
    textstr += f'Pareto-optimal: {len(pareto_designs)}\n'
    textstr += f'Dominated: {len(dominated_designs)}\n'
    textstr += '\nLower values are better'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', bbox=props)
    
    # ========================================================================
    # STEP 8: SAVE THE PLOT
    # ========================================================================
    """
    WHY SAVE TO FILE?
    -----------------
    - Can be included in reports
    - Can be shared with others
    - Can be version controlled
    - Doesn't require display (works on servers)
    """
    
    import os
    os.makedirs(os.path.dirname(outpath) if os.path.dirname(outpath) else '.', exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()  # Close to free memory
    
    print(f"Pareto plot saved to: {outpath}")


def plot_frame_deformed(
    nodes: Dict[int, Node],
    elements: List[Frame2D],
    d: np.ndarray,
    outpath: str,
    scale: float = 50.0,
    title: str = "Portal Frame: Deformed Shape",
) -> None:
    """
    Plot a portal frame showing both undeformed and deformed shapes.
    
    WHY THIS FUNCTION?
    ------------------
    After solving a structure, we want to visualize how it deforms under load.
    This helps engineers:
    - Verify results make physical sense
    - Understand structural behavior
    - Communicate findings to others
    - Identify potential issues (excessive deflections, unexpected shapes)
    
    HOW IT WORKS (PSEUDOCODE):
    --------------------------
    
    STEP 1: EXTRACT COORDINATES
    ---------------------------
    For each node:
        x_undeformed = node.x (original position)
        y_undeformed = node.y (original position)
        ux = displacement in x direction (from d vector)
        uy = displacement in y direction (from d vector)
        x_deformed = x_undeformed + scale × ux
        y_deformed = y_undeformed + scale × uy
    
    STEP 2: DRAW ELEMENTS
    ---------------------
    For each element (connects node i to node j):
        Draw line from (xi, yi) to (xj, yj) for undeformed
        Draw line from (xi_def, yi_def) to (xj_def, yj_def) for deformed
    
    STEP 3: ADD DETAILS
    -------------------
    - Plot nodes (circles)
    - Plot supports (triangles at base)
    - Add labels, grid, legend
    - Save to file
    
    STEP 4: SCALING
    ---------------
    Real deflections are tiny (millimeters). We scale them up visually
    (e.g., ×50 or ×100) so we can see the deformation. This is called
    "exaggerated deformation" - it's for visualization only, not the
    actual magnitude.
    
    ENGINEERING INTERPRETATION:
    ---------------------------
    - **Undeformed shape**: Original structure (before loading)
    - **Deformed shape**: Structure after loading (how it actually moves)
    - **Scale factor**: How much we exaggerate deflections for visibility
    - **Physical meaning**: 
      - Beam sags down (gravity load)
      - Frame sways to the right (lateral load)
      - Columns bend (combined axial + bending)
    
    Parameters:
    -----------
    nodes : Dict[int, Node]
        Dictionary mapping node IDs to Node objects
        Example: {0: Node(0, 0.0, 0.0), 1: Node(1, 0.0, 3.0), ...}
    
    elements : List[Frame2D]
        List of frame elements
        Example: [Frame2D(0, 0, 1, E=..., A=..., I=...), ...]
    
    d : np.ndarray
        Global displacement vector from solve_linear
        Shape: (ndof,) where ndof = 3 × number_of_nodes
        Contains: [ux_0, uy_0, rz_0, ux_1, uy_1, rz_1, ...]
    
    outpath : str
        File path to save the plot (e.g., "artifacts/winner_frame.png")
        - Will create directory if needed
        - Supports .png, .pdf, .svg formats
    
    scale : float
        Scale factor for deformation visualization (default: 50.0)
        - Real deflections are tiny (millimeters)
        - Scale them up so we can see them
        - Example: scale=50 means 1mm deflection → 50mm on plot
        - This is for visualization only (not actual magnitude)
    
    title : str
        Plot title (default: "Portal Frame: Deformed Shape")
    
    Returns:
    --------
    None
        Saves plot to file (doesn't display)
    
    Example:
    --------
    >>> nodes, elements, fixed, udls, loads = make_portal(params, material, sections)
    >>> K = assemble_global_K(nodes, elements)
    >>> F = assemble_element_loads_global(nodes, elements, udls)
    >>> d, R, _ = solve_linear(K, F, fixed)
    >>> plot_frame_deformed(nodes, elements, d, "artifacts/frame.png")
    """
    
    # ========================================================================
    # STEP 1: EXTRACT UNDEFORMED COORDINATES
    # ========================================================================
    """
    WHY EXTRACT COORDINATES?
    ------------------------
    - Need node positions to draw elements
    - Store in arrays for easy plotting
    - Separate x and y for matplotlib
    """
    
    node_ids = sorted(nodes.keys())
    xs_undeformed = np.array([nodes[i].x for i in node_ids])
    ys_undeformed = np.array([nodes[i].y for i in node_ids])
    
    # ========================================================================
    # STEP 2: EXTRACT DISPLACEMENTS
    # ========================================================================
    """
    WHY EXTRACT DISPLACEMENTS?
    -------------------------
    - Displacements are stored in d vector
    - Need to map from global DOF index to node
    - Extract ux and uy for each node
    - rz (rotation) is not used for plotting (only translation matters)
    """
    
    ux = np.array([d[dof_index(node_id, 0)] for node_id in node_ids])
    uy = np.array([d[dof_index(node_id, 1)] for node_id in node_ids])
    
    # ========================================================================
    # STEP 3: COMPUTE DEFORMED COORDINATES
    # ========================================================================
    """
    WHY SCALE DEFLECTIONS?
    ----------------------
    - Real deflections are tiny (millimeters)
    - Can't see them at true scale
    - Scale them up visually (exaggerate)
    - This is standard practice in structural visualization
    
    PHYSICAL EXAMPLE:
    -----------------
    If a 3m tall column deflects 5mm horizontally:
    - True deflection: 5mm = 0.005m
    - At scale=50: 5mm × 50 = 250mm = 0.25m on plot
    - Now we can see it clearly!
    """
    
    xs_deformed = xs_undeformed + scale * ux
    ys_deformed = ys_undeformed + scale * uy
    
    # ========================================================================
    # STEP 4: CREATE THE FIGURE
    # ========================================================================
    """
    WHY CREATE FIGURE?
    -----------------
    - matplotlib requires a figure and axes
    - Can set size, style, etc.
    - Standard plotting workflow
    """
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # ========================================================================
    # STEP 5: DRAW UNDEFORMED ELEMENTS
    # ========================================================================
    """
    WHY DRAW UNDEFORMED FIRST?
    ---------------------------
    - Shows original structure (reference)
    - Makes deformed shape stand out
    - Helps understand how much it moved
    - Standard visualization practice
    """
    
    for element in elements:
        # Get node indices in our sorted list
        i_idx = node_ids.index(element.ni)
        j_idx = node_ids.index(element.nj)
        
        # Draw line from node i to node j
        ax.plot(
            [xs_undeformed[i_idx], xs_undeformed[j_idx]],
            [ys_undeformed[i_idx], ys_undeformed[j_idx]],
            'b-',  # Blue, solid line
            linewidth=3,
            alpha=0.7,
            label='Undeformed' if element == elements[0] else '',  # Label only once
            zorder=1,  # Behind deformed
        )
    
    # ========================================================================
    # STEP 6: DRAW DEFORMED ELEMENTS
    # ========================================================================
    """
    WHY DRAW DEFORMED?
    ------------------
    - Shows how structure actually moves
    - Helps verify results make sense
    - Shows structural behavior
    - Critical for understanding response
    """
    
    for element in elements:
        # Get node indices
        i_idx = node_ids.index(element.ni)
        j_idx = node_ids.index(element.nj)
        
        # Draw line from deformed node i to deformed node j
        ax.plot(
            [xs_deformed[i_idx], xs_deformed[j_idx]],
            [ys_deformed[i_idx], ys_deformed[j_idx]],
            'r--',  # Red, dashed line
            linewidth=2,
            alpha=0.9,
            label=f'Deformed (×{scale} scale)' if element == elements[0] else '',
            zorder=2,  # In front of undeformed
        )
    
    # ========================================================================
    # STEP 7: DRAW NODES
    # ========================================================================
    """
    WHY DRAW NODES?
    ---------------
    - Shows connection points
    - Helps identify structure
    - Makes plot clearer
    - Standard visualization element
    """
    
    # Undeformed nodes
    ax.plot(xs_undeformed, ys_undeformed, 'bo', markersize=10, label='Nodes', zorder=3)
    
    # Deformed nodes
    ax.plot(xs_deformed, ys_deformed, 'ro', markersize=8, alpha=0.7, zorder=3)
    
    # ========================================================================
    # STEP 8: DRAW SUPPORTS
    # ========================================================================
    """
    WHY DRAW SUPPORTS?
    ------------------
    - Shows boundary conditions
    - Helps understand structure
    - Standard engineering visualization
    - Makes plot complete
    """
    
    # Find base nodes (y = 0 or minimum y)
    base_y = min(ys_undeformed)
    base_nodes = [i for i in node_ids if abs(nodes[i].y - base_y) < 1e-6]
    
    # Draw support triangles
    support_size = 0.3  # Size of triangle
    for node_id in base_nodes:
        idx = node_ids.index(node_id)
        x_base = xs_undeformed[idx]
        y_base = ys_undeformed[idx]
        
        # Triangle pointing up (pinned support symbol)
        triangle_x = [x_base - support_size, x_base, x_base + support_size]
        triangle_y = [y_base - support_size, y_base, y_base - support_size]
        
        ax.plot(triangle_x, triangle_y, 'k-', linewidth=2, 
                label='Support (pinned)' if node_id == base_nodes[0] else '',
                zorder=4)
    
    # ========================================================================
    # STEP 9: ADD FORMATTING AND LABELS
    # ========================================================================
    """
    WHY FORMATTING?
    ---------------
    - Clear labels help interpretation
    - Grid makes it easier to read
    - Title explains what we're showing
    - Legend identifies elements
    """
    
    ax.set_xlabel('x (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('y (m)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.set_aspect('equal')  # Equal aspect ratio (important for structures)
    
    # ========================================================================
    # STEP 10: ADD ANNOTATIONS
    # ========================================================================
    """
    WHY ANNOTATIONS?
    ----------------
    - Show scale factor (important!)
    - Add context
    - Help interpretation
    """
    
    # Add text box with scale info
    textstr = f'Deformation scale: ×{scale}\n'
    textstr += f'(Deflections exaggerated for visibility)'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', bbox=props)
    
    # ========================================================================
    # STEP 11: SAVE THE PLOT
    # ========================================================================
    """
    WHY SAVE TO FILE?
    -----------------
    - Can be included in reports
    - Can be shared with others
    - Can be version controlled
    - Doesn't require display (works on servers)
    """
    
    os.makedirs(os.path.dirname(outpath) if os.path.dirname(outpath) else '.', exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()  # Close to free memory
    
    print(f"Frame plot saved to: {outpath}")


# =============================================================================
# ENHANCED ARCHITECTURAL VISUALIZATION FUNCTIONS
# =============================================================================

def plot_design_card(
    nodes: Dict[int, Node],
    elements: List[Frame2D],
    d: np.ndarray,
    params: dict,
    metrics: dict,
    outpath: str,
    scale: float = None,
    sections_catalog: List = None,
) -> None:
    """
    Create a beautiful design card showing the frame with loads, dimensions, and metrics.
    
    This is an architect-friendly visualization that combines:
    - Clear structural diagram with proper proportions
    - Load arrows showing applied forces
    - Dimension annotations
    - Performance metrics panel
    - Design parameters summary
    
    Parameters:
    -----------
    nodes : Dict[int, Node]
        Node dictionary from make_portal()
    elements : List[Frame2D]
        Element list from make_portal()
    d : np.ndarray
        Displacement vector from solve_linear()
    params : dict
        Design parameters (span, height, brace, sec_col, sec_beam, etc.)
    metrics : dict
        Performance metrics (drift, volume, max_moment, carbon)
    outpath : str
        Output file path
    scale : float, optional
        Deformation scale factor (auto-calculated if None)
    sections_catalog : List, optional
        Section catalog for displaying section names
    """
    
    # Extract geometry
    node_ids = sorted(nodes.keys())
    xs = np.array([nodes[i].x for i in node_ids])
    ys = np.array([nodes[i].y for i in node_ids])
    
    span = params.get('span', max(xs) - min(xs))
    height = params.get('height', max(ys) - min(ys))
    
    # Auto-calculate scale factor based on max displacement
    ux = np.array([d[dof_index(nid, 0)] for nid in node_ids])
    uy = np.array([d[dof_index(nid, 1)] for nid in node_ids])
    max_disp = max(np.max(np.abs(ux)), np.max(np.abs(uy)), 1e-6)
    
    if scale is None:
        # Scale so max visible deformation is ~15% of frame size
        target_visual_disp = 0.15 * max(span, height)
        scale = target_visual_disp / max_disp
        scale = min(scale, 100)  # Cap at 100x
    
    # Compute deformed positions
    xs_def = xs + scale * ux
    ys_def = ys + scale * uy
    
    # Create figure with two panels
    fig = plt.figure(figsize=(14, 8), facecolor=COLORS['background'])
    
    # Left panel: Frame diagram (larger)
    ax_frame = fig.add_axes([0.05, 0.1, 0.6, 0.8])
    ax_frame.set_facecolor(COLORS['background'])
    
    # Right panel: Metrics
    ax_metrics = fig.add_axes([0.68, 0.1, 0.28, 0.8])
    ax_metrics.set_facecolor(COLORS['background'])
    ax_metrics.axis('off')
    
    # =========================================================================
    # DRAW GROUND LINE
    # =========================================================================
    ground_y = min(ys) - 0.3
    ground_extend = 0.5
    ax_frame.fill_between(
        [min(xs) - ground_extend, max(xs) + ground_extend],
        [ground_y - 0.2, ground_y - 0.2],
        [ground_y, ground_y],
        color=COLORS['ground'],
        alpha=0.3,
        zorder=0
    )
    ax_frame.axhline(y=ground_y, color=COLORS['ground'], linewidth=2, zorder=1)
    
    # =========================================================================
    # DRAW UNDEFORMED FRAME (with proportional line weights)
    # =========================================================================
    # Calculate line weights based on section areas
    areas = [el.A for el in elements]
    max_area = max(areas) if areas else 1
    min_lw, max_lw = 4, 12
    
    for element in elements:
        i_idx = node_ids.index(element.ni)
        j_idx = node_ids.index(element.nj)
        
        # Line weight proportional to area
        lw = min_lw + (element.A / max_area) * (max_lw - min_lw)
        
        ax_frame.plot(
            [xs[i_idx], xs[j_idx]],
            [ys[i_idx], ys[j_idx]],
            color=COLORS['structure_primary'],
            linewidth=lw,
            solid_capstyle='round',
            zorder=2
        )
    
    # =========================================================================
    # DRAW DEFORMED FRAME
    # =========================================================================
    for element in elements:
        i_idx = node_ids.index(element.ni)
        j_idx = node_ids.index(element.nj)
        
        ax_frame.plot(
            [xs_def[i_idx], xs_def[j_idx]],
            [ys_def[i_idx], ys_def[j_idx]],
            color=COLORS['structure_secondary'],
            linewidth=2,
            linestyle='--',
            alpha=0.8,
            zorder=3
        )
    
    # =========================================================================
    # DRAW JOINTS (nodes)
    # =========================================================================
    for i, nid in enumerate(node_ids):
        # Undeformed joint
        circle = Circle((xs[i], ys[i]), 0.08, 
                        facecolor='white', 
                        edgecolor=COLORS['structure_primary'],
                        linewidth=2, zorder=5)
        ax_frame.add_patch(circle)
        
        # Deformed joint
        circle_def = Circle((xs_def[i], ys_def[i]), 0.06,
                           facecolor=COLORS['structure_secondary'],
                           edgecolor='white',
                           linewidth=1, zorder=5)
        ax_frame.add_patch(circle_def)
    
    # =========================================================================
    # DRAW SUPPORTS (pinned)
    # =========================================================================
    base_y = min(ys)
    support_size = 0.25
    for nid in node_ids:
        if abs(nodes[nid].y - base_y) < 1e-6:
            x_base = nodes[nid].x
            # Triangle support
            triangle = Polygon(
                [[x_base - support_size, ground_y],
                 [x_base + support_size, ground_y],
                 [x_base, base_y]],
                facecolor=COLORS['ground'],
                edgecolor=COLORS['structure_primary'],
                linewidth=2,
                zorder=4
            )
            ax_frame.add_patch(triangle)
            
            # Hatch marks under support
            for dx in np.linspace(-support_size, support_size, 5):
                ax_frame.plot(
                    [x_base + dx, x_base + dx - 0.1],
                    [ground_y, ground_y - 0.15],
                    color=COLORS['ground'],
                    linewidth=1,
                    zorder=1
                )
    
    # =========================================================================
    # DRAW LOADS
    # =========================================================================
    udl_w = params.get('udl_w', 0)
    wind_P = params.get('wind_P', 0)
    
    # Find beam element (horizontal at top)
    top_y = max(ys)
    beam_nodes = [nid for nid in node_ids if abs(nodes[nid].y - top_y) < 1e-6]
    
    if len(beam_nodes) >= 2 and udl_w != 0:
        # Draw UDL (distributed load) on beam
        x_left = min(nodes[nid].x for nid in beam_nodes)
        x_right = max(nodes[nid].x for nid in beam_nodes)
        
        n_arrows = 8
        arrow_spacing = (x_right - x_left) / (n_arrows - 1)
        arrow_len = 0.4 * (1 if udl_w < 0 else -1)  # Down if negative
        
        # Draw load line
        ax_frame.plot([x_left, x_right], 
                     [top_y + abs(arrow_len) + 0.1] * 2,
                     color=COLORS['load_gravity'], linewidth=2, zorder=6)
        
        # Draw arrows
        for i in range(n_arrows):
            x_arr = x_left + i * arrow_spacing
            ax_frame.annotate(
                '', 
                xy=(x_arr, top_y + 0.05),
                xytext=(x_arr, top_y + abs(arrow_len) + 0.1),
                arrowprops=dict(
                    arrowstyle='->,head_width=0.2,head_length=0.15',
                    color=COLORS['load_gravity'],
                    lw=1.5
                ),
                zorder=6
            )
        
        # Label
        ax_frame.text(
            (x_left + x_right) / 2, top_y + abs(arrow_len) + 0.25,
            f'UDL = {abs(udl_w):.0f} N/m',
            ha='center', va='bottom',
            fontsize=9, color=COLORS['load_gravity'],
            fontweight='bold'
        )
    
    # Draw wind load (lateral)
    if wind_P != 0 and len(beam_nodes) > 0:
        # Apply at top-left node
        top_left_x = min(nodes[nid].x for nid in beam_nodes)
        arrow_len = 0.6
        
        ax_frame.annotate(
            '',
            xy=(top_left_x + 0.05, top_y),
            xytext=(top_left_x - arrow_len, top_y),
            arrowprops=dict(
                arrowstyle='->,head_width=0.25,head_length=0.2',
                color=COLORS['load_lateral'],
                lw=2.5
            ),
            zorder=6
        )
        
        ax_frame.text(
            top_left_x - arrow_len - 0.1, top_y,
            f'P = {wind_P:.0f} N',
            ha='right', va='center',
            fontsize=9, color=COLORS['load_lateral'],
            fontweight='bold'
        )
    
    # =========================================================================
    # DRAW DIMENSIONS
    # =========================================================================
    dim_offset = 0.4
    dim_color = COLORS['dimension']
    
    # Span dimension (bottom)
    ax_frame.annotate(
        '', xy=(max(xs), ground_y - dim_offset),
        xytext=(min(xs), ground_y - dim_offset),
        arrowprops=dict(arrowstyle='<->', color=dim_color, lw=1)
    )
    ax_frame.text(
        (min(xs) + max(xs)) / 2, ground_y - dim_offset - 0.15,
        f'{span:.2f} m',
        ha='center', va='top', fontsize=10, color=dim_color
    )
    
    # Height dimension (side)
    ax_frame.annotate(
        '', xy=(max(xs) + dim_offset, max(ys)),
        xytext=(max(xs) + dim_offset, min(ys)),
        arrowprops=dict(arrowstyle='<->', color=dim_color, lw=1)
    )
    ax_frame.text(
        max(xs) + dim_offset + 0.1, (min(ys) + max(ys)) / 2,
        f'{height:.2f} m',
        ha='left', va='center', fontsize=10, color=dim_color,
        rotation=90
    )
    
    # =========================================================================
    # FORMAT FRAME AXES
    # =========================================================================
    margin = 1.2
    ax_frame.set_xlim(min(xs) - margin, max(xs) + margin)
    ax_frame.set_ylim(ground_y - 0.8, max(ys) + 1.0)
    ax_frame.set_aspect('equal')
    ax_frame.axis('off')
    
    # Legend
    legend_elements = [
        Line2D([0], [0], color=COLORS['structure_primary'], linewidth=6, label='Undeformed'),
        Line2D([0], [0], color=COLORS['structure_secondary'], linewidth=2, linestyle='--', label=f'Deformed (×{scale:.0f})'),
        Line2D([0], [0], color=COLORS['load_gravity'], linewidth=2, marker='v', markersize=8, label='Gravity Load'),
        Line2D([0], [0], color=COLORS['load_lateral'], linewidth=2, marker='>', markersize=8, label='Wind Load'),
    ]
    ax_frame.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9)
    
    # =========================================================================
    # METRICS PANEL
    # =========================================================================
    y_pos = 0.95
    line_height = 0.065
    
    # Title
    ax_metrics.text(0.5, y_pos, 'DESIGN METRICS', ha='center', va='top',
                   fontsize=14, fontweight='bold', color=COLORS['text'])
    y_pos -= 0.08
    
    # Divider
    ax_metrics.axhline(y=y_pos, xmin=0.1, xmax=0.9, color=COLORS['grid'], linewidth=1)
    y_pos -= 0.05
    
    # Parameters section
    ax_metrics.text(0.5, y_pos, 'Parameters', ha='center', va='top',
                   fontsize=11, fontweight='bold', color=COLORS['structure_accent'])
    y_pos -= line_height
    
    param_items = [
        ('Span', f"{params.get('span', 0):.2f} m"),
        ('Height', f"{params.get('height', 0):.2f} m"),
        ('Bracing', 'X-Brace' if params.get('brace', 0) == 1 else 'None'),
    ]
    
    # Add section names if catalog available
    if sections_catalog:
        sec_col_idx = params.get('sec_col', 0)
        sec_beam_idx = params.get('sec_beam', 0)
        if sec_col_idx < len(sections_catalog):
            param_items.append(('Column', sections_catalog[sec_col_idx].name))
        if sec_beam_idx < len(sections_catalog):
            param_items.append(('Beam', sections_catalog[sec_beam_idx].name))
    
    for label, value in param_items:
        ax_metrics.text(0.15, y_pos, label + ':', ha='left', va='top',
                       fontsize=10, color=COLORS['text'])
        ax_metrics.text(0.85, y_pos, value, ha='right', va='top',
                       fontsize=10, fontweight='bold', color=COLORS['text'])
        y_pos -= line_height * 0.8
    
    y_pos -= 0.03
    ax_metrics.axhline(y=y_pos, xmin=0.1, xmax=0.9, color=COLORS['grid'], linewidth=1)
    y_pos -= 0.05
    
    # Performance section
    ax_metrics.text(0.5, y_pos, 'Performance', ha='center', va='top',
                   fontsize=11, fontweight='bold', color=COLORS['structure_accent'])
    y_pos -= line_height
    
    drift_mm = metrics.get('drift', 0) * 1000  # Convert to mm
    metric_items = [
        ('Drift', f"{drift_mm:.1f} mm", COLORS['structure_secondary'] if drift_mm > 50 else COLORS['reaction']),
        ('Max Moment', f"{metrics.get('max_abs_M', 0):.0f} N·m", COLORS['text']),
        ('Volume', f"{metrics.get('volume', 0):.4f} m³", COLORS['text']),
        ('Carbon', f"{metrics.get('carbon', 0):.1f} kg CO₂", COLORS['text']),
    ]
    
    for label, value, color in metric_items:
        ax_metrics.text(0.15, y_pos, label + ':', ha='left', va='top',
                       fontsize=10, color=COLORS['text'])
        ax_metrics.text(0.85, y_pos, value, ha='right', va='top',
                       fontsize=10, fontweight='bold', color=color)
        y_pos -= line_height * 0.8
    
    y_pos -= 0.03
    ax_metrics.axhline(y=y_pos, xmin=0.1, xmax=0.9, color=COLORS['grid'], linewidth=1)
    y_pos -= 0.05
    
    # Loads section
    ax_metrics.text(0.5, y_pos, 'Applied Loads', ha='center', va='top',
                   fontsize=11, fontweight='bold', color=COLORS['structure_accent'])
    y_pos -= line_height
    
    load_items = [
        ('Gravity (UDL)', f"{abs(params.get('udl_w', 0)):.0f} N/m", COLORS['load_gravity']),
        ('Wind (Point)', f"{params.get('wind_P', 0):.0f} N", COLORS['load_lateral']),
    ]
    
    for label, value, color in load_items:
        ax_metrics.text(0.15, y_pos, label + ':', ha='left', va='top',
                       fontsize=10, color=COLORS['text'])
        ax_metrics.text(0.85, y_pos, value, ha='right', va='top',
                       fontsize=10, fontweight='bold', color=color)
        y_pos -= line_height * 0.8
    
    ax_metrics.set_xlim(0, 1)
    ax_metrics.set_ylim(0, 1)
    
    # =========================================================================
    # SAVE
    # =========================================================================
    os.makedirs(os.path.dirname(outpath) if os.path.dirname(outpath) else '.', exist_ok=True)
    plt.savefig(outpath, dpi=150, bbox_inches='tight', facecolor=COLORS['background'])
    plt.close()
    
    print(f"Design card saved to: {outpath}")


def plot_pareto_comparison(
    df,
    mask,
    outpath: str,
    make_portal_func,
    evaluate_func,
    material,
    sections,
    n_designs: int = 6,
) -> None:
    """
    Create a comparison grid showing top Pareto designs alongside the Pareto plot.
    
    This creates a visual summary showing:
    - Main Pareto scatter plot with frontier highlighted
    - Thumbnail frames of the top N Pareto-optimal designs
    - Each thumbnail shows the frame shape with key metrics
    
    Parameters:
    -----------
    df : pd.DataFrame
        Results dataframe from run_search()
    mask : pd.Series
        Pareto mask from pareto_mask()
    outpath : str
        Output file path
    make_portal_func : callable
        The make_portal function for regenerating frames
    evaluate_func : callable
        Function to re-evaluate designs (not needed if we have all data)
    material : Material
        Material for frame generation
    sections : List[Section]
        Section catalog
    n_designs : int
        Number of top designs to show (default: 6)
    """
    from .explore import PortalParams
    
    # Filter to Pareto-optimal designs
    successful = df[df['ok'] == True].copy()
    pareto_df = successful[mask[successful.index]].copy()
    
    # Sort by volume (lowest first) to get range from lightest to stiffest
    pareto_df = pareto_df.sort_values('volume').reset_index(drop=True)
    
    # Select designs evenly spaced across the frontier
    n_available = len(pareto_df)
    if n_available < n_designs:
        n_designs = n_available
    
    if n_designs > 0:
        indices = np.linspace(0, n_available - 1, n_designs, dtype=int)
        selected = pareto_df.iloc[indices]
    else:
        selected = pareto_df.head(n_designs)
    
    # Create figure layout
    fig = plt.figure(figsize=(16, 10), facecolor=COLORS['background'])
    
    # Main Pareto plot (left side, larger)
    ax_pareto = fig.add_axes([0.05, 0.35, 0.4, 0.55])
    
    # Frame thumbnails (right side, grid)
    n_cols = 3
    n_rows = (n_designs + n_cols - 1) // n_cols
    
    # =========================================================================
    # DRAW PARETO PLOT
    # =========================================================================
    ax_pareto.set_facecolor(COLORS['background'])
    
    # Dominated designs
    dominated = successful[~mask[successful.index]]
    ax_pareto.scatter(
        dominated['volume'], dominated['drift'] * 1000,  # Convert to mm
        c=COLORS['pareto_dominated'],
        s=30, alpha=0.4, edgecolors='none',
        label=f'Dominated ({len(dominated)})'
    )
    
    # All Pareto designs
    ax_pareto.scatter(
        pareto_df['volume'], pareto_df['drift'] * 1000,
        c=COLORS['pareto_optimal'],
        s=80, alpha=0.7, edgecolors='darkred', linewidths=1.5,
        label=f'Pareto-optimal ({len(pareto_df)})'
    )
    
    # Highlight selected designs with numbers
    for i, (idx, row) in enumerate(selected.iterrows()):
        ax_pareto.scatter(
            row['volume'], row['drift'] * 1000,
            c=COLORS['pareto_highlight'],
            s=200, edgecolors='black', linewidths=2, zorder=10
        )
        ax_pareto.annotate(
            str(i + 1),
            (row['volume'], row['drift'] * 1000),
            ha='center', va='center',
            fontsize=10, fontweight='bold', color='black',
            zorder=11
        )
    
    # Connect Pareto points with line
    pareto_sorted = pareto_df.sort_values('volume')
    ax_pareto.plot(
        pareto_sorted['volume'], pareto_sorted['drift'] * 1000,
        color=COLORS['pareto_optimal'], linewidth=1.5, alpha=0.5,
        linestyle='--', zorder=1
    )
    
    ax_pareto.set_xlabel('Volume (m³)', fontsize=11, fontweight='bold')
    ax_pareto.set_ylabel('Drift (mm)', fontsize=11, fontweight='bold')
    ax_pareto.set_title('Pareto Frontier: Volume vs Drift', fontsize=13, fontweight='bold')
    ax_pareto.grid(True, alpha=0.3, linestyle='--')
    ax_pareto.legend(loc='upper right', fontsize=9)
    
    # Add annotation
    ax_pareto.text(
        0.02, 0.98, '← Lower is better for both axes',
        transform=ax_pareto.transAxes,
        fontsize=9, va='top', ha='left',
        style='italic', color=COLORS['dimension']
    )
    
    # =========================================================================
    # DRAW FRAME THUMBNAILS
    # =========================================================================
    for i, (idx, row) in enumerate(selected.iterrows()):
        # Calculate subplot position
        col = i % n_cols
        r = i // n_cols
        
        # Position: right side of figure
        x_start = 0.48 + col * 0.17
        y_start = 0.65 - r * 0.32
        width = 0.15
        height = 0.28
        
        ax_thumb = fig.add_axes([x_start, y_start, width, height])
        ax_thumb.set_facecolor(COLORS['background'])
        
        # Recreate the frame
        params = PortalParams(
            span=row['span'],
            height=row['height'],
            brace=int(row['brace']),
            sec_col=int(row['sec_col']),
            sec_beam=int(row['sec_beam']),
            sec_brace=int(row['sec_brace']),
            udl_w=row['udl_w'],
            wind_P=row['wind_P'],
        )
        
        try:
            nodes, elements, fixed, udls, nodal_loads = make_portal_func(params, material, sections)
            
            # Get node coordinates
            node_ids = sorted(nodes.keys())
            xs = np.array([nodes[nid].x for nid in node_ids])
            ys = np.array([nodes[nid].y for nid in node_ids])
            
            # Draw elements with varying thickness
            areas = [el.A for el in elements]
            max_area = max(areas) if areas else 1
            
            for element in elements:
                i_idx = node_ids.index(element.ni)
                j_idx = node_ids.index(element.nj)
                lw = 2 + (element.A / max_area) * 4
                
                ax_thumb.plot(
                    [xs[i_idx], xs[j_idx]],
                    [ys[i_idx], ys[j_idx]],
                    color=COLORS['structure_primary'],
                    linewidth=lw,
                    solid_capstyle='round'
                )
            
            # Draw nodes
            ax_thumb.scatter(xs, ys, s=30, c='white', 
                           edgecolors=COLORS['structure_primary'], linewidths=1.5, zorder=5)
            
            # Ground line
            ax_thumb.axhline(y=-0.1, color=COLORS['ground'], linewidth=2)
            
            # Set limits with padding
            pad = 0.3
            ax_thumb.set_xlim(min(xs) - pad, max(xs) + pad)
            ax_thumb.set_ylim(-0.3, max(ys) + pad)
            ax_thumb.set_aspect('equal')
            
        except Exception as e:
            ax_thumb.text(0.5, 0.5, 'Error', ha='center', va='center')
        
        ax_thumb.axis('off')
        
        # Add design number and metrics
        ax_thumb.set_title(
            f'#{i+1}',
            fontsize=12, fontweight='bold',
            color=COLORS['pareto_highlight'],
            pad=2
        )
        
        # Metrics below thumbnail
        metrics_text = (
            f"V={row['volume']:.3f}m³  D={row['drift']*1000:.0f}mm\n"
            f"{row['span']:.1f}×{row['height']:.1f}m  "
            f"{'X-braced' if row['brace']==1 else 'Unbraced'}"
        )
        ax_thumb.text(
            0.5, -0.15, metrics_text,
            transform=ax_thumb.transAxes,
            ha='center', va='top',
            fontsize=8, color=COLORS['text']
        )
    
    # =========================================================================
    # TITLE AND FOOTER
    # =========================================================================
    fig.suptitle(
        'PORTAL FRAME DESIGN EXPLORATION',
        fontsize=16, fontweight='bold',
        color=COLORS['text'], y=0.97
    )
    
    fig.text(
        0.5, 0.02,
        f'Explored {len(df)} designs • {len(pareto_df)} Pareto-optimal • Showing {n_designs} representative designs',
        ha='center', fontsize=10, color=COLORS['dimension']
    )
    
    # =========================================================================
    # SAVE
    # =========================================================================
    os.makedirs(os.path.dirname(outpath) if os.path.dirname(outpath) else '.', exist_ok=True)
    plt.savefig(outpath, dpi=150, bbox_inches='tight', facecolor=COLORS['background'])
    plt.close()
    
    print(f"Pareto comparison saved to: {outpath}")


def plot_pareto_enhanced(
    df,
    mask,
    outpath: str,
    x: str = "volume",
    y: str = "drift",
) -> None:
    """
    Create an enhanced, architect-friendly Pareto plot with better aesthetics.
    
    Features:
    - Clean, modern color palette
    - Clear frontier visualization
    - Better typography
    - Informative annotations
    """
    
    # Filter successful designs
    successful = df[df['ok'] == True].copy()
    pareto_designs = successful[mask[successful.index]]
    dominated_designs = successful[~mask[successful.index]]
    
    # Convert drift to mm for readability
    y_pareto = pareto_designs[y].values * 1000
    y_dominated = dominated_designs[y].values * 1000
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    # Plot dominated designs
    ax.scatter(
        dominated_designs[x], y_dominated,
        c=COLORS['pareto_dominated'],
        s=40, alpha=0.4, edgecolors='none',
        label=f'Dominated designs ({len(dominated_designs)})'
    )
    
    # Plot Pareto designs
    scatter = ax.scatter(
        pareto_designs[x], y_pareto,
        c=COLORS['pareto_optimal'],
        s=120, alpha=0.85,
        edgecolors='#8B0000', linewidths=2,
        label=f'Pareto-optimal ({len(pareto_designs)})',
        zorder=5
    )
    
    # Connect Pareto points
    pareto_sorted = pareto_designs.sort_values(x)
    ax.plot(
        pareto_sorted[x], pareto_sorted[y] * 1000,
        color=COLORS['pareto_optimal'],
        linewidth=2, alpha=0.4, linestyle='-',
        zorder=4
    )
    
    # Fill area under Pareto frontier
    ax.fill_between(
        pareto_sorted[x],
        pareto_sorted[y] * 1000,
        ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1000,
        color=COLORS['pareto_optimal'],
        alpha=0.05
    )
    
    # Formatting
    ax.set_xlabel('Material Volume (m³)', fontsize=12, fontweight='bold', color=COLORS['text'])
    ax.set_ylabel('Lateral Drift (mm)', fontsize=12, fontweight='bold', color=COLORS['text'])
    ax.set_title('Design Trade-off: Material vs Stiffness', 
                fontsize=14, fontweight='bold', color=COLORS['text'], pad=20)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='-', color=COLORS['grid'])
    ax.set_axisbelow(True)
    
    # Legend
    legend = ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
    legend.get_frame().set_facecolor(COLORS['background'])
    
    # Annotations
    # Best volume (lightest)
    best_vol = pareto_designs.loc[pareto_designs[x].idxmin()]
    ax.annotate(
        f'Lightest\n{best_vol[x]:.3f}m³',
        xy=(best_vol[x], best_vol[y] * 1000),
        xytext=(best_vol[x] + 0.05, best_vol[y] * 1000 + 200),
        fontsize=9, ha='left',
        arrowprops=dict(arrowstyle='->', color=COLORS['structure_accent']),
        color=COLORS['structure_accent']
    )
    
    # Best stiffness (lowest drift)
    best_drift = pareto_designs.loc[pareto_designs[y].idxmin()]
    ax.annotate(
        f'Stiffest\n{best_drift[y]*1000:.1f}mm drift',
        xy=(best_drift[x], best_drift[y] * 1000),
        xytext=(best_drift[x] - 0.08, best_drift[y] * 1000 + 100),
        fontsize=9, ha='right',
        arrowprops=dict(arrowstyle='->', color=COLORS['reaction']),
        color=COLORS['reaction']
    )
    
    # Info box
    info_text = (
        f"Total designs explored: {len(successful)}\n"
        f"Pareto-optimal: {len(pareto_designs)} ({100*len(pareto_designs)/len(successful):.1f}%)\n"
        f"\n↙ Lower-left = ideal (less material, stiffer)"
    )
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor=COLORS['grid'])
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
           fontsize=9, va='top', ha='left', bbox=props, color=COLORS['text'])
    
    # Remove spines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color(COLORS['grid'])
    
    # Save
    os.makedirs(os.path.dirname(outpath) if os.path.dirname(outpath) else '.', exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight', facecolor=COLORS['background'])
    plt.close()
    
    print(f"Enhanced Pareto plot saved to: {outpath}")
