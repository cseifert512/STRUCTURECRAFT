# app/components - Reusable UI components
from .model_viewer import render_3d_model
from .metrics_panel import render_metrics_panel
from .parameter_inputs import (
    render_geometry_inputs,
    render_structure_inputs,
    render_material_inputs,
    check_stability_warning,
    get_stable_support_options,
)
from .pareto_plot import (
    render_pareto_plot,
    create_design_summary_table,
    get_design_details,
)

__all__ = [
    'render_3d_model',
    'render_metrics_panel',
    'render_geometry_inputs',
    'render_structure_inputs',
    'render_material_inputs',
    'check_stability_warning',
    'get_stable_support_options',
    'render_pareto_plot',
    'create_design_summary_table',
    'get_design_details',
]

