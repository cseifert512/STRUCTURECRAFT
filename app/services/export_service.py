# app/services/export_service.py
"""
Export service: handles file exports (CSV, JSON, HTML).
"""

import json
import csv
import io
from typing import Dict, Any, List
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mini_branch.v3d.elements import element_geometry_3d


class ExportService:
    """Service for exporting design data to various formats."""
    
    @staticmethod
    def generate_cutlist_csv(nodes: Dict, bars: List, forces: Dict) -> str:
        """
        Generate a CSV cut list for fabrication.
        
        Returns CSV content as a string.
        """
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Header
        writer.writerow([
            'bar_id', 'node_i', 'node_j', 
            'length_m', 'length_mm', 
            'force_kN', 'force_type'
        ])
        
        # Data rows
        rows = []
        for bar in bars:
            L, _, _, _ = element_geometry_3d(nodes, bar)
            force = forces.get(bar.id, 0)
            force_type = "T" if force > 0 else "C"
            
            rows.append({
                'bar_id': bar.id,
                'node_i': bar.ni,
                'node_j': bar.nj,
                'length_m': round(L, 4),
                'length_mm': round(L * 1000, 1),
                'force_kN': round(force / 1000, 2),
                'force_type': force_type,
            })
        
        # Sort by length
        rows.sort(key=lambda x: x['length_m'])
        
        for row in rows:
            writer.writerow([
                row['bar_id'], row['node_i'], row['node_j'],
                row['length_m'], row['length_mm'],
                row['force_kN'], row['force_type']
            ])
        
        return output.getvalue()
    
    @staticmethod
    def generate_model_json(
        nodes: Dict, 
        bars: List, 
        params: Dict,
        metrics: Dict,
        forces: Dict = None
    ) -> str:
        """
        Generate JSON model data for interchange.
        
        Returns JSON content as a string.
        """
        # Convert nodes to serializable format
        nodes_data = {
            str(nid): {'id': n.id, 'x': n.x, 'y': n.y, 'z': n.z}
            for nid, n in nodes.items()
        }
        
        # Convert bars to serializable format
        bars_data = [
            {
                'id': bar.id,
                'ni': bar.ni,
                'nj': bar.nj,
                'E': bar.E,
                'A': bar.A,
            }
            for bar in bars
        ]
        
        # Add lengths and forces
        for bar_data in bars_data:
            bar_id = bar_data['id']
            bar = next(b for b in bars if b.id == bar_id)
            L, _, _, _ = element_geometry_3d(nodes, bar)
            bar_data['length'] = round(L, 4)
            if forces:
                bar_data['force'] = round(forces.get(bar_id, 0), 2)
        
        model = {
            'version': '1.0',
            'type': 'canopy',
            'parameters': params,
            'metrics': metrics,
            'geometry': {
                'nodes': nodes_data,
                'bars': bars_data,
            }
        }
        
        return json.dumps(model, indent=2)
    
    @staticmethod
    def generate_summary_text(params: Dict, metrics: Dict) -> str:
        """Generate a text summary of the design."""
        lines = [
            "CANOPY DESIGN SUMMARY",
            "=" * 40,
            "",
            "GEOMETRY",
            f"  Footprint:     {params.get('width', 0):.1f} x {params.get('depth', 0):.1f} m",
            f"  Height:        {params.get('min_height', 0):.1f} - {params.get('max_height', 0):.1f} m",
            f"  Grid:          {params.get('nx', 0)} x {params.get('ny', 0)}",
            f"  Heightfield:   {params.get('heightfield', 'N/A')}",
            f"  Topology:      {params.get('topology', 'N/A')}",
            "",
            "STRUCTURE",
            f"  Nodes:         {metrics.get('n_nodes', 0)}",
            f"  Bars:          {metrics.get('n_bars', 0)}",
            f"  Supports:      {metrics.get('n_supports', 0)}",
            "",
            "PERFORMANCE",
            f"  Max disp:      {metrics.get('max_displacement', 0)*1000:.2f} mm",
            f"  Max tension:   {metrics.get('max_tension', 0)/1000:.1f} kN",
            f"  Max compress:  {metrics.get('max_compression', 0)/1000:.1f} kN",
            "",
            "FABRICATION",
            f"  Volume:        {metrics.get('volume', 0):.4f} m^3",
            f"  Total length:  {metrics.get('total_length', 0):.1f} m",
            f"  Length bins:   {metrics.get('n_length_bins', 0)}",
            f"  Longest bar:   {metrics.get('max_member_length', 0)*1000:.0f} mm",
            f"  Shortest bar:  {metrics.get('min_member_length', 0)*1000:.0f} mm",
        ]
        
        return "\n".join(lines)





