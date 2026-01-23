'use client'

import { useState } from 'react'
import { useDesignStore } from '@/store/designStore'

interface MetricProps {
  label: string
  value: string | number
  unit?: string
  warning?: boolean
  tooltip?: string
}

function Metric({ label, value, unit = '', warning = false, tooltip }: MetricProps) {
  const [showTooltip, setShowTooltip] = useState(false)
  
  return (
    <div className="relative">
      <div className="text-xs text-slate-500 mb-0.5 flex items-center gap-1">
        {label}
        {tooltip && (
          <span
            className="cursor-help text-slate-400 hover:text-slate-600 transition-colors"
            onMouseEnter={() => setShowTooltip(true)}
            onMouseLeave={() => setShowTooltip(false)}
          >
            <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </span>
        )}
      </div>
      <div className="flex items-baseline gap-1">
        <span className={`text-lg font-semibold tabular-nums ${warning ? 'text-red-600' : 'text-slate-900'}`}>
          {value}
        </span>
        {unit && <span className="text-xs text-slate-400">{unit}</span>}
      </div>
      
      {/* Tooltip */}
      {tooltip && showTooltip && (
        <div className="absolute z-20 bottom-full left-0 mb-2 px-2 py-1.5 bg-slate-800 text-white text-xs rounded shadow-lg max-w-[200px] whitespace-normal">
          {tooltip}
          <div className="absolute top-full left-3 w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-slate-800" />
        </div>
      )}
    </div>
  )
}

export function MetricsCard() {
  const { metrics, error } = useDesignStore()
  
  if (error || !metrics) {
    return (
      <div className="card rounded-lg p-4">
        <h3 className="text-sm font-semibold text-slate-900 uppercase tracking-wide mb-3">Metrics</h3>
        <div className="text-center py-6 text-slate-400 text-sm">
          {error ? 'Invalid configuration' : 'No data'}
        </div>
      </div>
    )
  }
  
  const shippingWarning = metrics.max_member_length_mm > 6000
  
  return (
    <div className="card rounded-lg p-4">
      {/* Structure */}
      <section className="mb-4">
        <h3 className="label">Structure</h3>
        <div className="grid grid-cols-3 gap-3">
          <Metric 
            label="Nodes" 
            value={metrics.n_nodes}
            tooltip="Total joint count in the grid. Calculated as (nx+1) × (ny+1) based on grid divisions."
          />
          <Metric 
            label="Bars" 
            value={metrics.n_bars}
            tooltip="Total member count. Depends on topology: grid adds diagonals, triangulated uses alternating pattern."
          />
          <Metric 
            label="Supports" 
            value={metrics.n_supports}
            tooltip="Number of pinned support nodes. Each support constrains 3 DOFs (X, Y, Z translation)."
          />
        </div>
      </section>
      
      {/* Performance */}
      <section className="mb-4 pt-3 border-t border-slate-100">
        <h3 className="label">Performance</h3>
        <div className="grid grid-cols-2 gap-3 mb-2">
          <Metric
            label="Max Displacement"
            value={metrics.max_displacement_mm.toFixed(2)}
            unit="mm"
            tooltip="Maximum nodal displacement from linear elastic FEA: √(ux² + uy² + uz²). Solved via K·d = F with partitioned DOFs."
          />
          <Metric
            label="Total Length"
            value={metrics.total_length.toFixed(1)}
            unit="m"
            tooltip="Sum of all bar lengths: Σ√[(xj-xi)² + (yj-yi)² + (zj-zi)²]. Proxy for material usage."
          />
        </div>
        <div className="grid grid-cols-2 gap-3">
          <Metric
            label="Max Tension"
            value={metrics.max_tension_kn.toFixed(1)}
            unit="kN"
            tooltip="Highest tensile axial force (N > 0). Computed as N = (EA/L) × ΔL where ΔL is elongation along bar axis."
          />
          <Metric
            label="Max Compression"
            value={metrics.max_compression_kn.toFixed(1)}
            unit="kN"
            tooltip="Highest compressive axial force (|N| where N < 0). May govern buckling checks for slender members."
          />
        </div>
      </section>
      
      {/* Fabrication */}
      <section className="pt-3 border-t border-slate-100">
        <h3 className="label">Fabrication</h3>
        <div className="grid grid-cols-3 gap-3">
          <Metric
            label="Volume"
            value={metrics.volume.toFixed(4)}
            unit="m³"
            tooltip="Total material volume: Σ(A × L) for all bars. Multiply by density (steel ≈ 7850 kg/m³) for weight."
          />
          <Metric
            label="Length Bins"
            value={metrics.n_length_bins}
            tooltip="Unique bar lengths within 10mm tolerance. Fewer bins = easier cutting, stocking, and assembly."
          />
          <Metric
            label="Max Bar"
            value={Math.round(metrics.max_member_length_mm)}
            unit="mm"
            warning={shippingWarning}
            tooltip="Longest member length. Bars >6m may require special shipping. Affects handling and erection logistics."
          />
        </div>
        
        {shippingWarning && (
          <div className="mt-3 px-2 py-1.5 bg-red-50 border border-red-200 rounded text-xs text-red-700">
            Bars exceed 6m shipping limit
          </div>
        )}
      </section>
    </div>
  )
}
