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
      
      {/* Engineering Checks */}
      <section className="mb-4 pt-3 border-t border-slate-100">
        <h3 className="label">Engineering Checks</h3>
        <div className="grid grid-cols-3 gap-3">
          <Metric
            label="Buckling Factor"
            value={metrics.buckling_factor !== null && metrics.buckling_factor !== undefined 
              ? metrics.buckling_factor >= 99 ? '>99' : metrics.buckling_factor.toFixed(2)
              : '—'}
            warning={metrics.buckling_factor !== null && metrics.buckling_factor !== undefined && metrics.buckling_factor < 1.5}
            tooltip="Critical buckling load factor (λ_cr). Values >1.0 indicate stability. Target >1.5 for safety margin."
          />
          <Metric
            label="Max Utilization"
            value={metrics.max_utilization !== null && metrics.max_utilization !== undefined 
              ? (metrics.max_utilization * 100).toFixed(0) + '%'
              : '—'}
            warning={metrics.max_utilization !== null && metrics.max_utilization !== undefined && metrics.max_utilization >= 1.0}
            tooltip={metrics.material_type === 'steel' 
              ? "AISC combined utilization (H1). Values <100% pass design checks."
              : "Timber axial utilization per NDS. Values <100% pass design checks."
            }
          />
          <Metric
            label="Failing Members"
            value={metrics.n_failing_members !== null && metrics.n_failing_members !== undefined 
              ? metrics.n_failing_members
              : '—'}
            warning={metrics.n_failing_members !== null && metrics.n_failing_members !== undefined && metrics.n_failing_members > 0}
            tooltip="Count of members exceeding allowable stress. Increase section area or reduce loads to pass."
          />
        </div>
        
        {/* P-Delta Results */}
        {metrics.pdelta_amplification !== null && metrics.pdelta_amplification !== undefined && (
          <div className="grid grid-cols-2 gap-3 mt-3">
            <Metric
              label="P-Δ Amplification"
              value={metrics.pdelta_amplification.toFixed(3)}
              unit="×"
              warning={metrics.pdelta_amplification > 1.5}
              tooltip="Displacement amplification from P-Delta analysis. Values >1.5 indicate significant second-order effects."
            />
            <Metric
              label="P-Δ Iterations"
              value={metrics.pdelta_iterations !== null && metrics.pdelta_iterations !== undefined 
                ? metrics.pdelta_iterations
                : '—'}
              tooltip="Number of iterations to converge. 0 means no compression (linear analysis sufficient)."
            />
          </div>
        )}
        
        {/* Material and Connection badges */}
        {(metrics.material_type || metrics.connection_type) && (
          <div className="mt-3 flex flex-wrap items-center gap-2">
            {metrics.material_type && (
              <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                metrics.material_type === 'steel'
                  ? 'bg-blue-100 text-blue-700'
                  : 'bg-amber-100 text-amber-700'
              }`}>
                {metrics.material_type === 'steel' ? 'AISC Steel' : 'NDS Timber'}
              </span>
            )}
            {metrics.steel_section_name && (
              <span className="text-xs text-slate-500">{metrics.steel_section_name}</span>
            )}
            {metrics.connection_type && (
              <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                metrics.connection_type === 'rigid'
                  ? 'bg-purple-100 text-purple-700'
                  : metrics.connection_type === 'semi_rigid'
                    ? 'bg-indigo-100 text-indigo-700'
                    : 'bg-slate-100 text-slate-600'
              }`}>
                {metrics.connection_type === 'rigid' ? 'Rigid Joints' 
                  : metrics.connection_type === 'semi_rigid' ? 'Semi-Rigid' 
                  : 'Pinned'}
              </span>
            )}
          </div>
        )}
        
        {/* Status indicator */}
        {metrics.max_utilization !== null && metrics.max_utilization !== undefined && (
          <div className={`mt-2 px-2 py-1.5 rounded text-xs ${
            metrics.max_utilization < 0.8 
              ? 'bg-green-50 border border-green-200 text-green-700'
              : metrics.max_utilization < 1.0 
                ? 'bg-yellow-50 border border-yellow-200 text-yellow-700'
                : 'bg-red-50 border border-red-200 text-red-700'
          }`}>
            {metrics.max_utilization < 0.8 
              ? '✓ All members within design limits'
              : metrics.max_utilization < 1.0
                ? '⚠ Some members near capacity'
                : `✗ ${metrics.n_failing_members} member(s) overstressed`
            }
          </div>
        )}
        
        {/* P-Delta warning */}
        {metrics.pdelta_amplification !== null && metrics.pdelta_amplification !== undefined && metrics.pdelta_amplification > 1.5 && (
          <div className="mt-2 px-2 py-1.5 bg-amber-50 border border-amber-200 rounded text-xs text-amber-700">
            ⚠ High P-Delta amplification ({metrics.pdelta_amplification.toFixed(2)}×) — consider stiffer sections
          </div>
        )}
        
        {/* Natural Frequencies */}
        {metrics.natural_frequencies_hz && metrics.natural_frequencies_hz.length > 0 && (
          <div className="mt-3 pt-3 border-t border-slate-100">
            <div className="text-xs text-slate-500 mb-2 flex items-center gap-1">
              Natural Frequencies
              <span className="text-slate-400">(Hz)</span>
            </div>
            <div className="flex flex-wrap gap-2">
              {metrics.natural_frequencies_hz.slice(0, 5).map((freq, idx) => (
                <div 
                  key={idx}
                  className={`px-2 py-1 rounded text-xs font-mono ${
                    idx === 0 
                      ? 'bg-violet-100 text-violet-700 border border-violet-200' 
                      : 'bg-slate-100 text-slate-600'
                  }`}
                  title={`Mode ${idx + 1}: ${freq.toFixed(3)} Hz`}
                >
                  <span className="text-slate-400 mr-1">f{idx + 1}:</span>
                  {freq.toFixed(2)}
                </div>
              ))}
            </div>
            {metrics.natural_frequencies_hz[0] < 1.0 && (
              <div className="mt-2 px-2 py-1.5 bg-violet-50 border border-violet-200 rounded text-xs text-violet-700">
                ⚠ Low fundamental frequency ({metrics.natural_frequencies_hz[0].toFixed(2)} Hz) — may need dynamic analysis
              </div>
            )}
          </div>
        )}
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
