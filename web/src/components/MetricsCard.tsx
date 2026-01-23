'use client'

import { useDesignStore } from '@/store/designStore'

interface MetricProps {
  label: string
  value: string | number
  unit?: string
  warning?: boolean
}

function Metric({ label, value, unit = '', warning = false }: MetricProps) {
  return (
    <div>
      <div className="text-xs text-slate-500 mb-0.5">{label}</div>
      <div className="flex items-baseline gap-1">
        <span className={`text-lg font-semibold tabular-nums ${warning ? 'text-red-600' : 'text-slate-900'}`}>
          {value}
        </span>
        {unit && <span className="text-xs text-slate-400">{unit}</span>}
      </div>
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
          <Metric label="Nodes" value={metrics.n_nodes} />
          <Metric label="Bars" value={metrics.n_bars} />
          <Metric label="Supports" value={metrics.n_supports} />
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
          />
          <Metric
            label="Total Length"
            value={metrics.total_length.toFixed(1)}
            unit="m"
          />
        </div>
        <div className="grid grid-cols-2 gap-3">
          <Metric
            label="Max Tension"
            value={metrics.max_tension_kn.toFixed(1)}
            unit="kN"
          />
          <Metric
            label="Max Compression"
            value={metrics.max_compression_kn.toFixed(1)}
            unit="kN"
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
          />
          <Metric
            label="Length Bins"
            value={metrics.n_length_bins}
          />
          <Metric
            label="Max Bar"
            value={Math.round(metrics.max_member_length_mm)}
            unit="mm"
            warning={shippingWarning}
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
