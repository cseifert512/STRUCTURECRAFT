'use client'

import { useDesignStore } from '@/store/designStore'
import { motion, AnimatePresence } from 'framer-motion'

interface MetricProps {
  label: string
  value: string | number
  unit?: string
  highlight?: boolean
  warning?: boolean
}

function Metric({ label, value, unit = '', highlight = false, warning = false }: MetricProps) {
  return (
    <div className="flex flex-col">
      <span className="text-xs text-stone-400 mb-0.5">{label}</span>
      <div className="flex items-baseline gap-1">
        <motion.span
          key={String(value)}
          initial={{ opacity: 0, y: 5 }}
          animate={{ opacity: 1, y: 0 }}
          className={`text-lg font-serif tabular-nums ${
            warning ? 'text-clay-500' : highlight ? 'text-sage-600' : 'text-forest-800'
          }`}
        >
          {value}
        </motion.span>
        {unit && <span className="text-xs text-stone-400">{unit}</span>}
      </div>
    </div>
  )
}

export function MetricsCard() {
  const { metrics, isLoading, error } = useDesignStore()
  
  if (error || !metrics) {
    return (
      <div className="card-organic">
        <h3 className="font-serif text-lg text-forest-800 mb-4">Metrics</h3>
        <div className="text-center py-8 text-stone-400">
          {error ? 'Adjust parameters for valid structure' : 'No data yet'}
        </div>
      </div>
    )
  }
  
  const shippingWarning = metrics.max_member_length_mm > 6000
  
  return (
    <div className="card-organic">
      <AnimatePresence mode="wait">
        <motion.div
          key="metrics"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
        >
          {/* Structure */}
          <section className="mb-5">
            <h3 className="section-label">Structure</h3>
            <div className="grid grid-cols-3 gap-4">
              <Metric label="Nodes" value={metrics.n_nodes} />
              <Metric label="Bars" value={metrics.n_bars} />
              <Metric label="Supports" value={metrics.n_supports} />
            </div>
          </section>
          
          {/* Performance */}
          <section className="mb-5">
            <h3 className="section-label">Performance</h3>
            <div className="grid grid-cols-2 gap-4 mb-3">
              <Metric
                label="Max Displacement"
                value={metrics.max_displacement_mm.toFixed(2)}
                unit="mm"
                highlight
              />
              <Metric
                label="Total Length"
                value={metrics.total_length.toFixed(1)}
                unit="m"
              />
            </div>
            <div className="grid grid-cols-2 gap-4">
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
          <section>
            <h3 className="section-label">Fabrication</h3>
            <div className="grid grid-cols-3 gap-4">
              <Metric
                label="Volume"
                value={metrics.volume.toFixed(4)}
                unit="m³"
              />
              <Metric
                label="Length Bins"
                value={metrics.n_length_bins}
                highlight
              />
              <Metric
                label="Longest Bar"
                value={Math.round(metrics.max_member_length_mm)}
                unit="mm"
                warning={shippingWarning}
              />
            </div>
            
            {shippingWarning && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                className="mt-3 px-3 py-2 rounded-xl bg-clay-50 border border-clay-200"
              >
                <p className="text-xs text-clay-600">
                  Bars exceed 6m shipping limit
                </p>
              </motion.div>
            )}
          </section>
        </motion.div>
      </AnimatePresence>
    </div>
  )
}

