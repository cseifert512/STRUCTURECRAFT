'use client'

import { useDesignStore } from '@/store/designStore'
import { Slider } from '@/components/ui/Slider'
import { Select } from '@/components/ui/Select'
import {
  PARAM_LIMITS,
  HEIGHTFIELD_OPTIONS,
  TOPOLOGY_OPTIONS,
  SUPPORT_OPTIONS,
} from '@/lib/types'
import { RotateCcw } from 'lucide-react'

export function ControlPanel() {
  const { params, setParam, reset, colorByForce, setColorByForce } = useDesignStore()
  
  // Filter support options based on topology and grid size
  const availableSupports = SUPPORT_OPTIONS.filter((option) => {
    // Diagrid + corners is always unstable
    if (params.topology === 'diagrid' && option.value === 'corners') return false
    // Small grids need edge support
    if ((params.nx < 3 || params.ny < 3) && option.value === 'corners') return false
    // Perimeter_4 needs larger grids
    if ((params.nx < 4 || params.ny < 4) && option.value === 'perimeter_4') return false
    return true
  })
  
  // Auto-fix support if current selection is invalid
  const currentSupport = availableSupports.find(s => s.value === params.support_layout)
    ? params.support_layout
    : 'edges'
  
  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="font-serif text-xl text-forest-800">Parameters</h2>
          <p className="text-sm text-stone-400 mt-0.5">Adjust to see live updates</p>
        </div>
        <button
          onClick={reset}
          className="p-2 rounded-xl text-stone-400 hover:text-forest-700 hover:bg-cream-200 transition-colors"
          title="Reset to defaults"
        >
          <RotateCcw className="w-4 h-4" />
        </button>
      </div>
      
      {/* Scrollable content */}
      <div className="flex-1 overflow-y-auto space-y-6 pr-2 -mr-2">
        {/* Geometry Section */}
        <section>
          <h3 className="section-label">Geometry</h3>
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <Slider
                label="Width"
                value={params.width}
                min={PARAM_LIMITS.width.min}
                max={PARAM_LIMITS.width.max}
                step={PARAM_LIMITS.width.step}
                unit="m"
                onChange={(v) => setParam('width', v)}
              />
              <Slider
                label="Depth"
                value={params.depth}
                min={PARAM_LIMITS.depth.min}
                max={PARAM_LIMITS.depth.max}
                step={PARAM_LIMITS.depth.step}
                unit="m"
                onChange={(v) => setParam('depth', v)}
              />
            </div>
            <div className="grid grid-cols-2 gap-4">
              <Slider
                label="Min Height"
                value={params.min_height}
                min={PARAM_LIMITS.min_height.min}
                max={PARAM_LIMITS.min_height.max}
                step={PARAM_LIMITS.min_height.step}
                unit="m"
                onChange={(v) => setParam('min_height', v)}
              />
              <Slider
                label="Max Height"
                value={params.max_height}
                min={PARAM_LIMITS.max_height.min}
                max={PARAM_LIMITS.max_height.max}
                step={PARAM_LIMITS.max_height.step}
                unit="m"
                onChange={(v) => setParam('max_height', Math.max(v, params.min_height + 0.25))}
              />
            </div>
            <div className="grid grid-cols-2 gap-4">
              <Slider
                label="Grid X"
                value={params.nx}
                min={PARAM_LIMITS.nx.min}
                max={PARAM_LIMITS.nx.max}
                step={PARAM_LIMITS.nx.step}
                onChange={(v) => setParam('nx', v)}
              />
              <Slider
                label="Grid Y"
                value={params.ny}
                min={PARAM_LIMITS.ny.min}
                max={PARAM_LIMITS.ny.max}
                step={PARAM_LIMITS.ny.step}
                onChange={(v) => setParam('ny', v)}
              />
            </div>
          </div>
        </section>
        
        {/* Structure Section */}
        <section>
          <h3 className="section-label">Structure</h3>
          <div className="space-y-4">
            <Select
              label="Shape"
              value={params.heightfield}
              options={HEIGHTFIELD_OPTIONS}
              onChange={(v) => setParam('heightfield', v as typeof params.heightfield)}
            />
            <Select
              label="Topology"
              value={params.topology}
              options={TOPOLOGY_OPTIONS}
              onChange={(v) => setParam('topology', v as typeof params.topology)}
            />
            <Select
              label="Supports"
              value={currentSupport}
              options={availableSupports}
              onChange={(v) => setParam('support_layout', v as typeof params.support_layout)}
            />
          </div>
        </section>
        
        {/* Material Section */}
        <section>
          <h3 className="section-label">Material & Load</h3>
          <div className="space-y-4">
            <Slider
              label="Section Area"
              value={params.A_cm2}
              min={PARAM_LIMITS.A_cm2.min}
              max={PARAM_LIMITS.A_cm2.max}
              step={PARAM_LIMITS.A_cm2.step}
              unit=" cm²"
              onChange={(v) => setParam('A_cm2', v)}
            />
            <Slider
              label="Gravity Load"
              value={params.gravity_kn}
              min={PARAM_LIMITS.gravity_kn.min}
              max={PARAM_LIMITS.gravity_kn.max}
              step={PARAM_LIMITS.gravity_kn.step}
              unit=" kN"
              onChange={(v) => setParam('gravity_kn', v)}
            />
          </div>
        </section>
        
        {/* Display Options */}
        <section>
          <h3 className="section-label">Display</h3>
          <label className="flex items-center gap-3 cursor-pointer group">
            <div className="relative">
              <input
                type="checkbox"
                checked={colorByForce}
                onChange={(e) => setColorByForce(e.target.checked)}
                className="sr-only peer"
              />
              <div className="w-10 h-6 rounded-full bg-cream-200 peer-checked:bg-sage-500 transition-colors" />
              <div className="absolute top-1 left-1 w-4 h-4 rounded-full bg-white shadow-sm transition-transform peer-checked:translate-x-4" />
            </div>
            <span className="text-sm text-forest-700 group-hover:text-forest-900 transition-colors">
              Color by force
            </span>
          </label>
          <p className="text-xs text-stone-400 mt-2 ml-[52px]">
            Tension in warm tones, compression in green
          </p>
        </section>
      </div>
    </div>
  )
}

