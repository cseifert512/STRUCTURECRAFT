'use client'

import { useDesignStore } from '@/store/designStore'
import { Slider } from '@/components/ui/Slider'
import { Select } from '@/components/ui/Select'
import {
  PARAM_LIMITS,
  HEIGHTFIELD_OPTIONS,
  TOPOLOGY_OPTIONS,
  SUPPORT_OPTIONS,
  MATERIAL_OPTIONS,
  STEEL_SECTION_OPTIONS,
  CONNECTION_OPTIONS,
} from '@/lib/types'
import { RotateCcw } from 'lucide-react'

export function ControlPanel() {
  const { params, setParam, reset, colorByForce, setColorByForce } = useDesignStore()
  
  // Filter support options based on topology and grid size
  const availableSupports = SUPPORT_OPTIONS.filter((option) => {
    if (params.topology === 'diagrid' && option.value === 'corners') return false
    if ((params.nx < 3 || params.ny < 3) && option.value === 'corners') return false
    if ((params.nx < 4 || params.ny < 4) && option.value === 'perimeter_4') return false
    return true
  })
  
  const currentSupport = availableSupports.find(s => s.value === params.support_layout)
    ? params.support_layout
    : 'edges'
  
  return (
    <div className="h-full flex flex-col p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-4 pb-3 border-b border-slate-200">
        <h2 className="text-sm font-semibold text-slate-900 uppercase tracking-wide">Parameters</h2>
        <button
          onClick={reset}
          className="p-1.5 text-slate-400 hover:text-slate-600 transition-colors"
          title="Reset"
        >
          <RotateCcw className="w-3.5 h-3.5" />
        </button>
      </div>
      
      {/* Scrollable content */}
      <div className="flex-1 overflow-y-auto space-y-5 pr-1 -mr-1">
        {/* Geometry Section */}
        <section>
          <h3 className="label">Geometry</h3>
          <div className="space-y-3">
            <div className="grid grid-cols-2 gap-3">
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
            <div className="grid grid-cols-2 gap-3">
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
            <div className="grid grid-cols-2 gap-3">
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
          <h3 className="label">Structure</h3>
          <div className="space-y-3">
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
            <Select
              label="Connections"
              value={params.connection_type}
              options={CONNECTION_OPTIONS}
              onChange={(v) => setParam('connection_type', v as typeof params.connection_type)}
            />
          </div>
        </section>
        
        {/* Material Section */}
        <section>
          <h3 className="label">Material</h3>
          <div className="space-y-3">
            <Select
              label="Type"
              value={params.material_type}
              options={MATERIAL_OPTIONS}
              onChange={(v) => setParam('material_type', v as typeof params.material_type)}
            />
            {params.material_type === 'steel' && (
              <Select
                label="Section"
                value={params.steel_section}
                options={STEEL_SECTION_OPTIONS}
                onChange={(v) => setParam('steel_section', v)}
              />
            )}
            {params.material_type === 'timber' && (
              <Slider
                label="Section Area"
                value={params.A_cm2}
                min={PARAM_LIMITS.A_cm2.min}
                max={PARAM_LIMITS.A_cm2.max}
                step={PARAM_LIMITS.A_cm2.step}
                unit=" cm²"
                onChange={(v) => setParam('A_cm2', v)}
              />
            )}
          </div>
        </section>
        
        {/* Loading Section */}
        <section>
          <h3 className="label">Loading</h3>
          <div className="space-y-3">
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
        
        {/* Advanced Analysis */}
        <section>
          <h3 className="label">Analysis</h3>
          <div className="space-y-2">
            <button
              onClick={() => setParam('pdelta_enabled', !params.pdelta_enabled)}
              className={`w-full flex items-center justify-between px-3 py-2.5 rounded border transition-colors ${
                params.pdelta_enabled 
                  ? 'bg-amber-600 border-amber-600 text-white' 
                  : 'bg-white border-slate-300 text-slate-700 hover:border-slate-400'
              }`}
            >
              <div className="flex items-center gap-2">
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
                <span className="text-sm font-medium">P-Delta</span>
              </div>
              <span className={`text-xs font-mono ${params.pdelta_enabled ? 'text-amber-200' : 'text-slate-400'}`}>
                {params.pdelta_enabled ? 'ON' : 'OFF'}
              </span>
            </button>
            
            <button
              onClick={() => setParam('modal_enabled', !params.modal_enabled)}
              className={`w-full flex items-center justify-between px-3 py-2.5 rounded border transition-colors ${
                params.modal_enabled 
                  ? 'bg-violet-600 border-violet-600 text-white' 
                  : 'bg-white border-slate-300 text-slate-700 hover:border-slate-400'
              }`}
            >
              <div className="flex items-center gap-2">
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
                </svg>
                <span className="text-sm font-medium">Modal</span>
              </div>
              <span className={`text-xs font-mono ${params.modal_enabled ? 'text-violet-200' : 'text-slate-400'}`}>
                {params.modal_enabled ? 'ON' : 'OFF'}
              </span>
            </button>
          </div>
          
          {params.pdelta_enabled && (
            <div className="mt-2 px-2 py-1.5 bg-amber-50 border border-amber-200 rounded text-xs text-amber-700">
              P-Δ: Second-order effects included
            </div>
          )}
          {params.modal_enabled && (
            <div className="mt-2 px-2 py-1.5 bg-violet-50 border border-violet-200 rounded text-xs text-violet-700">
              Modal: Natural frequencies computed
            </div>
          )}
        </section>
        
        {/* Display Options */}
        <section>
          <h3 className="label">Display</h3>
          <button
            onClick={() => setColorByForce(!colorByForce)}
            className={`w-full flex items-center justify-between px-3 py-2.5 rounded border transition-colors ${
              colorByForce 
                ? 'bg-slate-900 border-slate-900 text-white' 
                : 'bg-white border-slate-300 text-slate-700 hover:border-slate-400'
            }`}
          >
            <div className="flex items-center gap-2">
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 21a4 4 0 01-4-4V5a2 2 0 012-2h4a2 2 0 012 2v12a4 4 0 01-4 4zm0 0h12a2 2 0 002-2v-4a2 2 0 00-2-2h-2.343M11 7.343l1.657-1.657a2 2 0 012.828 0l2.829 2.829a2 2 0 010 2.828l-8.486 8.485M7 17h.01" />
              </svg>
              <span className="text-sm font-medium">Force Coloring</span>
            </div>
            <span className={`text-xs font-mono ${colorByForce ? 'text-slate-300' : 'text-slate-400'}`}>
              {colorByForce ? 'ON' : 'OFF'}
            </span>
          </button>
          {colorByForce && (
            <div className="mt-2 flex items-center justify-center gap-4 text-xs text-slate-500">
              <div className="flex items-center gap-1.5">
                <div className="w-3 h-1 bg-[#E63946] rounded-sm" />
                <span>Tension</span>
              </div>
              <div className="flex items-center gap-1.5">
                <div className="w-3 h-1 bg-[#457B9D] rounded-sm" />
                <span>Compression</span>
              </div>
            </div>
          )}
        </section>
      </div>
    </div>
  )
}
