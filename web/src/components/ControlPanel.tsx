'use client'

import { useDesignStore, ColorMode } from '@/store/designStore'
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
  FRAME2D_PARAM_LIMITS,
  AnalysisMode,
} from '@/lib/types'
import { RotateCcw } from 'lucide-react'

const COLOR_MODE_OPTIONS: { value: ColorMode; label: string }[] = [
  { value: 'none', label: 'None' },
  { value: 'force', label: 'Force (T/C)' },
  { value: 'utilization', label: 'Utilization' },
]

const BRACE_OPTIONS = [
  { value: 0, label: 'None', description: 'No bracing' },
  { value: 1, label: 'X-Brace', description: 'Diagonal braces' },
]

export function ControlPanel() {
  const { 
    // Mode
    analysisMode, 
    setAnalysisMode,
    // 3D
    params, 
    setParam, 
    reset, 
    colorMode, 
    setColorMode,
    showDeflectedShape,
    setShowDeflectedShape,
    deflectionScale3D,
    setDeflectionScale3D,
    // 2D
    frame2dParams,
    setFrame2DParam,
    resetFrame2D,
  } = useDesignStore()
  
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

  // Mode toggle component
  const ModeToggle = () => (
    <div className="flex bg-slate-100 rounded-lg p-1 mb-4">
      <button
        onClick={() => setAnalysisMode('3d')}
        className={`flex-1 px-3 py-2 text-xs font-medium rounded-md transition-all ${
          analysisMode === '3d'
            ? 'bg-white text-slate-900 shadow-sm'
            : 'text-slate-500 hover:text-slate-700'
        }`}
      >
        3D Spaceframe
      </button>
      <button
        onClick={() => setAnalysisMode('2d')}
        className={`flex-1 px-3 py-2 text-xs font-medium rounded-md transition-all ${
          analysisMode === '2d'
            ? 'bg-white text-slate-900 shadow-sm'
            : 'text-slate-500 hover:text-slate-700'
        }`}
      >
        2D Frame
      </button>
    </div>
  )

  // 2D Frame Controls
  const Frame2DControls = () => (
    <div className="space-y-5">
      {/* Geometry Section */}
      <section>
        <h3 className="label">Geometry</h3>
        <div className="space-y-3">
          <div className="grid grid-cols-2 gap-3">
            <Slider
              label="Span"
              value={frame2dParams.span}
              min={FRAME2D_PARAM_LIMITS.span.min}
              max={FRAME2D_PARAM_LIMITS.span.max}
              step={FRAME2D_PARAM_LIMITS.span.step}
              unit="m"
              onChange={(v) => setFrame2DParam('span', v)}
            />
            <Slider
              label="Height"
              value={frame2dParams.height}
              min={FRAME2D_PARAM_LIMITS.height.min}
              max={FRAME2D_PARAM_LIMITS.height.max}
              step={FRAME2D_PARAM_LIMITS.height.step}
              unit="m"
              onChange={(v) => setFrame2DParam('height', v)}
            />
          </div>
          
          <div className="flex gap-2">
            {BRACE_OPTIONS.map(opt => (
              <button
                key={opt.value}
                onClick={() => setFrame2DParam('brace', opt.value as 0 | 1)}
                className={`flex-1 px-3 py-2 text-xs rounded border transition-colors ${
                  frame2dParams.brace === opt.value
                    ? 'bg-slate-800 text-white border-slate-800'
                    : 'bg-white text-slate-600 border-slate-200 hover:border-slate-400'
                }`}
              >
                {opt.label}
              </button>
            ))}
          </div>
        </div>
      </section>
      
      {/* Sections */}
      <section>
        <h3 className="label">Section Indices</h3>
        <div className="space-y-3">
          <div className="grid grid-cols-3 gap-2">
            <Slider
              label="Column"
              value={frame2dParams.sec_col}
              min={0}
              max={10}
              step={1}
              onChange={(v) => setFrame2DParam('sec_col', v)}
            />
            <Slider
              label="Beam"
              value={frame2dParams.sec_beam}
              min={0}
              max={10}
              step={1}
              onChange={(v) => setFrame2DParam('sec_beam', v)}
            />
            <Slider
              label="Brace"
              value={frame2dParams.sec_brace}
              min={0}
              max={10}
              step={1}
              onChange={(v) => setFrame2DParam('sec_brace', v)}
            />
          </div>
          <p className="text-[10px] text-slate-400">
            0 = smallest, 10 = largest timber section
          </p>
        </div>
      </section>
      
      {/* Loading */}
      <section>
        <h3 className="label">Loading</h3>
        <div className="space-y-3">
          <Slider
            label="Gravity (UDL)"
            value={frame2dParams.udl_kn_m}
            min={FRAME2D_PARAM_LIMITS.udl_kn_m.min}
            max={FRAME2D_PARAM_LIMITS.udl_kn_m.max}
            step={FRAME2D_PARAM_LIMITS.udl_kn_m.step}
            unit=" kN/m"
            onChange={(v) => setFrame2DParam('udl_kn_m', v)}
          />
          <Slider
            label="Lateral (Wind)"
            value={frame2dParams.lateral_kn}
            min={FRAME2D_PARAM_LIMITS.lateral_kn.min}
            max={FRAME2D_PARAM_LIMITS.lateral_kn.max}
            step={FRAME2D_PARAM_LIMITS.lateral_kn.step}
            unit=" kN"
            onChange={(v) => setFrame2DParam('lateral_kn', v)}
          />
        </div>
      </section>
    </div>
  )

  // 3D Spaceframe Controls (existing)
  const Frame3DControls = () => (
    <div className="space-y-5">
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
        <div className="space-y-3">
          <Select
            label="Color Mode"
            value={colorMode}
            onChange={(value) => setColorMode(value as ColorMode)}
            options={COLOR_MODE_OPTIONS}
          />
          {colorMode === 'force' && (
            <div className="flex items-center justify-center gap-4 text-xs text-slate-500">
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
          {colorMode === 'utilization' && (
            <div className="flex items-center justify-center gap-4 text-xs text-slate-500">
              <div className="flex items-center gap-1.5">
                <div className="w-3 h-1 bg-[#2D6A4F] rounded-sm" />
                <span>&lt;70%</span>
              </div>
              <div className="flex items-center gap-1.5">
                <div className="w-3 h-1 bg-[#E9C46A] rounded-sm" />
                <span>70-100%</span>
              </div>
              <div className="flex items-center gap-1.5">
                <div className="w-3 h-1 bg-[#D62828] rounded-sm" />
                <span>&gt;100%</span>
              </div>
            </div>
          )}
          
          {/* Deflected shape toggle for 3D */}
          <div className="pt-2 border-t border-slate-200">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={showDeflectedShape}
                onChange={(e) => setShowDeflectedShape(e.target.checked)}
                className="w-4 h-4 rounded border-slate-300 text-slate-800 focus:ring-slate-500"
              />
              <span className="text-sm text-slate-700">Show deflected shape</span>
            </label>
            {showDeflectedShape && (
              <div className="mt-2">
                <Slider
                  label="Scale"
                  value={deflectionScale3D}
                  min={1}
                  max={200}
                  step={5}
                  unit="×"
                  onChange={(v) => setDeflectionScale3D(v)}
                />
              </div>
            )}
          </div>
        </div>
      </section>
    </div>
  )
  
  return (
    <div className="h-full flex flex-col p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-4 pb-3 border-b border-slate-200">
        <h2 className="text-sm font-semibold text-slate-900 uppercase tracking-wide">Parameters</h2>
        <button
          onClick={analysisMode === '3d' ? reset : resetFrame2D}
          className="p-1.5 text-slate-400 hover:text-slate-600 transition-colors"
          title="Reset"
        >
          <RotateCcw className="w-3.5 h-3.5" />
        </button>
      </div>
      
      {/* Mode Toggle */}
      <ModeToggle />
      
      {/* Scrollable content */}
      <div className="flex-1 overflow-y-auto pr-1 -mr-1">
        {analysisMode === '3d' ? <Frame3DControls /> : <Frame2DControls />}
      </div>
    </div>
  )
}
