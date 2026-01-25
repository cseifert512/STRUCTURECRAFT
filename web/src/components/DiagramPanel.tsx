'use client'

import { useDesignStore } from '@/store/designStore'
import { DiagramType } from '@/lib/types'

const DIAGRAM_OPTIONS: { value: DiagramType; label: string; description: string; color: string }[] = [
  { value: 'none', label: 'None', description: 'Hide diagrams', color: '#666' },
  { value: 'N', label: 'Axial (N)', description: 'Tension/compression', color: '#E63946' },
  { value: 'V', label: 'Shear (V)', description: 'Cutting force', color: '#9B59B6' },
  { value: 'M', label: 'Moment (M)', description: 'Bending', color: '#457B9D' },
]

export function DiagramPanel() {
  const {
    frame2dDiagrams,
    frame2dMetrics,
    frame2dElements,
    frame2dReactions,
    diagramType,
    setDiagramType,
    showDeflectedShape2D,
    setShowDeflectedShape2D,
    frame2dParams,
    setFrame2DParam,
  } = useDesignStore()

  // Get max values for display
  const maxN = frame2dDiagrams.reduce((max, d) => Math.max(max, d.max_N), 0)
  const maxV = frame2dDiagrams.reduce((max, d) => Math.max(max, d.max_V), 0)
  const maxM = frame2dDiagrams.reduce((max, d) => Math.max(max, d.max_M), 0)

  return (
    <div className="bg-white border border-slate-200 rounded-lg p-4 space-y-4">
      {/* Diagram Type Selection */}
      <div>
        <h3 className="text-sm font-semibold text-slate-700 mb-2">Force Diagram</h3>
        <div className="grid grid-cols-2 gap-2">
          {DIAGRAM_OPTIONS.map(opt => (
            <button
              key={opt.value}
              onClick={() => setDiagramType(opt.value)}
              className={`px-3 py-2 text-xs rounded border transition-colors ${
                diagramType === opt.value
                  ? 'bg-slate-800 text-white border-slate-800'
                  : 'bg-white text-slate-600 border-slate-200 hover:border-slate-400'
              }`}
            >
              <div className="font-medium">{opt.label}</div>
              <div className="text-[10px] opacity-70">{opt.description}</div>
            </button>
          ))}
        </div>
      </div>

      {/* Deflected Shape Toggle */}
      <div>
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={showDeflectedShape2D}
            onChange={(e) => setShowDeflectedShape2D(e.target.checked)}
            className="w-4 h-4 rounded border-slate-300 text-slate-800 focus:ring-slate-500"
          />
          <span className="text-sm text-slate-700">Show deflected shape</span>
        </label>
      </div>

      {/* Deflection Scale */}
      <div>
        <label className="block text-xs font-medium text-slate-500 mb-1">
          Deflection Scale: ×{frame2dParams.deflection_scale}
        </label>
        <input
          type="range"
          min={1}
          max={200}
          step={5}
          value={frame2dParams.deflection_scale}
          onChange={(e) => setFrame2DParam('deflection_scale', Number(e.target.value))}
          className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer"
        />
      </div>

      {/* Force Summary */}
      {frame2dDiagrams.length > 0 && (
        <div className="border-t border-slate-200 pt-3">
          <h3 className="text-sm font-semibold text-slate-700 mb-2">Max Internal Forces</h3>
          <div className="grid grid-cols-3 gap-2 text-center">
            <div className={`p-2 rounded ${diagramType === 'N' ? 'bg-red-50 ring-1 ring-red-200' : 'bg-slate-50'}`}>
              <div className="text-xs text-slate-500">N (axial)</div>
              <div className="text-sm font-semibold text-slate-800">{(maxN / 1000).toFixed(1)} kN</div>
            </div>
            <div className={`p-2 rounded ${diagramType === 'V' ? 'bg-purple-50 ring-1 ring-purple-200' : 'bg-slate-50'}`}>
              <div className="text-xs text-slate-500">V (shear)</div>
              <div className="text-sm font-semibold text-slate-800">{(maxV / 1000).toFixed(1)} kN</div>
            </div>
            <div className={`p-2 rounded ${diagramType === 'M' ? 'bg-blue-50 ring-1 ring-blue-200' : 'bg-slate-50'}`}>
              <div className="text-xs text-slate-500">M (moment)</div>
              <div className="text-sm font-semibold text-slate-800">{(maxM / 1000).toFixed(1)} kN·m</div>
            </div>
          </div>
        </div>
      )}

      {/* Metrics Summary */}
      {frame2dMetrics && (
        <div className="border-t border-slate-200 pt-3">
          <h3 className="text-sm font-semibold text-slate-700 mb-2">Results</h3>
          <div className="space-y-1 text-xs">
            <div className="flex justify-between">
              <span className="text-slate-500">Lateral drift:</span>
              <span className={`font-medium ${frame2dMetrics.drift_passes ? 'text-green-600' : 'text-red-600'}`}>
                {frame2dMetrics.drift_mm.toFixed(2)} mm
                {frame2dMetrics.drift_passes ? ' ✓' : ' ✗'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Drift ratio:</span>
              <span className="font-medium text-slate-700">
                1/{Math.round(1 / frame2dMetrics.drift_ratio)} (H/{Math.round(1 / frame2dMetrics.drift_ratio)})
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Max displacement:</span>
              <span className="font-medium text-slate-700">{frame2dMetrics.max_displacement_mm.toFixed(2)} mm</span>
            </div>
          </div>
        </div>
      )}

      {/* Reactions */}
      {frame2dReactions.length > 0 && (
        <div className="border-t border-slate-200 pt-3">
          <h3 className="text-sm font-semibold text-slate-700 mb-2">Reactions</h3>
          <div className="space-y-2 text-xs">
            {frame2dReactions.map(r => (
              <div key={r.node_id} className="flex justify-between bg-slate-50 rounded p-2">
                <span className="text-slate-500">Node {r.node_id}:</span>
                <span className="font-mono text-slate-700">
                  Rx={(r.Rx / 1000).toFixed(1)}, Ry={(r.Ry / 1000).toFixed(1)} kN
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Element Details */}
      {frame2dElements.length > 0 && (
        <div className="border-t border-slate-200 pt-3">
          <details className="text-xs">
            <summary className="text-sm font-semibold text-slate-700 mb-2 cursor-pointer">
              Element Details ({frame2dElements.length})
            </summary>
            <div className="space-y-1 mt-2 max-h-40 overflow-y-auto">
              {frame2dElements.map(elem => (
                <div key={elem.id} className="flex justify-between bg-slate-50 rounded p-2">
                  <span className="text-slate-500 capitalize">{elem.element_type} {elem.id}</span>
                  <span className="font-mono text-slate-600">
                    L={elem.length.toFixed(2)}m | M={(Math.max(Math.abs(elem.M_i), Math.abs(elem.M_j)) / 1000).toFixed(1)}kN·m
                  </span>
                </div>
              ))}
            </div>
          </details>
        </div>
      )}
    </div>
  )
}

