'use client'

import { useState } from 'react'
import { useDesignStore } from '@/store/designStore'
import { ParetoPlot } from './ParetoPlot'
import { Slider } from '@/components/ui/Slider'
import { ChevronDown, ChevronUp, Play, Trash2, Check, Search, Filter } from 'lucide-react'

export function ExplorePanel() {
  const {
    params,
    exploreSettings,
    setExploreSettings,
    isExploring,
    exploreError,
    exploreResults,
    nTotal,
    nSuccessful,
    nPareto,
    selectedDesignIndex,
    selectDesign,
    applySelectedDesign,
    runExploration,
    clearExploration,
    showExplorePanel,
    setShowExplorePanel,
    filters,
    setFilters,
  } = useDesignStore()
  
  const [showFilters, setShowFilters] = useState(false)
  
  const selectedDesign = selectedDesignIndex !== null 
    ? exploreResults.find(d => d.index === selectedDesignIndex)
    : null
  
  // Get max values for filter sliders
  const maxDisp = Math.max(...exploreResults.filter(d => d.ok && d.max_displacement_mm).map(d => d.max_displacement_mm!), 100)
  const maxLen = Math.max(...exploreResults.filter(d => d.ok && d.max_member_length_mm).map(d => d.max_member_length_mm!), 10000)
  const topologies = [...new Set(exploreResults.filter(d => d.ok).map(d => d.topology))]
  
  return (
    <div className="card rounded-lg overflow-hidden">
      {/* Header - always visible */}
      <button
        onClick={() => setShowExplorePanel(!showExplorePanel)}
        className="w-full flex items-center justify-between p-4 hover:bg-slate-50 transition-colors"
      >
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-red-500 to-orange-500 flex items-center justify-center">
            <Search className="w-4 h-4 text-white" />
          </div>
          <div className="text-left">
            <h3 className="text-sm font-semibold text-slate-900">Explore Design Space</h3>
            <p className="text-xs text-slate-500">
              {exploreResults.length > 0 
                ? `${nSuccessful} designs • ${nPareto} Pareto-optimal`
                : 'Generate variants to find optimal trade-offs'}
            </p>
          </div>
        </div>
        {showExplorePanel ? (
          <ChevronUp className="w-5 h-5 text-slate-400" />
        ) : (
          <ChevronDown className="w-5 h-5 text-slate-400" />
        )}
      </button>
      
      {/* Expandable content */}
      {showExplorePanel && (
        <div className="border-t border-slate-200">
          {/* Controls */}
          <div className="p-4 bg-slate-50 border-b border-slate-200">
            <div className="flex items-end gap-4">
              <div className="flex-1 grid grid-cols-3 gap-3">
                <Slider
                  label="Variants"
                  value={exploreSettings.n_designs}
                  min={20}
                  max={300}
                  step={10}
                  onChange={(v) => setExploreSettings({ n_designs: v })}
                />
                <Slider
                  label="Seed"
                  value={exploreSettings.seed}
                  min={1}
                  max={999}
                  step={1}
                  onChange={(v) => setExploreSettings({ seed: v })}
                />
                <Slider
                  label="Variation"
                  value={Math.round(exploreSettings.variation_pct * 100)}
                  min={10}
                  max={50}
                  step={5}
                  unit="%"
                  onChange={(v) => setExploreSettings({ variation_pct: v / 100 })}
                />
              </div>
              
              <div className="flex gap-2">
                {exploreResults.length > 0 && (
                  <button
                    onClick={clearExploration}
                    className="px-3 py-2 text-sm text-slate-600 hover:text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                    title="Clear results"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                )}
                <button
                  onClick={runExploration}
                  disabled={isExploring}
                  className="flex items-center gap-2 px-4 py-2 bg-slate-900 text-white text-sm font-medium rounded-lg hover:bg-slate-800 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  {isExploring ? (
                    <>
                      <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                      <span>Exploring...</span>
                    </>
                  ) : (
                    <>
                      <Play className="w-4 h-4" />
                      <span>Generate</span>
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>
          
          {/* Error state */}
          {exploreError && (
            <div className="p-4 bg-red-50 text-red-700 text-sm">
              <strong>Error:</strong> {exploreError}
            </div>
          )}
          
          {/* Results */}
          {exploreResults.length > 0 && (
            <>
              {/* Stats bar */}
              <div className="flex items-center justify-between px-4 py-2 bg-white border-b border-slate-200">
                <div className="flex items-center gap-4 text-xs">
                  <span className="text-slate-500">
                    <span className="font-medium text-slate-700">{nTotal}</span> total
                  </span>
                  <span className="text-slate-500">
                    <span className="font-medium text-green-600">{nSuccessful}</span> successful
                  </span>
                  <span className="text-slate-500">
                    <span className="font-medium text-red-600">{nPareto}</span> Pareto
                  </span>
                </div>
                
                <button
                  onClick={() => setShowFilters(!showFilters)}
                  className={`flex items-center gap-1.5 px-2.5 py-1 text-xs rounded transition-colors ${
                    showFilters ? 'bg-slate-200 text-slate-700' : 'text-slate-500 hover:bg-slate-100'
                  }`}
                >
                  <Filter className="w-3 h-3" />
                  <span>Filters</span>
                </button>
              </div>
              
              {/* Filters */}
              {showFilters && (
                <div className="p-4 bg-slate-50 border-b border-slate-200">
                  <div className="grid grid-cols-4 gap-4">
                    <div>
                      <Slider
                        label="Max Displacement"
                        value={filters.maxDisplacement ?? maxDisp}
                        min={0}
                        max={Math.ceil(maxDisp)}
                        step={1}
                        unit=" mm"
                        onChange={(v) => setFilters({ maxDisplacement: v >= maxDisp ? null : v })}
                      />
                    </div>
                    <div>
                      <Slider
                        label="Max Bar Length"
                        value={filters.maxMemberLength ?? maxLen}
                        min={0}
                        max={Math.ceil(maxLen / 100) * 100}
                        step={100}
                        unit=" mm"
                        onChange={(v) => setFilters({ maxMemberLength: v >= maxLen ? null : v })}
                      />
                    </div>
                    <div>
                      <label className="block text-xs font-medium text-slate-600 mb-1.5">Topology</label>
                      <select
                        value={filters.topology ?? ''}
                        onChange={(e) => setFilters({ topology: e.target.value || null })}
                        className="w-full px-2.5 py-1.5 text-sm border border-slate-300 rounded-lg bg-white focus:outline-none focus:ring-2 focus:ring-slate-400"
                      >
                        <option value="">All</option>
                        {topologies.map(t => (
                          <option key={t} value={t}>{t}</option>
                        ))}
                      </select>
                    </div>
                    <div>
                      <label className="block text-xs font-medium text-slate-600 mb-1.5">Show</label>
                      <button
                        onClick={() => setFilters({ paretoOnly: !filters.paretoOnly })}
                        className={`w-full px-2.5 py-1.5 text-sm font-medium rounded-lg border transition-colors ${
                          filters.paretoOnly
                            ? 'bg-red-50 border-red-200 text-red-700'
                            : 'bg-white border-slate-300 text-slate-600 hover:bg-slate-50'
                        }`}
                      >
                        {filters.paretoOnly ? '★ Pareto Only' : 'All Designs'}
                      </button>
                    </div>
                  </div>
                </div>
              )}
              
              {/* Plot and selection */}
              <div className="p-4">
                <div className="flex gap-4">
                  {/* Pareto plot */}
                  <div className="flex-1">
                    <ParetoPlot 
                      designs={exploreResults} 
                      width={480}
                      height={320}
                    />
                    <div className="flex items-center justify-center gap-4 mt-2 text-xs text-slate-500">
                      <div className="flex items-center gap-1.5">
                        <div className="w-3 h-3 rounded-full bg-slate-300 border border-slate-400" />
                        <span>Dominated</span>
                      </div>
                      <div className="flex items-center gap-1.5">
                        <div className="w-3 h-3 rounded-full bg-red-500 border-2 border-red-700" />
                        <span>Pareto-optimal</span>
                      </div>
                      <div className="flex items-center gap-1.5">
                        <div className="w-3 h-3 rounded-full bg-green-500 border-2 border-green-700" />
                        <span>Selected</span>
                      </div>
                    </div>
                  </div>
                  
                  {/* Selected design info */}
                  <div className="w-64 shrink-0">
                    {selectedDesign ? (
                      <div className="h-full flex flex-col">
                        <div className="flex-1 p-4 bg-slate-50 rounded-lg">
                          <div className="flex items-center gap-2 mb-3">
                            <h4 className="text-sm font-semibold text-slate-900">
                              Design #{selectedDesign.index}
                            </h4>
                            {selectedDesign.is_pareto && (
                              <span className="px-1.5 py-0.5 text-xs font-medium bg-red-100 text-red-700 rounded">
                                ★ Pareto
                              </span>
                            )}
                          </div>
                          
                          <div className="space-y-2 text-xs">
                            <div className="grid grid-cols-2 gap-x-3 gap-y-1">
                              <span className="text-slate-500">Volume</span>
                              <span className="font-mono text-slate-700">{selectedDesign.volume?.toFixed(4)} m³</span>
                              
                              <span className="text-slate-500">Displacement</span>
                              <span className="font-mono text-slate-700">{selectedDesign.max_displacement_mm?.toFixed(2)} mm</span>
                              
                              <span className="text-slate-500">Length bins</span>
                              <span className="font-mono text-slate-700">{selectedDesign.n_length_bins}</span>
                            </div>
                            
                            <div className="pt-2 border-t border-slate-200 grid grid-cols-2 gap-x-3 gap-y-1">
                              <span className="text-slate-500">Size</span>
                              <span className="font-mono text-slate-700">{selectedDesign.width}×{selectedDesign.depth} m</span>
                              
                              <span className="text-slate-500">Grid</span>
                              <span className="font-mono text-slate-700">{selectedDesign.nx}×{selectedDesign.ny}</span>
                              
                              <span className="text-slate-500">Shape</span>
                              <span className="text-slate-700 capitalize">{selectedDesign.heightfield}</span>
                              
                              <span className="text-slate-500">Topology</span>
                              <span className="text-slate-700 capitalize">{selectedDesign.topology}</span>
                            </div>
                          </div>
                        </div>
                        
                        <button
                          onClick={applySelectedDesign}
                          className="mt-3 w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-green-600 text-white text-sm font-medium rounded-lg hover:bg-green-700 transition-colors"
                        >
                          <Check className="w-4 h-4" />
                          Apply This Design
                        </button>
                      </div>
                    ) : (
                      <div className="h-full flex items-center justify-center p-4 bg-slate-50 rounded-lg text-center">
                        <div>
                          <div className="w-10 h-10 mx-auto mb-2 rounded-full bg-slate-200 flex items-center justify-center">
                            <Search className="w-5 h-5 text-slate-400" />
                          </div>
                          <p className="text-sm text-slate-500">
                            Click a point on the chart to view design details
                          </p>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </>
          )}
          
          {/* Empty state */}
          {!isExploring && exploreResults.length === 0 && !exploreError && (
            <div className="p-8 text-center">
              <div className="w-12 h-12 mx-auto mb-3 rounded-full bg-gradient-to-br from-red-100 to-orange-100 flex items-center justify-center">
                <Search className="w-6 h-6 text-red-500" />
              </div>
              <h4 className="text-sm font-medium text-slate-900 mb-1">Explore Trade-offs</h4>
              <p className="text-xs text-slate-500 max-w-sm mx-auto">
                Generate design variants around your current parameters to find optimal trade-offs 
                between material volume, stiffness, and fabrication complexity.
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}





