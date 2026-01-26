'use client'

import { useEffect } from 'react'
import dynamic from 'next/dynamic'
import { useDesignStore } from '@/store/designStore'
import { ControlPanel } from '@/components/ControlPanel'
import { MetricsCard } from '@/components/MetricsCard'
import { ExportMenu } from '@/components/ExportMenu'
import { ExplorePanel } from '@/components/ExplorePanel'
import { DiagramPanel } from '@/components/DiagramPanel'
import SectionControls from '@/components/SectionControls'

// Dynamic import for Canvas3D to avoid SSR issues with Three.js
const Canvas3D = dynamic(() => import('@/components/Canvas3D').then(mod => mod.Canvas3D), {
  ssr: false,
  loading: () => (
    <div className="w-full h-full min-h-[400px] bg-slate-100 flex items-center justify-center border border-slate-200">
      <div className="text-slate-400 text-sm">Loading viewer...</div>
    </div>
  ),
})

// Dynamic import for Canvas2D (SVG-based, less critical but consistent)
const Canvas2D = dynamic(() => import('@/components/Canvas2D').then(mod => mod.Canvas2D), {
  ssr: false,
  loading: () => (
    <div className="w-full h-full min-h-[400px] bg-slate-100 flex items-center justify-center border border-slate-200">
      <div className="text-slate-400 text-sm">Loading 2D viewer...</div>
    </div>
  ),
})

// Dynamic import for Section Canvas2D (reuses Canvas2D logic but with section data)
const SectionCanvas2D = dynamic(() => import('@/components/SectionCanvas2D').then(mod => mod.SectionCanvas2D), {
  ssr: false,
  loading: () => (
    <div className="w-full h-full min-h-[400px] bg-slate-100 flex items-center justify-center border border-slate-200">
      <div className="text-slate-400 text-sm">Loading section viewer...</div>
    </div>
  ),
})

export default function Home() {
  const { generate, generateFrame2D, analysisMode, isLoading, frame2dLoading, sectionMode, sectionLoading, sectionResult } = useDesignStore()
  
  // Initial generation on mount
  useEffect(() => {
    if (analysisMode === '3d') {
      generate()
    } else {
      generateFrame2D()
    }
  }, [generate, generateFrame2D, analysisMode])

  const loading = analysisMode === '3d' ? (isLoading || sectionLoading) : frame2dLoading
  
  // Check if we're in section view mode (3D with section enabled)
  const showSectionView = analysisMode === '3d' && sectionMode
  
  return (
    <main className="min-h-screen bg-slate-50">
      {/* Header */}
      <header className="h-14 border-b border-slate-200 bg-white sticky top-0 z-40">
        <div className="h-full max-w-screen-2xl mx-auto px-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-slate-900 flex items-center justify-center">
              <svg className="w-4 h-4 text-white" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M12 2L2 7l10 5 10-5-10-5z" />
                <path d="M2 17l10 5 10-5" />
                <path d="M2 12l10 5 10-5" />
              </svg>
            </div>
            <div>
              <h1 className="text-sm font-semibold text-slate-900 tracking-tight">STRUCTURECRAFT</h1>
            </div>
          </div>
          
          <div className="flex items-center gap-4">
            {/* Mode indicator */}
            <div className="hidden sm:flex items-center gap-2 px-2 py-1 bg-slate-100 rounded text-xs">
              <span className={analysisMode === '3d' ? 'font-medium text-slate-900' : 'text-slate-400'}>3D</span>
              <span className="text-slate-300">|</span>
              <span className={analysisMode === '2d' ? 'font-medium text-slate-900' : 'text-slate-400'}>2D</span>
            </div>
            
            <div className="flex items-center gap-2">
              <div className={`w-1.5 h-1.5 rounded-full ${loading ? 'bg-amber-500 animate-pulse' : 'bg-green-500'}`} />
              <span className="text-xs text-slate-500 font-mono">
                {loading ? 'COMPUTING' : 'READY'}
              </span>
            </div>
          </div>
        </div>
      </header>
      
      {/* Main content */}
      <div className="max-w-screen-2xl mx-auto p-4">
        {/* Desktop layout */}
        <div className="hidden lg:flex lg:flex-col gap-4">
          {/* Main row */}
          <div className="grid grid-cols-12 gap-4" style={{ height: 'calc(100vh - 200px)', minHeight: '500px' }}>
            
            {/* Left panel - Controls */}
            <aside className={`${showSectionView ? 'col-span-2' : 'col-span-3'} card rounded-lg overflow-hidden flex flex-col`}>
              <div className="flex-1 overflow-y-auto">
                <ControlPanel />
                {/* Section controls in 3D mode */}
                {analysisMode === '3d' && <SectionControls />}
              </div>
            </aside>
            
            {/* Center - Canvas (3D or 2D based on mode) */}
            {/* When section mode is enabled, show side-by-side 3D and 2D */}
            {showSectionView ? (
              <>
                {/* 3D View with cutting plane */}
                <div className="col-span-5 flex flex-col gap-2">
                  <div className="flex-1 card rounded-lg overflow-hidden">
                    <Canvas3D />
                  </div>
                  <div className="text-center text-xs text-slate-500 py-1">
                    3D Spaceframe with Section Plane
                  </div>
                </div>
                
                {/* 2D Section View */}
                <div className="col-span-5 flex flex-col gap-2">
                  <div className="flex-1 card rounded-lg overflow-hidden relative">
                    {sectionResult?.success && sectionResult.frame_result ? (
                      <SectionCanvas2D frameResult={sectionResult.frame_result} />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center bg-slate-100">
                        <div className="text-center text-slate-400 text-sm">
                          {sectionLoading ? (
                            <div className="flex flex-col items-center gap-2">
                              <div className="w-5 h-5 border-2 border-slate-300 border-t-slate-600 rounded-full animate-spin" />
                              <span>Extracting section...</span>
                            </div>
                          ) : (
                            <span>Enable section view to see 2D analysis</span>
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                  <div className="text-center text-xs text-slate-500 py-1">
                    Extracted 2D Section with Force Diagrams
                  </div>
                </div>
              </>
            ) : (
              <div className="col-span-6 flex flex-col gap-4">
                <div className="flex-1 card rounded-lg overflow-hidden">
                  {analysisMode === '3d' ? <Canvas3D /> : <Canvas2D />}
                </div>
                
                {/* Legend - different for 3D and 2D */}
                {analysisMode === '3d' ? (
                  <div className="flex items-center justify-center gap-6 text-xs text-slate-500 py-2">
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 rounded-full bg-slate-900" />
                      <span>Support</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-0.5 bg-[#E63946]" />
                      <span>Tension</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-0.5 bg-[#457B9D]" />
                      <span>Compression</span>
                    </div>
                  </div>
                ) : (
                  <div className="flex items-center justify-center gap-6 text-xs text-slate-500 py-2">
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-0.5 bg-[#2C3E50]" />
                      <span>Undeformed</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-0.5 bg-[#E74C3C]" style={{ borderBottom: '2px dashed #E74C3C' }} />
                      <span>Deflected</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 bg-[#9B59B6] opacity-30 border border-[#9B59B6]" />
                      <span>Force Diagram</span>
                    </div>
                  </div>
                )}
              </div>
            )}
            
            {/* Right panel - Metrics & Export (3D) or DiagramPanel (2D) */}
            <aside className={`${showSectionView ? 'hidden' : 'col-span-3'} space-y-4 overflow-y-auto`}>
              {analysisMode === '3d' ? (
                <>
                  <MetricsCard />
                  <ExportMenu />
                </>
              ) : (
                <DiagramPanel />
              )}
            </aside>
            
          </div>
          
          {/* Explore panel - full width below main content (3D mode only, not in section view) */}
          {analysisMode === '3d' && !sectionMode && <ExplorePanel />}
        </div>
        
        {/* Mobile/Tablet layout */}
        <div className="lg:hidden space-y-4">
          <div className="card rounded-lg overflow-hidden h-[50vh] min-h-[300px]">
            {analysisMode === '3d' ? <Canvas3D /> : <Canvas2D />}
          </div>
          
          {/* Legend */}
          {analysisMode === '3d' ? (
            <div className="flex items-center justify-center gap-4 text-xs text-slate-500 py-2">
              <div className="flex items-center gap-1.5">
                <div className="w-2 h-2 rounded-full bg-slate-900" />
                <span>Support</span>
              </div>
              <div className="flex items-center gap-1.5">
                <div className="w-2.5 h-0.5 bg-[#E63946]" />
                <span>Tension</span>
              </div>
              <div className="flex items-center gap-1.5">
                <div className="w-2.5 h-0.5 bg-[#457B9D]" />
                <span>Compression</span>
              </div>
            </div>
          ) : (
            <div className="flex items-center justify-center gap-4 text-xs text-slate-500 py-2">
              <div className="flex items-center gap-1.5">
                <div className="w-2.5 h-0.5 bg-[#2C3E50]" />
                <span>Original</span>
              </div>
              <div className="flex items-center gap-1.5">
                <div className="w-2.5 h-0.5 bg-[#E74C3C]" />
                <span>Deflected</span>
              </div>
            </div>
          )}
          
          <div className="grid md:grid-cols-2 gap-4">
            <div className="card rounded-lg overflow-hidden">
              <ControlPanel />
            </div>
            <div className="space-y-4">
              {analysisMode === '3d' ? (
                <>
                  <MetricsCard />
                  <ExportMenu />
                </>
              ) : (
                <DiagramPanel />
              )}
            </div>
          </div>
          
          {/* Explore panel (3D mode only) */}
          {analysisMode === '3d' && <ExplorePanel />}
        </div>
      </div>
    </main>
  )
}
