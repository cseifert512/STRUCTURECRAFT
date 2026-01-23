'use client'

import { useEffect } from 'react'
import dynamic from 'next/dynamic'
import { useDesignStore } from '@/store/designStore'
import { ControlPanel } from '@/components/ControlPanel'
import { MetricsCard } from '@/components/MetricsCard'
import { ExportMenu } from '@/components/ExportMenu'
import { ExplorePanel } from '@/components/ExplorePanel'

// Dynamic import for Canvas3D to avoid SSR issues with Three.js
const Canvas3D = dynamic(() => import('@/components/Canvas3D').then(mod => mod.Canvas3D), {
  ssr: false,
  loading: () => (
    <div className="w-full h-full min-h-[400px] bg-slate-100 flex items-center justify-center border border-slate-200">
      <div className="text-slate-400 text-sm">Loading viewer...</div>
    </div>
  ),
})

export default function Home() {
  const { generate, isLoading } = useDesignStore()
  
  // Initial generation on mount
  useEffect(() => {
    generate()
  }, [generate])
  
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
            <div className="flex items-center gap-2">
              <div className={`w-1.5 h-1.5 rounded-full ${isLoading ? 'bg-amber-500 animate-pulse' : 'bg-green-500'}`} />
              <span className="text-xs text-slate-500 font-mono">
                {isLoading ? 'COMPUTING' : 'READY'}
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
            <aside className="col-span-3 card rounded-lg overflow-hidden flex flex-col">
              <div className="flex-1 overflow-y-auto">
                <ControlPanel />
              </div>
            </aside>
            
            {/* Center - 3D Canvas */}
            <div className="col-span-6 flex flex-col gap-4">
              <div className="flex-1 card rounded-lg overflow-hidden">
                <Canvas3D />
              </div>
              
              {/* Legend */}
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
            </div>
            
            {/* Right panel - Metrics & Export */}
            <aside className="col-span-3 space-y-4 overflow-y-auto">
              <MetricsCard />
              <ExportMenu />
            </aside>
            
          </div>
          
          {/* Explore panel - full width below main content */}
          <ExplorePanel />
        </div>
        
        {/* Mobile/Tablet layout */}
        <div className="lg:hidden space-y-4">
          <div className="card rounded-lg overflow-hidden h-[50vh] min-h-[300px]">
            <Canvas3D />
          </div>
          
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
          
          <div className="grid md:grid-cols-2 gap-4">
            <div className="card rounded-lg overflow-hidden">
              <ControlPanel />
            </div>
            <div className="space-y-4">
              <MetricsCard />
              <ExportMenu />
            </div>
          </div>
          
          {/* Explore panel */}
          <ExplorePanel />
        </div>
      </div>
    </main>
  )
}
