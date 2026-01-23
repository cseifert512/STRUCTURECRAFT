'use client'

import { useEffect } from 'react'
import dynamic from 'next/dynamic'
import { useDesignStore } from '@/store/designStore'
import { ControlPanel } from '@/components/ControlPanel'
import { MetricsCard } from '@/components/MetricsCard'
import { ExportMenu } from '@/components/ExportMenu'
import { motion } from 'framer-motion'

// Dynamic import for Canvas3D to avoid SSR issues with Three.js
const Canvas3D = dynamic(() => import('@/components/Canvas3D').then(mod => mod.Canvas3D), {
  ssr: false,
  loading: () => (
    <div className="w-full h-full min-h-[400px] rounded-3xl bg-cream-200 flex items-center justify-center">
      <div className="text-stone-400">Loading 3D viewer...</div>
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
    <main className="min-h-screen bg-cream-100">
      {/* Header */}
      <header className="border-b border-cream-200 bg-white/50 backdrop-blur-sm sticky top-0 z-40">
        <div className="max-w-screen-2xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            {/* Logo */}
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-sage-400 to-forest-600 flex items-center justify-center shadow-soft">
              <svg className="w-5 h-5 text-white" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M12 2L2 7l10 5 10-5-10-5z" />
                <path d="M2 17l10 5 10-5" />
                <path d="M2 12l10 5 10-5" />
              </svg>
            </div>
            <div>
              <h1 className="font-serif text-xl text-forest-800">StructureCraft</h1>
              <p className="text-xs text-stone-400">3D Canopy Designer</p>
            </div>
          </div>
          
          {/* Status indicator */}
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${isLoading ? 'bg-clay-400 animate-pulse' : 'bg-sage-500'}`} />
            <span className="text-xs text-stone-500">
              {isLoading ? 'Computing...' : 'Ready'}
            </span>
          </div>
        </div>
      </header>
      
      {/* Main content */}
      <div className="max-w-screen-2xl mx-auto px-4 md:px-6 py-4 md:py-6">
        {/* Desktop layout: 3 columns */}
        <div className="hidden lg:grid grid-cols-12 gap-6 h-[calc(100vh-120px)]">
          
          {/* Left panel - Controls */}
          <motion.aside
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
            className="col-span-3 card-organic overflow-hidden"
          >
            <ControlPanel />
          </motion.aside>
          
          {/* Center - 3D Canvas */}
          <motion.div
            initial={{ opacity: 0, scale: 0.98 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="col-span-6"
          >
            <Canvas3D />
            
            {/* Legend */}
            <div className="mt-4 flex items-center justify-center gap-6 text-xs text-stone-500">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-[#4A6741]" />
                <span>Supports</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-[#C4846C]" />
                <span>Tension</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-[#6B8F5B]" />
                <span>Compression</span>
              </div>
            </div>
          </motion.div>
          
          {/* Right panel - Metrics & Export */}
          <motion.aside
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
            className="col-span-3 space-y-4 overflow-y-auto"
          >
            <MetricsCard />
            <ExportMenu />
            
            {/* Help card */}
            <div className="card-organic bg-sage-50/50">
              <h4 className="text-sm font-medium text-forest-700 mb-2">Quick Tips</h4>
              <ul className="text-xs text-stone-500 space-y-1.5">
                <li className="flex items-start gap-2">
                  <span className="text-sage-500">•</span>
                  <span>Drag to rotate the 3D view</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-sage-500">•</span>
                  <span>Scroll to zoom in/out</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-sage-500">•</span>
                  <span>Use &quot;Grid&quot; topology for stability</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-sage-500">•</span>
                  <span>Fewer length bins = easier fabrication</span>
                </li>
              </ul>
            </div>
          </motion.aside>
          
        </div>
        
        {/* Tablet/Mobile layout: stacked */}
        <div className="lg:hidden space-y-4">
          {/* 3D Canvas - Full width on top */}
          <motion.div
            initial={{ opacity: 0, scale: 0.98 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5 }}
            className="h-[50vh] min-h-[300px]"
          >
            <Canvas3D />
          </motion.div>
          
          {/* Legend */}
          <div className="flex items-center justify-center gap-4 text-xs text-stone-500 py-2">
            <div className="flex items-center gap-1.5">
              <div className="w-2.5 h-2.5 rounded-full bg-[#4A6741]" />
              <span>Supports</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-2.5 h-2.5 rounded-full bg-[#C4846C]" />
              <span>Tension</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-2.5 h-2.5 rounded-full bg-[#6B8F5B]" />
              <span>Compression</span>
            </div>
          </div>
          
          {/* Controls and Metrics side by side on tablet */}
          <div className="grid md:grid-cols-2 gap-4">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.1 }}
              className="card-organic"
            >
              <ControlPanel />
            </motion.div>
            
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="space-y-4"
            >
              <MetricsCard />
              <ExportMenu />
            </motion.div>
          </div>
        </div>
      </div>
      
      {/* Footer */}
      <footer className="border-t border-cream-200 bg-white/30">
        <div className="max-w-screen-2xl mx-auto px-6 py-3 flex items-center justify-between text-xs text-stone-400">
          <span>StructureCraft v1.0</span>
          <span>Real-time 3D Spaceframe Design</span>
        </div>
      </footer>
    </main>
  )
}

