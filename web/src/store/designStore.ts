// Zustand store for design state

import { create } from 'zustand'
import { DesignParams, DesignResult, NodeData, BarData, MetricsData, DEFAULT_PARAMS } from '@/lib/types'
import { generateDesign } from '@/lib/api'

interface DesignState {
  // Parameters
  params: DesignParams
  
  // Results
  isLoading: boolean
  error: string | null
  nodes: NodeData[]
  bars: BarData[]
  supportNodes: number[]
  metrics: MetricsData | null
  
  // Color mode for 3D view
  colorByForce: boolean
  
  // Actions
  setParam: <K extends keyof DesignParams>(key: K, value: DesignParams[K]) => void
  setParams: (params: Partial<DesignParams>) => void
  generate: () => Promise<void>
  setColorByForce: (value: boolean) => void
  reset: () => void
}

// Debounce helper
let debounceTimer: NodeJS.Timeout | null = null

export const useDesignStore = create<DesignState>((set, get) => ({
  // Initial state
  params: DEFAULT_PARAMS,
  isLoading: false,
  error: null,
  nodes: [],
  bars: [],
  supportNodes: [],
  metrics: null,
  colorByForce: false,
  
  // Set a single parameter
  setParam: (key, value) => {
    set((state) => ({
      params: { ...state.params, [key]: value },
    }))
    
    // Debounced generate
    if (debounceTimer) clearTimeout(debounceTimer)
    debounceTimer = setTimeout(() => {
      get().generate()
    }, 200)
  },
  
  // Set multiple parameters
  setParams: (params) => {
    set((state) => ({
      params: { ...state.params, ...params },
    }))
    
    if (debounceTimer) clearTimeout(debounceTimer)
    debounceTimer = setTimeout(() => {
      get().generate()
    }, 200)
  },
  
  // Generate design from API
  generate: async () => {
    const { params } = get()
    set({ isLoading: true, error: null })
    
    try {
      const result = await generateDesign(params)
      
      // Always update geometry if we have it (even if solve failed)
      if (result.nodes && result.bars) {
        set({
          isLoading: false,
          error: result.error || null,
          nodes: result.nodes,
          bars: result.bars,
          supportNodes: result.support_nodes || [],
          metrics: result.metrics || null,
        })
      } else {
        set({
          isLoading: false,
          error: result.error || 'Failed to generate geometry',
          nodes: [],
          bars: [],
          supportNodes: [],
          metrics: null,
        })
      }
    } catch (err) {
      set({
        isLoading: false,
        error: err instanceof Error ? err.message : 'Failed to generate',
        nodes: [],
        bars: [],
        supportNodes: [],
        metrics: null,
      })
    }
  },
  
  // Toggle force coloring
  setColorByForce: (value) => set({ colorByForce: value }),
  
  // Reset to defaults
  reset: () => {
    set({
      params: DEFAULT_PARAMS,
      nodes: [],
      bars: [],
      supportNodes: [],
      metrics: null,
      error: null,
    })
    get().generate()
  },
}))

