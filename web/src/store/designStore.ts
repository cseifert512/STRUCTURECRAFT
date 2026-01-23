// Zustand store for design state

import { create } from 'zustand'
import { 
  DesignParams, 
  DesignResult, 
  NodeData, 
  BarData, 
  MetricsData, 
  DEFAULT_PARAMS,
  ExploreDesignData,
  ExploreResult,
  DEFAULT_EXPLORE_SETTINGS,
} from '@/lib/types'
import { generateDesign, exploreDesigns } from '@/lib/api'

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
  
  // Exploration state
  exploreSettings: {
    n_designs: number
    seed: number
    variation_pct: number
  }
  isExploring: boolean
  exploreError: string | null
  exploreResults: ExploreDesignData[]
  nTotal: number
  nSuccessful: number
  nPareto: number
  selectedDesignIndex: number | null
  showExplorePanel: boolean
  
  // Filter state
  filters: {
    maxDisplacement: number | null
    maxMemberLength: number | null
    topology: string | null
    paretoOnly: boolean
  }
  
  // Actions
  setParam: <K extends keyof DesignParams>(key: K, value: DesignParams[K]) => void
  setParams: (params: Partial<DesignParams>) => void
  generate: () => Promise<void>
  setColorByForce: (value: boolean) => void
  reset: () => void
  
  // Exploration actions
  setExploreSettings: (settings: Partial<typeof DEFAULT_EXPLORE_SETTINGS>) => void
  runExploration: () => Promise<void>
  clearExploration: () => void
  selectDesign: (index: number | null) => void
  applySelectedDesign: () => void
  setShowExplorePanel: (show: boolean) => void
  setFilters: (filters: Partial<DesignState['filters']>) => void
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
  
  // Exploration initial state
  exploreSettings: { ...DEFAULT_EXPLORE_SETTINGS },
  isExploring: false,
  exploreError: null,
  exploreResults: [],
  nTotal: 0,
  nSuccessful: 0,
  nPareto: 0,
  selectedDesignIndex: null,
  showExplorePanel: false,
  filters: {
    maxDisplacement: null,
    maxMemberLength: null,
    topology: null,
    paretoOnly: false,
  },
  
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
  
  // Exploration actions
  setExploreSettings: (settings) => {
    set((state) => ({
      exploreSettings: { ...state.exploreSettings, ...settings },
    }))
  },
  
  runExploration: async () => {
    const { params, exploreSettings } = get()
    set({ 
      isExploring: true, 
      exploreError: null,
      showExplorePanel: true,
    })
    
    try {
      const result = await exploreDesigns({
        base_params: params,
        n_designs: exploreSettings.n_designs,
        seed: exploreSettings.seed,
        variation_pct: exploreSettings.variation_pct,
      })
      
      if (result.success) {
        set({
          isExploring: false,
          exploreResults: result.designs,
          nTotal: result.n_total,
          nSuccessful: result.n_successful,
          nPareto: result.n_pareto,
          selectedDesignIndex: null,
        })
      } else {
        set({
          isExploring: false,
          exploreError: result.error || 'Exploration failed',
          exploreResults: [],
        })
      }
    } catch (err) {
      set({
        isExploring: false,
        exploreError: err instanceof Error ? err.message : 'Exploration failed',
        exploreResults: [],
      })
    }
  },
  
  clearExploration: () => {
    set({
      exploreResults: [],
      nTotal: 0,
      nSuccessful: 0,
      nPareto: 0,
      selectedDesignIndex: null,
      exploreError: null,
      showExplorePanel: false,
    })
  },
  
  selectDesign: (index) => {
    set({ selectedDesignIndex: index })
  },
  
  applySelectedDesign: () => {
    const { exploreResults, selectedDesignIndex, params } = get()
    if (selectedDesignIndex === null) return
    
    const design = exploreResults.find(d => d.index === selectedDesignIndex)
    if (!design) return
    
    // Apply design parameters (keep material, connection, and analysis settings from current params)
    set({
      params: {
        width: design.width,
        depth: design.depth,
        nx: design.nx,
        ny: design.ny,
        min_height: design.min_height,
        max_height: design.max_height,
        heightfield: design.heightfield as DesignParams['heightfield'],
        topology: design.topology as DesignParams['topology'],
        support_layout: design.support_layout as DesignParams['support_layout'],
        A_cm2: design.A_cm2,
        gravity_kn: design.gravity_kn,
        material_type: params.material_type,
        steel_section: params.steel_section,
        connection_type: params.connection_type,
        pdelta_enabled: params.pdelta_enabled,
        modal_enabled: params.modal_enabled,
      },
      selectedDesignIndex: null,
    })
    
    // Regenerate with new params
    get().generate()
  },
  
  setShowExplorePanel: (show) => {
    set({ showExplorePanel: show })
  },
  
  setFilters: (filters) => {
    set((state) => ({
      filters: { ...state.filters, ...filters },
    }))
  },
}))

