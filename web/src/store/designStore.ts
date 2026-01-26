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
  // 2D Frame types
  Frame2DParams,
  Frame2DResult,
  Frame2DNodeData,
  Frame2DElementData,
  Frame2DMetrics,
  ElementDiagramData,
  ReactionData,
  DEFAULT_FRAME2D_PARAMS,
  DiagramType,
  AnalysisMode,
  // Section extraction types
  SliceAxis,
  SectionExtractResult,
  DEFAULT_SECTION_PARAMS,
} from '@/lib/types'
import { generateDesign, exploreDesigns, generateFrame2D, extractSection } from '@/lib/api'

// Color mode for 3D visualization
export type ColorMode = 'none' | 'force' | 'utilization'

interface DesignState {
  // Analysis mode
  analysisMode: AnalysisMode
  
  // 3D Parameters
  params: DesignParams
  
  // 3D Results
  isLoading: boolean
  error: string | null
  nodes: NodeData[]
  bars: BarData[]
  supportNodes: number[]
  metrics: MetricsData | null
  
  // 3D Visualization
  colorMode: ColorMode
  showDeflectedShape: boolean
  deflectionScale3D: number
  
  // 2D Frame Parameters
  frame2dParams: Frame2DParams
  
  // 2D Frame Results
  frame2dLoading: boolean
  frame2dError: string | null
  frame2dNodes: Frame2DNodeData[]
  frame2dElements: Frame2DElementData[]
  frame2dReactions: ReactionData[]
  frame2dMetrics: Frame2DMetrics | null
  frame2dDiagrams: ElementDiagramData[]
  
  // 2D Visualization
  diagramType: DiagramType
  showDeflectedShape2D: boolean
  
  // Section extraction state
  sectionMode: boolean
  sliceAxis: SliceAxis
  slicePosition: number
  udlOverride: number | null
  sectionDeflectionScale: number
  sectionResult: SectionExtractResult | null
  sectionLoading: boolean
  sectionError: string | null
  
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
  
  // Mode actions
  setAnalysisMode: (mode: AnalysisMode) => void
  
  // 3D Actions
  setParam: <K extends keyof DesignParams>(key: K, value: DesignParams[K]) => void
  setParams: (params: Partial<DesignParams>) => void
  generate: () => Promise<void>
  setColorMode: (value: ColorMode) => void
  setShowDeflectedShape: (show: boolean) => void
  setDeflectionScale3D: (scale: number) => void
  reset: () => void
  
  // 2D Frame Actions
  setFrame2DParam: <K extends keyof Frame2DParams>(key: K, value: Frame2DParams[K]) => void
  setFrame2DParams: (params: Partial<Frame2DParams>) => void
  generateFrame2D: () => Promise<void>
  setDiagramType: (type: DiagramType) => void
  setShowDeflectedShape2D: (show: boolean) => void
  resetFrame2D: () => void
  
  // Section extraction actions
  setSectionMode: (enabled: boolean) => void
  setSliceAxis: (axis: SliceAxis) => void
  setSlicePosition: (position: number) => void
  setUdlOverride: (udl: number | null) => void
  setSectionDeflectionScale: (scale: number) => void
  extractSectionFromModel: () => Promise<void>
  
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
let debounceTimer2D: NodeJS.Timeout | null = null

export const useDesignStore = create<DesignState>((set, get) => ({
  // Initial state
  analysisMode: '3d',
  
  // 3D state
  params: DEFAULT_PARAMS,
  isLoading: false,
  error: null,
  nodes: [],
  bars: [],
  supportNodes: [],
  metrics: null,
  colorMode: 'none',
  showDeflectedShape: false,
  deflectionScale3D: 50,
  
  // 2D Frame state
  frame2dParams: DEFAULT_FRAME2D_PARAMS,
  frame2dLoading: false,
  frame2dError: null,
  frame2dNodes: [],
  frame2dElements: [],
  frame2dReactions: [],
  frame2dMetrics: null,
  frame2dDiagrams: [],
  diagramType: 'none',
  showDeflectedShape2D: true,
  
  // Section extraction initial state
  sectionMode: false,
  sliceAxis: DEFAULT_SECTION_PARAMS.slice_axis,
  slicePosition: DEFAULT_SECTION_PARAMS.slice_position,
  udlOverride: DEFAULT_SECTION_PARAMS.udl_override_kn_m ?? null,
  sectionDeflectionScale: DEFAULT_SECTION_PARAMS.deflection_scale,
  sectionResult: null,
  sectionLoading: false,
  sectionError: null,
  
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
  
  // Mode actions
  setAnalysisMode: (mode) => {
    set({ analysisMode: mode })
    // Trigger generation for the new mode
    if (mode === '3d') {
      get().generate()
    } else {
      get().generateFrame2D()
    }
  },
  
  // Set a single 3D parameter
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
  
  // Set multiple 3D parameters
  setParams: (params) => {
    set((state) => ({
      params: { ...state.params, ...params },
    }))
    
    if (debounceTimer) clearTimeout(debounceTimer)
    debounceTimer = setTimeout(() => {
      get().generate()
    }, 200)
  },
  
  // Generate 3D design from API
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
  
  // Set color mode
  setColorMode: (value) => set({ colorMode: value }),
  
  // 3D deflected shape toggle
  setShowDeflectedShape: (show) => set({ showDeflectedShape: show }),
  setDeflectionScale3D: (scale) => set({ deflectionScale3D: scale }),
  
  // Reset 3D to defaults
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
  
  // 2D Frame Actions
  setFrame2DParam: (key, value) => {
    set((state) => ({
      frame2dParams: { ...state.frame2dParams, [key]: value },
    }))
    
    // Debounced generate
    if (debounceTimer2D) clearTimeout(debounceTimer2D)
    debounceTimer2D = setTimeout(() => {
      get().generateFrame2D()
    }, 200)
  },
  
  setFrame2DParams: (params) => {
    set((state) => ({
      frame2dParams: { ...state.frame2dParams, ...params },
    }))
    
    if (debounceTimer2D) clearTimeout(debounceTimer2D)
    debounceTimer2D = setTimeout(() => {
      get().generateFrame2D()
    }, 200)
  },
  
  generateFrame2D: async () => {
    const { frame2dParams } = get()
    set({ frame2dLoading: true, frame2dError: null })
    
    try {
      const result = await generateFrame2D(frame2dParams)
      
      if (result.success && result.nodes && result.elements) {
        set({
          frame2dLoading: false,
          frame2dError: null,
          frame2dNodes: result.nodes,
          frame2dElements: result.elements,
          frame2dReactions: result.reactions || [],
          frame2dMetrics: result.metrics || null,
          frame2dDiagrams: result.diagrams || [],
        })
      } else {
        set({
          frame2dLoading: false,
          frame2dError: result.error || 'Failed to generate 2D frame',
          frame2dNodes: [],
          frame2dElements: [],
          frame2dReactions: [],
          frame2dMetrics: null,
          frame2dDiagrams: [],
        })
      }
    } catch (err) {
      set({
        frame2dLoading: false,
        frame2dError: err instanceof Error ? err.message : 'Failed to generate 2D frame',
        frame2dNodes: [],
        frame2dElements: [],
        frame2dReactions: [],
        frame2dMetrics: null,
        frame2dDiagrams: [],
      })
    }
  },
  
  setDiagramType: (type) => set({ diagramType: type }),
  setShowDeflectedShape2D: (show) => set({ showDeflectedShape2D: show }),
  
  resetFrame2D: () => {
    set({
      frame2dParams: DEFAULT_FRAME2D_PARAMS,
      frame2dNodes: [],
      frame2dElements: [],
      frame2dReactions: [],
      frame2dMetrics: null,
      frame2dDiagrams: [],
      frame2dError: null,
    })
    get().generateFrame2D()
  },
  
  // Section extraction actions
  setSectionMode: (enabled) => {
    set({ sectionMode: enabled })
    if (enabled) {
      // Trigger section extraction when enabled
      get().extractSectionFromModel()
    }
  },
  
  setSliceAxis: (axis) => {
    set({ sliceAxis: axis })
    if (get().sectionMode) {
      get().extractSectionFromModel()
    }
  },
  
  setSlicePosition: (position) => {
    set({ slicePosition: position })
    // Debounced extraction
    if (debounceTimer2D) clearTimeout(debounceTimer2D)
    debounceTimer2D = setTimeout(() => {
      if (get().sectionMode) {
        get().extractSectionFromModel()
      }
    }, 200)
  },
  
  setUdlOverride: (udl) => {
    set({ udlOverride: udl })
    if (get().sectionMode) {
      get().extractSectionFromModel()
    }
  },
  
  setSectionDeflectionScale: (scale) => {
    set({ sectionDeflectionScale: scale })
    if (get().sectionMode) {
      get().extractSectionFromModel()
    }
  },
  
  extractSectionFromModel: async () => {
    const { params, sliceAxis, slicePosition, udlOverride, sectionDeflectionScale } = get()
    set({ sectionLoading: true, sectionError: null })
    
    try {
      const result = await extractSection({
        // 3D design params
        width: params.width,
        depth: params.depth,
        nx: params.nx,
        ny: params.ny,
        min_height: params.min_height,
        max_height: params.max_height,
        heightfield: params.heightfield,
        topology: params.topology,
        support_layout: params.support_layout,
        A_cm2: params.A_cm2,
        gravity_kn: params.gravity_kn,
        // Section params
        slice_axis: sliceAxis,
        slice_position: slicePosition,
        udl_override_kn_m: udlOverride ?? undefined,
        deflection_scale: sectionDeflectionScale,
        n_diagram_points: 21,
      })
      
      if (result.success) {
        set({
          sectionLoading: false,
          sectionError: null,
          sectionResult: result,
        })
      } else {
        set({
          sectionLoading: false,
          sectionError: result.error || 'Failed to extract section',
          sectionResult: null,
        })
      }
    } catch (err) {
      set({
        sectionLoading: false,
        sectionError: err instanceof Error ? err.message : 'Failed to extract section',
        sectionResult: null,
      })
    }
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
        load_factor: params.load_factor,
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
