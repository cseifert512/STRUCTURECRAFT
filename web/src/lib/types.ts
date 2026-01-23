// Type definitions for StructureCraft

export interface DesignParams {
  width: number
  depth: number
  nx: number
  ny: number
  min_height: number
  max_height: number
  heightfield: 'flat' | 'paraboloid' | 'ridge' | 'saddle'
  topology: 'grid' | 'diagrid' | 'triangulated'
  support_layout: 'edges' | 'corners' | 'perimeter_4'
  A_cm2: number
  gravity_kn: number
  // Load factor for strength/serviceability
  load_factor: number  // 1.0 = service, 1.4/1.6 = ultimate
  // Material options
  material_type: 'timber' | 'steel'
  steel_section: string
  // Connection type
  connection_type: 'pinned' | 'semi_rigid' | 'rigid'
  // Advanced analysis options
  pdelta_enabled: boolean
  modal_enabled: boolean
}

export interface NodeData {
  id: number
  x: number
  y: number
  z: number
}

export interface BarData {
  id: number
  ni: number
  nj: number
  length: number
  force: number
  utilization?: number    // Stress utilization ratio (< 1.0 = pass)
  status?: 'PASS' | 'FAIL'
}

// Deflection check interfaces
export interface DeflectionLimitCheck {
  limit_mm: number
  actual_mm: number
  ratio: number
  status: 'PASS' | 'FAIL'
}

export interface DeflectionCheck {
  L_360: DeflectionLimitCheck
  L_240: DeflectionLimitCheck
  L_180: DeflectionLimitCheck
  governing: string
  overall_pass: boolean
}

// Design report card
export interface DesignReportCard {
  deflection_pass: boolean
  strength_pass: boolean
  buckling_pass: boolean
  shipping_pass: boolean
  overall_pass: boolean
  summary: string
}

export interface MetricsData {
  n_nodes: number
  n_bars: number
  n_supports: number
  max_displacement: number
  max_displacement_mm: number
  max_tension: number
  max_tension_kn: number
  max_compression: number
  max_compression_kn: number
  volume: number
  total_length: number
  n_length_bins: number
  max_member_length: number
  max_member_length_mm: number
  min_member_length: number
  // Engineering checks
  buckling_factor?: number       // Critical load factor (> 1.0 = safe)
  max_utilization?: number       // Worst member stress ratio
  n_failing_members?: number     // Count of members with util > 1.0
  critical_member_id?: number    // ID of worst member
  // P-Delta analysis
  pdelta_iterations?: number     // Iterations to converge (0 = linear)
  pdelta_amplification?: number  // Displacement amplification factor
  // Modal analysis
  natural_frequencies_hz?: number[]  // First N natural frequencies in Hz
  // Material info
  material_type?: string
  steel_section_name?: string
  // Connection info
  connection_type?: string
  // Load factor applied
  load_factor?: number
  // Deflection check results
  deflection_check?: DeflectionCheck
  // Design report card
  design_report_card?: DesignReportCard
}

export interface DesignResult {
  success: boolean
  error?: string
  nodes?: NodeData[]
  bars?: BarData[]
  support_nodes?: number[]
  metrics?: MetricsData
  params?: DesignParams
}

// Default parameter values
export const DEFAULT_PARAMS: DesignParams = {
  width: 10,
  depth: 8,
  nx: 5,
  ny: 4,
  min_height: 2.5,
  max_height: 4.0,
  heightfield: 'paraboloid',
  topology: 'grid',
  support_layout: 'edges',
  A_cm2: 8.0,
  gravity_kn: 50.0,
  load_factor: 1.0,
  material_type: 'timber',
  steel_section: 'HSS4x4x1/4',
  connection_type: 'pinned',
  pdelta_enabled: false,
  modal_enabled: false,
}

// Parameter constraints
export const PARAM_LIMITS = {
  width: { min: 4, max: 20, step: 0.5 },
  depth: { min: 4, max: 16, step: 0.5 },
  nx: { min: 2, max: 10, step: 1 },
  ny: { min: 2, max: 10, step: 1 },
  min_height: { min: 1.5, max: 6, step: 0.25 },
  max_height: { min: 1.5, max: 6, step: 0.25 },
  A_cm2: { min: 3, max: 20, step: 0.5 },
  gravity_kn: { min: 20, max: 150, step: 5 },
}

// Options for select fields
export const HEIGHTFIELD_OPTIONS = [
  { value: 'flat', label: 'Flat', description: 'Horizontal surface' },
  { value: 'paraboloid', label: 'Dome', description: 'Curved bowl shape' },
  { value: 'ridge', label: 'Ridge', description: 'Tent-like peak' },
  { value: 'saddle', label: 'Saddle', description: 'Hyperbolic surface' },
] as const

export const TOPOLOGY_OPTIONS = [
  { value: 'grid', label: 'Grid', description: 'Orthogonal with diagonals' },
  { value: 'diagrid', label: 'Diagrid', description: 'Diamond pattern' },
  { value: 'triangulated', label: 'Triangulated', description: 'Alternating mesh' },
] as const

export const SUPPORT_OPTIONS = [
  { value: 'edges', label: 'All Edges', description: 'Most stable' },
  { value: 'corners', label: 'Corners Only', description: 'Minimal supports' },
  { value: 'perimeter_4', label: 'Perimeter', description: 'Every 4th node' },
] as const

export const MATERIAL_OPTIONS = [
  { value: 'timber', label: 'Timber', description: 'Douglas Fir lumber' },
  { value: 'steel', label: 'Steel', description: 'AISC structural steel' },
] as const

export const STEEL_SECTION_OPTIONS = [
  { value: 'HSS6x6x1/4', label: 'HSS 6×6×1/4', description: 'Heavy HSS tube' },
  { value: 'HSS4x4x1/4', label: 'HSS 4×4×1/4', description: 'Medium HSS tube' },
  { value: 'HSS3x3x3/16', label: 'HSS 3×3×3/16', description: 'Light HSS tube' },
  { value: 'HSS4.000x0.250', label: 'HSS 4.0 Round', description: 'Round HSS' },
  { value: 'TUBE50x3', label: 'Tube 50×3', description: 'Small square tube' },
  { value: 'L4x4x1/4', label: 'L 4×4×1/4', description: 'Equal leg angle' },
] as const

export const CONNECTION_OPTIONS = [
  { value: 'pinned', label: 'Pinned', description: 'Truss connections (axial only)' },
  { value: 'semi_rigid', label: 'Semi-Rigid', description: 'Partial moment transfer' },
  { value: 'rigid', label: 'Rigid', description: 'Full moment connections' },
] as const

// =============================================================================
// Exploration Types
// =============================================================================

export interface ExploreParams {
  base_params: DesignParams
  n_designs: number
  seed: number
  variation_pct: number
}

export interface ExploreDesignData {
  index: number
  ok: boolean
  reason?: string
  // Parameters
  width: number
  depth: number
  nx: number
  ny: number
  min_height: number
  max_height: number
  heightfield: string
  topology: string
  support_layout: string
  A_cm2: number
  gravity_kn: number
  // Metrics (optional if failed)
  volume?: number
  max_displacement?: number
  max_displacement_mm?: number
  n_length_bins?: number
  max_member_length_mm?: number
  max_tension_kn?: number
  max_compression_kn?: number
  n_nodes?: number
  n_bars?: number
  // Pareto flag
  is_pareto: boolean
}

export interface ExploreResult {
  success: boolean
  error?: string
  n_total: number
  n_successful: number
  n_pareto: number
  designs: ExploreDesignData[]
}

// Default exploration settings
export const DEFAULT_EXPLORE_SETTINGS = {
  n_designs: 100,
  seed: 42,
  variation_pct: 0.3,
}

