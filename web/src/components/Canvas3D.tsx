'use client'

import { useMemo } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, Environment, ContactShadows, Line, Plane } from '@react-three/drei'
import * as THREE from 'three'
import { useDesignStore, ColorMode } from '@/store/designStore'
import { NodeData, BarData, SliceAxis } from '@/lib/types'

// Color palette - architectural/engineering style
const COLORS = {
  tension: new THREE.Color('#E63946'),      // Red for tension
  compression: new THREE.Color('#457B9D'),  // Steel blue for compression
  neutral: new THREE.Color('#6C757D'),      // Neutral gray
  support: new THREE.Color('#212529'),      // Dark for supports
  node: new THREE.Color('#495057'),         // Dark gray for nodes
  ground: new THREE.Color('#F8F9FA'),       // Light gray ground
  deflected: new THREE.Color('#E74C3C'),    // Coral red for deflected shape
  // Utilization colors
  utilSafe: new THREE.Color('#2D6A4F'),     // Green for safe (<70%)
  utilWarning: new THREE.Color('#E9C46A'),  // Yellow for warning (70-100%)
  utilFail: new THREE.Color('#D62828'),     // Red for failing (>100%)
  // Section plane
  sectionPlane: new THREE.Color('#4CAF50'), // Green for section plane
  sectionHighlight: new THREE.Color('#81C784'), // Lighter green for highlighted bars
}

interface StructureProps {
  nodes: NodeData[]
  bars: BarData[]
  supportNodes: number[]
  colorMode: ColorMode
  showDeflected: boolean
  deflectionScale: number
  maxDisplacement: number
  // Section mode props
  sectionMode?: boolean
  sliceAxis?: SliceAxis
  slicePosition?: number
  width?: number
  depth?: number
}

function Structure({ 
  nodes, bars, supportNodes, colorMode, showDeflected, deflectionScale, maxDisplacement,
  sectionMode, sliceAxis, slicePosition, width, depth 
}: StructureProps) {
  // Create node lookup map
  const nodeMap = useMemo(() => {
    const map = new Map<number, NodeData>()
    nodes.forEach(node => map.set(node.id, node))
    return map
  }, [nodes])
  
  // Calculate force range for color mapping (separate for tension and compression)
  const forceRange = useMemo(() => {
    if (bars.length === 0) return { maxTension: 1, maxCompression: 1 }
    const forces = bars.map(b => b.force)
    const tensions = forces.filter(f => f > 0)
    const compressions = forces.filter(f => f < 0)
    return {
      maxTension: tensions.length > 0 ? Math.max(...tensions) : 1,
      maxCompression: compressions.length > 0 ? Math.abs(Math.min(...compressions)) : 1,
    }
  }, [bars])
  
  // Calculate center of structure to center the model
  const center = useMemo(() => {
    if (nodes.length === 0) return { x: 0, y: 0 }
    const xs = nodes.map(n => n.x)
    const ys = nodes.map(n => n.y)
    return {
      x: (Math.min(...xs) + Math.max(...xs)) / 2,
      y: (Math.min(...ys) + Math.max(...ys)) / 2,
    }
  }, [nodes])
  
  // Get color for a bar based on color mode
  const getBarColor = (bar: BarData) => {
    if (colorMode === 'none') return COLORS.neutral
    
    if (colorMode === 'utilization') {
      // Color by stress utilization ratio
      const util = bar.utilization ?? 0
      if (util >= 1.0) {
        // Failing - red, intensity based on how much over
        const overUtil = Math.min((util - 1.0) * 2, 1)
        return COLORS.utilFail.clone().lerp(new THREE.Color('#8B0000'), overUtil)
      } else if (util >= 0.7) {
        // Warning - yellow to orange gradient
        const warningIntensity = (util - 0.7) / 0.3
        return COLORS.utilSafe.clone().lerp(COLORS.utilWarning, warningIntensity)
      } else {
        // Safe - green, brighter for lower utilization
        const safeIntensity = util / 0.7
        return COLORS.utilSafe.clone().lerp(COLORS.neutral, 1 - safeIntensity * 0.5)
      }
    }
    
    // Force-based coloring (fallback)
    const force = bar.force
    if (force > 0) {
      // Tension (positive) - red, scale by max tension
      const intensity = Math.min(force / forceRange.maxTension, 1)
      return COLORS.neutral.clone().lerp(COLORS.tension, 0.3 + intensity * 0.7)
    } else if (force < 0) {
      // Compression (negative) - blue, scale by max compression  
      const intensity = Math.min(Math.abs(force) / forceRange.maxCompression, 1)
      return COLORS.neutral.clone().lerp(COLORS.compression, 0.3 + intensity * 0.7)
    }
    return COLORS.neutral
  }
  
  // Transform coordinates: engineering (X, Y, Z-up) to Three.js (X, Z, Y-up)
  const toThreeCoords = (x: number, y: number, z: number): [number, number, number] => {
    return [x - center.x, z, y - center.y]
  }

  // Calculate slice coordinate for section mode
  const sliceCoord = useMemo(() => {
    if (!sectionMode || slicePosition === undefined) return 0
    const maxCoord = sliceAxis === 'x' ? (width ?? 10) : (depth ?? 8)
    return slicePosition * maxCoord
  }, [sectionMode, sliceAxis, slicePosition, width, depth])

  // Check if a bar is on the slice plane (both nodes near the slice)
  const isBarOnSlice = (bar: BarData): boolean => {
    if (!sectionMode) return false
    const startNode = nodeMap.get(bar.ni)
    const endNode = nodeMap.get(bar.nj)
    if (!startNode || !endNode) return false
    
    const tolerance = 0.1 // 10cm tolerance
    if (sliceAxis === 'x') {
      return Math.abs(startNode.x - sliceCoord) < tolerance && 
             Math.abs(endNode.x - sliceCoord) < tolerance
    } else {
      return Math.abs(startNode.y - sliceCoord) < tolerance && 
             Math.abs(endNode.y - sliceCoord) < tolerance
    }
  }

  // Calculate simulated deflected position (using force to estimate deflection direction)
  // In a real implementation, actual displacement data would come from the solver
  const getDeflectedCoords = (node: NodeData, bar?: BarData): [number, number, number] => {
    if (!showDeflected || maxDisplacement === 0) {
      return toThreeCoords(node.x, node.y, node.z)
    }
    
    // For visualization purposes, simulate deflection based on gravity (z direction)
    // Nodes at the top (high z) will appear to move down more
    // This is a rough approximation - real implementation would use solver output
    const isSupport = supportNodes.includes(node.id)
    if (isSupport) {
      return toThreeCoords(node.x, node.y, node.z)
    }
    
    // Calculate node height relative to supports
    const maxZ = Math.max(...nodes.map(n => n.z))
    const minZ = Math.min(...nodes.filter(n => supportNodes.includes(n.id)).map(n => n.z))
    const heightRatio = (node.z - minZ) / (maxZ - minZ || 1)
    
    // Apply deflection primarily in Z (gravity direction)
    const scaledDeflection = maxDisplacement * deflectionScale
    const dz = -heightRatio * scaledDeflection * 0.5  // Downward deflection
    
    return toThreeCoords(node.x, node.y, node.z + dz)
  }
  
  // Calculate section plane geometry
  const sectionPlaneGeometry = useMemo(() => {
    if (!sectionMode || !width || !depth) return null
    
    const maxZ = Math.max(...nodes.map(n => n.z), 5)
    const minZ = Math.min(...nodes.map(n => n.z), 0)
    const planeHeight = maxZ - minZ + 1
    
    if (sliceAxis === 'y') {
      // Slice at constant Y - plane perpendicular to Y axis
      // Plane extends in X and Z directions
      const planeWidth = width + 2
      return {
        position: toThreeCoords(center.x, sliceCoord, (maxZ + minZ) / 2),
        rotation: [0, 0, 0] as [number, number, number],
        args: [planeWidth, planeHeight] as [number, number],
      }
    } else {
      // Slice at constant X - plane perpendicular to X axis
      // Plane extends in Y and Z directions
      const planeWidth = depth + 2
      return {
        position: toThreeCoords(sliceCoord, center.y, (maxZ + minZ) / 2),
        rotation: [0, Math.PI / 2, 0] as [number, number, number],
        args: [planeWidth, planeHeight] as [number, number],
      }
    }
  }, [sectionMode, sliceAxis, sliceCoord, width, depth, nodes, center])
  
  return (
    <group>
      {/* Section cutting plane */}
      {sectionMode && sectionPlaneGeometry && (
        <mesh 
          position={sectionPlaneGeometry.position}
          rotation={sectionPlaneGeometry.rotation}
        >
          <planeGeometry args={sectionPlaneGeometry.args} />
          <meshStandardMaterial
            color={COLORS.sectionPlane}
            transparent
            opacity={0.15}
            side={THREE.DoubleSide}
            depthWrite={false}
          />
        </mesh>
      )}

      {/* Section plane edge lines */}
      {sectionMode && sectionPlaneGeometry && (
        <lineSegments position={sectionPlaneGeometry.position} rotation={sectionPlaneGeometry.rotation}>
          <edgesGeometry args={[new THREE.PlaneGeometry(...sectionPlaneGeometry.args)]} />
          <lineBasicMaterial color={COLORS.sectionPlane} linewidth={2} />
        </lineSegments>
      )}

      {/* Render bars as lines - original structure */}
      {bars.map((bar) => {
        const startNode = nodeMap.get(bar.ni)
        const endNode = nodeMap.get(bar.nj)
        if (!startNode || !endNode) return null
        
        const onSlice = isBarOnSlice(bar)
        const baseColor = getBarColor(bar)
        const color = onSlice ? COLORS.sectionHighlight : baseColor
        // Thicker lines for failing members or bars on slice
        const lineWidth = onSlice ? 3 : ((bar.utilization ?? 0) >= 1.0 ? 2.5 : 1.5)
        
        return (
          <Line
            key={bar.id}
            points={[
              toThreeCoords(startNode.x, startNode.y, startNode.z),
              toThreeCoords(endNode.x, endNode.y, endNode.z),
            ]}
            color={showDeflected && maxDisplacement > 0 ? COLORS.neutral : color}
            lineWidth={showDeflected && maxDisplacement > 0 ? 1 : lineWidth}
            opacity={showDeflected && maxDisplacement > 0 ? 0.3 : (sectionMode && !onSlice ? 0.4 : 1)}
            transparent={showDeflected && maxDisplacement > 0 || (sectionMode && !onSlice)}
          />
        )
      })}
      
      {/* Render deflected shape if enabled */}
      {showDeflected && maxDisplacement > 0 && bars.map((bar) => {
        const startNode = nodeMap.get(bar.ni)
        const endNode = nodeMap.get(bar.nj)
        if (!startNode || !endNode) return null
        
        return (
          <Line
            key={`deflected-${bar.id}`}
            points={[
              getDeflectedCoords(startNode, bar),
              getDeflectedCoords(endNode, bar),
            ]}
            color={COLORS.deflected}
            lineWidth={2}
            dashed
            dashSize={0.1}
            gapSize={0.05}
          />
        )
      })}
      
      {/* Render nodes as small spheres */}
      {nodes.map((node) => {
        const isSupport = supportNodes.includes(node.id)
        const color = isSupport ? COLORS.support : COLORS.node
        const size = isSupport ? 0.08 : 0.03
        
        return (
          <mesh key={node.id} position={toThreeCoords(node.x, node.y, node.z)}>
            <sphereGeometry args={[size, 12, 12]} />
            <meshStandardMaterial
              color={color}
              roughness={0.6}
              metalness={0.3}
            />
          </mesh>
        )
      })}
      
      {/* Render deflected nodes if enabled */}
      {showDeflected && maxDisplacement > 0 && nodes.map((node) => {
        const isSupport = supportNodes.includes(node.id)
        if (isSupport) return null  // Supports don't move
        
        return (
          <mesh key={`deflected-node-${node.id}`} position={getDeflectedCoords(node)}>
            <sphereGeometry args={[0.03, 12, 12]} />
            <meshStandardMaterial
              color={COLORS.deflected}
              roughness={0.6}
              metalness={0.3}
            />
          </mesh>
        )
      })}
    </group>
  )
}

function Scene() {
  const { 
    nodes, 
    bars, 
    supportNodes, 
    colorMode, 
    isLoading, 
    error,
    showDeflectedShape,
    deflectionScale3D,
    metrics,
    // Section mode
    sectionMode,
    sliceAxis,
    slicePosition,
    params,
  } = useDesignStore()
  
  // Calculate camera distance based on structure size
  const cameraDistance = useMemo(() => {
    if (nodes.length === 0) return 15
    const xs = nodes.map(n => n.x)
    const ys = nodes.map(n => n.y)
    const maxDim = Math.max(
      Math.max(...xs) - Math.min(...xs),
      Math.max(...ys) - Math.min(...ys)
    )
    return Math.max(maxDim * 1.5, 10)
  }, [nodes])

  // Get max displacement for deflected shape visualization
  const maxDisplacement = metrics?.max_displacement ?? 0
  
  return (
    <>
      {/* Lighting */}
      <ambientLight intensity={0.6} />
      <directionalLight
        position={[10, 15, 10]}
        intensity={0.8}
        castShadow
        shadow-mapSize={[2048, 2048]}
      />
      <directionalLight
        position={[-5, 10, -5]}
        intensity={0.3}
      />
      
      {/* Environment for reflections */}
      <Environment preset="apartment" />
      
      {/* Ground plane with shadow */}
      <ContactShadows
        position={[0, -0.01, 0]}
        opacity={0.25}
        scale={30}
        blur={2}
        far={10}
      />
      
      {/* Ground gradient */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.02, 0]} receiveShadow>
        <planeGeometry args={[50, 50]} />
        <meshStandardMaterial
          color={COLORS.ground}
          roughness={0.9}
          metalness={0}
        />
      </mesh>
      
      {/* Structure - show even if there's an error (geometry only) */}
      {nodes.length > 0 && (
        <Structure
          nodes={nodes}
          bars={bars}
          supportNodes={supportNodes}
          colorMode={colorMode}
          showDeflected={showDeflectedShape}
          deflectionScale={deflectionScale3D}
          maxDisplacement={maxDisplacement}
          sectionMode={sectionMode}
          sliceAxis={sliceAxis}
          slicePosition={slicePosition}
          width={params.width}
          depth={params.depth}
        />
      )}
      
      {/* Camera controls */}
      <OrbitControls
        makeDefault
        enableDamping
        dampingFactor={0.05}
        minDistance={5}
        maxDistance={50}
        maxPolarAngle={Math.PI / 2 - 0.1}
        target={[0, 2, 0]}
      />
    </>
  )
}

export function Canvas3D() {
  const { isLoading, error, nodes, showDeflectedShape, deflectionScale3D, metrics, sectionMode, sliceAxis, slicePosition, params } = useDesignStore()
  
  // Calculate slice coordinate for display
  const sliceCoord = sectionMode 
    ? (slicePosition * (sliceAxis === 'x' ? params.width : params.depth)).toFixed(2)
    : null
  
  return (
    <div className="relative w-full h-full min-h-[400px] overflow-hidden bg-slate-100">
      {/* Loading overlay */}
      {isLoading && (
        <div className="absolute inset-0 z-10 flex items-center justify-center bg-slate-100/90">
          <div className="flex flex-col items-center gap-2">
            <div className="w-6 h-6 border-2 border-slate-200 border-t-slate-600 rounded-full animate-spin" />
            <span className="text-xs text-slate-500 font-mono uppercase">Computing</span>
          </div>
        </div>
      )}
      
      {/* Error banner - shows on top, doesn't hide structure */}
      {error && !isLoading && nodes.length > 0 && (
        <div className="absolute top-3 left-3 right-3 z-10">
          <div className="bg-amber-50 border border-amber-200 rounded px-3 py-2 flex items-center gap-2">
            <svg className="w-4 h-4 text-amber-600 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
            <span className="text-xs text-amber-800">Unstable - geometry only (no forces)</span>
          </div>
        </div>
      )}

      {/* Deflected shape legend - bottom left */}
      {showDeflectedShape && nodes.length > 0 && metrics?.max_displacement && metrics.max_displacement > 0 && (
        <div className="absolute bottom-3 left-3 z-10 bg-white/90 rounded px-3 py-2 text-xs">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-1.5">
              <div className="w-4 h-0.5 bg-[#6C757D]" />
              <span className="text-slate-500">Original</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-4 h-0.5 bg-[#E74C3C]" style={{ borderStyle: 'dashed', borderWidth: '1px', borderColor: '#E74C3C' }} />
              <span className="text-slate-500">Deflected (×{deflectionScale3D})</span>
            </div>
          </div>
          <div className="mt-1 text-[10px] text-slate-400">
            Max δ = {(metrics.max_displacement * 1000).toFixed(2)} mm
          </div>
        </div>
      )}

      {/* Section mode legend - bottom right */}
      {sectionMode && nodes.length > 0 && (
        <div className="absolute bottom-3 right-3 z-10 bg-white/90 rounded px-3 py-2 text-xs">
          <div className="flex items-center gap-2 mb-1">
            <div className="w-3 h-3 bg-[#4CAF50] opacity-30 border border-[#4CAF50]" />
            <span className="text-slate-600 font-medium">Section Plane</span>
          </div>
          <div className="text-[10px] text-slate-500">
            {sliceAxis?.toUpperCase()} = {sliceCoord} m
          </div>
          <div className="flex items-center gap-2 mt-1">
            <div className="w-4 h-0.5 bg-[#81C784]" style={{ height: '3px' }} />
            <span className="text-[10px] text-slate-400">Extracted bars</span>
          </div>
        </div>
      )}
      
      {/* Error state when no geometry */}
      {error && !isLoading && nodes.length === 0 && (
        <div className="absolute inset-0 z-10 flex items-center justify-center bg-slate-100/95 p-6">
          <div className="text-center max-w-sm">
            <div className="w-10 h-10 rounded bg-red-100 flex items-center justify-center mx-auto mb-3">
              <svg className="w-5 h-5 text-red-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
            </div>
            <h3 className="text-sm font-semibold text-slate-900 mb-1">Error</h3>
            <p className="text-xs text-slate-500">{error}</p>
          </div>
        </div>
      )}
      
      {/* Empty state */}
      {nodes.length === 0 && !isLoading && !error && (
        <div className="absolute inset-0 z-10 flex items-center justify-center">
          <div className="text-center">
            <p className="text-sm text-slate-400">Waiting for parameters...</p>
          </div>
        </div>
      )}
      
      {/* Three.js Canvas */}
      <div className="absolute inset-0">
        <Canvas
          shadows
          camera={{
            position: [12, 8, 12],
            fov: 45,
            near: 0.1,
            far: 1000,
          }}
          gl={{
            antialias: true,
            alpha: true,
            powerPreference: 'high-performance',
          }}
          style={{ width: '100%', height: '100%' }}
        >
          <Scene />
        </Canvas>
      </div>
    </div>
  )
}
