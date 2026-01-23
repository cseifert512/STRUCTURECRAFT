'use client'

import { useMemo } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, Environment, ContactShadows, Line } from '@react-three/drei'
import * as THREE from 'three'
import { useDesignStore } from '@/store/designStore'
import { NodeData, BarData } from '@/lib/types'

// Color palette - architectural/engineering style
const COLORS = {
  tension: new THREE.Color('#E63946'),      // Red for tension
  compression: new THREE.Color('#457B9D'),  // Steel blue for compression
  neutral: new THREE.Color('#6C757D'),      // Neutral gray
  support: new THREE.Color('#212529'),      // Dark for supports
  node: new THREE.Color('#495057'),         // Dark gray for nodes
  ground: new THREE.Color('#F8F9FA'),       // Light gray ground
}

interface StructureProps {
  nodes: NodeData[]
  bars: BarData[]
  supportNodes: number[]
  colorByForce: boolean
}

function Structure({ nodes, bars, supportNodes, colorByForce }: StructureProps) {
  // Create node lookup map
  const nodeMap = useMemo(() => {
    const map = new Map<number, NodeData>()
    nodes.forEach(node => map.set(node.id, node))
    return map
  }, [nodes])
  
  // Calculate force range for color mapping
  const forceRange = useMemo(() => {
    if (bars.length === 0) return { min: 0, max: 1 }
    const forces = bars.map(b => b.force)
    const maxAbs = Math.max(...forces.map(Math.abs), 0.001)
    return { min: -maxAbs, max: maxAbs }
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
  
  // Get color for a bar based on force
  const getBarColor = (force: number) => {
    if (!colorByForce) return COLORS.neutral
    
    const t = force / forceRange.max
    if (force > 0) {
      // Tension - interpolate from neutral to tension color
      return COLORS.neutral.clone().lerp(COLORS.tension, Math.abs(t))
    } else {
      // Compression - interpolate from neutral to compression color
      return COLORS.neutral.clone().lerp(COLORS.compression, Math.abs(t))
    }
  }
  
  // Transform coordinates: engineering (X, Y, Z-up) to Three.js (X, Z, Y-up)
  const toThreeCoords = (x: number, y: number, z: number): [number, number, number] => {
    return [x - center.x, z, y - center.y]
  }
  
  return (
    <group>
      {/* Render bars as lines */}
      {bars.map((bar) => {
        const startNode = nodeMap.get(bar.ni)
        const endNode = nodeMap.get(bar.nj)
        if (!startNode || !endNode) return null
        
        const color = getBarColor(bar.force)
        
        return (
          <Line
            key={bar.id}
            points={[
              toThreeCoords(startNode.x, startNode.y, startNode.z),
              toThreeCoords(endNode.x, endNode.y, endNode.z),
            ]}
            color={color}
            lineWidth={1.5}
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
    </group>
  )
}

function Scene() {
  const { nodes, bars, supportNodes, colorByForce, isLoading, error } = useDesignStore()
  
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
          colorByForce={colorByForce}
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
  const { isLoading, error, nodes } = useDesignStore()
  
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

