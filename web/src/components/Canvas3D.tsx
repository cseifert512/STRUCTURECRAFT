'use client'

import { useMemo } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, Environment, ContactShadows, Line } from '@react-three/drei'
import * as THREE from 'three'
import { useDesignStore } from '@/store/designStore'
import { NodeData, BarData } from '@/lib/types'

// Color palette for forces
const COLORS = {
  tension: new THREE.Color('#C4846C'),      // Clay/terracotta for tension
  compression: new THREE.Color('#6B8F5B'),  // Sage for compression
  neutral: new THREE.Color('#9CA89C'),      // Stone gray for neutral
  support: new THREE.Color('#4A6741'),      // Forest for supports
  node: new THREE.Color('#D4A574'),         // Clay for regular nodes
  ground: new THREE.Color('#F5F0E8'),       // Cream for ground
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
            lineWidth={2.5}
          />
        )
      })}
      
      {/* Render nodes as spheres */}
      {nodes.map((node) => {
        const isSupport = supportNodes.includes(node.id)
        const color = isSupport ? COLORS.support : COLORS.node
        const size = isSupport ? 0.12 : 0.06
        
        return (
          <mesh key={node.id} position={toThreeCoords(node.x, node.y, node.z)}>
            <sphereGeometry args={[size, 16, 16]} />
            <meshStandardMaterial
              color={color}
              roughness={0.4}
              metalness={0.1}
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
      
      {/* Structure */}
      {nodes.length > 0 && !error && (
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
    <div className="relative w-full h-[500px] rounded-3xl overflow-hidden bg-gradient-to-b from-cream-100 to-cream-200">
      {/* Loading overlay */}
      {isLoading && (
        <div className="absolute inset-0 z-10 flex items-center justify-center bg-cream-100/80 backdrop-blur-sm">
          <div className="flex flex-col items-center gap-3">
            <div className="w-8 h-8 border-3 border-sage-200 border-t-sage-500 rounded-full animate-spin" />
            <span className="text-sm text-stone-500">Computing structure...</span>
          </div>
        </div>
      )}
      
      {/* Error state */}
      {error && !isLoading && (
        <div className="absolute inset-0 z-10 flex items-center justify-center bg-cream-100/90 backdrop-blur-sm p-8">
          <div className="text-center max-w-md">
            <div className="w-12 h-12 rounded-full bg-clay-100 flex items-center justify-center mx-auto mb-4">
              <svg className="w-6 h-6 text-clay-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
            </div>
            <h3 className="font-serif text-lg text-forest-800 mb-2">Structure Unstable</h3>
            <p className="text-sm text-stone-500">{error}</p>
          </div>
        </div>
      )}
      
      {/* Empty state */}
      {nodes.length === 0 && !isLoading && !error && (
        <div className="absolute inset-0 z-10 flex items-center justify-center">
          <div className="text-center">
            <div className="w-16 h-16 rounded-full bg-sage-100 flex items-center justify-center mx-auto mb-4">
              <svg className="w-8 h-8 text-sage-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
              </svg>
            </div>
            <p className="text-stone-400">Adjust parameters to generate a structure</p>
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
      
      {/* Gradient overlay at bottom */}
      <div className="absolute bottom-0 left-0 right-0 h-24 bg-gradient-to-t from-cream-200/50 to-transparent pointer-events-none" />
    </div>
  )
}

