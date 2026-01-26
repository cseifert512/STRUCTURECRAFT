'use client'

/**
 * SectionCanvas2D.tsx
 * 
 * Renders a 2D frame section extracted from a 3D spaceframe.
 * Similar to Canvas2D but takes Frame2DResult directly as a prop
 * rather than reading from the store.
 */

import { useMemo, useState } from 'react'
import { Frame2DResult, Frame2DNodeData, Frame2DElementData, ElementDiagramData, DiagramType } from '@/lib/types'
import { useDesignStore } from '@/store/designStore'

// Color palette
const COLORS = {
  structure: '#2C3E50',      // Dark blue-gray for undeformed
  deflected: '#E74C3C',      // Coral red for deformed
  support: '#27AE60',        // Green for supports
  load_gravity: '#9B59B6',   // Purple for gravity
  ground: '#8B7355',         // Earth brown
  // Diagram colors
  tension: '#E63946',        // Red for tension
  compression: '#457B9D',    // Blue for compression
  moment_pos: '#E63946',     // Red for positive moment
  moment_neg: '#457B9D',     // Blue for negative moment
  shear: '#9B59B6',          // Purple for shear
}

interface SectionCanvas2DProps {
  frameResult: Frame2DResult
  width?: number
  height?: number
}

export function SectionCanvas2D({ frameResult, width = 800, height = 500 }: SectionCanvas2DProps) {
  const { sectionDeflectionScale, sectionResult } = useDesignStore()
  const [diagramType, setDiagramType] = useState<DiagramType>('M')
  const [showDeflected, setShowDeflected] = useState(true)

  const nodes = frameResult.nodes || []
  const elements = frameResult.elements || []
  const diagrams = frameResult.diagrams || []
  const metrics = frameResult.metrics

  // Compute view bounds and transform
  const viewTransform = useMemo(() => {
    if (nodes.length === 0) {
      return { scale: 50, offsetX: 100, offsetY: height - 100, minX: 0, maxX: 10, minY: 0, maxY: 5 }
    }

    const xs = nodes.map(n => n.x)
    const ys = nodes.map(n => n.y)
    const minX = Math.min(...xs)
    const maxX = Math.max(...xs)
    const minY = Math.min(...ys)
    const maxY = Math.max(...ys)

    const spanX = maxX - minX || 1
    const spanY = maxY - minY || 1

    // Add margins
    const marginX = 80
    const marginY = 60
    const availableWidth = width - 2 * marginX
    const availableHeight = height - 2 * marginY

    // Scale to fit
    const scaleX = availableWidth / spanX
    const scaleY = availableHeight / spanY
    const scale = Math.min(scaleX, scaleY)

    // Center the frame
    const offsetX = marginX + (availableWidth - spanX * scale) / 2 - minX * scale
    const offsetY = height - marginY - (availableHeight - spanY * scale) / 2 + minY * scale

    return { scale, offsetX, offsetY, minX, maxX, minY, maxY }
  }, [nodes, width, height])

  // Transform engineering coords (Y up) to SVG coords (Y down)
  const toSVG = (x: number, y: number): [number, number] => {
    const { scale, offsetX, offsetY } = viewTransform
    return [
      offsetX + x * scale,
      offsetY - y * scale,
    ]
  }

  // Get diagram scale factor for visualization
  const diagramScale = useMemo(() => {
    if (diagramType === 'none' || diagrams.length === 0) return 0

    let maxVal = 0
    for (const diag of diagrams) {
      if (diagramType === 'N') maxVal = Math.max(maxVal, diag.max_N)
      if (diagramType === 'V') maxVal = Math.max(maxVal, diag.max_V)
      if (diagramType === 'M') maxVal = Math.max(maxVal, diag.max_M)
    }

    // Scale diagram to be visible but not overwhelming
    const targetSize = Math.min(viewTransform.scale * 0.6, 50)
    return maxVal > 0 ? targetSize / maxVal : 0
  }, [diagramType, diagrams, viewTransform.scale])

  // Render element line
  const renderElement = (elem: Frame2DElementData, deflectedMode: boolean = false) => {
    const ni = nodes.find(n => n.id === elem.ni)
    const nj = nodes.find(n => n.id === elem.nj)
    if (!ni || !nj) return null

    let x1: number, y1: number, x2: number, y2: number

    if (deflectedMode) {
      const scale = sectionDeflectionScale
      ;[x1, y1] = toSVG(ni.x + ni.ux * scale, ni.y + ni.uy * scale)
      ;[x2, y2] = toSVG(nj.x + nj.ux * scale, nj.y + nj.uy * scale)
    } else {
      ;[x1, y1] = toSVG(ni.x, ni.y)
      ;[x2, y2] = toSVG(nj.x, nj.y)
    }

    const lineWidth = elem.element_type === 'column' ? 5 : elem.element_type === 'beam' ? 4 : 2

    return (
      <line
        key={`elem-${elem.id}-${deflectedMode ? 'def' : 'undef'}`}
        x1={x1}
        y1={y1}
        x2={x2}
        y2={y2}
        stroke={deflectedMode ? COLORS.deflected : COLORS.structure}
        strokeWidth={lineWidth}
        strokeLinecap="round"
        opacity={deflectedMode ? 0.8 : 1}
        strokeDasharray={deflectedMode ? '6,3' : undefined}
      />
    )
  }

  // Render curved deflected shape
  const renderDeflectedCurve = (diagram: ElementDiagramData) => {
    if (!showDeflected || diagram.deflected_shape.length < 2) return null

    const points = diagram.deflected_shape.map(p => {
      const [x, y] = toSVG(p.x, p.y)
      return `${x},${y}`
    }).join(' ')

    return (
      <polyline
        key={`curve-${diagram.element_id}`}
        points={points}
        fill="none"
        stroke={COLORS.deflected}
        strokeWidth={2}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    )
  }

  // Render force diagram overlay
  const renderDiagram = (diagram: ElementDiagramData) => {
    if (diagramType === 'none' || diagram.points.length < 2) return null

    const ni = nodes.find(n => n.id === diagram.ni)
    const nj = nodes.find(n => n.id === diagram.nj)
    if (!ni || !nj) return null

    const dx = nj.x - ni.x
    const dy = nj.y - ni.y
    const L = Math.sqrt(dx * dx + dy * dy)
    if (L === 0) return null

    // Normal direction (perpendicular to element)
    const nx = -dy / L
    const ny = dx / L

    // Build polygon points for diagram
    const polygonPoints: string[] = []

    for (const pt of diagram.points) {
      let val = 0
      if (diagramType === 'N') val = pt.N
      if (diagramType === 'V') val = pt.V
      if (diagramType === 'M') val = pt.M

      const offsetX = nx * val * diagramScale
      const offsetY = ny * val * diagramScale

      const [sx, sy] = toSVG(pt.x_global + offsetX, pt.y_global + offsetY)
      polygonPoints.push(`${sx},${sy}`)
    }

    // Close polygon by going back along the element
    for (let i = diagram.points.length - 1; i >= 0; i--) {
      const pt = diagram.points[i]
      const [sx, sy] = toSVG(pt.x_global, pt.y_global)
      polygonPoints.push(`${sx},${sy}`)
    }

    // Determine color based on force direction
    let color = COLORS.tension
    const avgVal = diagram.points.reduce((sum, p) => {
      if (diagramType === 'N') return sum + p.N
      if (diagramType === 'V') return sum + p.V
      if (diagramType === 'M') return sum + p.M
      return sum
    }, 0) / diagram.points.length

    if (diagramType === 'N') {
      color = avgVal > 0 ? COLORS.tension : COLORS.compression
    } else if (diagramType === 'V') {
      color = COLORS.shear
    } else if (diagramType === 'M') {
      color = avgVal > 0 ? COLORS.moment_pos : COLORS.moment_neg
    }

    return (
      <g key={`diagram-${diagram.element_id}`}>
        <polygon
          points={polygonPoints.join(' ')}
          fill={color}
          fillOpacity={0.2}
          stroke={color}
          strokeWidth={1.5}
        />
      </g>
    )
  }

  // Render support symbol
  const renderSupport = (node: Frame2DNodeData) => {
    if (!node.is_support) return null

    const [x, y] = toSVG(node.x, node.y)
    const size = 12

    return (
      <g key={`support-${node.id}`}>
        <polygon
          points={`${x},${y} ${x - size},${y + size} ${x + size},${y + size}`}
          fill={COLORS.ground}
          stroke={COLORS.structure}
          strokeWidth={2}
        />
        <line x1={x - size - 4} y1={y + size + 4} x2={x - size + 4} y2={y + size + 12} stroke={COLORS.ground} strokeWidth={1.5} />
        <line x1={x - 4} y1={y + size + 4} x2={x + 4} y2={y + size + 12} stroke={COLORS.ground} strokeWidth={1.5} />
        <line x1={x + size - 4} y1={y + size + 4} x2={x + size + 4} y2={y + size + 12} stroke={COLORS.ground} strokeWidth={1.5} />
      </g>
    )
  }

  // Render node
  const renderNode = (node: Frame2DNodeData) => {
    const [x, y] = toSVG(node.x, node.y)
    const radius = node.is_support ? 5 : 3

    return (
      <circle
        key={`node-${node.id}`}
        cx={x}
        cy={y}
        r={radius}
        fill="white"
        stroke={COLORS.structure}
        strokeWidth={2}
      />
    )
  }

  // Render UDL arrows
  const renderLoads = () => {
    if (!sectionResult?.applied_udl_kn_m) return null

    const maxY = Math.max(...nodes.map(n => n.y))
    const beamNodes = nodes.filter(n => Math.abs(n.y - maxY) < 0.1)
    
    if (beamNodes.length < 2) return null

    const xMin = Math.min(...beamNodes.map(n => n.x))
    const xMax = Math.max(...beamNodes.map(n => n.x))
    const [x1, y1] = toSVG(xMin, maxY)
    const [x2] = toSVG(xMax, maxY)

    const elements: JSX.Element[] = []
    const nArrows = 5
    const spacing = (x2 - x1) / (nArrows - 1)
    
    for (let i = 0; i < nArrows; i++) {
      const ax = x1 + i * spacing
      elements.push(
        <g key={`udl-arrow-${i}`}>
          <line x1={ax} y1={y1 - 30} x2={ax} y2={y1 - 5} stroke={COLORS.load_gravity} strokeWidth={2} />
          <polygon
            points={`${ax},${y1 - 5} ${ax - 4},${y1 - 12} ${ax + 4},${y1 - 12}`}
            fill={COLORS.load_gravity}
          />
        </g>
      )
    }

    // UDL line
    elements.push(
      <line key="udl-line" x1={x1} y1={y1 - 30} x2={x2} y2={y1 - 30} stroke={COLORS.load_gravity} strokeWidth={2} />
    )

    // Label
    const midX = (x1 + x2) / 2
    elements.push(
      <text key="udl-label" x={midX} y={y1 - 38} textAnchor="middle" fontSize={11} fill={COLORS.load_gravity}>
        {sectionResult.applied_udl_kn_m.toFixed(1)} kN/m
      </text>
    )

    return elements
  }

  if (!frameResult.success) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-slate-100">
        <div className="text-red-500 text-sm">Error: {frameResult.error}</div>
      </div>
    )
  }

  return (
    <div className="relative w-full h-full bg-slate-50">
      {/* Controls overlay */}
      <div className="absolute top-2 left-2 z-10 flex gap-1">
        {(['none', 'N', 'V', 'M'] as DiagramType[]).map(type => (
          <button
            key={type}
            onClick={() => setDiagramType(type)}
            className={`px-2 py-1 text-xs rounded ${
              diagramType === type 
                ? 'bg-slate-700 text-white' 
                : 'bg-white/80 text-slate-600 hover:bg-slate-200'
            }`}
          >
            {type === 'none' ? 'Off' : type}
          </button>
        ))}
        <button
          onClick={() => setShowDeflected(!showDeflected)}
          className={`px-2 py-1 text-xs rounded ml-2 ${
            showDeflected 
              ? 'bg-red-500 text-white' 
              : 'bg-white/80 text-slate-600 hover:bg-slate-200'
          }`}
        >
          δ
        </button>
      </div>

      {/* Metrics overlay */}
      {metrics && (
        <div className="absolute top-2 right-2 z-10 bg-white/90 rounded px-2 py-1 text-xs">
          <div className="text-slate-500">Max δ: <span className="font-mono">{metrics.max_displacement_mm.toFixed(2)} mm</span></div>
          <div className="text-slate-500">Max M: <span className="font-mono">{(metrics.max_moment / 1000).toFixed(2)} kN·m</span></div>
        </div>
      )}

      {/* SVG Canvas */}
      <svg 
        width="100%" 
        height="100%" 
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="xMidYMid meet"
        className="absolute inset-0"
      >
        {/* Background */}
        <rect width={width} height={height} fill="#f8fafc" />

        {/* Ground line */}
        <line 
          x1={0} 
          y1={viewTransform.offsetY + 20} 
          x2={width} 
          y2={viewTransform.offsetY + 20} 
          stroke={COLORS.ground} 
          strokeWidth={2} 
          strokeDasharray="8,4" 
        />

        {/* Supports */}
        {nodes.map(renderSupport)}

        {/* Undeformed structure */}
        {elements.map(elem => renderElement(elem, false))}

        {/* Force diagrams */}
        {diagrams.map(renderDiagram)}

        {/* Deflected curves */}
        {diagrams.map(renderDeflectedCurve)}

        {/* Loads */}
        {renderLoads()}

        {/* Nodes */}
        {nodes.map(renderNode)}
      </svg>
    </div>
  )
}

