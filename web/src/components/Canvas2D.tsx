'use client'

import { useMemo } from 'react'
import { useDesignStore } from '@/store/designStore'
import { Frame2DNodeData, Frame2DElementData, ElementDiagramData, DiagramType } from '@/lib/types'

// Color palette
const COLORS = {
  structure: '#2C3E50',      // Dark blue-gray for undeformed
  deflected: '#E74C3C',      // Coral red for deformed
  support: '#27AE60',        // Green for supports
  load_gravity: '#9B59B6',   // Purple for gravity
  load_lateral: '#E67E22',   // Orange for lateral
  reaction: '#27AE60',       // Green for reactions
  ground: '#8B7355',         // Earth brown
  // Diagram colors
  tension: '#E63946',        // Red for tension
  compression: '#457B9D',    // Blue for compression
  moment_pos: '#E63946',     // Red for positive moment
  moment_neg: '#457B9D',     // Blue for negative moment
  shear: '#9B59B6',          // Purple for shear
}

interface Canvas2DProps {
  width?: number
  height?: number
}

export function Canvas2D({ width = 800, height = 500 }: Canvas2DProps) {
  const {
    frame2dNodes,
    frame2dElements,
    frame2dDiagrams,
    frame2dMetrics,
    frame2dParams,
    frame2dLoading,
    frame2dError,
    diagramType,
    showDeflectedShape2D,
  } = useDesignStore()

  // Compute view bounds and transform
  const viewTransform = useMemo(() => {
    if (frame2dNodes.length === 0) {
      return { scale: 50, offsetX: 100, offsetY: height - 100, minX: 0, maxX: 10, minY: 0, maxY: 5 }
    }

    const xs = frame2dNodes.map(n => n.x)
    const ys = frame2dNodes.map(n => n.y)
    const minX = Math.min(...xs)
    const maxX = Math.max(...xs)
    const minY = Math.min(...ys)
    const maxY = Math.max(...ys)

    const spanX = maxX - minX || 1
    const spanY = maxY - minY || 1

    // Add margins
    const marginX = 100
    const marginY = 80
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
  }, [frame2dNodes, width, height])

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
    if (diagramType === 'none' || frame2dDiagrams.length === 0) return 0

    let maxVal = 0
    for (const diag of frame2dDiagrams) {
      if (diagramType === 'N') maxVal = Math.max(maxVal, diag.max_N)
      if (diagramType === 'V') maxVal = Math.max(maxVal, diag.max_V)
      if (diagramType === 'M') maxVal = Math.max(maxVal, diag.max_M)
    }

    // Scale diagram to be visible but not overwhelming (max ~30% of view)
    const targetSize = Math.min(viewTransform.scale * 0.8, 60)
    return maxVal > 0 ? targetSize / maxVal : 0
  }, [diagramType, frame2dDiagrams, viewTransform.scale])

  // Render element line
  const renderElement = (elem: Frame2DElementData, deflectedMode: boolean = false) => {
    const ni = frame2dNodes.find(n => n.id === elem.ni)
    const nj = frame2dNodes.find(n => n.id === elem.nj)
    if (!ni || !nj) return null

    let x1: number, y1: number, x2: number, y2: number

    if (deflectedMode) {
      const scale = frame2dParams.deflection_scale
      ;[x1, y1] = toSVG(ni.x + ni.ux * scale, ni.y + ni.uy * scale)
      ;[x2, y2] = toSVG(nj.x + nj.ux * scale, nj.y + nj.uy * scale)
    } else {
      ;[x1, y1] = toSVG(ni.x, ni.y)
      ;[x2, y2] = toSVG(nj.x, nj.y)
    }

    const lineWidth = elem.element_type === 'column' ? 6 : elem.element_type === 'beam' ? 5 : 3

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
        strokeDasharray={deflectedMode ? '8,4' : undefined}
      />
    )
  }

  // Render curved deflected shape
  const renderDeflectedCurve = (diagram: ElementDiagramData) => {
    if (!showDeflectedShape2D || diagram.deflected_shape.length < 2) return null

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

    // Get element direction
    const ni = frame2dNodes.find(n => n.id === diagram.ni)
    const nj = frame2dNodes.find(n => n.id === diagram.nj)
    if (!ni || !nj) return null

    const dx = nj.x - ni.x
    const dy = nj.y - ni.y
    const L = Math.sqrt(dx * dx + dy * dy)
    if (L === 0) return null

    // Normal direction (perpendicular to element, pointing outward)
    const nx = -dy / L
    const ny = dx / L

    // Build polygon points for diagram
    const polygonPoints: string[] = []
    const forcePoints: Array<{ x: number; y: number; val: number }> = []

    for (const pt of diagram.points) {
      let val = 0
      if (diagramType === 'N') val = pt.N
      if (diagramType === 'V') val = pt.V
      if (diagramType === 'M') val = pt.M

      const offsetX = nx * val * diagramScale
      const offsetY = ny * val * diagramScale

      const [sx, sy] = toSVG(pt.x_global + offsetX, pt.y_global + offsetY)
      polygonPoints.push(`${sx},${sy}`)
      forcePoints.push({ x: sx, y: sy, val })
    }

    // Close polygon by going back along the element
    for (let i = diagram.points.length - 1; i >= 0; i--) {
      const pt = diagram.points[i]
      const [sx, sy] = toSVG(pt.x_global, pt.y_global)
      polygonPoints.push(`${sx},${sy}`)
    }

    // Determine color based on dominant force direction
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
    const size = 15

    return (
      <g key={`support-${node.id}`}>
        {/* Triangle */}
        <polygon
          points={`${x},${y} ${x - size},${y + size} ${x + size},${y + size}`}
          fill={COLORS.ground}
          stroke={COLORS.structure}
          strokeWidth={2}
        />
        {/* Hatch marks */}
        <line x1={x - size - 5} y1={y + size + 5} x2={x - size + 5} y2={y + size + 15} stroke={COLORS.ground} strokeWidth={1.5} />
        <line x1={x - 5} y1={y + size + 5} x2={x + 5} y2={y + size + 15} stroke={COLORS.ground} strokeWidth={1.5} />
        <line x1={x + size - 5} y1={y + size + 5} x2={x + size + 5} y2={y + size + 15} stroke={COLORS.ground} strokeWidth={1.5} />
      </g>
    )
  }

  // Render node
  const renderNode = (node: Frame2DNodeData) => {
    const [x, y] = toSVG(node.x, node.y)
    const radius = node.is_support ? 6 : 4

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

  // Render loads
  const renderLoads = () => {
    if (frame2dNodes.length === 0) return null

    const elements: JSX.Element[] = []
    const { udl_kn_m, lateral_kn } = frame2dParams

    // Find beam element (horizontal at top)
    const maxY = Math.max(...frame2dNodes.map(n => n.y))
    const beamNodes = frame2dNodes.filter(n => Math.abs(n.y - maxY) < 0.01)
    
    if (beamNodes.length >= 2 && udl_kn_m > 0) {
      const xMin = Math.min(...beamNodes.map(n => n.x))
      const xMax = Math.max(...beamNodes.map(n => n.x))
      const [x1, y1] = toSVG(xMin, maxY)
      const [x2] = toSVG(xMax, maxY)

      // UDL arrows
      const nArrows = 7
      const spacing = (x2 - x1) / (nArrows - 1)
      
      for (let i = 0; i < nArrows; i++) {
        const ax = x1 + i * spacing
        elements.push(
          <g key={`udl-arrow-${i}`}>
            <line x1={ax} y1={y1 - 40} x2={ax} y2={y1 - 5} stroke={COLORS.load_gravity} strokeWidth={2} />
            <polygon
              points={`${ax},${y1 - 5} ${ax - 5},${y1 - 15} ${ax + 5},${y1 - 15}`}
              fill={COLORS.load_gravity}
            />
          </g>
        )
      }

      // UDL line
      elements.push(
        <line key="udl-line" x1={x1} y1={y1 - 40} x2={x2} y2={y1 - 40} stroke={COLORS.load_gravity} strokeWidth={2} />
      )

      // UDL label
      elements.push(
        <text
          key="udl-label"
          x={(x1 + x2) / 2}
          y={y1 - 50}
          textAnchor="middle"
          fill={COLORS.load_gravity}
          fontSize={12}
          fontWeight="bold"
        >
          {udl_kn_m.toFixed(1)} kN/m
        </text>
      )
    }

    // Lateral load at top-left
    if (lateral_kn > 0 && beamNodes.length > 0) {
      const leftNode = beamNodes.reduce((a, b) => a.x < b.x ? a : b)
      const [x, y] = toSVG(leftNode.x, leftNode.y)

      elements.push(
        <g key="lateral-load">
          <line x1={x - 60} y1={y} x2={x - 10} y2={y} stroke={COLORS.load_lateral} strokeWidth={3} />
          <polygon
            points={`${x - 10},${y} ${x - 20},${y - 6} ${x - 20},${y + 6}`}
            fill={COLORS.load_lateral}
          />
          <text
            x={x - 70}
            y={y - 10}
            textAnchor="end"
            fill={COLORS.load_lateral}
            fontSize={12}
            fontWeight="bold"
          >
            {lateral_kn.toFixed(0)} kN
          </text>
        </g>
      )
    }

    return elements
  }

  // Loading state
  if (frame2dLoading) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-slate-100">
        <div className="flex flex-col items-center gap-2">
          <div className="w-6 h-6 border-2 border-slate-200 border-t-slate-600 rounded-full animate-spin" />
          <span className="text-xs text-slate-500 font-mono uppercase">Computing</span>
        </div>
      </div>
    )
  }

  // Error state
  if (frame2dError && frame2dNodes.length === 0) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-slate-100 p-6">
        <div className="text-center max-w-sm">
          <div className="w-10 h-10 rounded bg-red-100 flex items-center justify-center mx-auto mb-3">
            <svg className="w-5 h-5 text-red-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
          </div>
          <h3 className="text-sm font-semibold text-slate-900 mb-1">Error</h3>
          <p className="text-xs text-slate-500">{frame2dError}</p>
        </div>
      </div>
    )
  }

  // Empty state
  if (frame2dNodes.length === 0) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-slate-100">
        <p className="text-sm text-slate-400">Waiting for parameters...</p>
      </div>
    )
  }

  return (
    <div className="relative w-full h-full bg-slate-50">
      {/* Error banner */}
      {frame2dError && (
        <div className="absolute top-3 left-3 right-3 z-10">
          <div className="bg-amber-50 border border-amber-200 rounded px-3 py-2 flex items-center gap-2">
            <svg className="w-4 h-4 text-amber-600 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
            <span className="text-xs text-amber-800">{frame2dError}</span>
          </div>
        </div>
      )}

      {/* SVG Canvas */}
      <svg
        width="100%"
        height="100%"
        viewBox={`0 0 ${width} ${height}`}
        className="bg-slate-50"
      >
        {/* Ground line */}
        <line
          x1={0}
          y1={viewTransform.offsetY + 20}
          x2={width}
          y2={viewTransform.offsetY + 20}
          stroke={COLORS.ground}
          strokeWidth={3}
        />

        {/* Undeformed elements */}
        {frame2dElements.map(elem => renderElement(elem, false))}

        {/* Force diagrams (rendered before deflected shape for layering) */}
        {frame2dDiagrams.map(diag => renderDiagram(diag))}

        {/* Deflected shape (curved) */}
        {showDeflectedShape2D && frame2dDiagrams.map(diag => renderDeflectedCurve(diag))}

        {/* Loads */}
        {renderLoads()}

        {/* Supports */}
        {frame2dNodes.map(node => renderSupport(node))}

        {/* Nodes */}
        {frame2dNodes.map(node => renderNode(node))}

        {/* Legend */}
        <g transform={`translate(${width - 150}, 20)`}>
          <rect x={0} y={0} width={140} height={100} fill="white" fillOpacity={0.9} rx={4} />
          
          <line x1={10} y1={20} x2={40} y2={20} stroke={COLORS.structure} strokeWidth={4} />
          <text x={50} y={24} fontSize={11} fill="#333">Undeformed</text>
          
          <line x1={10} y1={40} x2={40} y2={40} stroke={COLORS.deflected} strokeWidth={2} strokeDasharray="6,3" />
          <text x={50} y={44} fontSize={11} fill="#333">Deflected (×{frame2dParams.deflection_scale})</text>
          
          {diagramType !== 'none' && (
            <>
              <rect x={10} y={55} width={30} height={15} fill={diagramType === 'N' ? COLORS.tension : diagramType === 'V' ? COLORS.shear : COLORS.moment_pos} fillOpacity={0.3} stroke={diagramType === 'N' ? COLORS.tension : diagramType === 'V' ? COLORS.shear : COLORS.moment_pos} />
              <text x={50} y={67} fontSize={11} fill="#333">{diagramType} Diagram</text>
            </>
          )}
          
          {frame2dMetrics && (
            <text x={10} y={90} fontSize={10} fill="#666">
              Drift: {frame2dMetrics.drift_mm.toFixed(1)} mm
            </text>
          )}
        </g>
      </svg>
    </div>
  )
}

