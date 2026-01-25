'use client'

import { useMemo } from 'react'
import { ExploreDesignData } from '@/lib/types'
import { useDesignStore } from '@/store/designStore'

interface ParetoPlotProps {
  designs: ExploreDesignData[]
  width?: number
  height?: number
}

export function ParetoPlot({ designs, width = 500, height = 350 }: ParetoPlotProps) {
  const { selectedDesignIndex, selectDesign, filters } = useDesignStore()
  
  // Filter designs based on current filters
  const filteredDesigns = useMemo(() => {
    return designs.filter(d => {
      if (!d.ok || d.volume === undefined || d.max_displacement_mm === undefined) return false
      if (filters.paretoOnly && !d.is_pareto) return false
      if (filters.maxDisplacement !== null && d.max_displacement_mm > filters.maxDisplacement) return false
      if (filters.maxMemberLength !== null && d.max_member_length_mm && d.max_member_length_mm > filters.maxMemberLength) return false
      if (filters.topology !== null && d.topology !== filters.topology) return false
      return true
    })
  }, [designs, filters])
  
  // Calculate axis ranges
  const { xMin, xMax, yMin, yMax } = useMemo(() => {
    if (filteredDesigns.length === 0) {
      return { xMin: 0, xMax: 1, yMin: 0, yMax: 100 }
    }
    
    const volumes = filteredDesigns.map(d => d.volume!)
    const displacements = filteredDesigns.map(d => d.max_displacement_mm!)
    
    const xPad = (Math.max(...volumes) - Math.min(...volumes)) * 0.1 || 0.01
    const yPad = (Math.max(...displacements) - Math.min(...displacements)) * 0.1 || 1
    
    return {
      xMin: Math.max(0, Math.min(...volumes) - xPad),
      xMax: Math.max(...volumes) + xPad,
      yMin: Math.max(0, Math.min(...displacements) - yPad),
      yMax: Math.max(...displacements) + yPad,
    }
  }, [filteredDesigns])
  
  // Padding for axes
  const padding = { top: 20, right: 30, bottom: 45, left: 60 }
  const plotWidth = width - padding.left - padding.right
  const plotHeight = height - padding.top - padding.bottom
  
  // Scale functions
  const scaleX = (v: number) => padding.left + ((v - xMin) / (xMax - xMin)) * plotWidth
  const scaleY = (v: number) => padding.top + plotHeight - ((v - yMin) / (yMax - yMin)) * plotHeight
  
  // Generate axis ticks
  const xTicks = useMemo(() => {
    const count = 5
    const step = (xMax - xMin) / count
    return Array.from({ length: count + 1 }, (_, i) => xMin + i * step)
  }, [xMin, xMax])
  
  const yTicks = useMemo(() => {
    const count = 5
    const step = (yMax - yMin) / count
    return Array.from({ length: count + 1 }, (_, i) => yMin + i * step)
  }, [yMin, yMax])
  
  // Pareto frontier line
  const paretoLine = useMemo(() => {
    const paretoDesigns = filteredDesigns
      .filter(d => d.is_pareto)
      .sort((a, b) => a.volume! - b.volume!)
    
    if (paretoDesigns.length < 2) return ''
    
    return paretoDesigns
      .map((d, i) => `${i === 0 ? 'M' : 'L'} ${scaleX(d.volume!)} ${scaleY(d.max_displacement_mm!)}`)
      .join(' ')
  }, [filteredDesigns, scaleX, scaleY])
  
  // Get marker size based on length bins
  const getMarkerSize = (d: ExploreDesignData) => {
    if (!d.n_length_bins) return 5
    const minBins = Math.min(...filteredDesigns.map(x => x.n_length_bins || 1))
    const maxBins = Math.max(...filteredDesigns.map(x => x.n_length_bins || 1))
    const range = maxBins - minBins || 1
    return 4 + ((d.n_length_bins - minBins) / range) * 6
  }
  
  if (filteredDesigns.length === 0) {
    return (
      <div 
        className="flex items-center justify-center text-slate-400 text-sm"
        style={{ width, height }}
      >
        No designs match current filters
      </div>
    )
  }
  
  return (
    <svg width={width} height={height} className="font-sans">
      {/* Background */}
      <rect 
        x={padding.left} 
        y={padding.top} 
        width={plotWidth} 
        height={plotHeight} 
        fill="#fafafa" 
        stroke="#e2e8f0" 
      />
      
      {/* Grid lines */}
      {xTicks.map((tick, i) => (
        <line
          key={`x-grid-${i}`}
          x1={scaleX(tick)}
          y1={padding.top}
          x2={scaleX(tick)}
          y2={padding.top + plotHeight}
          stroke="#e2e8f0"
          strokeDasharray="2,2"
        />
      ))}
      {yTicks.map((tick, i) => (
        <line
          key={`y-grid-${i}`}
          x1={padding.left}
          y1={scaleY(tick)}
          x2={padding.left + plotWidth}
          y2={scaleY(tick)}
          stroke="#e2e8f0"
          strokeDasharray="2,2"
        />
      ))}
      
      {/* Pareto frontier line */}
      {paretoLine && (
        <path
          d={paretoLine}
          fill="none"
          stroke="#dc2626"
          strokeWidth={2}
          strokeDasharray="4,4"
          opacity={0.5}
        />
      )}
      
      {/* Data points - dominated (gray) */}
      {filteredDesigns
        .filter(d => !d.is_pareto)
        .map(d => (
          <circle
            key={`point-${d.index}`}
            cx={scaleX(d.volume!)}
            cy={scaleY(d.max_displacement_mm!)}
            r={getMarkerSize(d)}
            fill={selectedDesignIndex === d.index ? '#22c55e' : '#cbd5e1'}
            stroke={selectedDesignIndex === d.index ? '#16a34a' : '#94a3b8'}
            strokeWidth={selectedDesignIndex === d.index ? 2 : 1}
            opacity={0.7}
            className="cursor-pointer hover:opacity-100 transition-opacity"
            onClick={() => selectDesign(d.index)}
          >
            <title>
              {`Design #${d.index}\nVolume: ${d.volume?.toFixed(4)} m³\nDisplacement: ${d.max_displacement_mm?.toFixed(2)} mm\nBins: ${d.n_length_bins}\nTopology: ${d.topology}`}
            </title>
          </circle>
        ))}
      
      {/* Data points - Pareto (red) */}
      {filteredDesigns
        .filter(d => d.is_pareto)
        .map(d => (
          <circle
            key={`pareto-${d.index}`}
            cx={scaleX(d.volume!)}
            cy={scaleY(d.max_displacement_mm!)}
            r={getMarkerSize(d) + 2}
            fill={selectedDesignIndex === d.index ? '#22c55e' : '#dc2626'}
            stroke={selectedDesignIndex === d.index ? '#16a34a' : '#991b1b'}
            strokeWidth={2}
            className="cursor-pointer hover:brightness-110 transition-all"
            onClick={() => selectDesign(d.index)}
          >
            <title>
              {`★ Pareto Design #${d.index}\nVolume: ${d.volume?.toFixed(4)} m³\nDisplacement: ${d.max_displacement_mm?.toFixed(2)} mm\nBins: ${d.n_length_bins}\nTopology: ${d.topology}`}
            </title>
          </circle>
        ))}
      
      {/* X-axis */}
      <line
        x1={padding.left}
        y1={padding.top + plotHeight}
        x2={padding.left + plotWidth}
        y2={padding.top + plotHeight}
        stroke="#64748b"
        strokeWidth={1}
      />
      {xTicks.map((tick, i) => (
        <g key={`x-tick-${i}`}>
          <line
            x1={scaleX(tick)}
            y1={padding.top + plotHeight}
            x2={scaleX(tick)}
            y2={padding.top + plotHeight + 5}
            stroke="#64748b"
          />
          <text
            x={scaleX(tick)}
            y={padding.top + plotHeight + 18}
            textAnchor="middle"
            fontSize={10}
            fill="#64748b"
          >
            {tick.toFixed(3)}
          </text>
        </g>
      ))}
      <text
        x={padding.left + plotWidth / 2}
        y={height - 8}
        textAnchor="middle"
        fontSize={11}
        fontWeight={500}
        fill="#475569"
      >
        Volume (m³)
      </text>
      
      {/* Y-axis */}
      <line
        x1={padding.left}
        y1={padding.top}
        x2={padding.left}
        y2={padding.top + plotHeight}
        stroke="#64748b"
        strokeWidth={1}
      />
      {yTicks.map((tick, i) => (
        <g key={`y-tick-${i}`}>
          <line
            x1={padding.left - 5}
            y1={scaleY(tick)}
            x2={padding.left}
            y2={scaleY(tick)}
            stroke="#64748b"
          />
          <text
            x={padding.left - 8}
            y={scaleY(tick) + 3}
            textAnchor="end"
            fontSize={10}
            fill="#64748b"
          >
            {tick.toFixed(1)}
          </text>
        </g>
      ))}
      <text
        x={15}
        y={padding.top + plotHeight / 2}
        textAnchor="middle"
        fontSize={11}
        fontWeight={500}
        fill="#475569"
        transform={`rotate(-90, 15, ${padding.top + plotHeight / 2})`}
      >
        Displacement (mm)
      </text>
    </svg>
  )
}




