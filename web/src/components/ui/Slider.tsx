'use client'

import * as React from 'react'
import * as SliderPrimitive from '@radix-ui/react-slider'
import { cn } from '@/lib/utils'

interface SliderProps {
  label: string
  value: number
  min: number
  max: number
  step: number
  unit?: string
  onChange: (value: number) => void
  className?: string
}

export function Slider({
  label,
  value,
  min,
  max,
  step,
  unit = '',
  onChange,
  className,
}: SliderProps) {
  return (
    <div className={cn('space-y-2', className)}>
      <div className="flex items-center justify-between">
        <label className="text-xs font-medium text-slate-600">{label}</label>
        <span className="text-xs font-mono text-slate-900 tabular-nums">
          {value}{unit}
        </span>
      </div>
      <SliderPrimitive.Root
        className="relative flex w-full touch-none select-none items-center"
        value={[value]}
        min={min}
        max={max}
        step={step}
        onValueChange={([v]) => onChange(v)}
      >
        <SliderPrimitive.Track className="relative h-1 w-full grow overflow-hidden rounded-full bg-slate-200">
          <SliderPrimitive.Range className="absolute h-full bg-slate-900" />
        </SliderPrimitive.Track>
        <SliderPrimitive.Thumb
          className="block h-3 w-3 rounded-full border border-slate-400 bg-white shadow-sm
                     transition-colors duration-150
                     hover:border-slate-600
                     focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-500 focus-visible:ring-offset-2
                     disabled:pointer-events-none disabled:opacity-50"
        />
      </SliderPrimitive.Root>
    </div>
  )
}
