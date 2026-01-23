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
        <label className="text-sm font-medium text-forest-700">{label}</label>
        <span className="text-sm font-medium text-sage-600 tabular-nums">
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
        <SliderPrimitive.Track className="relative h-2 w-full grow overflow-hidden rounded-full bg-cream-200">
          <SliderPrimitive.Range className="absolute h-full bg-gradient-to-r from-sage-400 to-sage-500" />
        </SliderPrimitive.Track>
        <SliderPrimitive.Thumb
          className="block h-5 w-5 rounded-full border-2 border-sage-500 bg-white shadow-soft
                     ring-offset-cream-100 transition-all duration-150
                     hover:scale-110 hover:border-sage-600 hover:shadow-soft-lg
                     focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sage-400 focus-visible:ring-offset-2
                     disabled:pointer-events-none disabled:opacity-50"
        />
      </SliderPrimitive.Root>
    </div>
  )
}

