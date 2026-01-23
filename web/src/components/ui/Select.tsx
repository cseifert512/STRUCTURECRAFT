'use client'

import * as React from 'react'
import * as SelectPrimitive from '@radix-ui/react-select'
import { ChevronDown, Check } from 'lucide-react'
import { cn } from '@/lib/utils'

interface SelectOption {
  value: string
  label: string
  description?: string
}

interface SelectProps {
  label: string
  value: string
  options: readonly SelectOption[]
  onChange: (value: string) => void
  className?: string
}

export function Select({ label, value, options, onChange, className }: SelectProps) {
  const selectedOption = options.find(o => o.value === value)
  
  return (
    <div className={cn('space-y-2', className)}>
      <label className="text-xs font-medium text-slate-600">{label}</label>
      <SelectPrimitive.Root value={value} onValueChange={onChange}>
        <SelectPrimitive.Trigger
          className="flex w-full items-center justify-between rounded border border-slate-300
                     bg-white px-3 py-2 text-sm text-slate-900
                     transition-colors duration-150
                     hover:border-slate-400
                     focus:outline-none focus:border-slate-500 focus:ring-1 focus:ring-slate-500
                     data-[placeholder]:text-slate-400"
        >
          <SelectPrimitive.Value>
            {selectedOption?.label}
          </SelectPrimitive.Value>
          <SelectPrimitive.Icon>
            <ChevronDown className="h-4 w-4 text-slate-400" />
          </SelectPrimitive.Icon>
        </SelectPrimitive.Trigger>
        
        <SelectPrimitive.Portal>
          <SelectPrimitive.Content
            className="relative z-50 min-w-[10rem] overflow-hidden rounded border border-slate-200
                       bg-white shadow-card
                       data-[state=open]:animate-in data-[state=closed]:animate-out
                       data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0
                       data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95"
            position="popper"
            sideOffset={4}
          >
            <SelectPrimitive.Viewport className="p-1">
              {options.map((option) => (
                <SelectPrimitive.Item
                  key={option.value}
                  value={option.value}
                  className="relative flex cursor-pointer select-none items-center rounded
                             px-2 py-1.5 text-sm outline-none
                             transition-colors duration-100
                             focus:bg-slate-100
                             data-[disabled]:pointer-events-none data-[disabled]:opacity-50"
                >
                  <div className="flex-1">
                    <SelectPrimitive.ItemText>
                      {option.label}
                    </SelectPrimitive.ItemText>
                    {option.description && (
                      <p className="text-xs text-slate-400">{option.description}</p>
                    )}
                  </div>
                  <SelectPrimitive.ItemIndicator className="ml-2">
                    <Check className="h-3.5 w-3.5 text-slate-600" />
                  </SelectPrimitive.ItemIndicator>
                </SelectPrimitive.Item>
              ))}
            </SelectPrimitive.Viewport>
          </SelectPrimitive.Content>
        </SelectPrimitive.Portal>
      </SelectPrimitive.Root>
    </div>
  )
}
