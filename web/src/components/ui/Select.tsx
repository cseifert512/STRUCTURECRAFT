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
      <label className="text-sm font-medium text-forest-700">{label}</label>
      <SelectPrimitive.Root value={value} onValueChange={onChange}>
        <SelectPrimitive.Trigger
          className="flex w-full items-center justify-between rounded-xl border border-cream-300
                     bg-cream-50 px-4 py-2.5 text-sm text-forest-800
                     transition-all duration-200
                     hover:border-sage-300 hover:bg-cream-100
                     focus:outline-none focus:border-sage-400 focus:ring-2 focus:ring-sage-200
                     data-[placeholder]:text-stone-400"
        >
          <SelectPrimitive.Value>
            {selectedOption?.label}
          </SelectPrimitive.Value>
          <SelectPrimitive.Icon>
            <ChevronDown className="h-4 w-4 text-stone-400" />
          </SelectPrimitive.Icon>
        </SelectPrimitive.Trigger>
        
        <SelectPrimitive.Portal>
          <SelectPrimitive.Content
            className="relative z-50 min-w-[12rem] overflow-hidden rounded-xl border border-cream-200
                       bg-white shadow-soft-lg
                       data-[state=open]:animate-in data-[state=closed]:animate-out
                       data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0
                       data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95"
            position="popper"
            sideOffset={5}
          >
            <SelectPrimitive.Viewport className="p-1.5">
              {options.map((option) => (
                <SelectPrimitive.Item
                  key={option.value}
                  value={option.value}
                  className="relative flex cursor-pointer select-none items-center rounded-lg
                             px-3 py-2.5 text-sm outline-none
                             transition-colors duration-150
                             focus:bg-sage-50 focus:text-sage-700
                             data-[disabled]:pointer-events-none data-[disabled]:opacity-50"
                >
                  <div className="flex-1">
                    <SelectPrimitive.ItemText>
                      {option.label}
                    </SelectPrimitive.ItemText>
                    {option.description && (
                      <p className="text-xs text-stone-400 mt-0.5">{option.description}</p>
                    )}
                  </div>
                  <SelectPrimitive.ItemIndicator className="ml-2">
                    <Check className="h-4 w-4 text-sage-600" />
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

