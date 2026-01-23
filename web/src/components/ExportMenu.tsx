'use client'

import { useState } from 'react'
import { useDesignStore } from '@/store/designStore'
import { exportCSV, exportJSON, downloadBlob } from '@/lib/api'
import { Download, FileJson, FileSpreadsheet, Loader2 } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'

export function ExportMenu() {
  const { params, metrics } = useDesignStore()
  const [isExporting, setIsExporting] = useState<string | null>(null)
  const [showSuccess, setShowSuccess] = useState(false)
  
  const handleExport = async (type: 'csv' | 'json') => {
    setIsExporting(type)
    try {
      const blob = type === 'csv' 
        ? await exportCSV(params)
        : await exportJSON(params)
      
      if (blob) {
        const filename = type === 'csv' ? 'canopy_cutlist.csv' : 'canopy_model.json'
        downloadBlob(blob, filename)
        setShowSuccess(true)
        setTimeout(() => setShowSuccess(false), 2000)
      }
    } finally {
      setIsExporting(null)
    }
  }
  
  const disabled = !metrics
  
  return (
    <div className="card-organic">
      <div className="flex items-center gap-2 mb-4">
        <Download className="w-4 h-4 text-sage-500" />
        <h3 className="font-serif text-lg text-forest-800">Export</h3>
      </div>
      
      <div className="space-y-2">
        <button
          onClick={() => handleExport('csv')}
          disabled={disabled || isExporting !== null}
          className="w-full flex items-center gap-3 px-4 py-3 rounded-xl
                     bg-cream-50 border border-cream-200 text-left
                     transition-all duration-200
                     hover:bg-cream-100 hover:border-sage-300
                     disabled:opacity-50 disabled:cursor-not-allowed
                     group"
        >
          {isExporting === 'csv' ? (
            <Loader2 className="w-5 h-5 text-sage-500 animate-spin" />
          ) : (
            <FileSpreadsheet className="w-5 h-5 text-sage-500 group-hover:text-sage-600" />
          )}
          <div className="flex-1">
            <span className="text-sm font-medium text-forest-700">Cut List</span>
            <p className="text-xs text-stone-400">CSV with bar lengths and forces</p>
          </div>
        </button>
        
        <button
          onClick={() => handleExport('json')}
          disabled={disabled || isExporting !== null}
          className="w-full flex items-center gap-3 px-4 py-3 rounded-xl
                     bg-cream-50 border border-cream-200 text-left
                     transition-all duration-200
                     hover:bg-cream-100 hover:border-sage-300
                     disabled:opacity-50 disabled:cursor-not-allowed
                     group"
        >
          {isExporting === 'json' ? (
            <Loader2 className="w-5 h-5 text-sage-500 animate-spin" />
          ) : (
            <FileJson className="w-5 h-5 text-sage-500 group-hover:text-sage-600" />
          )}
          <div className="flex-1">
            <span className="text-sm font-medium text-forest-700">Model Data</span>
            <p className="text-xs text-stone-400">JSON with geometry and metrics</p>
          </div>
        </button>
      </div>
      
      {/* Success toast */}
      <AnimatePresence>
        {showSuccess && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="mt-3 px-3 py-2 rounded-xl bg-sage-50 border border-sage-200 text-center"
          >
            <span className="text-xs text-sage-700">Downloaded successfully</span>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

