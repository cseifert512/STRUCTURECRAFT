'use client'

import { useState } from 'react'
import { useDesignStore } from '@/store/designStore'
import { exportCSV, exportJSON, downloadBlob } from '@/lib/api'
import { Download, FileSpreadsheet, FileJson, Loader2 } from 'lucide-react'

export function ExportMenu() {
  const { params, metrics } = useDesignStore()
  const [isExporting, setIsExporting] = useState<string | null>(null)
  
  const handleExport = async (type: 'csv' | 'json') => {
    setIsExporting(type)
    try {
      const blob = type === 'csv' 
        ? await exportCSV(params)
        : await exportJSON(params)
      
      if (blob) {
        const filename = type === 'csv' ? 'cutlist.csv' : 'model.json'
        downloadBlob(blob, filename)
      }
    } finally {
      setIsExporting(null)
    }
  }
  
  const disabled = !metrics
  
  return (
    <div className="card rounded-lg p-4">
      <div className="flex items-center gap-2 mb-3">
        <Download className="w-3.5 h-3.5 text-slate-400" />
        <h3 className="text-sm font-semibold text-slate-900 uppercase tracking-wide">Export</h3>
      </div>
      
      <div className="space-y-2">
        <button
          onClick={() => handleExport('csv')}
          disabled={disabled || isExporting !== null}
          className="w-full flex items-center gap-2.5 px-3 py-2 rounded border border-slate-200
                     text-left text-sm
                     transition-colors duration-150
                     hover:bg-slate-50 hover:border-slate-300
                     disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isExporting === 'csv' ? (
            <Loader2 className="w-4 h-4 text-slate-400 animate-spin" />
          ) : (
            <FileSpreadsheet className="w-4 h-4 text-slate-400" />
          )}
          <div className="flex-1">
            <div className="font-medium text-slate-700">Cut List</div>
            <div className="text-xs text-slate-400">CSV format</div>
          </div>
        </button>
        
        <button
          onClick={() => handleExport('json')}
          disabled={disabled || isExporting !== null}
          className="w-full flex items-center gap-2.5 px-3 py-2 rounded border border-slate-200
                     text-left text-sm
                     transition-colors duration-150
                     hover:bg-slate-50 hover:border-slate-300
                     disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isExporting === 'json' ? (
            <Loader2 className="w-4 h-4 text-slate-400 animate-spin" />
          ) : (
            <FileJson className="w-4 h-4 text-slate-400" />
          )}
          <div className="flex-1">
            <div className="font-medium text-slate-700">Model Data</div>
            <div className="text-xs text-slate-400">JSON format</div>
          </div>
        </button>
      </div>
    </div>
  )
}
