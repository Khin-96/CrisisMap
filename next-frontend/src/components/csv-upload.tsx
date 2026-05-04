'use client'

import React, { useState, useCallback } from 'react'
import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'

type DataType = 'acled_events' | 'cast_predictions'

interface ColumnMapping {
  [standardColumn: string]: string
}

interface CSVAnalysis {
  total_rows: number
  columns: string[]
  detected_mappings: ColumnMapping
  data_quality: any
  sample_data: any[]
  missing_required: string[]
}

interface UploadStatus {
  upload_id?: string
  fetch_id?: string
  status: string
  progress: number
  message: string
  records_processed?: number
  records_stored?: number
  analysis?: CSVAnalysis
}

const DATA_TYPE_OPTIONS = [
  {
    value: 'acled_events' as DataType,
    label: 'ACLED Conflict Events',
    description: 'Historical conflict events with location, actors, fatalities.',
    color: 'border-red-500 bg-red-50',
    activeColor: 'border-red-600 bg-red-100 ring-2 ring-red-400',
    requiredColumns: [
      { key: 'event_date', label: 'Event Date', description: 'Date of the event (YYYY-MM-DD)' },
      { key: 'latitude', label: 'Latitude', description: 'Decimal latitude coordinate' },
      { key: 'longitude', label: 'Longitude', description: 'Decimal longitude coordinate' },
      { key: 'event_type', label: 'Event Type', description: 'Category of conflict event' },
      { key: 'fatalities', label: 'Fatalities', description: 'Number of fatalities (numeric)' },
    ],
    optionalColumns: [
      { key: 'location', label: 'Location', description: 'Place name' },
      { key: 'actor1', label: 'Actor 1', description: 'Primary actor' },
      { key: 'actor2', label: 'Actor 2', description: 'Secondary actor' },
      { key: 'country', label: 'Country', description: 'Country name' },
      { key: 'notes', label: 'Notes', description: 'Additional context' },
    ],
    uploadEndpoint: '/api/upload/csv',
    analyzeEndpoint: '/api/upload/analyze',
    pollEndpoint: (id: string) => `/api/upload/status/${id}`,
    idField: 'upload_id',
  },
  {
    value: 'cast_predictions' as DataType,
    label: 'CAST Predictions',
    description: 'ACLED CAST conflict forecast data (expected, low, high forecasts by period).',
    color: 'border-blue-500 bg-blue-50',
    activeColor: 'border-blue-600 bg-blue-100 ring-2 ring-blue-400',
    requiredColumns: [
      { key: 'country', label: 'Country', description: 'Country name' },
      { key: 'period', label: 'Period', description: 'Forecast period date (YYYY-MM-DD)' },
      { key: 'expected_forecast', label: 'Expected Forecast', description: 'Expected number of events' },
    ],
    optionalColumns: [
      { key: 'id', label: 'ID', description: 'Geographic ID (e.g. global/Afghanistan)' },
      { key: 'level', label: 'Level', description: 'Geographic level (global, country, admin1)' },
      { key: 'admin1', label: 'Admin1', description: 'First-level administrative division' },
      { key: 'outcome', label: 'Outcome', description: 'Type of violence outcome' },
      { key: 'low_forecast', label: 'Low Forecast', description: 'Lower bound forecast' },
      { key: 'high_forecast', label: 'High Forecast', description: 'Upper bound forecast' },
    ],
    uploadEndpoint: '/api/upload/cast-csv',
    analyzeEndpoint: '/api/upload/analyze',
    pollEndpoint: (id: string) => `/api/cast/fetch/${id}`,
    idField: 'fetch_id',
  },
]

export default function CSVUpload({ onUploadComplete }: { onUploadComplete?: (uploadId: string) => void }) {
  const [dataType, setDataType] = useState<DataType>('acled_events')
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [uploadStatus, setUploadStatus] = useState<UploadStatus | null>(null)
  const [analysis, setAnalysis] = useState<CSVAnalysis | null>(null)
  const [customMappings, setCustomMappings] = useState<ColumnMapping>({})
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [showMappingInterface, setShowMappingInterface] = useState(false)
  const [isUploading, setIsUploading] = useState(false)

  const activeConfig = DATA_TYPE_OPTIONS.find(o => o.value === dataType)!

  const resetState = useCallback(() => {
    setSelectedFile(null)
    setAnalysis(null)
    setUploadStatus(null)
    setShowMappingInterface(false)
    setCustomMappings({})
    setIsUploading(false)
    setIsAnalyzing(false)
  }, [])

  const handleDataTypeChange = useCallback((type: DataType) => {
    setDataType(type)
    resetState()
  }, [resetState])

  const handleFileSelect = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      setSelectedFile(file)
      setAnalysis(null)
      setUploadStatus(null)
      setShowMappingInterface(false)
      setCustomMappings({})
    }
  }, [])

  const analyzeFile = useCallback(async () => {
    if (!selectedFile) return
    setIsAnalyzing(true)
    try {
      const formData = new FormData()
      formData.append('file', selectedFile)
      formData.append('data_type', dataType)
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
      const response = await fetch(`${apiUrl}${activeConfig.analyzeEndpoint}`, {
        method: 'POST',
        body: formData,
      })
      if (!response.ok) throw new Error(`Analysis failed: ${response.statusText}`)
      const result: CSVAnalysis = await response.json()
      setAnalysis(result)
      if (result.missing_required.length > 0) {
        setShowMappingInterface(true)
        setCustomMappings(result.detected_mappings)
      }
    } catch (error) {
      console.error('Analysis error:', error)
      alert(`Failed to analyze file: ${error}`)
    } finally {
      setIsAnalyzing(false)
    }
  }, [selectedFile, activeConfig])

  const uploadFile = useCallback(async () => {
    if (!selectedFile) return
    setIsUploading(true)
    try {
      const formData = new FormData()
      formData.append('file', selectedFile)
      if (Object.keys(customMappings).length > 0) {
        formData.append('custom_mappings', JSON.stringify(customMappings))
      }

      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
      const response = await fetch(`${apiUrl}${activeConfig.uploadEndpoint}`, {
        method: 'POST',
        body: formData,
      })
      if (!response.ok) throw new Error(`Upload failed: ${response.statusText}`)
      const result = await response.json()
      const jobId = result[activeConfig.idField] || result.upload_id || result.fetch_id

      setUploadStatus({
        ...result,
        progress: result.progress ?? 0,
        status: result.status ?? 'processing',
        message: result.message ?? 'Processing...',
      })

      pollStatus(jobId)
    } catch (error) {
      console.error('Upload error:', error)
      alert(`Upload failed: ${error}`)
      setIsUploading(false)
    }
  }, [selectedFile, customMappings, activeConfig])

  const pollStatus = useCallback(async (jobId: string) => {
    const poll = async () => {
      try {
        const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
        const response = await fetch(`${apiUrl}${activeConfig.pollEndpoint(jobId)}`)
        if (response.ok) {
          const status: UploadStatus = await response.json()
          setUploadStatus(status)
          if (status.status === 'completed') {
            setIsUploading(false)
            onUploadComplete?.(jobId)
          } else if (status.status === 'error') {
            setIsUploading(false)
          } else if (['processing', 'fetching', 'queued'].includes(status.status)) {
            setTimeout(poll, 2000)
          } else if (status.status === 'needs_mapping') {
            setIsUploading(false)
            if (status.analysis) {
              setAnalysis(status.analysis)
              setShowMappingInterface(true)
              setCustomMappings(status.analysis.detected_mappings)
            }
          }
        }
      } catch (error) {
        console.error('Poll error:', error)
        setIsUploading(false)
      }
    }
    poll()
  }, [activeConfig, onUploadComplete])

  const updateMapping = useCallback((standardColumn: string, sourceColumn: string) => {
    setCustomMappings(prev => ({ ...prev, [standardColumn]: sourceColumn }))
  }, [])

  const canUpload = selectedFile && (!showMappingInterface ||
    activeConfig.requiredColumns.every(col => customMappings[col.key]))

  const completedRecords = uploadStatus?.records_processed ?? uploadStatus?.records_stored

  return (
    <div className="space-y-6">
      {/* Data Type Selector */}
      <Card className="p-6">
        <h3 className="text-lg font-semibold mb-1">Upload Data</h3>
        <p className="text-sm text-gray-500 mb-4">Select the type of data you are uploading.</p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {DATA_TYPE_OPTIONS.map(opt => (
            <button
              key={opt.value}
              onClick={() => handleDataTypeChange(opt.value)}
              className={`text-left p-4 rounded-xl border-2 transition-all ${dataType === opt.value ? opt.activeColor : opt.color
                }`}
            >
              <div className="flex items-center gap-2 mb-1">
                <span className="text-2xl">{opt.icon}</span>
                <span className="font-semibold text-sm">{opt.label}</span>
                {dataType === opt.value && (
                  <Badge variant="secondary" className="ml-auto text-xs">Selected</Badge>
                )}
              </div>
              <p className="text-xs text-gray-600">{opt.description}</p>
            </button>
          ))}
        </div>
      </Card>


      {/* File Selection */}
      <Card className="p-6">
        <h3 className="text-lg font-semibold mb-4">
          Select {activeConfig.label} File
        </h3>
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-2">
              CSV or Excel file
            </label>
            <input
              type="file"
              onChange={handleFileSelect}
              disabled={isUploading}
              className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100 disabled:opacity-50"
            />
          </div>

          {selectedFile && (
            <div className="flex items-center space-x-4">
              <span className="text-sm text-gray-600">
                {selectedFile.name} ({(selectedFile.size / 1024 / 1024).toFixed(2)} MB)
              </span>
              <Button
                onClick={analyzeFile}
                disabled={isAnalyzing || isUploading}
                variant="outline"
                size="sm"
              >
                {isAnalyzing ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600 mr-2" />
                    Analyzing...
                  </>
                ) : 'Analyze Structure'}
              </Button>
            </div>
          )}
        </div>
      </Card>

      {/* Analysis Results */}
      {analysis && (
        <Card className="p-6">
          <h3 className="text-lg font-semibold mb-4">File Analysis</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
            <div className="bg-blue-50 p-3 rounded">
              <div className="text-2xl font-bold text-blue-600">{analysis.total_rows.toLocaleString()}</div>
              <div className="text-sm text-blue-800">Total Rows</div>
            </div>
            <div className="bg-green-50 p-3 rounded">
              <div className="text-2xl font-bold text-green-600">{analysis.columns.length}</div>
              <div className="text-sm text-green-800">Columns</div>
            </div>
            <div className="bg-orange-50 p-3 rounded">
              <div className="text-2xl font-bold text-orange-600">{analysis.missing_required.length}</div>
              <div className="text-sm text-orange-800">Missing Required</div>
            </div>
          </div>

          {analysis.missing_required.length > 0 && (
            <div className="bg-yellow-50 border border-yellow-200 rounded p-4 mb-4">
              <h4 className="font-medium text-yellow-800 mb-1">Manual Column Mapping Required</h4>
              <p className="text-sm text-yellow-700">
                Missing: <span className="font-mono">{analysis.missing_required.join(', ')}</span>
              </p>
            </div>
          )}

          {/* Sample Preview */}
          <div>
            <h4 className="font-medium mb-2">Sample Data Preview</h4>
            <div className="overflow-x-auto">
              <table className="min-w-full text-sm border border-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    {analysis.columns.slice(0, 8).map(col => (
                      <th key={col} className="px-3 py-2 text-left border-b">{col}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {analysis.sample_data.slice(0, 3).map((row, idx) => (
                    <tr key={idx} className="border-b">
                      {analysis.columns.slice(0, 8).map(col => (
                        <td key={col} className="px-3 py-2 border-r text-xs">
                          {String(row[col] ?? '').substring(0, 40)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </Card>
      )}

      {/* Column Mapping */}
      {showMappingInterface && analysis && (
        <Card className="p-6">
          <h3 className="text-lg font-semibold mb-4">Column Mapping</h3>
          <p className="text-sm text-gray-600 mb-4">
            Map your file&apos;s columns to the required <strong>{activeConfig.label}</strong> format:
          </p>
          <div className="space-y-6">
            <div>
              <h4 className="font-medium text-red-600 mb-3">Required Columns</h4>
              <div className="grid gap-4">
                {activeConfig.requiredColumns.map(col => (
                  <div key={col.key} className="flex items-center space-x-4">
                    <div className="w-1/3">
                      <label className="block text-sm font-medium">{col.label}</label>
                      <p className="text-xs text-gray-500">{col.description}</p>
                    </div>
                    <div className="w-1/3">
                      <select
                        value={customMappings[col.key] || ''}
                        onChange={(e) => updateMapping(col.key, e.target.value)}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
                      >
                        <option value="">Select column...</option>
                        {analysis.columns.map(sourceCol => (
                          <option key={sourceCol} value={sourceCol}>{sourceCol}</option>
                        ))}
                      </select>
                    </div>
                    <div className="w-1/3 text-xs">
                      {analysis.detected_mappings[col.key] && (
                        <span className="text-green-600">Auto: {analysis.detected_mappings[col.key]}</span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div>
              <h4 className="font-medium text-blue-600 mb-3">Optional Columns</h4>
              <div className="grid gap-4">
                {activeConfig.optionalColumns.map(col => (
                  <div key={col.key} className="flex items-center space-x-4">
                    <div className="w-1/3">
                      <label className="block text-sm font-medium">{col.label}</label>
                      <p className="text-xs text-gray-500">{col.description}</p>
                    </div>
                    <div className="w-1/3">
                      <select
                        value={customMappings[col.key] || ''}
                        onChange={(e) => updateMapping(col.key, e.target.value)}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
                      >
                        <option value="">Skip column</option>
                        {analysis.columns.map(sourceCol => (
                          <option key={sourceCol} value={sourceCol}>{sourceCol}</option>
                        ))}
                      </select>
                    </div>
                    <div className="w-1/3 text-xs">
                      {analysis.detected_mappings[col.key] && (
                        <span className="text-green-600">Auto: {analysis.detected_mappings[col.key]}</span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </Card>
      )}

      {/* Upload Button + Status */}
      <Card className="p-6">
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <Button
              onClick={uploadFile}
              disabled={!canUpload || isUploading}
              className="px-8"
            >
              {isUploading ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2" />
                  Processing...
                </>
              ) : (
                `Upload ${activeConfig.label}`
              )}
            </Button>

            {uploadStatus && (
              <Badge variant={
                uploadStatus.status === 'completed' ? 'secondary' :
                  uploadStatus.status === 'error' ? 'destructive' :
                    uploadStatus.status === 'needs_mapping' ? 'default' : 'outline'
              }>
                {uploadStatus.status}
              </Badge>
            )}
          </div>

          {uploadStatus && (
            <div className="space-y-3">
              <div className="flex items-center justify-between text-sm">
                <span className="font-medium">{uploadStatus.message}</span>
                <span className="text-gray-500">{uploadStatus.progress ?? 0}%</span>
              </div>
              <Progress value={uploadStatus.progress ?? 0} className="w-full" />
              {completedRecords != null && completedRecords > 0 && (
                <p className="text-sm text-green-600">
                  Successfully processed {completedRecords.toLocaleString()} records
                  {dataType === 'cast_predictions' && ' — CAST data will be used to improve hotspot predictions'}
                </p>
              )}
              {uploadStatus.status === 'error' && (
                <p className="text-sm text-red-600">{uploadStatus.message}</p>
              )}
            </div>
          )}
        </div>
      </Card>
    </div>
  )
}