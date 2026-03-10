'use client'

import React, { useState, useCallback } from 'react'
import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'

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
  upload_id: string
  status: string
  progress: number
  message: string
  records_processed?: number
  analysis?: CSVAnalysis
}

export default function CSVUpload({ onUploadComplete }: { onUploadComplete?: (uploadId: string) => void }) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [uploadStatus, setUploadStatus] = useState<UploadStatus | null>(null)
  const [analysis, setAnalysis] = useState<CSVAnalysis | null>(null)
  const [customMappings, setCustomMappings] = useState<ColumnMapping>({})
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [showMappingInterface, setShowMappingInterface] = useState(false)
  const [isUploading, setIsUploading] = useState(false)

  const requiredColumns = [
    { key: 'event_date', label: 'Event Date', description: 'Date of the conflict event (YYYY-MM-DD)' },
    { key: 'latitude', label: 'Latitude', description: 'Latitude coordinate (decimal degrees)' },
    { key: 'longitude', label: 'Longitude', description: 'Longitude coordinate (decimal degrees)' },
    { key: 'event_type', label: 'Event Type', description: 'Type of conflict event' },
    { key: 'fatalities', label: 'Fatalities', description: 'Number of fatalities (numeric)' }
  ]

  const optionalColumns = [
    { key: 'location', label: 'Location', description: 'Place name or location description' },
    { key: 'actor1', label: 'Actor 1', description: 'Primary conflict actor' },
    { key: 'actor2', label: 'Actor 2', description: 'Secondary conflict actor' },
    { key: 'country', label: 'Country', description: 'Country where event occurred' },
    { key: 'notes', label: 'Notes', description: 'Additional context or description' }
  ]

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

      const response = await fetch('http://localhost:8000/api/upload/analyze', {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        throw new Error(`Analysis failed: ${response.statusText}`)
      }

      const analysisResult: CSVAnalysis = await response.json()
      setAnalysis(analysisResult)

      // If there are missing required columns, show mapping interface
      if (analysisResult.missing_required.length > 0) {
        setShowMappingInterface(true)
        // Initialize custom mappings with detected mappings
        setCustomMappings(analysisResult.detected_mappings)
      }
    } catch (error) {
      console.error('Analysis failed:', error)
      alert(`Failed to analyze file: ${error}`)
    } finally {
      setIsAnalyzing(false)
    }
  }, [selectedFile])

  const uploadFile = useCallback(async () => {
    if (!selectedFile) return

    setIsUploading(true)
    try {
      const formData = new FormData()
      formData.append('file', selectedFile)
      
      // Add custom mappings if any
      if (Object.keys(customMappings).length > 0) {
        formData.append('custom_mappings', JSON.stringify(customMappings))
      }

      const response = await fetch('http://localhost:8000/api/upload/csv', {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`)
      }

      const result = await response.json()
      setUploadStatus(result)

      // Poll for status updates
      pollUploadStatus(result.upload_id)
    } catch (error) {
      console.error('Upload failed:', error)
      alert(`Upload failed: ${error}`)
      setIsUploading(false)
    }
  }, [selectedFile, customMappings])

  const pollUploadStatus = useCallback(async (uploadId: string) => {
    const poll = async () => {
      try {
        const response = await fetch(`http://localhost:8000/api/upload/status/${uploadId}`)
        if (response.ok) {
          const status: UploadStatus = await response.json()
          setUploadStatus(status)

          if (status.status === 'completed') {
            setIsUploading(false)
            onUploadComplete?.(uploadId)
          } else if (status.status === 'error') {
            setIsUploading(false)
          } else if (status.status === 'processing') {
            setTimeout(poll, 2000) // Poll every 2 seconds
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
        console.error('Failed to get upload status:', error)
        setIsUploading(false)
      }
    }
    poll()
  }, [onUploadComplete])

  const updateMapping = useCallback((standardColumn: string, sourceColumn: string) => {
    setCustomMappings(prev => ({
      ...prev,
      [standardColumn]: sourceColumn
    }))
  }, [])

  const canUpload = selectedFile && (!showMappingInterface || 
    requiredColumns.every(col => customMappings[col.key]))

  return (
    <div className="space-y-6">
      {/* File Selection */}
      <Card className="p-6">
        <h3 className="text-lg font-semibold mb-4">Upload Conflict Data</h3>
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-2">
              Select CSV or Excel file
            </label>
            <input
              type="file"
              accept=".csv,.xlsx,.xls"
              onChange={handleFileSelect}
              disabled={isUploading}
              className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100 disabled:opacity-50"
            />
          </div>

          {selectedFile && (
            <div className="flex items-center space-x-4">
              <span className="text-sm text-gray-600">
                Selected: {selectedFile.name} ({(selectedFile.size / 1024 / 1024).toFixed(2)} MB)
              </span>
              <Button 
                onClick={analyzeFile} 
                disabled={isAnalyzing || isUploading}
                variant="outline"
                size="sm"
              >
                {isAnalyzing ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600 mr-2"></div>
                    Analyzing...
                  </>
                ) : (
                  'Analyze Structure'
                )}
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
              <h4 className="font-medium text-yellow-800 mb-2">Manual Column Mapping Required</h4>
              <p className="text-sm text-yellow-700">
                Some required columns could not be automatically detected. Please map them manually below.
              </p>
            </div>
          )}

          {/* Sample Data Preview */}
          <div className="mb-4">
            <h4 className="font-medium mb-2">Sample Data Preview</h4>
            <div className="overflow-x-auto">
              <table className="min-w-full text-sm border border-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    {analysis.columns.slice(0, 8).map(col => (
                      <th key={col} className="px-3 py-2 text-left border-b">
                        {col}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {analysis.sample_data.slice(0, 3).map((row, idx) => (
                    <tr key={idx} className="border-b">
                      {analysis.columns.slice(0, 8).map(col => (
                        <td key={col} className="px-3 py-2 border-r">
                          {String(row[col] || '').substring(0, 50)}
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

      {/* Column Mapping Interface */}
      {showMappingInterface && analysis && (
        <Card className="p-6">
          <h3 className="text-lg font-semibold mb-4">Column Mapping</h3>
          <p className="text-sm text-gray-600 mb-4">
            Map your file columns to the required CrisisMap format:
          </p>

          <div className="space-y-4">
            {/* Required Columns */}
            <div>
              <h4 className="font-medium text-red-600 mb-3">Required Columns</h4>
              <div className="grid gap-4">
                {requiredColumns.map(col => (
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
                          <option key={sourceCol} value={sourceCol}>
                            {sourceCol}
                          </option>
                        ))}
                      </select>
                    </div>
                    <div className="w-1/3 text-sm">
                      {analysis.detected_mappings[col.key] && (
                        <span className="text-green-600">
                          Auto-detected: {analysis.detected_mappings[col.key]}
                        </span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Optional Columns */}
            <div>
              <h4 className="font-medium text-blue-600 mb-3">Optional Columns</h4>
              <div className="grid gap-4">
                {optionalColumns.map(col => (
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
                        <option value="">Skip this column</option>
                        {analysis.columns.map(sourceCol => (
                          <option key={sourceCol} value={sourceCol}>
                            {sourceCol}
                          </option>
                        ))}
                      </select>
                    </div>
                    <div className="w-1/3 text-sm">
                      {analysis.detected_mappings[col.key] && (
                        <span className="text-green-600">
                          Auto-detected: {analysis.detected_mappings[col.key]}
                        </span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </Card>
      )}

      {/* Upload Button and Status */}
      <Card className="p-6">
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <Button
              onClick={uploadFile}
              disabled={!canUpload || isUploading}
              className="px-6"
            >
              {isUploading ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                  Processing...
                </>
              ) : (
                'Upload and Process'
              )}
            </Button>

            {uploadStatus && (
              <div className="flex items-center space-x-2">
                <Badge variant={
                  uploadStatus.status === 'completed' ? 'secondary' :
                  uploadStatus.status === 'error' ? 'destructive' :
                  uploadStatus.status === 'needs_mapping' ? 'default' :
                  'outline'
                }>
                  {uploadStatus.status}
                </Badge>
              </div>
            )}
          </div>

          {uploadStatus && (
            <div className="space-y-3">
              <div className="flex items-center justify-between text-sm">
                <span className="font-medium">{uploadStatus.message}</span>
                <span className="text-gray-500">{uploadStatus.progress}%</span>
              </div>
              
              <Progress value={uploadStatus.progress} className="w-full" />
              
              {uploadStatus.records_processed && (
                <p className="text-sm text-green-600">
                  ✅ Successfully processed {uploadStatus.records_processed.toLocaleString()} records
                </p>
              )}
              
              {uploadStatus.status === 'error' && (
                <p className="text-sm text-red-600">
                  ❌ {uploadStatus.message}
                </p>
              )}
            </div>
          )}
        </div>
      </Card>
    </div>
  )
}