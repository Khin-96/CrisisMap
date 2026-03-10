'use client'

import React, { useState, useEffect } from 'react'
import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'

interface SystemStatus {
  backend: 'online' | 'offline' | 'error'
  database: 'connected' | 'disconnected' | 'error'
  lastUpdate: string
  dataStats: {
    totalEvents: number
    totalUploads: number
    lastUpload: string | null
    dataQuality: 'good' | 'warning' | 'error'
  }
}

interface UploadHistory {
  upload_id: string
  filename: string
  status: string
  records_processed?: number
  created_at: string
  message: string
}

export default function SystemStatus() {
  const [status, setStatus] = useState<SystemStatus | null>(null)
  const [uploadHistory, setUploadHistory] = useState<UploadHistory[]>([])
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)

  const checkSystemStatus = async () => {
    try {
      setRefreshing(true)
      
      // Check backend health
      const backendResponse = await fetch('http://localhost:8000/api/')
      const backendStatus = backendResponse.ok ? 'online' : 'error'
      
      // Get system status
      const statusResponse = await fetch('http://localhost:8000/api/system/status')
      const systemStatus = statusResponse.ok ? await statusResponse.json() : null
      
      // Mock upload history - in real app this would come from backend
      const mockUploadHistory: UploadHistory[] = [
        {
          upload_id: 'upload_001',
          filename: 'Africa_aggregated_data_up_to-2026-02-28.xlsx',
          status: 'completed',
          records_processed: 266828,
          created_at: new Date().toISOString(),
          message: 'Successfully processed 266,828 records'
        }
      ]
      
      setStatus(systemStatus || {
        backend: backendStatus,
        database: backendStatus === 'online' ? 'connected' : 'disconnected',
        lastUpdate: new Date().toISOString(),
        dataStats: {
          totalEvents: 0,
          totalUploads: 0,
          lastUpload: null,
          dataQuality: 'warning'
        }
      })
      
      setUploadHistory(mockUploadHistory)
      
    } catch (error) {
      console.error('Failed to check system status:', error)
      setStatus({
        backend: 'error',
        database: 'error',
        lastUpdate: new Date().toISOString(),
        dataStats: {
          totalEvents: 0,
          totalUploads: 0,
          lastUpload: null,
          dataQuality: 'error'
        }
      })
    } finally {
      setLoading(false)
      setRefreshing(false)
    }
  }

  useEffect(() => {
    checkSystemStatus()
    
    // Auto-refresh every 30 seconds
    const interval = setInterval(checkSystemStatus, 30000)
    return () => clearInterval(interval)
  }, [])

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'online':
      case 'connected':
      case 'completed':
      case 'good':
        return <Badge variant="secondary" className="bg-green-100 text-green-800">✓ {status}</Badge>
      case 'offline':
      case 'disconnected':
      case 'warning':
        return <Badge variant="default" className="bg-yellow-100 text-yellow-800">⚠ {status}</Badge>
      case 'error':
      case 'failed':
        return <Badge variant="destructive">✗ {status}</Badge>
      case 'processing':
        return <Badge variant="outline" className="bg-blue-100 text-blue-800">⟳ {status}</Badge>
      default:
        return <Badge variant="outline">{status}</Badge>
    }
  }

  if (loading) {
    return (
      <Card className="p-6">
        <div className="flex items-center justify-center h-32">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
        </div>
      </Card>
    )
  }

  return (
    <div className="space-y-6">
      {/* System Health Overview */}
      <Card className="p-6">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-semibold">System Status</h3>
          <div className="flex items-center space-x-2">
            <Button
              variant="outline"
              size="sm"
              onClick={checkSystemStatus}
              disabled={refreshing}
            >
              {refreshing ? '⟳' : '↻'} Refresh
            </Button>
            <span className="text-sm text-gray-500">
              Last updated: {status ? new Date(status.lastUpdate).toLocaleTimeString() : 'Never'}
            </span>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <div className="text-center">
            <div className="text-sm text-gray-600 mb-2">Backend API</div>
            {status && getStatusBadge(status.backend)}
            <div className="text-xs text-gray-500 mt-1">
              {status?.backend === 'online' ? 'Responding normally' : 'Connection issues'}
            </div>
          </div>

          <div className="text-center">
            <div className="text-sm text-gray-600 mb-2">Database</div>
            {status && getStatusBadge(status.database)}
            <div className="text-xs text-gray-500 mt-1">
              {status?.database === 'connected' ? 'MongoDB connected' : 'Database unavailable'}
            </div>
          </div>

          <div className="text-center">
            <div className="text-sm text-gray-600 mb-2">Data Quality</div>
            {status && getStatusBadge(status.dataStats.dataQuality)}
            <div className="text-xs text-gray-500 mt-1">
              {status?.dataStats.totalEvents.toLocaleString()} events loaded
            </div>
          </div>

          <div className="text-center">
            <div className="text-sm text-gray-600 mb-2">Last Upload</div>
            {status?.dataStats.lastUpload ? (
              <>
                <Badge variant="secondary">Recent</Badge>
                <div className="text-xs text-gray-500 mt-1">
                  {new Date(status.dataStats.lastUpload).toLocaleDateString()}
                </div>
              </>
            ) : (
              <>
                <Badge variant="outline">None</Badge>
                <div className="text-xs text-gray-500 mt-1">No uploads yet</div>
              </>
            )}
          </div>
        </div>
      </Card>

      {/* Data Statistics */}
      <Card className="p-6">
        <h3 className="text-lg font-semibold mb-4">Data Overview</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-blue-50 p-4 rounded-lg">
            <div className="text-2xl font-bold text-blue-600">
              {status?.dataStats.totalEvents.toLocaleString() || '0'}
            </div>
            <div className="text-sm text-blue-800">Total Events</div>
            <div className="text-xs text-blue-600 mt-1">
              Across all datasets
            </div>
          </div>

          <div className="bg-green-50 p-4 rounded-lg">
            <div className="text-2xl font-bold text-green-600">
              {status?.dataStats.totalUploads || '0'}
            </div>
            <div className="text-sm text-green-800">Successful Uploads</div>
            <div className="text-xs text-green-600 mt-1">
              Data files processed
            </div>
          </div>

          <div className="bg-purple-50 p-4 rounded-lg">
            <div className="text-2xl font-bold text-purple-600">
              {status?.backend === 'online' ? '100%' : '0%'}
            </div>
            <div className="text-sm text-purple-800">System Uptime</div>
            <div className="text-xs text-purple-600 mt-1">
              Current session
            </div>
          </div>
        </div>
      </Card>

      {/* Upload History */}
      <Card className="p-6">
        <h3 className="text-lg font-semibold mb-4">Recent Upload History</h3>
        {uploadHistory.length > 0 ? (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Filename</TableHead>
                <TableHead>Status</TableHead>
                <TableHead className="text-right">Records</TableHead>
                <TableHead>Upload Date</TableHead>
                <TableHead>Message</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {uploadHistory.map((upload) => (
                <TableRow key={upload.upload_id}>
                  <TableCell className="font-medium max-w-xs truncate" title={upload.filename}>
                    {upload.filename}
                  </TableCell>
                  <TableCell>
                    {getStatusBadge(upload.status)}
                  </TableCell>
                  <TableCell className="text-right">
                    {upload.records_processed?.toLocaleString() || 'N/A'}
                  </TableCell>
                  <TableCell>
                    {new Date(upload.created_at).toLocaleDateString('en-US', {
                      year: 'numeric',
                      month: 'short',
                      day: 'numeric',
                      hour: '2-digit',
                      minute: '2-digit'
                    })}
                  </TableCell>
                  <TableCell className="max-w-xs truncate" title={upload.message}>
                    {upload.message}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        ) : (
          <div className="text-center py-8 text-gray-500">
            <p>No upload history available</p>
            <p className="text-sm mt-1">Upload your first dataset to see history here</p>
          </div>
        )}
      </Card>

      {/* System Information */}
      <Card className="p-6">
        <h3 className="text-lg font-semibold mb-4">System Information</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-medium mb-3">Configuration</h4>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-600">Version:</span>
                <Badge variant="outline">v2.0.0</Badge>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Environment:</span>
                <Badge variant="secondary">Development</Badge>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Database:</span>
                <span>MongoDB (Crisis)</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Backend:</span>
                <span>FastAPI + Python</span>
              </div>
            </div>
          </div>

          <div>
            <h4 className="font-medium mb-3">Capabilities</h4>
            <div className="space-y-2 text-sm">
              <div className="flex items-center space-x-2">
                <Badge variant="secondary" className="bg-green-100 text-green-800">✓</Badge>
                <span>Adaptive CSV Processing</span>
              </div>
              <div className="flex items-center space-x-2">
                <Badge variant="secondary" className="bg-green-100 text-green-800">✓</Badge>
                <span>Real-time Map Visualization</span>
              </div>
              <div className="flex items-center space-x-2">
                <Badge variant="secondary" className="bg-green-100 text-green-800">✓</Badge>
                <span>Advanced Analytics</span>
              </div>
              <div className="flex items-center space-x-2">
                <Badge variant="secondary" className="bg-green-100 text-green-800">✓</Badge>
                <span>Data Quality Validation</span>
              </div>
            </div>
          </div>
        </div>
      </Card>
    </div>
  )
}