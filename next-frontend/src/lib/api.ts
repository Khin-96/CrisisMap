import axios from 'axios'

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: API_BASE,
  timeout: 10000,
})

export interface ConflictEvent {
  event_id: string
  event_date: string
  location: string
  latitude: number
  longitude: number
  event_type: string
  actor1: string
  fatalities: number
  country: string
}

export interface TrendAnalysis {
  period: string
  total_events: number
  total_fatalities: number
  hotspot_locations: string[]
  trend_direction: 'increasing' | 'decreasing' | 'stable'
  temporal_data?: Array<{
    period: string
    total_events: number
    total_fatalities: number
  }>
}

export async function fetchEvents(params?: {
  country?: string
  start_date?: string
  end_date?: string
  limit?: number
}): Promise<ConflictEvent[]> {
  const response = await api.get('/api/events', { params })
  return response.data
}

export async function fetchTrends(params?: {
  country?: string
  period?: string
}): Promise<TrendAnalysis> {
  const response = await api.get('/api/trends', { params })
  return response.data
}

export async function fetchHotspots(params?: {
  country?: string
  threshold?: number
}) {
  const response = await api.get('/api/hotspots', { params })
  return response.data
}

export async function uploadCSV(file: File, metadata?: any) {
  const formData = new FormData()
  formData.append('file', file)
  if (metadata) {
    formData.append('metadata', JSON.stringify(metadata))
  }
  
  const response = await api.post('/api/upload/csv', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  })
  return response.data
}

export async function getUploadStatus(uploadId: string) {
  const response = await api.get(`/api/upload/status/${uploadId}`)
  return response.data
}

export async function trainModel(datasetId: string, modelConfig?: any) {
  const response = await api.post('/api/ml/train', {
    dataset_id: datasetId,
    config: modelConfig,
  })
  return response.data
}

export async function getModelMetrics(modelId: string) {
  const response = await api.get(`/api/ml/models/${modelId}/metrics`)
  return response.data
}