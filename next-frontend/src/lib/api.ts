import axios from 'axios'

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
})

// ------------------------------------------------------------------ //
// Types
// ------------------------------------------------------------------ //

export interface ConflictEvent {
  event_id: string
  event_date: string
  location: string
  latitude: number
  longitude: number
  event_type: string
  sub_event_type?: string
  actor1: string
  actor2?: string
  fatalities: number
  country: string
  admin1?: string
  admin2?: string
  disorder_type?: string
  source?: string
  notes?: string
  confidence_score?: number
}

export interface CastForecast {
  country: string
  admin1?: string
  month: string
  year: number
  total_forecast: number
  battles_forecast: number
  erv_forecast: number
  vac_forecast: number
  total_observed: number
  battles_observed: number
  erv_observed: number
  vac_observed: number
  timestamp?: number
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
  predictions?: any
}

export interface ACLEDFetchRequest {
  country?: string
  countries?: string[]
  start_date?: string
  end_date?: string
  event_type?: string
  year?: number
  max_records?: number
}

export interface CASTFetchRequest {
  country?: string
  year?: number
  max_records?: number
}

export interface FetchJob {
  fetch_id: string
  status: 'queued' | 'fetching' | 'completed' | 'error'
  progress: number
  records_fetched: number
  records_stored: number
  message: string
  started_at: string
  completed_at?: string
  filters: Record<string, any>
  max_records: number
}

// ------------------------------------------------------------------ //
// Events
// ------------------------------------------------------------------ //

export async function fetchEvents(params?: {
  country?: string
  start_date?: string
  end_date?: string
  event_type?: string
  limit?: number
}): Promise<ConflictEvent[]> {
  const response = await api.get('/api/events', { params })
  return response.data
}

export async function fetchTrends(params?: {
  country?: string
  period?: string
  include_predictions?: boolean
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

// ------------------------------------------------------------------ //
// ACLED Live API
// ------------------------------------------------------------------ //

export async function triggerACLEDFetch(request: ACLEDFetchRequest): Promise<{ fetch_id: string; status: string; message: string }> {
  const response = await api.post('/api/acled/fetch', request)
  return response.data
}

export async function getACLEDFetchStatus(fetchId: string): Promise<FetchJob> {
  const response = await api.get(`/api/acled/fetch/${fetchId}`)
  return response.data
}

export async function listACLEDFetchJobs(): Promise<FetchJob[]> {
  const response = await api.get('/api/acled/fetch')
  return response.data
}

export async function getACLEDTokenStatus(): Promise<{ authenticated: boolean; email?: string; expires_in?: number }> {
  const response = await api.get('/api/acled/token-status')
  return response.data
}

export async function refreshACLEDToken(): Promise<{ message: string }> {
  const response = await api.post('/api/acled/authenticate')
  return response.data
}

// ------------------------------------------------------------------ //
// CAST Forecasts
// ------------------------------------------------------------------ //

export async function triggerCASTFetch(request: CASTFetchRequest): Promise<{ fetch_id: string; status: string }> {
  const response = await api.post('/api/cast/fetch', request)
  return response.data
}

export async function getCASTFetchStatus(fetchId: string): Promise<FetchJob> {
  const response = await api.get(`/api/cast/fetch/${fetchId}`)
  return response.data
}

export async function fetchCASTPredictions(params?: {
  country?: string
  year?: number
  limit?: number
}): Promise<CastForecast[]> {
  const response = await api.get('/api/cast/predictions', { params })
  return response.data
}

// ------------------------------------------------------------------ //
// CSV Upload (legacy / manual)
// ------------------------------------------------------------------ //

export async function uploadCSV(file: File, metadata?: any) {
  const formData = new FormData()
  formData.append('file', file)
  if (metadata) formData.append('metadata', JSON.stringify(metadata))
  const response = await api.post('/api/upload/csv', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return response.data
}

export async function getUploadStatus(uploadId: string) {
  const response = await api.get(`/api/upload/status/${uploadId}`)
  return response.data
}

// ------------------------------------------------------------------ //
// ML
// ------------------------------------------------------------------ //

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