export interface ConflictEvent {
  event_id: string
  event_date: string
  location: string
  latitude: number
  longitude: number
  event_type: string
  actor1?: string
  actor2?: string
  fatalities: number
  country: string
  confidence_score?: number
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
  predictions?: {
    model_id: string
    predictions: Prediction[]
  }
}

export interface Prediction {
  date: string
  location: string
  latitude: number
  longitude: number
  predicted_fatalities: number
  confidence: number
  risk_level: 'low' | 'medium' | 'high'
}

export interface UploadStatus {
  upload_id: string
  status: 'uploading' | 'processing' | 'completed' | 'error'
  progress: number
  message: string
  records_processed?: number
}

export interface ModelMetrics {
  model_id: string
  model_type: string
  metrics: {
    mse: number
    mae: number
    r2: number
    rmse: number
  }
  training_samples: number
  test_samples: number
  created_at: string
}

export interface Anomaly {
  type: 'high_fatalities' | 'geographic_clustering' | 'temporal_spike'
  event_id?: string
  location: string
  fatalities?: number
  event_count?: number
  date: string
  severity: 'low' | 'medium' | 'high'
  description: string
}