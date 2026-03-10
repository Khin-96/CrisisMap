'use client'

import { useState, useEffect } from 'react'
import { Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { MyBarChart } from '@/components/bar-chart'
import { TrendChart } from '@/components/trend-chart'
import { Brain, TrendingUp, AlertTriangle, Target } from 'lucide-react'

interface MLModel {
  model_id: string
  model_type: string
  status: string
  metrics: {
    r2: number
    mse: number
    mae: number
    rmse: number
  }
  training_samples: number
  created_at: string
}

interface Prediction {
  date: string
  location: string
  latitude: number
  longitude: number
  predicted_fatalities: number
  confidence: number
  risk_level: string
}

export default function MLDashboard() {
  const [models, setModels] = useState<MLModel[]>([])
  const [predictions, setPredictions] = useState<Prediction[]>([])
  const [loading, setLoading] = useState(true)
  const [trainingModel, setTrainingModel] = useState(false)
  const [trainingStatus, setTrainingStatus] = useState<string>('')

  useEffect(() => {
    loadModels()
    loadPredictions()
  }, [])

  const loadModels = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/ml/models')
      if (response.ok) {
        const data = await response.json()
        setModels(data.models || [])
      }
    } catch (error) {
      console.error('Failed to load models:', error)
    }
  }

  const loadPredictions = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/ml/predictions?horizon_days=7')
      if (response.ok) {
        const data = await response.json()
        setPredictions(data.predictions || [])
      }
    } catch (error) {
      console.error('Failed to load predictions:', error)
    } finally {
      setLoading(false)
    }
  }

  const trainNewModel = async () => {
    setTrainingModel(true)
    setTrainingStatus('Starting training on current database...')
    
    try {
      const response = await fetch('http://localhost:8000/api/ml/train-auto', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        }
      })

      if (response.ok) {
        const data = await response.json()
        const trainingId = data.training_id
        setTrainingStatus(`Training started (ID: ${trainingId.slice(0, 8)}...)`)
        
        // Poll for training status
        const pollInterval = setInterval(async () => {
          try {
            const statusResponse = await fetch(`http://localhost:8000/api/ml/train-status/${trainingId}`)
            if (statusResponse.ok) {
              const statusData = await statusResponse.json()
              
              if (statusData.status === 'completed') {
                clearInterval(pollInterval)
                setTrainingStatus(`Training completed! Best R² score: ${statusData.best_model?.r2_score?.toFixed(3) || 'N/A'}`)
                loadModels()
                loadPredictions()
                
                // Clear status after 5 seconds
                setTimeout(() => {
                  setTrainingModel(false)
                  setTrainingStatus('')
                }, 5000)
              } else {
                setTrainingStatus(`Training in progress... (${statusData.models_trained || 0} models completed)`)
              }
            }
          } catch (error) {
            console.error('Status check failed:', error)
          }
        }, 3000) // Check every 3 seconds
        
        // Timeout after 10 minutes
        setTimeout(() => {
          clearInterval(pollInterval)
          if (trainingModel) {
            setTrainingStatus('Training timeout - check manually')
            setTrainingModel(false)
          }
        }, 600000)
        
      } else {
        throw new Error('Failed to start training')
      }
    } catch (error) {
      console.error('Failed to start training:', error)
      setTrainingStatus('Training failed - check backend logs')
      setTrainingModel(false)
    }
  }

  const getRiskColor = (level: string) => {
    switch (level) {
      case 'high': return 'destructive'
      case 'medium': return 'default'
      case 'low': return 'secondary'
      default: return 'outline'
    }
  }

  // Prepare chart data
  const predictionChartData = predictions.map(p => ({
    name: p.location,
    value: p.predicted_fatalities,
    confidence: p.confidence
  }))

  const riskLevelData = predictions.reduce((acc, p) => {
    const existing = acc.find(item => item.name === p.risk_level)
    if (existing) {
      existing.value++
    } else {
      acc.push({ name: p.risk_level, value: 1 })
    }
    return acc
  }, [] as Array<{ name: string; value: number }>)

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {[...Array(3)].map((_, i) => (
            <Card key={i} className="p-6">
              <div className="animate-pulse">
                <div className="h-4 bg-gray-200 rounded w-3/4 mb-2"></div>
                <div className="h-8 bg-gray-200 rounded w-1/2"></div>
              </div>
            </Card>
          ))}
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Machine Learning Dashboard</h2>
          <p className="text-muted-foreground">
            Predictive analytics and model management for conflict forecasting
          </p>
        </div>
        <Button onClick={trainNewModel} disabled={trainingModel}>
          <Brain className="h-4 w-4 mr-2" />
          {trainingModel ? 'Training...' : 'Train Models on Current Data'}
        </Button>
      </div>

      {/* Training Status */}
      {trainingStatus && (
        <Card className="p-4 bg-blue-50 border-blue-200">
          <div className="flex items-center gap-2">
            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
            <span className="text-sm text-blue-800">{trainingStatus}</span>
          </div>
        </Card>
      )}

      {/* Model Overview */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card className="p-6">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-primary/10 rounded-lg">
              <Brain className="h-5 w-5 text-primary" />
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Active Models</p>
              <p className="text-2xl font-bold">{models.length}</p>
            </div>
          </div>
        </Card>

        <Card className="p-6">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-green-100 rounded-lg">
              <Target className="h-5 w-5 text-green-600" />
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Best Model R²</p>
              <p className="text-2xl font-bold">
                {models.length > 0 
                  ? Math.max(...models.map(m => m.metrics?.r2 || 0)).toFixed(3)
                  : '0.000'
                }
              </p>
            </div>
          </div>
        </Card>

        <Card className="p-6">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-orange-100 rounded-lg">
              <AlertTriangle className="h-5 w-5 text-orange-600" />
            </div>
            <div>
              <p className="text-sm text-muted-foreground">High Risk Predictions</p>
              <p className="text-2xl font-bold">
                {predictions.filter(p => p.risk_level === 'high').length}
              </p>
            </div>
          </div>
        </Card>
      </div>

      {/* Models Table */}
      <Card className="p-6">
        <h3 className="text-lg font-semibold mb-4">Trained Models</h3>
        {models.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b">
                  <th className="text-left py-2">Model ID</th>
                  <th className="text-left py-2">Type</th>
                  <th className="text-left py-2">Status</th>
                  <th className="text-left py-2">R² Score</th>
                  <th className="text-left py-2">RMSE</th>
                  <th className="text-left py-2">Training Samples</th>
                </tr>
              </thead>
              <tbody>
                {models.map((model) => (
                  <tr key={model.model_id} className="border-b">
                    <td className="py-2 font-mono text-sm">{model.model_id.slice(0, 12)}...</td>
                    <td className="py-2">
                      <Badge variant="outline">{model.model_type}</Badge>
                    </td>
                    <td className="py-2">
                      <Badge variant={model.status === 'completed' ? 'secondary' : 'default'}>
                        {model.status}
                      </Badge>
                    </td>
                    <td className="py-2">{model.metrics?.r2?.toFixed(3) || 'N/A'}</td>
                    <td className="py-2">{model.metrics?.rmse?.toFixed(2) || 'N/A'}</td>
                    <td className="py-2">{model.training_samples?.toLocaleString() || 'N/A'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="text-center py-8 text-muted-foreground">
            <Brain className="h-12 w-12 mx-auto mb-4 opacity-50" />
            <p>No trained models available</p>
            <p className="text-sm">Train your first model to get started with predictions</p>
          </div>
        )}
      </Card>

      {/* Predictions */}
      {predictions.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card className="p-6">
            <h3 className="text-lg font-semibold mb-4">Predicted Fatalities by Location</h3>
            <MyBarChart 
              data={predictionChartData.slice(0, 10)} 
              dataKey="value"
              nameKey="name"
              height={300}
              color="hsl(var(--destructive))"
            />
          </Card>

          <Card className="p-6">
            <h3 className="text-lg font-semibold mb-4">Risk Level Distribution</h3>
            <MyBarChart 
              data={riskLevelData} 
              dataKey="value"
              nameKey="name"
              height={300}
              color="hsl(var(--primary))"
            />
          </Card>
        </div>
      )}

      {/* Recent Predictions Table */}
      {predictions.length > 0 && (
        <Card className="p-6">
          <h3 className="text-lg font-semibold mb-4">Recent Predictions</h3>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b">
                  <th className="text-left py-2">Date</th>
                  <th className="text-left py-2">Location</th>
                  <th className="text-left py-2">Predicted Fatalities</th>
                  <th className="text-left py-2">Confidence</th>
                  <th className="text-left py-2">Risk Level</th>
                </tr>
              </thead>
              <tbody>
                {predictions.slice(0, 10).map((prediction, index) => (
                  <tr key={index} className="border-b">
                    <td className="py-2">{new Date(prediction.date).toLocaleDateString()}</td>
                    <td className="py-2">{prediction.location}</td>
                    <td className="py-2 font-semibold">{prediction.predicted_fatalities.toFixed(1)}</td>
                    <td className="py-2">{(prediction.confidence * 100).toFixed(1)}%</td>
                    <td className="py-2">
                      <Badge variant={getRiskColor(prediction.risk_level) as any}>
                        {prediction.risk_level}
                      </Badge>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  )
}