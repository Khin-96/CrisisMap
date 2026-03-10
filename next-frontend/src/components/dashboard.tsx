'use client'

import { useState, useEffect } from 'react'
import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'
import { SidebarProvider, SidebarTrigger } from '@/components/ui/sidebar'
import { AppSidebar } from '@/components/app-sidebar'
import { MetricCard } from '@/components/metric-card'
import { TrendChart } from '@/components/trend-chart'
import { AlertFeed } from '@/components/alert-feed'
import RealMap from '@/components/real-map'
import CSVUpload from '@/components/csv-upload'
import EventsTable from '@/components/events-table'
import AnalyticsDashboard from '@/components/analytics-dashboard'
import AIAssistant from '@/components/ai-assistant'
import MLDashboard from '@/components/ml-dashboard'
import SystemStatus from '@/components/system-status'

interface ConflictEvent {
  event_id: string
  latitude: number
  longitude: number
  event_type: string
  fatalities: number
  event_date: string
  location?: string
  country?: string
  actor1?: string
  actor2?: string
}

interface TrendData {
  period: string
  total_events: number
  total_fatalities: number
  hotspot_locations: string[]
  trend_direction: 'increasing' | 'decreasing' | 'stable'
  temporal_data?: any[]
  predictions?: any
}

export default function Dashboard() {
  const [events, setEvents] = useState<ConflictEvent[]>([])
  const [trends, setTrends] = useState<TrendData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedCountry, setSelectedCountry] = useState<string>('')
  const [selectedEventType, setSelectedEventType] = useState<string>('')
  const [dateRange, setDateRange] = useState({
    start: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
    end: new Date().toISOString().split('T')[0]
  })
  const [activeTab, setActiveTab] = useState<'dashboard' | 'upload' | 'analytics' | 'events' | 'status' | 'ai' | 'ml'>('dashboard')
  const [showHeatmap, setShowHeatmap] = useState(false)
  const [selectedEvent, setSelectedEvent] = useState<ConflictEvent | null>(null)

  // Fetch events data
  const fetchEvents = async () => {
    try {
      const params = new URLSearchParams({
        limit: '5000',
        start_date: dateRange.start,
        end_date: dateRange.end
      })
      
      if (selectedCountry) {
        params.append('country', selectedCountry)
      }
      
      if (selectedEventType) {
        params.append('event_type', selectedEventType)
      }

      // Add timeout to prevent infinite loading
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), 10000) // 10 second timeout

      const response = await fetch(`http://localhost:8000/api/events?${params}`, {
        signal: controller.signal
      })
      
      clearTimeout(timeoutId)
      
      if (response.ok) {
        const data = await response.json()
        setEvents(Array.isArray(data) ? data : [])
      } else {
        console.error('Failed to fetch events:', response.status, response.statusText)
        setEvents([]) // Set empty array on error
      }
    } catch (error: any) {
      if (error.name === 'AbortError') {
        console.error('Request timed out')
      } else {
        console.error('Failed to fetch events:', error)
      }
      setEvents([]) // Set empty array on error
    }
  }

  // Fetch trends data
  const fetchTrends = async () => {
    try {
      const params = new URLSearchParams({
        period: 'monthly',
        include_predictions: 'true'
      })
      
      if (selectedCountry) {
        params.append('country', selectedCountry)
      }

      const response = await fetch(`http://localhost:8000/api/trends?${params}`)
      if (response.ok) {
        const data = await response.json()
        setTrends(data)
      } else {
        // If trends endpoint doesn't exist, create mock trends data
        console.log('Trends endpoint not available, using mock data')
        setTrends({
          period: 'monthly',
          total_events: events.length,
          total_fatalities: events.reduce((sum, e) => sum + (e.fatalities || 0), 0),
          hotspot_locations: [...new Set(events.map(e => e.location).filter((loc): loc is string => Boolean(loc)))].slice(0, 5),
          trend_direction: 'increasing',
          temporal_data: generateMockTemporalData(),
          predictions: null
        })
      }
    } catch (error) {
      console.error('Failed to fetch trends:', error)
      // Create fallback trends data
      setTrends({
        period: 'monthly',
        total_events: events.length,
        total_fatalities: events.reduce((sum, e) => sum + (e.fatalities || 0), 0),
        hotspot_locations: [...new Set(events.map(e => e.location).filter((loc): loc is string => Boolean(loc)))].slice(0, 5),
        trend_direction: 'stable',
        temporal_data: generateMockTemporalData(),
        predictions: null
      })
    }
  }

  // Generate mock temporal data for charts
  const generateMockTemporalData = () => {
    const data = []
    const now = new Date()
    for (let i = 11; i >= 0; i--) {
      const date = new Date(now.getFullYear(), now.getMonth() - i, 1)
      data.push({
        period: date.toISOString(),
        total_events: Math.floor(Math.random() * 100) + 50,
        total_fatalities: Math.floor(Math.random() * 500) + 100,
      })
    }
    return data
  }

  // Initial data load
  useEffect(() => {
    const loadData = async () => {
      setLoading(true)
      setError(null)
      
      try {
        console.log('Starting to load dashboard data...')
        
        // First test basic connectivity
        console.log('Testing API connectivity...')
        const testResponse = await fetch('http://localhost:8000/health')
        if (!testResponse.ok) {
          throw new Error(`API not accessible: ${testResponse.status}`)
        }
        const healthData = await testResponse.json()
        console.log('API connectivity test passed:', healthData)
        
        // Set a maximum loading time of 15 seconds
        const loadingTimeout = setTimeout(() => {
          console.log('Loading timeout reached, stopping loading state')
          setLoading(false)
          setError('Loading timed out. Please refresh the page.')
        }, 15000)

        // Load events first
        console.log('Fetching events...')
        await fetchEvents()
        
        // Then load trends
        console.log('Fetching trends...')
        await fetchTrends()
        
        console.log('Data loading completed successfully')
        clearTimeout(loadingTimeout)
        setLoading(false)
      } catch (error: any) {
        console.error('Error during data loading:', error)
        setError(`Failed to load data: ${error.message}. Please check if the backend is running on http://localhost:8000`)
        setLoading(false)
      }
    }
    loadData()
  }, [selectedCountry, selectedEventType, dateRange])

  // Handle successful upload
  const handleUploadComplete = () => {
    fetchEvents()
    fetchTrends()
    setActiveTab('dashboard')
  }

  // Handle event selection
  const handleEventSelect = (event: ConflictEvent) => {
    setSelectedEvent(event)
  }

  // Calculate metrics from events
  const metrics = {
    totalEvents: events.length,
    totalFatalities: events.reduce((sum, event) => sum + event.fatalities, 0),
    activeHotspots: trends?.hotspot_locations.length || 0,
    trendDirection: trends?.trend_direction || 'stable',
    avgFatalitiesPerEvent: events.length > 0 ? Math.round(events.reduce((sum, event) => sum + event.fatalities, 0) / events.length) : 0
  }

  // Get unique values for filters
  const countries = Array.from(new Set(events.map(e => e.country).filter((country): country is string => Boolean(country)))).sort()
  const eventTypes = Array.from(new Set(events.map(e => e.event_type).filter((type): type is string => Boolean(type)))).sort()

  // Get risk level
  const getRiskLevel = (): { level: string; color: 'default' | 'secondary' | 'destructive' | 'outline' } => {
    if (metrics.totalFatalities > 5000) return { level: 'Critical', color: 'destructive' }
    if (metrics.totalFatalities > 2000) return { level: 'High', color: 'destructive' }
    if (metrics.totalFatalities > 500) return { level: 'Medium', color: 'default' }
    return { level: 'Low', color: 'secondary' }
  }

  const riskLevel = getRiskLevel()

  return (
    <SidebarProvider>
      <AppSidebar 
        activeTab={activeTab} 
        onTabChange={(tab: string) => setActiveTab(tab as typeof activeTab)}
        eventCount={events.length}
      />
      <main className="flex-1 overflow-hidden">
        <div className="flex h-screen flex-col">
          {/* Header */}
          <div className="flex h-16 items-center border-b bg-background px-6">
            <SidebarTrigger />
            <div className="ml-4">
              <h1 className="text-xl font-semibold">
                {activeTab === 'dashboard' && 'Dashboard'}
                {activeTab === 'upload' && 'Data Upload'}
                {activeTab === 'events' && 'Events Table'}
                {activeTab === 'analytics' && 'Analytics'}
                {activeTab === 'status' && 'System Status'}
                {activeTab === 'ai' && 'AI Assistant'}
                {activeTab === 'ml' && 'ML Models'}
              </h1>
              <p className="text-sm text-muted-foreground">
                {activeTab === 'dashboard' && 'Real-time conflict monitoring and early warning system'}
                {activeTab === 'upload' && 'Upload CSV or Excel files with conflict data'}
                {activeTab === 'events' && 'Detailed view of all conflict events'}
                {activeTab === 'analytics' && 'Comprehensive analysis of conflict patterns and trends'}
                {activeTab === 'status' && 'Monitor system health and data quality'}
                {activeTab === 'ai' && 'AI-powered insights and strategic recommendations'}
                {activeTab === 'ml' && 'Machine learning models and predictive analytics'}
              </p>
            </div>
          </div>

          {/* Content */}
          <div className="flex-1 overflow-auto p-6">
            {loading && activeTab === 'dashboard' ? (
              <div className="space-y-6">
                {error ? (
                  <Card className="p-8 text-center">
                    <div className="text-red-600 mb-4">
                      <svg className="h-12 w-12 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z" />
                      </svg>
                      <h3 className="text-lg font-semibold mb-2">Loading Error</h3>
                      <p className="text-gray-600 mb-4">{error}</p>
                      <button 
                        onClick={() => window.location.reload()} 
                        className="px-4 py-2 bg-primary text-white rounded hover:bg-primary/90"
                      >
                        Refresh Page
                      </button>
                    </div>
                  </Card>
                ) : (
                  <>
                    <Card className="p-4">
                      <div className="flex items-center gap-4">
                        <Skeleton className="h-16 w-32" />
                        <Skeleton className="h-16 w-32" />
                        <Skeleton className="h-16 w-32" />
                        <Skeleton className="h-16 w-32" />
                      </div>
                    </Card>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6">
                      {[...Array(5)].map((_, i) => (
                        <Card key={i} className="p-6">
                          <Skeleton className="h-4 w-24 mb-2" />
                          <Skeleton className="h-8 w-16" />
                        </Card>
                      ))}
                    </div>
                    
                    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                      <div className="lg:col-span-2">
                        <Card className="p-6">
                          <Skeleton className="h-96 w-full" />
                        </Card>
                      </div>
                      <div className="space-y-6">
                        <Card className="p-6">
                          <Skeleton className="h-48 w-full" />
                        </Card>
                      </div>
                    </div>
                    
                    <div className="text-center mt-8">
                      <div className="inline-flex items-center gap-2 text-sm text-gray-500">
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary"></div>
                        Loading dashboard data...
                      </div>
                    </div>
                  </>
                )}
              </div>
            ) : (
              <>
                {activeTab === 'upload' && (
                  <CSVUpload onUploadComplete={handleUploadComplete} />
                )}

                {activeTab === 'events' && (
                  <div className="space-y-6">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-4">
                        <Badge variant={riskLevel.color}>
                          Risk Level: {riskLevel.level}
                        </Badge>
                        <Badge variant="outline">
                          {events.length.toLocaleString()} Events
                        </Badge>
                      </div>
                    </div>
                    
                    <Card className="p-4">
                      <div className="flex flex-wrap items-center gap-4">
                        <div>
                          <label className="block text-sm font-medium mb-1">Country</label>
                          <select
                            value={selectedCountry}
                            onChange={(e) => setSelectedCountry(e.target.value)}
                            className="px-3 py-2 border border-gray-300 rounded-md text-sm"
                          >
                            <option value="">All Countries</option>
                            {countries.map(country => (
                              <option key={country} value={country}>{country}</option>
                            ))}
                          </select>
                        </div>
                        
                        <div>
                          <label className="block text-sm font-medium mb-1">Event Type</label>
                          <select
                            value={selectedEventType}
                            onChange={(e) => setSelectedEventType(e.target.value)}
                            className="px-3 py-2 border border-gray-300 rounded-md text-sm"
                          >
                            <option value="">All Types</option>
                            {eventTypes.map(type => (
                              <option key={type} value={type}>{type}</option>
                            ))}
                          </select>
                        </div>
                        
                        <div>
                          <label className="block text-sm font-medium mb-1">Start Date</label>
                          <input
                            type="date"
                            value={dateRange.start}
                            onChange={(e) => setDateRange(prev => ({ ...prev, start: e.target.value }))}
                            className="px-3 py-2 border border-gray-300 rounded-md text-sm"
                          />
                        </div>
                        
                        <div>
                          <label className="block text-sm font-medium mb-1">End Date</label>
                          <input
                            type="date"
                            value={dateRange.end}
                            onChange={(e) => setDateRange(prev => ({ ...prev, end: e.target.value }))}
                            className="px-3 py-2 border border-gray-300 rounded-md text-sm"
                          />
                        </div>
                      </div>
                    </Card>

                    <EventsTable 
                      events={events} 
                      onEventSelect={handleEventSelect}
                      maxRows={1000}
                    />
                  </div>
                )}

                {activeTab === 'analytics' && (
                  <AnalyticsDashboard events={events} />
                )}

                {activeTab === 'status' && (
                  <SystemStatus />
                )}

                {activeTab === 'ai' && (
                  <AIAssistant 
                    events={events}
                    currentContext={{
                      selectedCountry,
                      selectedEventType,
                      dateRange,
                      metrics,
                      trends
                    }}
                  />
                )}

                {activeTab === 'ml' && (
                  <MLDashboard />
                )}

                {activeTab === 'dashboard' && (
                  <div className="space-y-6">
                    {/* Filters Section */}
                    <Card className="p-4">
                      <h3 className="text-sm font-medium text-gray-700 mb-3">Filters</h3>
                      <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-6 gap-4">
                        <div>
                          <label className="block text-xs font-medium mb-1 text-gray-600">Country</label>
                          <select
                            value={selectedCountry}
                            onChange={(e) => setSelectedCountry(e.target.value)}
                            className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                          >
                            <option value="">All Countries</option>
                            {countries.map(country => (
                              <option key={country} value={country}>{country}</option>
                            ))}
                          </select>
                        </div>
                        
                        <div>
                          <label className="block text-xs font-medium mb-1 text-gray-600">Event Type</label>
                          <select
                            value={selectedEventType}
                            onChange={(e) => setSelectedEventType(e.target.value)}
                            className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                          >
                            <option value="">All Types</option>
                            {eventTypes.map(type => (
                              <option key={type} value={type}>{type}</option>
                            ))}
                          </select>
                        </div>
                        
                        <div>
                          <label className="block text-xs font-medium mb-1 text-gray-600">Start Date</label>
                          <input
                            type="date"
                            value={dateRange.start}
                            onChange={(e) => setDateRange(prev => ({ ...prev, start: e.target.value }))}
                            className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                          />
                        </div>
                        
                        <div>
                          <label className="block text-xs font-medium mb-1 text-gray-600">End Date</label>
                          <input
                            type="date"
                            value={dateRange.end}
                            onChange={(e) => setDateRange(prev => ({ ...prev, end: e.target.value }))}
                            className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                          />
                        </div>

                        <div>
                          <label className="block text-xs font-medium mb-1 text-gray-600">View Mode</label>
                          <label className="flex items-center space-x-2 mt-2">
                            <input
                              type="checkbox"
                              checked={showHeatmap}
                              onChange={(e) => setShowHeatmap(e.target.checked)}
                              className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                            />
                            <span className="text-sm text-gray-700">Heatmap</span>
                          </label>
                        </div>

                        <div>
                          <label className="block text-xs font-medium mb-1 text-gray-600">Risk Level</label>
                          <Badge variant={riskLevel.color} className="mt-2">
                            {riskLevel.level}
                          </Badge>
                        </div>
                      </div>
                    </Card>

                    {/* Key Metrics */}
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                      <MetricCard
                        title="Total Events"
                        value={metrics.totalEvents}
                        trend={trends?.trend_direction}
                      />
                      <MetricCard
                        title="Total Fatalities"
                        value={metrics.totalFatalities}
                        trend="increasing"
                      />
                      <MetricCard
                        title="Active Hotspots"
                        value={metrics.activeHotspots}
                        trend="increasing"
                      />
                      <MetricCard
                        title="Avg Fatalities/Event"
                        value={metrics.avgFatalitiesPerEvent}
                        trend="stable"
                      />
                    </div>

                    {/* Main Content - Map and Event Types */}
                    <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
                      {/* Map Section - Takes 3 columns */}
                      <div className="lg:col-span-3">
                        <Card className="h-full">
                          <div className="p-4 border-b">
                            <h3 className="text-lg font-semibold">Geographic Distribution</h3>
                            <p className="text-sm text-gray-600">Interactive map showing conflict events and hotspots</p>
                          </div>
                          <div className="p-0">
                            <RealMap
                              events={events}
                              height="600px"
                              showHeatmap={showHeatmap}
                              filterByDateRange={dateRange}
                              filterByType={selectedEventType ? [selectedEventType] : undefined}
                              onEventClick={handleEventSelect}
                            />
                          </div>
                        </Card>
                      </div>

                      {/* Event Types Sidebar - Takes 1 column */}
                      <div className="lg:col-span-1">
                        <Card className="h-full">
                          <div className="p-4 border-b">
                            <h3 className="text-lg font-semibold">Event Types</h3>
                            <p className="text-sm text-gray-600">Distribution by conflict type</p>
                          </div>
                          <div className="p-4">
                            <div className="space-y-3">
                              {eventTypes.slice(0, 8).map(type => {
                                const typeEvents = events.filter(e => e.event_type === type)
                                const typeFatalities = typeEvents.reduce((sum, e) => sum + e.fatalities, 0)
                                return (
                                  <div key={type} className="space-y-2">
                                    <div className="flex items-center justify-between">
                                      <Badge 
                                        variant={
                                          type.toLowerCase().includes('violence') ? 'destructive' :
                                          type.toLowerCase().includes('protest') ? 'default' :
                                          'secondary'
                                        }
                                        className="text-xs"
                                      >
                                        {type}
                                      </Badge>
                                      <span className="text-xs text-gray-500">{typeEvents.length}</span>
                                    </div>
                                    <div className="text-xs text-gray-600">
                                      {typeFatalities.toLocaleString()} fatalities
                                    </div>
                                    <div className="w-full bg-gray-200 rounded-full h-1">
                                      <div 
                                        className="bg-blue-600 h-1 rounded-full" 
                                        style={{ 
                                          width: `${Math.min((typeEvents.length / events.length) * 100, 100)}%` 
                                        }}
                                      />
                                    </div>
                                  </div>
                                )
                              })}
                            </div>
                          </div>
                        </Card>
                      </div>
                    </div>

                    {/* Bottom Section - Charts and Recent Events */}
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                      {/* Trend Chart */}
                      <Card className="p-6">
                        <div className="mb-4">
                          <h3 className="text-lg font-semibold">Temporal Trends</h3>
                          <p className="text-sm text-gray-600">Events and fatalities over time</p>
                        </div>
                        <TrendChart data={trends?.temporal_data || []} />
                      </Card>
                      
                      {/* Recent High-Impact Events */}
                      <Card className="p-6">
                        <div className="mb-4">
                          <h3 className="text-lg font-semibold">Recent High-Impact Events</h3>
                          <p className="text-sm text-gray-600">Events with significant casualties</p>
                        </div>
                        <EventsTable 
                          events={events
                            .filter(e => e.fatalities > 10)
                            .sort((a, b) => new Date(b.event_date).getTime() - new Date(a.event_date).getTime())
                          } 
                          onEventSelect={handleEventSelect}
                          maxRows={5}
                        />
                      </Card>
                    </div>

                    {/* Alert Feed */}
                    <Card className="p-6">
                      <div className="mb-4">
                        <h3 className="text-lg font-semibold">Live Alert Feed</h3>
                        <p className="text-sm text-gray-600">Recent conflict events and updates</p>
                      </div>
                      <AlertFeed events={events} />
                    </Card>
                  </div>
                )}
              </>
            )}
          </div>
        </div>

        {selectedEvent && (
          <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-[9999] p-4">
            <div className="relative z-[10000] max-w-2xl w-full max-h-[90vh] overflow-y-auto">
              <Card className="p-6 shadow-2xl">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold">Event Details</h3>
                  <button
                    onClick={() => setSelectedEvent(null)}
                    className="text-gray-500 hover:text-gray-700 text-xl font-bold w-8 h-8 flex items-center justify-center rounded-full hover:bg-gray-100 transition-colors"
                  >
                    ×
                  </button>
                </div>
                <div className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <strong className="text-sm text-gray-600">Date:</strong>
                      <div className="text-base">{new Date(selectedEvent.event_date).toLocaleDateString()}</div>
                    </div>
                    <div>
                      <strong className="text-sm text-gray-600">Event Type:</strong>
                      <div className="mt-1">
                        <Badge variant={
                          selectedEvent.event_type.toLowerCase().includes('violence') ? 'destructive' :
                          selectedEvent.event_type.toLowerCase().includes('protest') ? 'default' :
                          'secondary'
                        }>
                          {selectedEvent.event_type}
                        </Badge>
                      </div>
                    </div>
                    <div>
                      <strong className="text-sm text-gray-600">Location:</strong>
                      <div className="text-base">{selectedEvent.location || 'Unknown'}</div>
                    </div>
                    <div>
                      <strong className="text-sm text-gray-600">Country:</strong>
                      <div className="text-base">{selectedEvent.country || 'Unknown'}</div>
                    </div>
                    <div>
                      <strong className="text-sm text-gray-600">Fatalities:</strong>
                      <div className="text-base font-semibold text-red-600">{selectedEvent.fatalities}</div>
                    </div>
                    <div>
                      <strong className="text-sm text-gray-600">Coordinates:</strong>
                      <div className="text-sm text-gray-500">
                        {selectedEvent.latitude.toFixed(4)}, {selectedEvent.longitude.toFixed(4)}
                      </div>
                    </div>
                  </div>
                  
                  {(selectedEvent.actor1 || selectedEvent.actor2) && (
                    <div className="border-t pt-4">
                      <strong className="text-sm text-gray-600">Conflict Actors:</strong>
                      <div className="mt-2 space-y-2">
                        {selectedEvent.actor1 && (
                          <div className="flex items-center space-x-2">
                            <Badge variant="outline">Actor 1</Badge>
                            <span className="text-sm">{selectedEvent.actor1}</span>
                          </div>
                        )}
                        {selectedEvent.actor2 && (
                          <div className="flex items-center space-x-2">
                            <Badge variant="outline">Actor 2</Badge>
                            <span className="text-sm">{selectedEvent.actor2}</span>
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              </Card>
            </div>
          </div>
        )}
      </main>
    </SidebarProvider>
  )
}