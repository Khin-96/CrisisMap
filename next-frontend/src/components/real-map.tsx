'use client'

import React, { useEffect, useRef, useState } from 'react'
import { Card } from '@/components/ui/card'

// Define types for our data
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

interface RealMapProps {
  events: ConflictEvent[]
  height?: string
  onEventClick?: (event: ConflictEvent) => void
  showHeatmap?: boolean
  filterByType?: string[]
  filterByDateRange?: { start: string; end: string }
}

export default function RealMap({ 
  events, 
  height = "600px", 
  onEventClick,
  showHeatmap = false,
  filterByType,
  filterByDateRange
}: RealMapProps) {
  const mapRef = useRef<HTMLDivElement>(null)
  const mapInstanceRef = useRef<any>(null)
  const markersRef = useRef<any[]>([])
  const heatmapLayerRef = useRef<any>(null)
  const [isLoaded, setIsLoaded] = useState(false)
  const [filteredEvents, setFilteredEvents] = useState<ConflictEvent[]>(events)

  // Filter events based on props
  useEffect(() => {
    let filtered = [...events]

    // Filter by event type
    if (filterByType && filterByType.length > 0) {
      filtered = filtered.filter(event => 
        filterByType.includes(event.event_type)
      )
    }

    // Filter by date range
    if (filterByDateRange) {
      filtered = filtered.filter(event => {
        const eventDate = new Date(event.event_date)
        const startDate = new Date(filterByDateRange.start)
        const endDate = new Date(filterByDateRange.end)
        return eventDate >= startDate && eventDate <= endDate
      })
    }

    setFilteredEvents(filtered)
  }, [events, filterByType, filterByDateRange])

  // Load Leaflet dynamically
  useEffect(() => {
    const loadLeaflet = async () => {
      if (typeof window === 'undefined') return

      // Load Leaflet CSS
      if (!document.querySelector('link[href*="leaflet.css"]')) {
        const link = document.createElement('link')
        link.rel = 'stylesheet'
        link.href = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.css'
        document.head.appendChild(link)
      }

      // Load Leaflet JS
      if (!window.L) {
        const script = document.createElement('script')
        script.src = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.js'
        script.onload = () => {
          // Load heatmap plugin if needed
          if (showHeatmap) {
            const heatmapScript = document.createElement('script')
            heatmapScript.src = 'https://unpkg.com/leaflet.heat@0.2.0/dist/leaflet-heat.js'
            heatmapScript.onload = () => setIsLoaded(true)
            document.head.appendChild(heatmapScript)
          } else {
            setIsLoaded(true)
          }
        }
        document.head.appendChild(script)
      } else {
        setIsLoaded(true)
      }
    }

    loadLeaflet()
  }, [showHeatmap])

  // Initialize map
  useEffect(() => {
    if (!isLoaded || !mapRef.current || mapInstanceRef.current) return

    const L = (window as any).L
    
    // Create map
    const map = L.map(mapRef.current, {
      center: [0, 20], // Center on Africa
      zoom: 3,
      zoomControl: true,
      scrollWheelZoom: true
    })

    // Add tile layer
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '© OpenStreetMap contributors',
      maxZoom: 18
    }).addTo(map)

    mapInstanceRef.current = map

    return () => {
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove()
        mapInstanceRef.current = null
      }
    }
  }, [isLoaded])

  // Update markers when events change
  useEffect(() => {
    if (!mapInstanceRef.current || !isLoaded) return

    const L = (window as any).L
    const map = mapInstanceRef.current

    // Clear existing markers
    markersRef.current.forEach(marker => map.removeLayer(marker))
    markersRef.current = []

    // Clear existing heatmap
    if (heatmapLayerRef.current) {
      map.removeLayer(heatmapLayerRef.current)
      heatmapLayerRef.current = null
    }

    if (showHeatmap && L.heatLayer) {
      // Create heatmap
      const heatmapData = filteredEvents.map(event => [
        event.latitude,
        event.longitude,
        Math.min(event.fatalities / 10, 1) // Normalize intensity
      ])

      heatmapLayerRef.current = L.heatLayer(heatmapData, {
        radius: 25,
        blur: 15,
        maxZoom: 10,
        gradient: {
          0.0: 'blue',
          0.2: 'cyan',
          0.4: 'lime',
          0.6: 'yellow',
          0.8: 'orange',
          1.0: 'red'
        }
      }).addTo(map)
    } else {
      // Create individual markers
      filteredEvents.forEach(event => {
        // Determine marker color based on fatalities
        let color = '#3388ff' // Default blue
        if (event.fatalities > 100) color = '#ff0000' // Red for high fatalities
        else if (event.fatalities > 50) color = '#ff8800' // Orange for medium
        else if (event.fatalities > 10) color = '#ffff00' // Yellow for low-medium
        else if (event.fatalities > 0) color = '#88ff00' // Light green for low

        // Create custom icon
        const icon = L.divIcon({
          className: 'custom-marker',
          html: `<div style="
            background-color: ${color};
            width: ${Math.max(8, Math.min(20, event.fatalities / 5 + 8))}px;
            height: ${Math.max(8, Math.min(20, event.fatalities / 5 + 8))}px;
            border-radius: 50%;
            border: 2px solid white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
          "></div>`,
          iconSize: [20, 20],
          iconAnchor: [10, 10]
        })

        const marker = L.marker([event.latitude, event.longitude], { icon })
          .bindPopup(`
            <div style="min-width: 200px;">
              <h3 style="margin: 0 0 8px 0; color: #333;">${event.event_type}</h3>
              <p style="margin: 4px 0;"><strong>Location:</strong> ${event.location || 'Unknown'}</p>
              <p style="margin: 4px 0;"><strong>Date:</strong> ${new Date(event.event_date).toLocaleDateString()}</p>
              <p style="margin: 4px 0;"><strong>Fatalities:</strong> ${event.fatalities}</p>
              ${event.country ? `<p style="margin: 4px 0;"><strong>Country:</strong> ${event.country}</p>` : ''}
              ${event.actor1 ? `<p style="margin: 4px 0;"><strong>Actor 1:</strong> ${event.actor1}</p>` : ''}
              ${event.actor2 ? `<p style="margin: 4px 0;"><strong>Actor 2:</strong> ${event.actor2}</p>` : ''}
            </div>
          `)
          .addTo(map)

        // Add click handler
        if (onEventClick) {
          marker.on('click', () => onEventClick(event))
        }

        markersRef.current.push(marker)
      })
    }

    // Fit map to show all events if there are any
    if (filteredEvents.length > 0) {
      const group = new L.featureGroup(markersRef.current)
      if (markersRef.current.length > 0) {
        map.fitBounds(group.getBounds().pad(0.1))
      }
    }
  }, [filteredEvents, isLoaded, showHeatmap, onEventClick])

  if (!isLoaded) {
    return (
      <Card className="p-6" style={{ height }}>
        <div className="flex items-center justify-center h-full">
          <div className="text-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-4"></div>
            <p className="text-gray-600">Loading map...</p>
          </div>
        </div>
      </Card>
    )
  }

  return (
    <Card className="overflow-hidden">
      <div className="p-4 border-b bg-gray-50">
        <div className="flex justify-between items-center">
          <h3 className="text-lg font-semibold">Conflict Events Map</h3>
          <div className="flex items-center space-x-4 text-sm text-gray-600">
            <span>{filteredEvents.length} events shown</span>
            {showHeatmap && (
              <span className="px-2 py-1 bg-blue-100 text-blue-800 rounded">
                Heatmap Mode
              </span>
            )}
          </div>
        </div>
      </div>
      
      <div 
        ref={mapRef} 
        style={{ height, width: '100%' }}
        className="relative"
      />
      
      {/* Legend */}
      <div className="p-4 border-t bg-gray-50">
        <div className="flex items-center justify-between text-sm">
          <div className="flex items-center space-x-4">
            <span className="font-medium">Fatality Scale:</span>
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 rounded-full bg-blue-500"></div>
              <span>0</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 rounded-full bg-green-400"></div>
              <span>1-10</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 rounded-full bg-yellow-400"></div>
              <span>11-50</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 rounded-full bg-orange-500"></div>
              <span>51-100</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 rounded-full bg-red-500"></div>
              <span>100+</span>
            </div>
          </div>
          <span className="text-gray-500">Click markers for details</span>
        </div>
      </div>
    </Card>
  )
}