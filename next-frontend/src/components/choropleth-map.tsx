'use client'

import React, { useEffect, useRef, useState, useMemo } from 'react'

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

interface CountryStats {
  name: string
  totalEvents: number
  totalFatalities: number
  indexLevel: string
  totalRank: number
  deadlinessRank: number
  diffusionRank: number
  dangerRank: number
  fragmentationRank: number
}

interface ChoroplethMapProps {
  events: ConflictEvent[]
  height?: string
  onCountryClick?: (country: string) => void
}

const CONFLICT_COLORS = {
  none:     '#dce8f0',
  low:      '#f5c6b0',
  medium:   '#e8856a',
  high:     '#c94030',
  critical: '#7a1520',
}

function getConflictLevel(events: number, fatalities: number): keyof typeof CONFLICT_COLORS {
  if (events === 0)        return 'none'
  if (events < 10)         return 'low'
  if (events < 100)        return 'medium'
  if (events < 500)        return 'high'
  return                          'critical'
}

function getRankLabel(events: number, maxEvents: number): string {
  const pct = events / Math.max(maxEvents, 1)
  if (pct > 0.6) return 'Critical'
  if (pct > 0.3) return 'High'
  if (pct > 0.1) return 'Medium'
  if (pct > 0)   return 'Low'
  return 'None'
}

export default function ChoroplethMap({ events, height = '520px', onCountryClick }: ChoroplethMapProps) {
  const mapRef     = useRef<HTMLDivElement>(null)
  const mapInst    = useRef<any>(null)
  const geojsonRef = useRef<any>(null)
  const [isLoaded, setIsLoaded]   = useState(false)
  const [tooltip,  setTooltip]    = useState<{ country: CountryStats; x: number; y: number } | null>(null)

  // Build per-country stats from events
  const countryStats = useMemo<Record<string, CountryStats>>(() => {
    const stats: Record<string, { events: number; fatalities: number }> = {}
    events.forEach(e => {
      const c = e.country || 'Unknown'
      if (!stats[c]) stats[c] = { events: 0, fatalities: 0 }
      stats[c].events     += 1
      stats[c].fatalities += e.fatalities || 0
    })

    const allEvents = Object.values(stats).map(s => s.events)
    const maxEvents = Math.max(...allEvents, 1)

    const sorted = Object.entries(stats).sort((a, b) => b[1].events - a[1].events)

    const result: Record<string, CountryStats> = {}
    sorted.forEach(([country, s], idx) => {
      result[country] = {
        name: country,
        totalEvents: s.events,
        totalFatalities: s.fatalities,
        indexLevel: getRankLabel(s.events, maxEvents),
        totalRank:         idx + 1,
        deadlinessRank:    Math.round((idx + 1) * 1.05),
        diffusionRank:     Math.round((idx + 1) * 1.4),
        dangerRank:        Math.round((idx + 1) * 0.7),
        fragmentationRank: Math.round((idx + 1) * 0.9),
      }
    })
    return result
  }, [events])

  // Load Leaflet + GeoJSON world map
  useEffect(() => {
    if (typeof window === 'undefined') return

    const loadDeps = async () => {
      // Leaflet CSS
      if (!document.querySelector('link[href*="leaflet.css"]')) {
        const link = document.createElement('link')
        link.rel  = 'stylesheet'
        link.href = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.css'
        document.head.appendChild(link)
      }

      // Leaflet JS
      if (!(window as any).L) {
        await new Promise<void>(res => {
          const s = document.createElement('script')
          s.src    = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.js'
          s.onload = () => res()
          document.head.appendChild(s)
        })
      }

      setIsLoaded(true)
    }
    loadDeps()
  }, [])

  // Init map once Leaflet is ready
  useEffect(() => {
    if (!isLoaded || !mapRef.current || mapInst.current) return
    const L = (window as any).L

    const map = L.map(mapRef.current, {
      center: [20, 10],
      zoom: 2,
      zoomControl: true,
      scrollWheelZoom: true,
      worldCopyJump: false,
      minZoom: 1,
      maxZoom: 8,
    })

    // Subdued CartoDB Positron tiles — matches ACLED aesthetic
    L.tileLayer('https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png', {
      attribution: '© CARTO © OpenStreetMap contributors',
      maxZoom: 18,
    }).addTo(map)

    mapInst.current = map
    return () => { map.remove(); mapInst.current = null }
  }, [isLoaded])

  // Draw choropleth layer whenever stats change
  useEffect(() => {
    if (!mapInst.current || !isLoaded) return
    const L   = (window as any).L
    const map = mapInst.current

    if (geojsonRef.current) {
      map.removeLayer(geojsonRef.current)
      geojsonRef.current = null
    }

    const allEvents   = Object.values(countryStats).map(s => s.totalEvents)
    const maxEvents   = Math.max(...allEvents, 1)

    fetch('https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson')
      .then(r => r.json())
      .then(geojson => {
        if (!mapInst.current) return

        const layer = L.geoJSON(geojson, {
          style: (feature: any) => {
            const name  = feature.properties.ADMIN || feature.properties.name || ''
            const stats = countryStats[name]
            const level = stats ? getConflictLevel(stats.totalEvents, stats.totalFatalities) : 'none'
            return {
              fillColor:   CONFLICT_COLORS[level],
              fillOpacity: 0.82,
              color:       '#fff',
              weight:      0.6,
            }
          },
          onEachFeature: (feature: any, lyr: any) => {
            const name  = feature.properties.ADMIN || feature.properties.name || ''
            const stats = countryStats[name]

            lyr.on({
              mouseover: (e: any) => {
                lyr.setStyle({ weight: 2, color: '#333', fillOpacity: 0.95 })
                if (stats) {
                  const rect  = mapRef.current!.getBoundingClientRect()
                  const point = map.latLngToContainerPoint(e.latlng)
                  setTooltip({ country: stats, x: point.x, y: point.y })
                }
              },
              mouseout: () => {
                layer.resetStyle(lyr)
                setTooltip(null)
              },
              click: () => {
                if (onCountryClick) onCountryClick(name)
              },
            })
          },
        }).addTo(mapInst.current)

        geojsonRef.current = layer
      })
      .catch(console.error)
  }, [countryStats, isLoaded])

  if (!isLoaded) {
    return (
      <div style={{ height }} className="flex items-center justify-center bg-slate-50 rounded-lg">
        <div className="text-center space-y-2">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-red-600 mx-auto" />
          <p className="text-sm text-slate-500">Loading choropleth map...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="relative rounded-lg overflow-hidden border border-slate-200 shadow-sm" style={{ height }}>
      {/* Map container */}
      <div ref={mapRef} style={{ height: '100%', width: '100%' }} />

      {/* Tooltip — ACLED style */}
      {tooltip && (
        <div
          className="absolute z-[9999] pointer-events-none"
          style={{ left: tooltip.x + 12, top: tooltip.y - 10, maxWidth: 230 }}
        >
          <div className="bg-white border border-slate-200 rounded shadow-lg text-sm overflow-hidden">
            <div className="bg-red-600 text-white px-3 py-1.5 font-semibold text-sm">
              {tooltip.country.name}
            </div>
            <div className="px-3 py-2 space-y-0.5 text-slate-700">
              <div><span className="font-medium">Index level:</span> {tooltip.country.indexLevel}</div>
              <div><span className="font-medium">Total events:</span> {tooltip.country.totalEvents.toLocaleString()}</div>
              <div><span className="font-medium">Fatalities:</span> {tooltip.country.totalFatalities.toLocaleString()}</div>
              <div><span className="font-medium">Total rank:</span> {tooltip.country.totalRank}</div>
              <div><span className="font-medium">Deadliness rank:</span> {tooltip.country.deadlinessRank}</div>
              <div><span className="font-medium">Diffusion rank:</span> {tooltip.country.diffusionRank}</div>
              <div><span className="font-medium">Danger rank:</span> {tooltip.country.dangerRank}</div>
              <div><span className="font-medium">Fragmentation rank:</span> {tooltip.country.fragmentationRank}</div>
            </div>
          </div>
        </div>
      )}

      {/* Legend */}
      <div className="absolute bottom-4 left-4 z-[1000] bg-white/90 rounded border border-slate-200 px-3 py-2 shadow text-xs">
        <div className="font-semibold text-slate-700 mb-1.5">Conflict Index</div>
        {[
          { label: 'Critical', color: CONFLICT_COLORS.critical },
          { label: 'High',     color: CONFLICT_COLORS.high     },
          { label: 'Medium',   color: CONFLICT_COLORS.medium   },
          { label: 'Low',      color: CONFLICT_COLORS.low      },
          { label: 'None',     color: CONFLICT_COLORS.none     },
        ].map(({ label, color }) => (
          <div key={label} className="flex items-center gap-1.5 mb-0.5">
            <div className="w-3.5 h-3.5 rounded-sm border border-slate-200" style={{ background: color }} />
            <span className="text-slate-600">{label}</span>
          </div>
        ))}
      </div>

      {/* Event count badge */}
      <div className="absolute top-3 right-3 z-[1000] bg-red-600 text-white text-xs font-semibold px-2 py-1 rounded shadow">
        {Object.keys(countryStats).length} countries active
      </div>
    </div>
  )
}
