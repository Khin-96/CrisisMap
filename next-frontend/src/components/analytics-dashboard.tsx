'use client'

import React, { useMemo } from 'react'
import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { MyBarChart } from '@/components/bar-chart'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'

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

interface AnalyticsDashboardProps {
  events: ConflictEvent[]
}

export default function AnalyticsDashboard({ events }: AnalyticsDashboardProps) {
  const analytics = useMemo(() => {
    if (!events.length) return null

    // Country statistics
    const countryStats = events.reduce((acc, event) => {
      const country = event.country || 'Unknown'
      if (!acc[country]) {
        acc[country] = { events: 0, fatalities: 0, types: new Set() }
      }
      acc[country].events++
      acc[country].fatalities += event.fatalities
      acc[country].types.add(event.event_type)
      return acc
    }, {} as Record<string, { events: number; fatalities: number; types: Set<string> }>)

    // Event type statistics
    const typeStats = events.reduce((acc, event) => {
      const type = event.event_type || 'Unknown'
      if (!acc[type]) {
        acc[type] = { events: 0, fatalities: 0, countries: new Set() }
      }
      acc[type].events++
      acc[type].fatalities += event.fatalities
      if (event.country) acc[type].countries.add(event.country)
      return acc
    }, {} as Record<string, { events: number; fatalities: number; countries: Set<string> }>)

    // Actor statistics
    const actorStats = events.reduce((acc, event) => {
      [event.actor1, event.actor2].forEach(actor => {
        if (actor && actor !== 'null' && actor !== 'undefined') {
          if (!acc[actor]) {
            acc[actor] = { events: 0, fatalities: 0, countries: new Set() }
          }
          acc[actor].events++
          acc[actor].fatalities += event.fatalities
          if (event.country) acc[actor].countries.add(event.country)
        }
      })
      return acc
    }, {} as Record<string, { events: number; fatalities: number; countries: Set<string> }>)

    // Temporal analysis
    const monthlyStats = events.reduce((acc, event) => {
      const date = new Date(event.event_date)
      const monthKey = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`
      if (!acc[monthKey]) {
        acc[monthKey] = { events: 0, fatalities: 0 }
      }
      acc[monthKey].events++
      acc[monthKey].fatalities += event.fatalities
      return acc
    }, {} as Record<string, { events: number; fatalities: number }>)

    // Hotspot analysis (by location)
    const locationStats = events.reduce((acc, event) => {
      const location = event.location || 'Unknown'
      if (!acc[location]) {
        acc[location] = { 
          events: 0, 
          fatalities: 0, 
          latitude: event.latitude, 
          longitude: event.longitude,
          country: event.country 
        }
      }
      acc[location].events++
      acc[location].fatalities += event.fatalities
      return acc
    }, {} as Record<string, { events: number; fatalities: number; latitude: number; longitude: number; country?: string }>)

    // Sort and get top entries
    const topCountries = Object.entries(countryStats)
      .sort(([,a], [,b]) => b.fatalities - a.fatalities)
      .slice(0, 10)

    const topEventTypes = Object.entries(typeStats)
      .sort(([,a], [,b]) => b.events - a.events)
      .slice(0, 10)

    const topActors = Object.entries(actorStats)
      .sort(([,a], [,b]) => b.events - a.events)
      .slice(0, 10)

    const topHotspots = Object.entries(locationStats)
      .sort(([,a], [,b]) => b.fatalities - a.fatalities)
      .slice(0, 10)

    const recentMonths = Object.entries(monthlyStats)
      .sort(([a], [b]) => b.localeCompare(a))
      .slice(0, 12)

    return {
      topCountries,
      topEventTypes,
      topActors,
      topHotspots,
      recentMonths,
      totalEvents: events.length,
      totalFatalities: events.reduce((sum, e) => sum + e.fatalities, 0),
      uniqueCountries: new Set(events.map(e => e.country).filter(Boolean)).size,
      uniqueEventTypes: new Set(events.map(e => e.event_type)).size,
      dateRange: {
        start: events.reduce((min, e) => e.event_date < min ? e.event_date : min, events[0]?.event_date),
        end: events.reduce((max, e) => e.event_date > max ? e.event_date : max, events[0]?.event_date)
      }
    }
  }, [events])

  if (!analytics) {
    return (
      <div className="space-y-6">
        <Card className="p-8 text-center">
          <h3 className="text-lg font-semibold mb-2">No Data Available</h3>
          <p className="text-gray-600">Upload conflict data to see detailed analytics.</p>
        </Card>
      </div>
    )
  }

  const getSeverityBadge = (fatalities: number) => {
    if (fatalities === 0) return <Badge variant="secondary">Low</Badge>
    if (fatalities <= 100) return <Badge variant="outline">Medium</Badge>
    if (fatalities <= 1000) return <Badge variant="default">High</Badge>
    return <Badge variant="destructive">Critical</Badge>
  }

  return (
    <div className="space-y-6">
      {/* Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card className="p-6">
          <div className="text-2xl font-bold text-blue-600">{analytics.totalEvents.toLocaleString()}</div>
          <div className="text-sm text-gray-600">Total Events</div>
          <div className="text-xs text-gray-500 mt-1">
            {new Date(analytics.dateRange.start).toLocaleDateString()} - {new Date(analytics.dateRange.end).toLocaleDateString()}
          </div>
        </Card>
        
        <Card className="p-6">
          <div className="text-2xl font-bold text-red-600">{analytics.totalFatalities.toLocaleString()}</div>
          <div className="text-sm text-gray-600">Total Fatalities</div>
          <div className="text-xs text-gray-500 mt-1">
            Avg: {Math.round(analytics.totalFatalities / analytics.totalEvents)} per event
          </div>
        </Card>
        
        <Card className="p-6">
          <div className="text-2xl font-bold text-green-600">{analytics.uniqueCountries}</div>
          <div className="text-sm text-gray-600">Countries Affected</div>
          <div className="text-xs text-gray-500 mt-1">
            {analytics.uniqueEventTypes} event types
          </div>
        </Card>
        
        <Card className="p-6">
          <div className="text-2xl font-bold text-purple-600">
            {Math.round(analytics.totalEvents / analytics.recentMonths.length)}
          </div>
          <div className="text-sm text-gray-600">Avg Events/Month</div>
          <div className="text-xs text-gray-500 mt-1">
            Last {analytics.recentMonths.length} months
          </div>
        </Card>
      </div>

      {/* Countries Chart */}
      <Card className="p-6">
        <h3 className="text-lg font-semibold mb-4">Most Affected Countries</h3>
        <div className="mb-6">
          <MyBarChart 
            data={analytics.topCountries.slice(0, 10).map(([country, stats]) => ({
              name: country,
              value: stats.fatalities,
              events: stats.events
            }))}
            dataKey="value"
            nameKey="name"
            height={300}
            color="hsl(var(--destructive))"
          />
        </div>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Country</TableHead>
              <TableHead className="text-right">Events</TableHead>
              <TableHead className="text-right">Fatalities</TableHead>
              <TableHead className="text-right">Avg per Event</TableHead>
              <TableHead>Severity</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {analytics.topCountries.slice(0, 8).map(([country, stats]) => (
              <TableRow key={country}>
                <TableCell className="font-medium">{country}</TableCell>
                <TableCell className="text-right">{stats.events.toLocaleString()}</TableCell>
                <TableCell className="text-right">{stats.fatalities.toLocaleString()}</TableCell>
                <TableCell className="text-right">{Math.round(stats.fatalities / stats.events)}</TableCell>
                <TableCell>{getSeverityBadge(stats.fatalities)}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </Card>

      {/* Event Types and Hotspots Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Event Types Chart */}
        <Card className="p-6">
          <h3 className="text-lg font-semibold mb-4">Event Types Distribution</h3>
          <div className="mb-4">
            <MyBarChart 
              data={analytics.topEventTypes.map(([type, stats]) => ({
                name: type.length > 15 ? type.substring(0, 15) + '...' : type,
                value: stats.events,
                fatalities: stats.fatalities
              }))}
              dataKey="value"
              nameKey="name"
              height={250}
              color="hsl(var(--primary))"
            />
          </div>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Type</TableHead>
                <TableHead className="text-right">Events</TableHead>
                <TableHead className="text-right">Fatalities</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {analytics.topEventTypes.slice(0, 5).map(([type, stats]) => (
                <TableRow key={type}>
                  <TableCell>
                    <Badge variant={
                      type.toLowerCase().includes('violence') ? 'destructive' :
                      type.toLowerCase().includes('protest') ? 'default' :
                      'secondary'
                    }>
                      {type}
                    </Badge>
                  </TableCell>
                  <TableCell className="text-right">{stats.events.toLocaleString()}</TableCell>
                  <TableCell className="text-right">{stats.fatalities.toLocaleString()}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </Card>

        {/* Geographic Hotspots Chart */}
        <Card className="p-6">
          <h3 className="text-lg font-semibold mb-4">Geographic Hotspots</h3>
          <div className="mb-4">
            <MyBarChart 
              data={analytics.topHotspots.slice(0, 8).map(([location, stats]) => ({
                name: location.length > 12 ? location.substring(0, 12) + '...' : location,
                value: stats.fatalities,
                events: stats.events
              }))}
              dataKey="value"
              nameKey="name"
              height={250}
              color="hsl(var(--chart-2))"
            />
          </div>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Location</TableHead>
                <TableHead className="text-right">Events</TableHead>
                <TableHead className="text-right">Fatalities</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {analytics.topHotspots.slice(0, 5).map(([location, stats]) => (
                <TableRow key={location}>
                  <TableCell>
                    <div>
                      <div className="font-medium">{location}</div>
                      {stats.country && (
                        <div className="text-xs text-gray-500">{stats.country}</div>
                      )}
                    </div>
                  </TableCell>
                  <TableCell className="text-right">{stats.events.toLocaleString()}</TableCell>
                  <TableCell className="text-right">{stats.fatalities.toLocaleString()}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </Card>
      </div>

      {/* Top Actors */}
      <Card className="p-6">
        <h3 className="text-lg font-semibold mb-4">Most Active Conflict Actors</h3>
        <div className="mb-6">
          <MyBarChart 
            data={analytics.topActors.slice(0, 10).map(([actor, stats]) => ({
              name: actor.length > 20 ? actor.substring(0, 20) + '...' : actor,
              value: stats.events,
              fatalities: stats.fatalities
            }))}
            dataKey="value"
            nameKey="name"
            height={300}
            color="hsl(var(--chart-3))"
          />
        </div>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Actor</TableHead>
              <TableHead className="text-right">Events Involved</TableHead>
              <TableHead className="text-right">Associated Fatalities</TableHead>
              <TableHead className="text-right">Countries</TableHead>
              <TableHead>Activity Level</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {analytics.topActors.slice(0, 8).map(([actor, stats]) => (
              <TableRow key={actor}>
                <TableCell className="font-medium max-w-xs truncate" title={actor}>
                  {actor}
                </TableCell>
                <TableCell className="text-right">{stats.events.toLocaleString()}</TableCell>
                <TableCell className="text-right">{stats.fatalities.toLocaleString()}</TableCell>
                <TableCell className="text-right">{stats.countries.size}</TableCell>
                <TableCell>
                  <Badge variant={
                    stats.events > 100 ? 'destructive' :
                    stats.events > 50 ? 'default' :
                    stats.events > 10 ? 'outline' :
                    'secondary'
                  }>
                    {stats.events > 100 ? 'Very High' :
                     stats.events > 50 ? 'High' :
                     stats.events > 10 ? 'Medium' : 'Low'}
                  </Badge>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </Card>

      {/* Temporal Trends */}
      <Card className="p-6">
        <h3 className="text-lg font-semibold mb-4">Monthly Trends (Recent 12 Months)</h3>
        <div className="mb-6">
          <MyBarChart 
            data={analytics.recentMonths.reverse().map(([month, stats]) => ({
              name: new Date(month + '-01').toLocaleDateString('en-US', { 
                month: 'short', 
                year: '2-digit' 
              }),
              value: stats.events,
              fatalities: stats.fatalities
            }))}
            dataKey="value"
            nameKey="name"
            height={250}
            color="hsl(var(--chart-4))"
          />
        </div>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Month</TableHead>
              <TableHead className="text-right">Events</TableHead>
              <TableHead className="text-right">Fatalities</TableHead>
              <TableHead className="text-right">Avg Fatalities/Event</TableHead>
              <TableHead>Trend</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {analytics.recentMonths.slice(0, 8).map(([month, stats], index) => {
              const prevStats = analytics.recentMonths[index + 1]?.[1]
              const eventTrend = prevStats ? 
                (stats.events > prevStats.events ? 'up' : 
                 stats.events < prevStats.events ? 'down' : 'stable') : 'stable'
              
              return (
                <TableRow key={month}>
                  <TableCell className="font-medium">
                    {new Date(month + '-01').toLocaleDateString('en-US', { 
                      year: 'numeric', 
                      month: 'long' 
                    })}
                  </TableCell>
                  <TableCell className="text-right">{stats.events.toLocaleString()}</TableCell>
                  <TableCell className="text-right">{stats.fatalities.toLocaleString()}</TableCell>
                  <TableCell className="text-right">
                    {stats.events > 0 ? Math.round(stats.fatalities / stats.events) : 0}
                  </TableCell>
                  <TableCell>
                    <Badge variant={
                      eventTrend === 'up' ? 'destructive' :
                      eventTrend === 'down' ? 'secondary' :
                      'outline'
                    }>
                      {eventTrend === 'up' ? '↗ Increasing' :
                       eventTrend === 'down' ? '↘ Decreasing' :
                       '→ Stable'}
                    </Badge>
                  </TableCell>
                </TableRow>
              )
            })}
          </TableBody>
        </Table>
      </Card>
    </div>
  )
}