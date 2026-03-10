'use client'

import React, { useState, useMemo } from 'react'
import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  Table,
  TableBody,
  TableCaption,
  TableCell,
  TableFooter,
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
  notes?: string
}

interface EventsTableProps {
  events: ConflictEvent[]
  onEventSelect?: (event: ConflictEvent) => void
  maxRows?: number
}

export default function EventsTable({ events, onEventSelect, maxRows = 50 }: EventsTableProps) {
  const [sortField, setSortField] = useState<keyof ConflictEvent>('event_date')
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('desc')
  const [filterType, setFilterType] = useState<string>('')
  const [filterCountry, setFilterCountry] = useState<string>('')
  const [currentPage, setCurrentPage] = useState(1)
  const itemsPerPage = 20

  // Get unique values for filters
  const eventTypes = useMemo(() => 
    Array.from(new Set(events.map(e => e.event_type).filter(Boolean))).sort(),
    [events]
  )
  
  const countries = useMemo(() => 
    Array.from(new Set(events.map(e => e.country).filter(Boolean))).sort(),
    [events]
  )

  // Filter and sort events
  const filteredAndSortedEvents = useMemo(() => {
    let filtered = events

    // Apply filters
    if (filterType) {
      filtered = filtered.filter(event => event.event_type === filterType)
    }
    if (filterCountry) {
      filtered = filtered.filter(event => event.country === filterCountry)
    }

    // Apply sorting
    filtered.sort((a, b) => {
      const aValue = a[sortField]
      const bValue = b[sortField]
      
      if (aValue === bValue) return 0
      
      const comparison = aValue < bValue ? -1 : 1
      return sortDirection === 'asc' ? comparison : -comparison
    })

    return filtered.slice(0, maxRows)
  }, [events, filterType, filterCountry, sortField, sortDirection, maxRows])

  // Paginated events
  const paginatedEvents = useMemo(() => {
    const startIndex = (currentPage - 1) * itemsPerPage
    return filteredAndSortedEvents.slice(startIndex, startIndex + itemsPerPage)
  }, [filteredAndSortedEvents, currentPage])

  const totalPages = Math.ceil(filteredAndSortedEvents.length / itemsPerPage)

  const handleSort = (field: keyof ConflictEvent) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc')
    } else {
      setSortField(field)
      setSortDirection('desc')
    }
  }

  const getFatalityBadge = (fatalities: number) => {
    if (fatalities === 0) return <Badge variant="secondary">0</Badge>
    if (fatalities <= 10) return <Badge variant="outline">{fatalities}</Badge>
    if (fatalities <= 50) return <Badge variant="default">{fatalities}</Badge>
    if (fatalities <= 100) return <Badge variant="destructive">{fatalities}</Badge>
    return <Badge variant="destructive" className="bg-red-700">{fatalities}</Badge>
  }

  const getEventTypeBadge = (eventType: string) => {
    const type = eventType?.toLowerCase() || ''
    if (type.includes('violence')) return <Badge variant="destructive">{eventType}</Badge>
    if (type.includes('protest')) return <Badge variant="default">{eventType}</Badge>
    if (type.includes('riot')) return <Badge variant="destructive">{eventType}</Badge>
    return <Badge variant="secondary">{eventType}</Badge>
  }

  const formatDate = (dateString: string) => {
    try {
      return new Date(dateString).toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
      })
    } catch {
      return dateString
    }
  }

  const totalFatalities = filteredAndSortedEvents.reduce((sum, event) => sum + event.fatalities, 0)

  return (
    <Card className="w-full">
      {/* Header and Filters */}
      <div className="p-6 border-b">
        <div className="flex flex-col space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold">Conflict Events</h3>
            <div className="flex items-center space-x-2 text-sm text-gray-600">
              <span>{filteredAndSortedEvents.length} events</span>
              <span>•</span>
              <span>{totalFatalities.toLocaleString()} total fatalities</span>
            </div>
          </div>
          
          {/* Filters */}
          <div className="flex flex-wrap items-center gap-4">
            <div className="flex items-center space-x-2">
              <label className="text-sm font-medium">Event Type:</label>
              <select
                value={filterType}
                onChange={(e) => setFilterType(e.target.value)}
                className="px-3 py-1 border border-gray-300 rounded-md text-sm"
              >
                <option value="">All Types</option>
                {eventTypes.map(type => (
                  <option key={type} value={type}>{type}</option>
                ))}
              </select>
            </div>
            
            <div className="flex items-center space-x-2">
              <label className="text-sm font-medium">Country:</label>
              <select
                value={filterCountry}
                onChange={(e) => setFilterCountry(e.target.value)}
                className="px-3 py-1 border border-gray-300 rounded-md text-sm"
              >
                <option value="">All Countries</option>
                {countries.map(country => (
                  <option key={country} value={country}>{country}</option>
                ))}
              </select>
            </div>

            {(filterType || filterCountry) && (
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  setFilterType('')
                  setFilterCountry('')
                  setCurrentPage(1)
                }}
              >
                Clear Filters
              </Button>
            )}
          </div>
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead 
                className="cursor-pointer hover:bg-gray-50"
                onClick={() => handleSort('event_date')}
              >
                Date {sortField === 'event_date' && (sortDirection === 'asc' ? '↑' : '↓')}
              </TableHead>
              <TableHead 
                className="cursor-pointer hover:bg-gray-50"
                onClick={() => handleSort('event_type')}
              >
                Event Type {sortField === 'event_type' && (sortDirection === 'asc' ? '↑' : '↓')}
              </TableHead>
              <TableHead 
                className="cursor-pointer hover:bg-gray-50"
                onClick={() => handleSort('location')}
              >
                Location {sortField === 'location' && (sortDirection === 'asc' ? '↑' : '↓')}
              </TableHead>
              <TableHead 
                className="cursor-pointer hover:bg-gray-50"
                onClick={() => handleSort('country')}
              >
                Country {sortField === 'country' && (sortDirection === 'asc' ? '↑' : '↓')}
              </TableHead>
              <TableHead 
                className="cursor-pointer hover:bg-gray-50 text-right"
                onClick={() => handleSort('fatalities')}
              >
                Fatalities {sortField === 'fatalities' && (sortDirection === 'asc' ? '↑' : '↓')}
              </TableHead>
              <TableHead>Actors</TableHead>
              <TableHead>Coordinates</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {paginatedEvents.map((event) => (
              <TableRow 
                key={event.event_id}
                className={onEventSelect ? "cursor-pointer hover:bg-blue-50" : ""}
                onClick={() => onEventSelect?.(event)}
              >
                <TableCell className="font-medium">
                  {formatDate(event.event_date)}
                </TableCell>
                <TableCell>
                  {getEventTypeBadge(event.event_type)}
                </TableCell>
                <TableCell>
                  <div className="max-w-xs truncate" title={event.location}>
                    {event.location || 'Unknown'}
                  </div>
                </TableCell>
                <TableCell>
                  {event.country && (
                    <Badge variant="outline">{event.country}</Badge>
                  )}
                </TableCell>
                <TableCell className="text-right">
                  {getFatalityBadge(event.fatalities)}
                </TableCell>
                <TableCell>
                  <div className="space-y-1">
                    {event.actor1 && (
                      <div className="text-xs text-gray-600 truncate max-w-32" title={event.actor1}>
                        {event.actor1}
                      </div>
                    )}
                    {event.actor2 && (
                      <div className="text-xs text-gray-500 truncate max-w-32" title={event.actor2}>
                        vs {event.actor2}
                      </div>
                    )}
                  </div>
                </TableCell>
                <TableCell className="text-xs text-gray-500">
                  {event.latitude.toFixed(3)}, {event.longitude.toFixed(3)}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
          {totalPages > 1 && (
            <TableFooter>
              <TableRow>
                <TableCell colSpan={7}>
                  <div className="flex items-center justify-between">
                    <div className="text-sm text-gray-600">
                      Showing {((currentPage - 1) * itemsPerPage) + 1} to {Math.min(currentPage * itemsPerPage, filteredAndSortedEvents.length)} of {filteredAndSortedEvents.length} events
                    </div>
                    <div className="flex items-center space-x-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
                        disabled={currentPage === 1}
                      >
                        Previous
                      </Button>
                      <span className="text-sm">
                        Page {currentPage} of {totalPages}
                      </span>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => setCurrentPage(Math.min(totalPages, currentPage + 1))}
                        disabled={currentPage === totalPages}
                      >
                        Next
                      </Button>
                    </div>
                  </div>
                </TableCell>
              </TableRow>
            </TableFooter>
          )}
        </Table>
      </div>

      {paginatedEvents.length === 0 && (
        <div className="p-8 text-center text-gray-500">
          <p>No events found matching the current filters.</p>
          {(filterType || filterCountry) && (
            <Button
              variant="outline"
              size="sm"
              className="mt-2"
              onClick={() => {
                setFilterType('')
                setFilterCountry('')
              }}
            >
              Clear Filters
            </Button>
          )}
        </div>
      )}
    </Card>
  )
}