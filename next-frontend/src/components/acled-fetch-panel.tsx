'use client'

import { useState, useEffect } from 'react'
import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'

interface FetchJob {
  fetch_id: string
  status: 'queued' | 'fetching' | 'completed' | 'error'
  progress: number
  records_fetched: number
  records_stored: number
  message: string
  started_at: string
  completed_at?: string
  filters: Record<string, any>
}

interface ACLEDFetchPanelProps {
  onFetchComplete?: () => void
}

const ACLED_EVENT_TYPES = [
  'Battles',
  'Explosions/Remote violence',
  'Violence against civilians',
  'Protests',
  'Riots',
  'Strategic developments',
]

export default function ACLEDFetchPanel({ onFetchComplete }: ACLEDFetchPanelProps) {
  const [loading,    setLoading]    = useState(false)
  const [jobs,       setJobs]       = useState<FetchJob[]>([])
  const [activeJob,  setActiveJob]  = useState<string | null>(null)
  const [tokenStatus, setTokenStatus] = useState<any>(null)

  // Form state
  const [country,    setCountry]    = useState('')
  const [countries,  setCountries]  = useState('')  // comma-separated
  const [startDate,  setStartDate]  = useState('')
  const [endDate,    setEndDate]    = useState('')
  const [eventType,  setEventType]  = useState('')
  const [year,       setYear]       = useState('')
  const [maxRecords, setMaxRecords] = useState('5000')
  const [fetchMode,  setFetchMode]  = useState<'events' | 'cast'>('events')

  const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

  // Check token status on mount
  useEffect(() => {
    fetch(`${API}/api/acled/token-status`)
      .then(r => r.json())
      .then(setTokenStatus)
      .catch(() => setTokenStatus(null))

    // Load existing jobs
    fetch(`${API}/api/acled/fetch`)
      .then(r => r.json())
      .then(setJobs)
      .catch(() => {})
  }, [])

  // Poll active job
  useEffect(() => {
    if (!activeJob) return
    const interval = setInterval(async () => {
      try {
        const r    = await fetch(`${API}/api/acled/fetch/${activeJob}`)
        const data = await r.json()
        setJobs(prev => {
          const idx = prev.findIndex(j => j.fetch_id === activeJob)
          if (idx >= 0) {
            const next = [...prev]
            next[idx]  = data
            return next
          }
          return [data, ...prev]
        })
        if (data.status === 'completed' || data.status === 'error') {
          setActiveJob(null)
          setLoading(false)
          if (data.status === 'completed' && onFetchComplete) onFetchComplete()
        }
      } catch {}
    }, 1500)
    return () => clearInterval(interval)
  }, [activeJob])

  const submitFetch = async () => {
    setLoading(true)
    const endpoint = fetchMode === 'cast' ? '/api/cast/fetch' : '/api/acled/fetch'
    const body: Record<string, any> = { max_records: parseInt(maxRecords) }

    if (country.trim())                  body.country    = country.trim()
    if (countries.trim()) {
      body.countries = countries.split(',').map(c => c.trim()).filter(Boolean)
      delete body.country
    }
    if (startDate)                       body.start_date = startDate
    if (endDate)                         body.end_date   = endDate
    if (eventType)                       body.event_type = eventType
    if (year)                            body.year       = parseInt(year)

    try {
      const r    = await fetch(`${API}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      const data = await r.json()
      if (data.fetch_id) {
        setActiveJob(data.fetch_id)
        setJobs(prev => [{ fetch_id: data.fetch_id, status: 'queued', progress: 0,
          records_fetched: 0, records_stored: 0, message: 'Queued',
          started_at: new Date().toISOString(), filters: body }, ...prev])
      } else {
        setLoading(false)
        alert('Failed to start fetch: ' + JSON.stringify(data))
      }
    } catch (err) {
      setLoading(false)
      alert('Request failed: ' + err)
    }
  }

  const refreshToken = async () => {
    const r = await fetch(`${API}/api/acled/authenticate`, { method: 'POST' })
    const d = await r.json()
    alert(d.message || JSON.stringify(d))
    const ts = await fetch(`${API}/api/acled/token-status`).then(r => r.json())
    setTokenStatus(ts)
  }

  const statusColor = (status: string) => {
    if (status === 'completed') return 'default'
    if (status === 'error')     return 'destructive'
    if (status === 'fetching')  return 'secondary'
    return 'outline'
  }

  return (
    <div className="space-y-6">
      {/* Token status bar */}
      <Card className="p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className={`w-2.5 h-2.5 rounded-full ${tokenStatus?.authenticated ? 'bg-green-500' : 'bg-red-500'}`} />
            <div>
              <div className="font-medium text-sm">ACLED API Connection</div>
              <div className="text-xs text-muted-foreground">
                {tokenStatus?.email || 'Not authenticated'} &middot; OAuth Bearer token
              </div>
            </div>
          </div>
          <button
            onClick={refreshToken}
            className="text-xs px-3 py-1.5 border rounded hover:bg-slate-50 transition-colors"
          >
            Refresh Token
          </button>
        </div>
      </Card>

      {/* Fetch form */}
      <Card className="p-6">
        <div className="mb-4">
          <h3 className="font-semibold text-base">Fetch from ACLED API</h3>
          <p className="text-sm text-muted-foreground mt-0.5">
            Pull live conflict data directly from acleddata.com using your OAuth credentials
          </p>
        </div>

        {/* Mode toggle */}
        <div className="flex gap-2 mb-5">
          {(['events', 'cast'] as const).map(mode => (
            <button
              key={mode}
              onClick={() => setFetchMode(mode)}
              className={`px-4 py-1.5 rounded text-sm font-medium transition-colors border ${
                fetchMode === mode
                  ? 'bg-red-600 text-white border-red-600'
                  : 'bg-white text-slate-600 border-slate-200 hover:bg-slate-50'
              }`}
            >
              {mode === 'events' ? 'Conflict Events' : 'CAST Forecasts'}
            </button>
          ))}
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {fetchMode === 'events' && (
            <>
              <div>
                <label className="block text-xs font-medium text-slate-600 mb-1">Single Country</label>
                <input
                  value={country}
                  onChange={e => setCountry(e.target.value)}
                  placeholder="e.g. Kenya"
                  className="w-full px-3 py-2 text-sm border rounded focus:outline-none focus:ring-2 focus:ring-red-500"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-slate-600 mb-1">Multiple Countries (comma-sep)</label>
                <input
                  value={countries}
                  onChange={e => setCountries(e.target.value)}
                  placeholder="Kenya, Somalia, Ethiopia"
                  className="w-full px-3 py-2 text-sm border rounded focus:outline-none focus:ring-2 focus:ring-red-500"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-slate-600 mb-1">Event Type</label>
                <select
                  value={eventType}
                  onChange={e => setEventType(e.target.value)}
                  className="w-full px-3 py-2 text-sm border rounded focus:outline-none focus:ring-2 focus:ring-red-500"
                >
                  <option value="">All types</option>
                  {ACLED_EVENT_TYPES.map(t => <option key={t} value={t}>{t}</option>)}
                </select>
              </div>
              <div>
                <label className="block text-xs font-medium text-slate-600 mb-1">Start Date</label>
                <input
                  type="date"
                  value={startDate}
                  onChange={e => setStartDate(e.target.value)}
                  className="w-full px-3 py-2 text-sm border rounded focus:outline-none focus:ring-2 focus:ring-red-500"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-slate-600 mb-1">End Date</label>
                <input
                  type="date"
                  value={endDate}
                  onChange={e => setEndDate(e.target.value)}
                  className="w-full px-3 py-2 text-sm border rounded focus:outline-none focus:ring-2 focus:ring-red-500"
                />
              </div>
            </>
          )}

          {fetchMode === 'cast' && (
            <div>
              <label className="block text-xs font-medium text-slate-600 mb-1">Country (optional)</label>
              <input
                value={country}
                onChange={e => setCountry(e.target.value)}
                placeholder="Leave blank for all countries"
                className="w-full px-3 py-2 text-sm border rounded focus:outline-none focus:ring-2 focus:ring-red-500"
              />
            </div>
          )}

          <div>
            <label className="block text-xs font-medium text-slate-600 mb-1">Year</label>
            <input
              type="number"
              value={year}
              onChange={e => setYear(e.target.value)}
              placeholder="e.g. 2025"
              className="w-full px-3 py-2 text-sm border rounded focus:outline-none focus:ring-2 focus:ring-red-500"
            />
          </div>

          <div>
            <label className="block text-xs font-medium text-slate-600 mb-1">Max Records</label>
            <select
              value={maxRecords}
              onChange={e => setMaxRecords(e.target.value)}
              className="w-full px-3 py-2 text-sm border rounded focus:outline-none focus:ring-2 focus:ring-red-500"
            >
              <option value="1000">1,000</option>
              <option value="5000">5,000</option>
              <option value="10000">10,000</option>
              <option value="25000">25,000</option>
              <option value="50000">50,000</option>
            </select>
          </div>
        </div>

        <div className="mt-5">
          <button
            onClick={submitFetch}
            disabled={loading}
            className="px-6 py-2.5 bg-red-600 text-white text-sm font-semibold rounded hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
          >
            {loading && <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />}
            {loading ? 'Fetching...' : `Fetch ${fetchMode === 'cast' ? 'CAST Forecasts' : 'Conflict Events'}`}
          </button>
        </div>
      </Card>

      {/* Jobs history */}
      {jobs.length > 0 && (
        <Card className="p-6">
          <h3 className="font-semibold text-base mb-4">Fetch History</h3>
          <div className="space-y-3">
            {jobs.slice(0, 10).map(job => (
              <div key={job.fetch_id} className="border rounded-lg p-4 space-y-2">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Badge variant={statusColor(job.status) as any}>
                      {job.status.toUpperCase()}
                    </Badge>
                    <span className="text-sm text-muted-foreground font-mono text-xs">
                      {job.fetch_id.slice(0, 16)}...
                    </span>
                  </div>
                  <span className="text-xs text-slate-400">
                    {new Date(job.started_at).toLocaleTimeString()}
                  </span>
                </div>

                {/* Progress bar */}
                {(job.status === 'fetching' || job.status === 'queued') && (
                  <div className="w-full bg-slate-100 rounded-full h-1.5">
                    <div
                      className="bg-red-600 h-1.5 rounded-full transition-all duration-500"
                      style={{ width: `${job.progress}%` }}
                    />
                  </div>
                )}

                <div className="text-sm text-slate-600">{job.message}</div>

                {job.status === 'completed' && (
                  <div className="flex gap-4 text-xs text-slate-500">
                    <span>Fetched: <strong>{job.records_fetched?.toLocaleString()}</strong></span>
                    <span>Stored: <strong>{job.records_stored?.toLocaleString()}</strong></span>
                  </div>
                )}

                {/* Applied filters */}
                <div className="flex flex-wrap gap-1 mt-1">
                  {Object.entries(job.filters || {}).filter(([k]) => !['_format'].includes(k)).map(([k, v]) => (
                    <span key={k} className="text-xs bg-slate-100 text-slate-600 px-2 py-0.5 rounded">
                      {k}: {String(v)}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  )
}
