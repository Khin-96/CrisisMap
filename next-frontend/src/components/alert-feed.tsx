'use client'

import { motion, AnimatePresence } from 'framer-motion'
import { AlertCircle } from 'lucide-react'
import { formatDate } from '../lib/utils'

interface AlertFeedProps {
  events: any[]
  loading?: boolean
}

export function AlertFeed({ events, loading }: AlertFeedProps) {
  if (loading) {
    return (
      <div className="space-y-2">
        {[...Array(5)].map((_, i) => (
          <div key={i} className="h-16 bg-muted animate-pulse rounded" />
        ))}
      </div>
    )
  }

  const recentEvents = events?.slice(0, 10) || []

  return (
    <div className="h-[300px] overflow-y-auto space-y-2 pr-2">
      <AnimatePresence>
        {recentEvents.map((event, index) => (
          <motion.div
            key={event.event_id || index}
            initial={{ x: -20, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: 20, opacity: 0 }}
            transition={{ delay: index * 0.05 }}
            className="p-3 rounded-lg border bg-card hover:bg-accent/50 transition-colors cursor-pointer"
          >
            <div className="flex items-start gap-3">
              <motion.div
                animate={{ scale: [1, 1.2, 1] }}
                transition={{ duration: 2, repeat: Infinity, repeatDelay: 3 }}
              >
                <AlertCircle className="h-5 w-5 text-destructive mt-0.5" />
              </motion.div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium truncate">{event.event_type}</p>
                <p className="text-xs text-muted-foreground">{event.location}</p>
                <div className="flex items-center gap-2 mt-1">
                  <span className="text-xs text-muted-foreground">
                    {formatDate(event.event_date)}
                  </span>
                  {event.fatalities > 0 && (
                    <span className="text-xs px-2 py-0.5 rounded-full bg-destructive/10 text-destructive">
                      {event.fatalities} fatalities
                    </span>
                  )}
                </div>
              </div>
            </div>
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  )
}
