'use client'

import { motion } from 'framer-motion'
import { Card, CardContent } from './ui/card'
import { TrendingUp, TrendingDown, Minus } from 'lucide-react'
import { formatNumber } from '../lib/utils'

interface MetricCardProps {
  title: string
  value: number | string
  trend?: 'increasing' | 'decreasing' | 'stable'
  loading?: boolean
  isText?: boolean
}

export function MetricCard({ title, value, trend, loading, isText }: MetricCardProps) {
  const getTrendIcon = () => {
    if (!trend) return null
    switch (trend) {
      case 'increasing':
        return <TrendingUp className="h-4 w-4 text-red-500" />
      case 'decreasing':
        return <TrendingDown className="h-4 w-4 text-green-500" />
      case 'stable':
        return <Minus className="h-4 w-4 text-yellow-500" />
    }
  }

  const getTrendColor = () => {
    if (!trend) return ''
    switch (trend) {
      case 'increasing':
        return 'text-red-500'
      case 'decreasing':
        return 'text-green-500'
      case 'stable':
        return 'text-yellow-500'
    }
  }

  return (
    <motion.div
      whileHover={{ scale: 1.02 }}
      transition={{ type: 'spring', stiffness: 300 }}
    >
      <Card className="relative overflow-hidden">
        <motion.div
          className="absolute inset-0 bg-gradient-to-br from-primary/5 to-transparent"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5 }}
        />
        <CardContent className="p-6 relative">
          <div className="flex items-center justify-between mb-2">
            <p className="text-sm text-muted-foreground">{title}</p>
            {getTrendIcon()}
          </div>
          {loading ? (
            <div className="h-8 w-24 bg-muted animate-pulse rounded" />
          ) : (
            <motion.p
              className={`text-3xl font-bold ${getTrendColor()}`}
              initial={{ scale: 0.5, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ type: 'spring', stiffness: 200 }}
            >
              {isText ? value : formatNumber(Number(value))}
            </motion.p>
          )}
        </CardContent>
      </Card>
    </motion.div>
  )
}
