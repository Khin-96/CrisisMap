'use client'

import { Bar, BarChart, XAxis, YAxis, CartesianGrid, ResponsiveContainer } from "recharts"
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"

interface BarChartProps {
  data: Array<{
    name: string
    value: number
    [key: string]: any
  }>
  dataKey?: string
  nameKey?: string
  height?: number
  color?: string
}

export function MyBarChart({ 
  data, 
  dataKey = "value", 
  nameKey = "name", 
  height = 300,
  color = "hsl(var(--primary))"
}: BarChartProps) {
  return (
    <ChartContainer className={`h-[${height}px]`}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
          <XAxis 
            dataKey={nameKey} 
            className="text-xs"
            tick={{ fontSize: 12 }}
          />
          <YAxis className="text-xs" tick={{ fontSize: 12 }} />
          <ChartTooltip content={<ChartTooltipContent />} />
          <Bar 
            dataKey={dataKey} 
            fill={color}
            radius={[4, 4, 0, 0]}
          />
        </BarChart>
      </ResponsiveContainer>
    </ChartContainer>
  )
}