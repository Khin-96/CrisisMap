'use client'

import { motion } from 'framer-motion'
import { AlertTriangle, Upload } from 'lucide-react'
import { Button } from './ui/button'
import Link from 'next/link'

export function Header() {
  return (
    <motion.header
      initial={{ y: -20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      className="border-b bg-card/50 backdrop-blur-sm sticky top-0 z-50"
    >
      <div className="container mx-auto px-4 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <motion.div
            animate={{ rotate: [0, 5, -5, 0] }}
            transition={{ duration: 2, repeat: Infinity, repeatDelay: 3 }}
          >
            <AlertTriangle className="h-6 w-6 text-primary" />
          </motion.div>
          <h1 className="text-2xl font-bold">CrisisMap</h1>
          <span className="text-sm text-muted-foreground">
            Conflict Early Warning System
          </span>
        </div>
        <nav className="flex items-center gap-4">
          <Link href="/">
            <Button variant="ghost">Dashboard</Button>
          </Link>
          <Link href="/analytics">
            <Button variant="ghost">Analytics</Button>
          </Link>
          <Link href="/upload">
            <Button variant="outline" className="gap-2">
              <Upload className="h-4 w-4" />
              Upload Data
            </Button>
          </Link>
        </nav>
      </div>
    </motion.header>
  )
}
