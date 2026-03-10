'use client'

import { motion } from 'framer-motion'
import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Sphere } from '@react-three/drei'

interface HotspotMapProps {
  events: any[]
  loading?: boolean
}

function Globe({ events }: { events: any[] }) {
  const meshRef = useRef<THREE.Mesh>(null)
  
  useFrame((state, delta) => {
    if (meshRef.current) {
      meshRef.current.rotation.y += delta * 0.1
    }
  })

  return (
    <group>
      <Sphere ref={meshRef} args={[2, 64, 64]}>
        <meshStandardMaterial
          color="#1e40af"
          transparent
          opacity={0.6}
          wireframe
        />
      </Sphere>
      {events?.slice(0, 20).map((event, index) => {
        const lat = (event.latitude || 0) * (Math.PI / 180)
        const lon = (event.longitude || 0) * (Math.PI / 180)
        const radius = 2.1
        
        const x = radius * Math.cos(lat) * Math.cos(lon)
        const y = radius * Math.sin(lat)
        const z = radius * Math.cos(lat) * Math.sin(lon)
        
        return (
          <motion.group
            key={event.event_id || index}
            position={[x, y, z]}
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ delay: index * 0.1 }}
          >
            <Sphere args={[0.05, 8, 8]}>
              <meshStandardMaterial
                color={event.fatalities > 10 ? "#ef4444" : "#f59e0b"}
                emissive={event.fatalities > 10 ? "#ef4444" : "#f59e0b"}
                emissiveIntensity={0.3}
              />
            </Sphere>
          </motion.group>
        )
      })}
    </group>
  )
}

export function HotspotMap({ events, loading }: HotspotMapProps) {
  if (loading) {
    return (
      <div className="h-[400px] flex items-center justify-center bg-muted/20 rounded-lg">
        <div className="animate-pulse text-muted-foreground">Loading 3D map...</div>
      </div>
    )
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="h-[400px] rounded-lg overflow-hidden bg-gradient-to-b from-slate-900 to-slate-800"
    >
      <Canvas camera={{ position: [0, 0, 5], fov: 60 }}>
        <ambientLight intensity={0.4} />
        <pointLight position={[10, 10, 10]} intensity={1} />
        <Globe events={events || []} />
        <OrbitControls enableZoom={true} enablePan={true} enableRotate={true} />
      </Canvas>
    </motion.div>
  )
}