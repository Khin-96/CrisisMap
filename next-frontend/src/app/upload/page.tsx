'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { useDropzone } from 'react-dropzone'
import { Upload, FileText, CheckCircle, AlertCircle } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '../../components/ui/card'
import { Button } from '../../components/ui/button'
import { uploadCSV, getUploadStatus } from '../../lib/api'
import { Header } from '../../components/header'

interface UploadStatus {
  id: string
  status: 'uploading' | 'processing' | 'completed' | 'error'
  progress: number
  message: string
}

export default function UploadPage() {
  const [uploads, setUploads] = useState<UploadStatus[]>([])
  const [isUploading, setIsUploading] = useState(false)

  const onDrop = async (acceptedFiles: File[]) => {
    for (const file of acceptedFiles) {
      if (file.type === 'text/csv' || file.name.endsWith('.csv')) {
        await handleFileUpload(file)
      }
    }
  }

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.ms-excel': ['.xls'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
    },
    multiple: true,
  })

  const handleFileUpload = async (file: File) => {
    const uploadId = Math.random().toString(36).substr(2, 9)
    
    setUploads(prev => [...prev, {
      id: uploadId,
      status: 'uploading',
      progress: 0,
      message: `Uploading ${file.name}...`,
    }])

    setIsUploading(true)

    try {
      const result = await uploadCSV(file, {
        filename: file.name,
        size: file.size,
        uploadId,
      })

      setUploads(prev => prev.map(upload => 
        upload.id === uploadId 
          ? { ...upload, status: 'processing', progress: 50, message: 'Processing data...' }
          : upload
      ))

      // Simulate processing time
      setTimeout(() => {
        setUploads(prev => prev.map(upload => 
          upload.id === uploadId 
            ? { ...upload, status: 'completed', progress: 100, message: 'Upload completed successfully!' }
            : upload
        ))
      }, 2000)

    } catch (error) {
      setUploads(prev => prev.map(upload => 
        upload.id === uploadId 
          ? { ...upload, status: 'error', progress: 0, message: 'Upload failed. Please try again.' }
          : upload
      ))
    } finally {
      setIsUploading(false)
    }
  }

  return (
    <div className="min-h-screen bg-background">
      <Header />
      <div className="container mx-auto px-4 py-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="max-w-4xl mx-auto space-y-6"
        >
          <div>
            <h1 className="text-3xl font-bold mb-2">Upload Training Data</h1>
            <p className="text-muted-foreground">
              Upload CSV files containing conflict event data to train and improve our ML models.
            </p>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>File Upload</CardTitle>
            </CardHeader>
            <CardContent>
              <motion.div
                {...getRootProps()}
                className={`
                  border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors
                  ${isDragActive ? 'border-primary bg-primary/5' : 'border-muted-foreground/25'}
                  ${isUploading ? 'pointer-events-none opacity-50' : 'hover:border-primary hover:bg-primary/5'}
                `}
                whileHover={{ scale: 1.01 }}
                whileTap={{ scale: 0.99 }}
              >
                <input {...getInputProps()} />
                <motion.div
                  animate={{ y: isDragActive ? -5 : 0 }}
                  className="flex flex-col items-center gap-4"
                >
                  <Upload className="h-12 w-12 text-muted-foreground" />
                  <div>
                    <p className="text-lg font-medium">
                      {isDragActive ? 'Drop files here' : 'Drag & drop CSV files here'}
                    </p>
                    <p className="text-sm text-muted-foreground">
                      or click to select files (CSV, XLS, XLSX)
                    </p>
                  </div>
                </motion.div>
              </motion.div>
            </CardContent>
          </Card>

          {uploads.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>Upload Progress</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {uploads.map((upload) => (
                    <motion.div
                      key={upload.id}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      className="flex items-center gap-3 p-3 rounded-lg border"
                    >
                      <div className="flex-shrink-0">
                        {upload.status === 'completed' && (
                          <CheckCircle className="h-5 w-5 text-green-500" />
                        )}
                        {upload.status === 'error' && (
                          <AlertCircle className="h-5 w-5 text-red-500" />
                        )}
                        {(upload.status === 'uploading' || upload.status === 'processing') && (
                          <motion.div
                            animate={{ rotate: 360 }}
                            transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                          >
                            <FileText className="h-5 w-5 text-primary" />
                          </motion.div>
                        )}
                      </div>
                      <div className="flex-1">
                        <p className="text-sm font-medium">{upload.message}</p>
                        {upload.status !== 'completed' && upload.status !== 'error' && (
                          <div className="mt-1 w-full bg-muted rounded-full h-2">
                            <motion.div
                              className="bg-primary h-2 rounded-full"
                              initial={{ width: 0 }}
                              animate={{ width: `${upload.progress}%` }}
                              transition={{ duration: 0.5 }}
                            />
                          </div>
                        )}
                      </div>
                    </motion.div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          <Card>
            <CardHeader>
              <CardTitle>Data Requirements</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <h4 className="font-medium mb-2">Required Columns</h4>
                  <ul className="text-sm text-muted-foreground space-y-1">
                    <li>• event_date (YYYY-MM-DD)</li>
                    <li>• latitude (decimal degrees)</li>
                    <li>• longitude (decimal degrees)</li>
                    <li>• event_type</li>
                    <li>• fatalities (numeric)</li>
                  </ul>
                </div>
                <div>
                  <h4 className="font-medium mb-2">Optional Columns</h4>
                  <ul className="text-sm text-muted-foreground space-y-1">
                    <li>• location (place name)</li>
                    <li>• actor1, actor2 (conflict actors)</li>
                    <li>• country</li>
                    <li>• notes (additional context)</li>
                  </ul>
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </div>
    </div>
  )
}