'use client'

import { useState, useRef, useEffect } from 'react'
import { Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'
import { Send, Bot, User, Lightbulb, AlertTriangle, TrendingUp } from 'lucide-react'

interface Message {
  id: string
  type: 'user' | 'assistant'
  content: string
  timestamp: Date
  context?: any
  isTyping?: boolean
}

interface AIAssistantProps {
  events?: any[]
  currentContext?: any
}

export default function AIAssistant({ events = [], currentContext }: AIAssistantProps) {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      type: 'assistant',
      content: 'Hello! I\'m your AI conflict analyst. I can help you understand patterns, assess risks, and provide strategic insights based on the current data. What would you like to know?',
      timestamp: new Date()
    }
  ])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const quickActions = [
    {
      icon: <TrendingUp className="h-4 w-4" />,
      label: 'Analyze Trends',
      query: 'What are the current conflict trends and patterns in the data?'
    },
    {
      icon: <AlertTriangle className="h-4 w-4" />,
      label: 'Risk Assessment',
      query: 'Provide a comprehensive risk assessment for the current situation'
    },
    {
      icon: <Lightbulb className="h-4 w-4" />,
      label: 'Recommendations',
      query: 'What are your strategic recommendations for conflict prevention and response?'
    }
  ]

  const formatAIResponse = (text: string): string => {
    // Convert markdown-style formatting to HTML
    let formatted = text
      // Convert **text** to <strong>text</strong>
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      // Convert *text* to <em>text</em>
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      // Convert bullet points (handle different bullet characters)
      .replace(/^â¢ /gm, '• ')
      .replace(/^\* /gm, '• ')
      .replace(/^- /gm, '• ')
      // Convert numbered lists
      .replace(/^(\d+)\. /gm, '$1. ')
      // Add proper line breaks and spacing
      .replace(/\n\n/g, '<br><br>')
      .replace(/\n/g, '<br>')
      // Add spacing after section headers
      .replace(/(<strong>[^<]*<\/strong>)(<br>)/g, '$1<br><br>')
    
    return formatted
  }

  const TypewriterText = ({ text, onComplete }: { text: string, onComplete?: () => void }) => {
    const [displayText, setDisplayText] = useState('')
    const [currentIndex, setCurrentIndex] = useState(0)
    const [showCursor, setShowCursor] = useState(true)

    useEffect(() => {
      if (currentIndex < text.length) {
        const timer = setTimeout(() => {
          setDisplayText(prev => prev + text[currentIndex])
          setCurrentIndex(prev => prev + 1)
        }, 15) // Faster typing speed

        return () => clearTimeout(timer)
      } else {
        // Hide cursor after completion
        setTimeout(() => setShowCursor(false), 1000)
        if (onComplete) {
          onComplete()
        }
      }
    }, [currentIndex, text, onComplete])

    return (
      <div className="ai-response">
        <span 
          dangerouslySetInnerHTML={{ __html: formatAIResponse(displayText) }}
        />
        {showCursor && currentIndex < text.length && (
          <span className="typewriter-cursor"></span>
        )}
      </div>
    )
  }

  const sendMessage = async (messageText: string) => {
    if (!messageText.trim() || isLoading) return

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: messageText,
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    setInput('')
    setIsLoading(true)

    // Add typing indicator
    const typingMessage: Message = {
      id: (Date.now() + 1).toString(),
      type: 'assistant',
      content: '',
      timestamp: new Date(),
      isTyping: true
    }
    setMessages(prev => [...prev, typingMessage])

    try {
      // Prepare context data
      const contextData = {
        totalEvents: events.length,
        recentEvents: events.slice(0, 10),
        countries: [...new Set(events.map(e => e.country).filter(Boolean))],
        eventTypes: [...new Set(events.map(e => e.event_type).filter(Boolean))],
        totalFatalities: events.reduce((sum, e) => sum + (e.fatalities || 0), 0),
        dateRange: {
          start: events.length > 0 ? Math.min(...events.map(e => new Date(e.event_date).getTime())) : null,
          end: events.length > 0 ? Math.max(...events.map(e => new Date(e.event_date).getTime())) : null
        },
        ...currentContext
      }

      const response = await fetch('http://localhost:8000/api/ai/insights', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: messageText,
          context: contextData
        })
      })

      if (!response.ok) {
        throw new Error('Failed to get AI response')
      }

      const data = await response.json()

      // Remove typing indicator
      setMessages(prev => prev.filter(msg => !msg.isTyping))

      const assistantMessage: Message = {
        id: (Date.now() + 2).toString(),
        type: 'assistant',
        content: data.response,
        timestamp: new Date(),
        context: data
      }

      setMessages(prev => [...prev, assistantMessage])

    } catch (error) {
      console.error('AI Assistant error:', error)
      
      // Remove typing indicator
      setMessages(prev => prev.filter(msg => !msg.isTyping))
      
      const errorMessage: Message = {
        id: (Date.now() + 2).toString(),
        type: 'assistant',
        content: 'I apologize, but I\'m having trouble processing your request right now. Please try again later or check if the AI service is available.',
        timestamp: new Date()
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const handleQuickAction = (query: string) => {
    sendMessage(query)
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    sendMessage(input)
  }

  return (
    <Card className="flex flex-col h-[600px]">
      {/* Header */}
      <div className="flex items-center gap-3 p-4 border-b">
        <div className="flex h-10 w-10 items-center justify-center rounded-full bg-primary text-primary-foreground">
          <Bot className="h-5 w-5" />
        </div>
        <div>
          <h3 className="font-semibold">Dr. Sarah Chen - AI Conflict Analyst</h3>
          <p className="text-sm text-muted-foreground">Powered by Groq AI • Expert Analysis</p>
        </div>
        <Badge variant="secondary" className="ml-auto bg-green-100 text-green-800">
          Online
        </Badge>
      </div>

      {/* Quick Actions */}
      <div className="p-4 border-b bg-gray-50">
        <p className="text-sm text-muted-foreground mb-2">Quick Analysis:</p>
        <div className="flex flex-wrap gap-2">
          {quickActions.map((action, index) => (
            <Button
              key={index}
              variant="outline"
              size="sm"
              onClick={() => handleQuickAction(action.query)}
              disabled={isLoading}
              className="text-xs hover:bg-primary hover:text-primary-foreground"
            >
              {action.icon}
              <span className="ml-1">{action.label}</span>
            </Button>
          ))}
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex gap-3 ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            {message.type === 'assistant' && (
              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary text-primary-foreground flex-shrink-0">
                <Bot className="h-4 w-4" />
              </div>
            )}
            
            <div
              className={`max-w-[85%] rounded-lg p-4 ${
                message.type === 'user'
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-gray-100 border'
              }`}
            >
              {message.isTyping ? (
                <div className="flex items-center gap-2">
                  <div className="flex space-x-1">
                    <div className="w-2 h-2 bg-primary rounded-full animate-bounce"></div>
                    <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                    <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                  </div>
                  <span className="text-sm text-muted-foreground">Analyzing data...</span>
                </div>
              ) : message.type === 'assistant' && message.id !== '1' ? (
                <div className="ai-response">
                  <TypewriterText text={message.content} />
                </div>
              ) : (
                <div 
                  className="text-sm whitespace-pre-wrap ai-response"
                  dangerouslySetInnerHTML={{ __html: formatAIResponse(message.content) }}
                />
              )}
              
              <p className={`text-xs mt-2 ${message.type === 'user' ? 'opacity-70' : 'text-gray-500'}`}>
                {message.timestamp.toLocaleTimeString()}
                {message.context?.data_quality && (
                  <span className="ml-2">• {message.context.data_quality} quality data</span>
                )}
              </p>
            </div>

            {message.type === 'user' && (
              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-secondary flex-shrink-0">
                <User className="h-4 w-4" />
              </div>
            )}
          </div>
        ))}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <form onSubmit={handleSubmit} className="p-4 border-t bg-white">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about conflict patterns, risks, or get strategic recommendations..."
            className="flex-1 px-3 py-2 border border-gray-300 rounded-md text-sm focus:ring-2 focus:ring-primary focus:border-primary"
            disabled={isLoading}
          />
          <Button type="submit" disabled={isLoading || !input.trim()}>
            <Send className="h-4 w-4" />
          </Button>
        </div>
        <p className="text-xs text-gray-500 mt-1">
          AI has access to {events.length.toLocaleString()} conflict events for analysis
        </p>
      </form>
    </Card>
  )
}