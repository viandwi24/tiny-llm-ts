import React, { useEffect, useState } from 'react'
import { Box, Text, useApp } from 'ink'

export interface WikipediaScraperProps {
  topics: string[]
  language: string
  outputDir: string
  searchLimit: number
  onRun: (callbacks: WikipediaScraperCallbacks) => Promise<void>
}

export interface WikipediaScraperCallbacks {
  onTopicStart: (topic: string, index: number, total: number) => void
  onSearchDone: (topic: string, count: number) => void
  onFileWritten: (title: string, file: string, topic: string) => void
  onFileSkipped: (title: string, reason: string) => void
  onTopicDone: (topic: string) => void
  onDone: (total: number) => void
}

interface TopicStatus {
  topic: string
  state: 'pending' | 'searching' | 'fetching' | 'done'
  found: number
  written: number
}

export function WikipediaScraper({ topics, language, outputDir, searchLimit, onRun }: WikipediaScraperProps) {
  const { exit } = useApp()
  const [statuses, setStatuses] = useState<TopicStatus[]>(
    topics.map(t => ({ topic: t, state: 'pending', found: 0, written: 0 }))
  )
  const [currentTitle, setCurrentTitle] = useState<string>('')
  const [totalWritten, setTotalWritten] = useState(0)
  const [done, setDone] = useState(false)
  const [log, setLog] = useState<string[]>([])

  const appendLog = (msg: string) => setLog(prev => [...prev.slice(-6), msg])

  useEffect(() => {
    onRun({
      onTopicStart(topic, index) {
        setStatuses(prev => prev.map(s =>
          s.topic === topic ? { ...s, state: 'searching' } : s
        ))
        appendLog(`Searching: "${topic}"`)
      },
      onSearchDone(topic, count) {
        setStatuses(prev => prev.map(s =>
          s.topic === topic ? { ...s, state: count > 0 ? 'fetching' : 'done', found: count } : s
        ))
      },
      onFileWritten(title, _file, topic) {
        setCurrentTitle(title)
        setTotalWritten(n => n + 1)
        setStatuses(prev => prev.map(s =>
          s.topic === topic ? { ...s, written: s.written + 1 } : s
        ))
      },
      onFileSkipped(title, reason) {
        appendLog(`skip: ${title} (${reason})`)
      },
      onTopicDone(topic) {
        setStatuses(prev => prev.map(s =>
          s.topic === topic ? { ...s, state: 'done' } : s
        ))
      },
      onDone(total) {
        setDone(true)
        setTotalWritten(total)
        appendLog(`Done! ${total} files written.`)
        setTimeout(() => exit(), 500)
      },
    })
  }, [])

  const stateIcon = (state: TopicStatus['state']) => {
    if (state === 'pending') return '○'
    if (state === 'searching') return '⟳'
    if (state === 'fetching') return '↓'
    return '✓'
  }

  const stateColor = (state: TopicStatus['state']): string => {
    if (state === 'pending') return 'gray'
    if (state === 'searching') return 'yellow'
    if (state === 'fetching') return 'cyan'
    return 'green'
  }

  return (
    <Box flexDirection="column" padding={1} gap={1}>
      {/* Header */}
      <Box gap={2}>
        <Text bold color="cyan">Wikipedia Scraper</Text>
        <Text color="gray">lang={language}  limit={searchLimit}/topic  →  {outputDir}</Text>
      </Box>

      {/* Topics */}
      <Box flexDirection="column">
        {statuses.map(s => (
          <Box key={s.topic} gap={2}>
            <Text color={stateColor(s.state)}>{stateIcon(s.state)}</Text>
            <Text color={stateColor(s.state)}>{s.topic.padEnd(24)}</Text>
            {s.state === 'fetching' && (
              <Text color="gray">{s.written}/{s.found} articles</Text>
            )}
            {s.state === 'done' && (
              <Text color="green">{s.written} articles</Text>
            )}
          </Box>
        ))}
      </Box>

      {/* Current file */}
      {!done && currentTitle ? (
        <Text color="gray">fetching: <Text color="white">{currentTitle}</Text></Text>
      ) : null}

      {/* Log */}
      <Box flexDirection="column">
        {log.map((l, i) => (
          <Text key={i} color="gray" dimColor>{l}</Text>
        ))}
      </Box>

      {/* Footer */}
      {done && (
        <Text color="green" bold>✓ {totalWritten} files written to {outputDir}</Text>
      )}
    </Box>
  )
}
