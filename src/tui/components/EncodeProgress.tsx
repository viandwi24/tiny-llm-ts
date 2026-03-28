import React, { useEffect, useState } from 'react'
import { Box, Text, useApp } from 'ink'
import { ProgressBar } from './ProgressBar'
import { useMemoryUsage } from '../hooks/useMemoryUsage'
import { formatBytes } from '../utils'

export interface EncodeCallbacks {
  onFile: (file: string, index: number, total: number, tokens: number) => void
  onDone: (totalTokens: number, fileSizeBytes: number, elapsed: number) => void
}

export interface EncodeProgressProps {
  onRun: (cb: EncodeCallbacks) => Promise<void>
}

export function EncodeProgress({ onRun }: EncodeProgressProps) {
  const { exit } = useApp()
  const [currentFile, setCurrentFile] = useState('')
  const [index, setIndex] = useState(0)
  const [total, setTotal] = useState(0)
  const [totalTokens, setTotalTokens] = useState(0)
  const [done, setDone] = useState<{ tokens: number; bytes: number; elapsed: number } | null>(null)
  const heapUsed = useMemoryUsage()

  useEffect(() => {
    onRun({
      onFile(file, idx, tot, tokens) {
        setCurrentFile(file)
        setIndex(idx)
        setTotal(tot)
        setTotalTokens(n => n + tokens)
      },
      onDone(tokens, bytes, elapsed) {
        setDone({ tokens, bytes, elapsed })
        setTimeout(() => exit(), 300)
      },
    })
  }, [])

  const progress = total > 0 ? index / total : 0

  return (
    <Box flexDirection="column" padding={1} gap={1}>
      <Box justifyContent="space-between">
        <Text bold color="cyan">Encoder — BPE Encode</Text>
        <Text color="gray">mem: <Text color={heapUsed > 512 * 1024 * 1024 ? 'red' : 'green'}>{formatBytes(heapUsed)}</Text></Text>
      </Box>

      {total > 0 && (
        <ProgressBar
          value={progress}
          label={`${index}/${total} files`}
          color="magenta"
        />
      )}

      {currentFile && !done && (
        <Text color="gray">encoding: <Text color="white">{currentFile}</Text></Text>
      )}

      <Text color="gray">tokens so far: <Text color="white">{totalTokens.toLocaleString()}</Text></Text>

      {done && (
        <Box flexDirection="column">
          <Text color="green" bold>✓ Done in {done.elapsed.toFixed(2)}s</Text>
          <Text color="gray">
            total tokens: <Text color="white">{done.tokens.toLocaleString()}</Text>
            {'  '}output size: <Text color="white">{formatBytes(done.bytes)}</Text>
          </Text>
        </Box>
      )}
    </Box>
  )
}
