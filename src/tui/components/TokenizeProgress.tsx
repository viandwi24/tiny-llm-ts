import React, { useEffect, useState } from 'react'
import { Box, Text, useApp } from 'ink'
import { ProgressBar } from './ProgressBar'

export interface TokenizeCallbacks {
  onPhase: (phase: string) => void
  onBpeTick: (iteration: number, total: number, elapsed: number, eta: number, vocabSize: number) => void
  onDone: (vocabSize: number, merges: number, elapsed: number) => void
}

export interface TokenizeProgressProps {
  vocabSize: number
  onRun: (cb: TokenizeCallbacks) => Promise<void>
}

export function TokenizeProgress({ vocabSize, onRun }: TokenizeProgressProps) {
  const { exit } = useApp()
  const [phase, setPhase] = useState('Initializing...')
  const [iteration, setIteration] = useState(0)
  const [total, setTotal] = useState(0)
  const [elapsed, setElapsed] = useState(0)
  const [eta, setEta] = useState(0)
  const [currentVocabSize, setCurrentVocabSize] = useState(0)
  const [done, setDone] = useState<{ vocabSize: number; merges: number; elapsed: number } | null>(null)

  useEffect(() => {
    onRun({
      onPhase(p) {
        setPhase(p)
      },
      onBpeTick(iter, tot, elap, e, vs) {
        setIteration(iter)
        setTotal(tot)
        setElapsed(elap)
        setEta(e)
        setCurrentVocabSize(vs)
      },
      onDone(vs, merges, elap) {
        setDone({ vocabSize: vs, merges, elapsed: elap })
        setTimeout(() => exit(), 300)
      },
    })
  }, [])

  const progress = total > 0 ? iteration / total : 0

  return (
    <Box flexDirection="column" padding={1} gap={1}>
      <Text bold color="cyan">Tokenizer — BPE Training</Text>
      <Text color="gray">target vocab size: {vocabSize}</Text>

      <Box gap={1}>
        <Text color="gray">phase:</Text>
        <Text color="yellow">{phase}</Text>
      </Box>

      {total > 0 && (
        <Box flexDirection="column" gap={0}>
          <ProgressBar
            value={progress}
            label={`${iteration}/${total} merges`}
          />
          <Box gap={2} marginTop={0}>
            <Text color="gray">vocab: <Text color="white">{currentVocabSize}</Text></Text>
            <Text color="gray">elapsed: <Text color="white">{elapsed.toFixed(1)}s</Text></Text>
            <Text color="gray">eta: <Text color={eta < 10 ? 'green' : 'yellow'}>{eta.toFixed(1)}s</Text></Text>
          </Box>
        </Box>
      )}

      {done && (
        <Box flexDirection="column">
          <Text color="green" bold>✓ Done in {done.elapsed.toFixed(2)}s</Text>
          <Text color="gray">vocab size: <Text color="white">{done.vocabSize}</Text>  merges: <Text color="white">{done.merges}</Text></Text>
        </Box>
      )}
    </Box>
  )
}
