import React from 'react'
import { Box, Text } from 'ink'

interface ProgressBarProps {
  value: number   // 0–1
  width?: number
  label?: string
  color?: string
}

export function ProgressBar({ value, width = 30, label, color = 'cyan' }: ProgressBarProps) {
  const filled = Math.round(value * width)
  const empty = width - filled
  const bar = '█'.repeat(filled) + '░'.repeat(empty)
  const pct = Math.floor(value * 100)

  return (
    <Box gap={1}>
      <Text color={color}>{bar}</Text>
      <Text color={color}>{String(pct).padStart(3)}%</Text>
      {label ? <Text color="gray">{label}</Text> : null}
    </Box>
  )
}
