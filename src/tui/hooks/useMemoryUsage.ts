import { useState, useEffect } from 'react'

export function useMemoryUsage(intervalMs = 500) {
  const [heapUsed, setHeapUsed] = useState(() => process.memoryUsage().heapUsed)

  useEffect(() => {
    const id = setInterval(() => {
      setHeapUsed(process.memoryUsage().heapUsed)
    }, intervalMs)
    return () => clearInterval(id)
  }, [])

  return heapUsed
}
