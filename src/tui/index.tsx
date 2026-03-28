import React from 'react'
import { render } from 'ink'

export function renderTUI(component: React.ReactElement) {
  const { waitUntilExit } = render(component)
  return waitUntilExit
}
