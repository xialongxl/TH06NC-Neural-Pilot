// Only included by the separate benchmark build; never runs in the console.
import { Profiler } from 'react'
import type { ReactNode } from 'react'
import { createRoot } from 'react-dom/client'

export function mountProfile(element: HTMLElement, children: ReactNode) {
  const output = document.createElement('output')
  output.id = 'render-profile'
  output.hidden = true
  document.body.append(output)
  const stats = { commits: 0, actualDurationMs: 0, samples: [] as number[] }
  createRoot(element).render(<Profiler id="console" onRender={(_id, _phase, duration) => {
    stats.commits++
    stats.actualDurationMs += duration
    stats.samples.push(duration)
    stats.samples = stats.samples.slice(-500)
    output.textContent = JSON.stringify(stats)
  }}>{children}</Profiler>)
}
