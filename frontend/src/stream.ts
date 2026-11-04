import { message } from './i18n'
import type { Snapshot } from './types'

type Path = string[]
export type HistoryPoint = Snapshot['gpu_history'][number]
export type StreamFrame =
  | { protocol: 2; kind: 'snapshot'; revision: number; state: Snapshot }
  | { protocol: 2; kind: 'heartbeat'; revision: number }
  | { protocol: 2; kind: 'delta'; base: number; revision: number; set: [Path, unknown][]; remove: Path[];
      history?: { mode: 'reset' | 'append'; after?: number | null; points: HistoryPoint[] } }
export interface StreamState { revision: number; state: Snapshot | null }

// Reconnects and cold replacements retain existing object/array references
// when values are equal, so memoized views do not redraw unchanged content.
export function reconcile<T>(previous: T, next: T): T {
  if (Object.is(previous, next)) return previous
  if (Array.isArray(previous) && Array.isArray(next)) {
    const result = next.map((item, i) => reconcile(previous[i], item))
    return result.length === previous.length && result.every((item, i) => item === previous[i]) ? previous : result as T
  }
  if (previous && next && typeof previous === 'object' && typeof next === 'object' && !Array.isArray(previous) && !Array.isArray(next)) {
    const old = previous as Record<string, unknown>, incoming = next as Record<string, unknown>
    const result: Record<string, unknown> = {}
    let same = Object.keys(old).length === Object.keys(incoming).length
    for (const key of Object.keys(incoming)) {
      result[key] = reconcile(old[key], incoming[key])
      same = same && Object.hasOwn(old, key) && result[key] === old[key]
    }
    return same ? previous : result as T
  }
  return next
}

function updatePath(value: unknown, path: Path, incoming: unknown, remove = false): unknown {
  if (!path.length || path.some(key => ['__proto__', 'prototype', 'constructor'].includes(key))) throw new Error(message("console.invalid_state_path"))
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error(message("console.state_resync_required"))
  const old = value as Record<string, unknown>, [key, ...rest] = path
  if (!rest.length) {
    if (remove) {
      if (!Object.hasOwn(old, key)) return value
      const result = { ...old }; delete result[key]; return result
    }
    const item = reconcile(old[key], incoming)
    return Object.hasOwn(old, key) && item === old[key] ? value : { ...old, [key]: item }
  }
  if (!Object.hasOwn(old, key)) throw new Error(message("console.state_path_missing"))
  const item = updatePath(old[key], rest, incoming, remove)
  return item === old[key] ? value : { ...old, [key]: item }
}

export function applyFrame(previous: StreamState, frame: StreamFrame): StreamState {
  if (frame.protocol !== 2 || !Number.isSafeInteger(frame.revision) || frame.revision < 1) throw new Error(message("console.incompatible_state_protocol"))
  if (frame.kind === 'snapshot') {
    return { revision: frame.revision, state: reconcile(previous.state, frame.state) }
  }
  if (!previous.state) throw new Error(message("console.initial_snapshot_missing"))
  if (frame.kind === 'heartbeat') {
    if (frame.revision !== previous.revision) throw new Error(message("console.state_version_resync_required"))
    return previous
  }
  if (frame.kind !== 'delta' || frame.base !== previous.revision || frame.revision <= frame.base) throw new Error(message("console.discontinuous_state_version"))
  let next: unknown = previous.state
  for (const path of frame.remove) next = updatePath(next, path, undefined, true)
  for (const [path, value] of frame.set) next = updatePath(next, path, value)
  if (frame.history) {
    const old = previous.state.gpu_history
    const h = frame.history
    if (h.mode === 'append' && (!old.length || old.at(-1)!.stamp !== h.after)) throw new Error(message("console.history_cursor_resync_required"))
    const points = h.mode === 'reset' ? h.points.slice(-180) : [...old, ...h.points].slice(-180)
    next = updatePath(next, ['gpu_history'], points)
  }
  return { revision: frame.revision, state: next as Snapshot }
}

export const streamUrl = (hidden: boolean) => `/api/events?protocol=2&hidden=${hidden ? 'true' : 'false'}`
export const pageSchedule = (hidden: boolean) => ({ reconnectMs: hidden ? 10000 : 3000, staleMs: hidden ? 15000 : 8000 })
